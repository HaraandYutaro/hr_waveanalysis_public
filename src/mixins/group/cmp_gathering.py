"""
CMP gatheringモジュール

GroupProcesser に継承させるための Mixin。
single 側の BackscatterAnalysis(PlotterWrapperMixin) と同じ構成パターンに従う。

Public API:
    cmp_gathering() — cmp_gather() へのエイリアス。後方互換のため維持。
    cmp_gather()    — CMP 集約のメインエントリ
    show()          — CMP ギャザーの描画
    get_all_target_height() — CMP ターゲット位置の標高補間

TODO: show() / _plot_cmp() の描画を PlotterWrapperMixin に CMP 用メソッド
      (cmp_image 等) として追加し、CmpGathering から委譲する
TODO: get_all_target_height() が肥大化したら CmpGeometry mixin に分割する
"""

from __future__ import annotations

import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class CmpGathering(PlotterWrapperMixin):
    """
    CMP (Common Midpoint) データ集約 Mixin。

    GroupProcesser に多重継承させることで group.cmp_gathering() 等が使えるようになる。
    single 側の BackscatterAnalysis と同様に PlotterWrapperMixin を継承し、
    描画の初期化パターン (self.init_plotter, self._plotter) に準拠する。
    """

    def cmp_gathering(
        self,
        axis: str,
        closs_corr: bool = False,
        integrate: bool = False,
        average: bool = True,
        show: bool = True,
        save_name: str | None = None,
        t: list[float] = [0, 99],
        plot_mode: str = "fill",
        **kw,
    ):
        """
        Perform CMP (Common Midpoint) gathering on the loaded data.

        Parameters
        ----------
        axis : {"x", "y", "z"}
            Analysis axis used to extract waveform data from each record.
        closs_corr : bool, default False
            If True, apply cross-correlation relative to the source channel
            before gathering.
        integrate : bool, default False
            If True, convert half-offsets to their absolute values, merging
            forward and backward offset groups.
        average : bool, default True
            If True, average waveforms that share the same half-offset within
            each CMP bin.
        show : bool, default True
            If True, display the CMP gather plot after computation.
        save_name : str or None, optional
            Path used to save the figure.  If None, the figure is not saved.
        t : list of float, default [0, 99]
            Time window [t_start, t_end] in seconds displayed in the plot.
        plot_mode : str, default "fill"
            Waveform rendering style passed to the CMP plotter
            (e.g. ``"fill"`` or ``"wiggle"``).

        Notes
        -----
        Sets the following instance attributes upon return:
        ``self.cmp`` — dict mapping each CMP target position to a
        ``(n_traces, n_samples)`` waveform array;
        ``self.offsets`` — dict mapping each target to a ``(n_traces,)``
        half-offset array;
        ``self.targets`` — sorted array of unique CMP midpoint positions;
        ``self.src_h``, ``self.rec_h`` — dicts of source and receiver
        surface heights per target (populated only when ``z_distance`` is
        available on the underlying records);
        ``self.axis``, ``self.interval``, ``self.fs``,
        ``self.analysis_name``.

        Raises
        ------
        ValueError
            If *axis* is not present in all records of ``self.datalist``.
        """

        # backward compat
        if "time_range" in kw:
            t = kw.pop("time_range")
        if "show_mode" in kw:
            plot_mode = kw.pop("show_mode")

        # 1) Check axis
        if not self.axis_bool[axis]:
            raise ValueError(
                f"Error. some of your data don't have axis{axis} component."
            )

        # 2) Reference data for shape and analysis name
        self.analysis_axis, self.analysis_name = self.datalist[0].getax_analysis(axis)
        self.analysis_name += "_CMP"
        self.axis = axis

        # 3) Gather unique CMP targets and their trace-counts
        self.targets = self._collect_targets(self.datalist)
        counts = self._count_per_target(self.datalist, self.targets)

        # 4) Build CMP data dictionaries
        (self.cmp, self.offsets, self.src_h, self.rec_h) = self._build_cmp_dicts(
            self.datalist, self.targets, counts, closs_corr
        )

        # 5) Optionally integrate and average offsets
        if integrate:
            self._integrate_offsets()
        if average:
            self._average_duplicates()

        # 6) Store metadata
        self.axis = axis
        self.interval = 2 * self.datalist[0].interval
        self.fs = self.datalist[0].fs

        if show or save_name:
            self._ensure_plotter()
            self.cmp_image(
                self.cmp,
                self.offsets,
                self.fs,
                self.targets,
                t,
                show=show,
                save_name=save_name,
                color=kw.get("color", "black"),
                plot_mode=plot_mode,
            )

    def _collect_targets(self, datallist: list) -> np.ndarray:
        """
        Collect unique CMP midpoint positions from all Hr2c data.
        Returns sorted 1D array of targets (float).
        """
        all_targets = np.concatenate([hr.get_all_CMP() for hr in datallist])
        return np.unique(all_targets)

    def _count_per_target(
        self, datallist: list, targets: np.ndarray
    ) -> dict[float, int]:
        """
        Count number of traces per CMP target across datallist.
        Returns dict mapping target -> count.
        """
        counts = {t: 0 for t in targets}
        for hr in datallist:
            for t in hr.get_all_CMP():
                counts[t] += 1
        return counts

    def _build_cmp_dicts(
        self,
        datallist: list,
        targets: np.ndarray,
        counts: dict[float, int],
        closs_corr: bool,
    ) -> tuple[dict, dict, dict, dict]:
        """
        Extract waveforms, half-offsets, and heights per CMP target.
        Returns four dicts:
          cmp_dict[t]: ndarray (n_tr, n_samp)
          off_dict[t]: ndarray (n_tr,)
          src_dict[t]: ndarray (n_tr,)
          rec_dict[t]: ndarray (n_tr,)
        """
        cmp_dict, off_dict, src_dict, rec_dict = {}, {}, {}, {}

        if closs_corr:
            wavelet = self.analysis_axis[datallist[0].get_source_ch()]
            for hr in datallist:
                hr.closs_corr(axis=self.axis, wavelet=wavelet)

        n_samp = datallist[0].getax_analysis(self.axis)[0].shape[1]

        for t in targets:
            n = counts[t]
            wf = np.zeros((n, n_samp))
            off = np.zeros(n)
            sh = np.zeros(n)
            rh = np.zeros(n)

            idx = 0
            for hr in datallist:
                cmps = hr.get_all_CMP()
                if t in cmps:
                    i = int(np.where(cmps == t)[0][0])
                    d = hr.get_all_CMPdist()[i]
                    data_ax, _ = hr.getax_analysis(self.axis)
                    wf[idx] = data_ax[i, :n_samp]
                    off[idx] = d
                    if hasattr(hr, "z_distance"):
                        sh[idx] = np.mean(
                            [
                                hr.z_distance[hr.get_source_ch()],
                                hr.z_distance[hr.get_source_ch() - 1],
                            ]
                        )
                        rh[idx] = hr.z_distance[i]
                    idx += 1

            cmp_dict[t] = wf
            off_dict[t] = off
            src_dict[t] = sh
            rec_dict[t] = rh

        return cmp_dict, off_dict, src_dict, rec_dict

    def _integrate_offsets(self) -> None:
        """
        Convert half-offsets to absolute values for integration.
        """
        for t, off in self.offsets.items():
            self.offsets[t] = np.abs(off)

    def _average_duplicates(self) -> None:
        """
        average waveforms with identical offsets.
        """
        for t, off in self.offsets.items():
            wf = self.cmp[t]
            norms = np.max(np.abs(wf), axis=1, keepdims=True)
            wf_norm = wf / (norms + 1e-12)

            unique_off, inv = np.unique(off, return_inverse=True)
            order = np.argsort(inv)
            wf_sorted = wf_norm[order]
            inv_sorted = inv[order]

            _, counts = np.unique(inv_sorted, return_counts=True)
            idx = np.concatenate(([0], np.cumsum(counts)[:-1]))

            sums = np.add.reduceat(wf_sorted, idx, axis=0)
            avg_wf = sums / counts[:, None]

            self.cmp[t] = avg_wf
            self.offsets[t] = unique_off

    def get_all_target_height(self) -> np.ndarray:
        """
        Compute linear interpolation of surface height at each CMP target.
        Returns array of heights matching self.targets.
        """
        hr0 = self.datalist[0]
        dist = hr0.distance
        zdist = (
            hr0.z_distance
            if hasattr(self, "z_distance")
            else np.zeros_like(hr0.distance)
        )

        if hasattr(self, "targets"):
            heights = np.zeros(len(self.targets))
        else:
            raise ValueError(
                "Error. target height must be called after method 'cmp_gathering'."
            )

        for i, t in enumerate(self.targets):
            idx1 = np.argmin(np.abs(dist - t))
            if dist[idx1] >= t and idx1 > 0:
                idx2 = idx1 - 1
            elif dist[idx1] < t and idx1 < len(dist) - 1:
                idx2 = idx1 + 1
            else:
                idx2 = idx1
            d1, d2 = dist[idx1], dist[idx2]
            h1, h2 = zdist[idx1], zdist[idx2]
            denom = d2 - d1 or 1e-6
            heights[i] = h1 + (h2 - h1) * (t - d1) / denom

        return heights


if __name__ == "__main__":
    pass
