"""
waveanalysis.pyから分離したキルヒホッフマイグレーションに関するクラス
"""

import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class Kirchhoff_MG(PlotterWrapperMixin):
    def Kirchhoff_migration(
        self,
        axis,
        v_min,
        v_max,
        n_vel,
        gate_len,
        gate_step,
        depth_vel,
        vs_model,
        z_max,
        dz,
        save_name,
        show,
        show_nmo=False,
        aperture_half: float | None = None,
        amplitude_weight: bool = True,
        backend: str = "mpl",
        title: str = "Kirchhoff Depth Migration",
        cmap: str = "gray",
        figsize: tuple = (10, 6),
        aspect: str = "auto",
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        num_ticks: int = 8,
    ):
        """
        NMO補正済みCMP断面を使って Post-stack Kirchhoff 深度マイグレーションを行う。

        Parameters
        ----------
        axis : str
            データ軸 ('x', 'y', 'z')
        v_min, v_max, n_vel : 速度推定用試行速度範囲とサンプル数
        gate_len, gate_step : 速度推定窓 [s]
        depth_vel : 時刻→深度変換速度 [m/s]
        vs_model : array-like, shape (nz_vs,)
            1D SH波速度モデル [m/s]
        z_max : float
            マイグレーション最大深度 [m]
        dz : float
            深度サンプリング [m]
        save_name : str | Path | None
            保存先パス。None / '' のとき保存しない
        show : bool
        show_nmo : bool
            NMO断面を表示するか
        aperture_half : float | None
            マイグレーションアパーチャ半幅 [m]
        amplitude_weight : bool
            幾何発散・傾斜補正を行うか
        backend : str
            描画バックエンド ('mpl' or 'plotly')
        """
        image, z_img, cmp_pos, targets_sorted = self._Kirchhoff_migration_core(
            axis,
            v_min,
            v_max,
            n_vel,
            gate_len,
            gate_step,
            depth_vel,
            vs_model,
            z_max,
            dz,
            show_nmo,
            aperture_half,
            amplitude_weight,
        )

        # 3) 描画 / 保存 — PlotterWrapperMixin.reflection_image に委譲
        #    深度軸を負の elevation として渡すことで depth-down 表示を実現
        if show or save_name:
            self._ensure_plotter()
            depth_as_elev = -z_img  # 0(地表) → -z_max(最大深度)
            self.reflection_image(
                image,
                cmp_pos,
                depth_as_elev,
                targets_sorted,
                title=title,
                num_ticks=num_ticks,
                cmap=cmap,
                figsize=figsize,
                aspect=aspect,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                save_name=save_name,
                show=show,
            )

        return image, z_img, cmp_pos, targets_sorted

    def _Kirchhoff_migration_core(
        self,
        axis,
        v_min,
        v_max,
        n_vel,
        gate_len,
        gate_step,
        depth_vel,
        vs_model,
        z_max,
        dz,
        show_nmo,
        aperture_half,
        amplitude_weight,
    ):
        """
        Kirchhoff_migration() の純粋計算部 (描画なし)。

        Section 1: NMO_correction で CMP 重合 + NMO 補正
        Section 1.5: NaN → 0 に置換 (表面マスク部分が NaN のため)
        Section 2: kirchhoff_migration_from_NMO で Kirchhoff 深度マイグレーション

        Returns
        -------
        image          : np.ndarray, 深度マイグレーション後イメージ (nx, nz_img)
        z_img          : np.ndarray, 深度軸 [m]
        cmp_pos        : np.ndarray, CMP 水平位置 [m]
        targets_sorted : np.ndarray, ソート済み CMP target キー
        """
        # 1) CMP重合&NMO補正
        stack_horiz, cmp_pos, elev_axis, targets_sorted = self.NMO_correction(
            axis,
            v_min,
            v_max,
            n_vel,
            gate_len,
            gate_step,
            depth_vel,
            criterion=2,
            show_est_vel=False,
            show=show_nmo,
        )

        dt = 1.0 / self.fs

        # NaN → 0 (表面マスク部分がNaNのため)
        stack_horiz_clean = np.nan_to_num(stack_horiz, nan=0.0)

        # 2) Kirchhoff 深度マイグレーション
        image, z_img = Kirchhoff_MG.kirchhoff_migration_from_NMO(
            stack_horiz_clean,
            cmp_pos,
            dt,
            np.asarray(vs_model, dtype=float),
            z_max,
            dz,
            aperture_half=aperture_half,
            amplitude_weight=amplitude_weight,
        )

        return image, z_img, cmp_pos, targets_sorted

    @staticmethod
    def kirchhoff_migration_from_NMO(
        stack_horiz: np.ndarray,
        cmp_pos: np.ndarray,
        dt: float,
        vs_model: np.ndarray,
        z_max: float,
        dz: float,
        elev_datum: float = 0.0,
        aperture_half: float | None = None,
        amplitude_weight: bool = True,
    ):
        """
        SH-wave Post-stack Kirchhoff depth migration (2D, zero-offset).

        Parameters
        ----------
        stack_horiz : (nx, nt)
                Elev+NMO 補正後のゼロオフセット断面 (stack_horiz)。
                横軸 = CMP 位置 index, 縦軸 = two-way time サンプル
        cmp_pos : (nx,)
                CMP の水平位置 [m]。
        dt : float
                時間サンプリング [s]。
        vs_model : (nz_vs,)
                深度方向の SH 波速度モデル [m/s]。1D とする。
        z_max : float
                出力イメージの最大深度 [m]。
        dz : float
                出力深度サンプリング [m]。
        elev_datum : float, optional
                Datum (0 m) とする高さ。既に shift_to_surface 済みなら 0 でOK。
        aperture_half : float, optional
                マイグレーションアパーチャ半幅 [m]。None なら全 CMP を使用。
        amplitude_weight : bool
                2D 幾何発散・傾斜補正を行うかどうか。

        Returns
        -------
        image : (nx, nz_img)
                深度マイグレーション後のイメージ。
        z_img : (nz_img,)
                深度軸 [m]。
        """
        stack_horiz = np.asarray(stack_horiz, float)
        cmp_pos = np.asarray(cmp_pos, float)
        vs_model = np.asarray(vs_model, float)

        nx, nt = stack_horiz.shape
        t_max = (nt - 1) * dt

        # --- 深度軸と Vs(z) の補間関数 ---
        nz_vs = len(vs_model)
        z_vs = np.linspace(0.0, z_max, nz_vs)

        def vs_func(z):
            z = np.clip(z, 0.0, z_max)
            return np.interp(z, z_vs, vs_model)

        nz_img = int(np.floor(z_max / dz)) + 1
        z_img = np.arange(nz_img) * dz

        # 時間微分 (∂P/∂t)
        d_stack = np.zeros_like(stack_horiz)
        d_stack[:, 1:-1] = (stack_horiz[:, 2:] - stack_horiz[:, :-2]) / (2.0 * dt)
        d_stack[:, 0] = (stack_horiz[:, 1] - stack_horiz[:, 0]) / dt
        d_stack[:, -1] = (stack_horiz[:, -1] - stack_horiz[:, -2]) / dt

        image = np.zeros((nx, nz_img), float)
        dx_med = np.median(np.abs(np.diff(cmp_pos))) if nx > 1 else 1.0
        t_axis = np.arange(nt) * dt

        for ix_m in range(nx):
            x_m = cmp_pos[ix_m]

            # アパーチャ制限
            if aperture_half is None:
                mask = np.ones(nx, bool)
            else:
                mask = np.abs(cmp_pos - x_m) <= aperture_half

            x_src = cmp_pos[mask]
            idx_src = np.where(mask)[0]

            for iz, z in enumerate(z_img):
                if z == 0.0:
                    continue

                vs_z = vs_func(z)

                # 片道走時 tau_i = r_i / Vs(z)
                dx = x_m - x_src
                dz_geom = z  # datum=0 ならこれでOK
                r = np.sqrt(dx**2 + dz_geom**2)
                tau = r / vs_z  # one-way time
                t_d = 2.0 * tau  # two-way time

                # stack_horiz から t_d に対応するサンプルを線形補間
                t_idx = t_d / dt
                valid = (t_idx >= 0.0) & (t_idx < nt - 1)
                if not np.any(valid):
                    continue

                it0 = t_idx[valid].astype(int)
                frac = t_idx[valid] - it0
                i_src = idx_src[valid]

                amp = (
                    d_stack[i_src, it0] * (1.0 - frac) + d_stack[i_src, it0 + 1] * frac
                )

                if amplitude_weight:
                    r_v = r[valid]
                    dz_v = dz_geom
                    # obliquity: cos θ = dz / r
                    cos_th = np.where(r_v > 0.0, dz_v / r_v, 0.0)
                    w = cos_th / np.sqrt(r_v + 1e-12)
                    amp *= w

                image[ix_m, iz] += np.sum(amp) * dx_med

        image /= 2.0 * np.pi
        return image, z_img
