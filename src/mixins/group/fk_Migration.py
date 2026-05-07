"""
waveanalysis.py から分離した FK（Stolt）マイグレーションに関するクラス。
kirchhoff_MG.py と同一の呼び出し規約・スタイルで記述。

Reference:
  Stolt, R.H. (1978). Migration by Fourier transform.
  Geophysics, 43(1), 23-48.
"""

import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class Fk_Migration(PlotterWrapperMixin):
    def FK_migration(
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
        show_nmo: bool = False,
        velocity_mode: str = "rms",
        v_const: float | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        pad_t: int = 2,
        pad_x: int = 2,
        mute_evanescent: bool = True,
        jacobian_weight: bool = True,
        backend: str = "mpl",
        title: str = "FK Depth Migration",
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
        NMO補正済みCMP断面を使って Post-stack FK（Stolt）深度マイグレーションを行う。

        Parameters
        ----------
        axis : str
            データ軸 ('x', 'y', 'z')
            'y' → SH 波スカラー FK。  'x','z' → 将来的に P-SV 拡張予定。
        v_min, v_max, n_vel : float, float, int
            速度推定用試行速度範囲とサンプル数
        gate_len, gate_step : float
            速度推定窓 [s]
        depth_vel : float
            時刻→深度変換速度 [m/s]（NMO_correction 内で使用）
        vs_model : array-like, shape (nz_vs,)
            1D SH 波速度モデル [m/s]。
            FK マイグレーションに使う代表速度は velocity_mode で決定。
        z_max : float
            マイグレーション最大深度 [m]
        dz : float
            深度サンプリング [m]
        save_name : str | Path | None
            保存先パス。None / '' のとき保存しない
        show : bool
            プロットを表示するか
        show_nmo : bool
            NMO 断面を表示するか
        velocity_mode : str
            FK に使う代表速度の算出方法。
            'rms'   : RMS 速度（推奨）
            'mean'  : 算術平均
            'median': 中央値
            'const' : v_const を直接使用
        v_const : float | None
            velocity_mode='const' のときの速度 [m/s]
        fmin : float | None
            マイグレーション周波数下限 [Hz]。None → DC
        fmax : float | None
            マイグレーション周波数上限 [Hz]。None → Nyquist
        pad_t, pad_x : int
            時間軸・CMP 軸のゼロパディング倍率（2^n に丸め）
        mute_evanescent : bool
            True のとき evanescent 領域（|ω| < v|kx|）をゼロミュート
        jacobian_weight : bool
            True のとき Stolt Jacobian（obliquity factor）v²kz/ω を乗算
        backend : str
            描画バックエンド ('mpl' or 'plotly')
        title : str
            プロットタイトル
        cmap : str
            カラーマップ
        figsize : tuple
        aspect : str
        xmin, xmax, ymin, ymax : float | None
        num_ticks : int

        Returns
        -------
        image : ndarray, shape (nx, nz_img)
            深度マイグレーション後のイメージ
        z_img : ndarray, shape (nz_img,)
            深度軸 [m]
        cmp_pos : ndarray, shape (nx,)
            CMP 水平位置 [m]
        targets_sorted : list
            ソート済み CMP ターゲットキー
        """
        image, z_img, cmp_pos, targets_sorted = self._FK_migration_core(
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
            velocity_mode,
            v_const,
            fmin,
            fmax,
            pad_t,
            pad_x,
            mute_evanescent,
            jacobian_weight,
        )

        # 3) 描画 / 保存 ─── PlotterWrapperMixin.reflection_image に委譲
        #    深度軸を負の elevation として渡すことで depth-down 表示を実現
        if show or save_name:
            self._ensure_plotter()
            depth_as_elev = -z_img  # 0（地表）→ -z_max（最大深度）
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

    def _FK_migration_core(
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
        velocity_mode,
        v_const,
        fmin,
        fmax,
        pad_t,
        pad_x,
        mute_evanescent,
        jacobian_weight,
    ):
        """
        FK_migration() の純粋計算部 (描画なし)。

        Section 1: NMO_correction で CMP 重合 + NMO 補正
        Section 1.5: NaN → 0 に置換 (表面マスク部分が NaN のため)
        Section 2: fk_migration_from_NMO で FK 深度マイグレーション

        Returns
        -------
        image          : np.ndarray, 深度マイグレーション後イメージ (nx, nz_img)
        z_img          : np.ndarray, 深度軸 [m]
        cmp_pos        : np.ndarray, CMP 水平位置 [m]
        targets_sorted : np.ndarray, ソート済み CMP target キー
        """
        # 1) CMP 重合 & NMO 補正 ─── Kirchhoff 版と完全共通
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

        # NaN → 0  （表面マスク部分が NaN のため）
        stack_horiz_clean = np.nan_to_num(stack_horiz, nan=0.0)

        # 2) FK 深度マイグレーション
        image, z_img = self.fk_migration_from_NMO(
            stack_horiz_clean,
            cmp_pos,
            dt,
            np.asarray(vs_model, dtype=float),
            z_max,
            dz,
            velocity_mode=velocity_mode,
            v_const=v_const,
            fmin=fmin,
            fmax=fmax,
            pad_t=pad_t,
            pad_x=pad_x,
            mute_evanescent=mute_evanescent,
            jacobian_weight=jacobian_weight,
        )

        return image, z_img, cmp_pos, targets_sorted

    # ------------------------------------------------------------------
    @staticmethod
    def fk_migration_from_NMO(
        stack_horiz: np.ndarray,
        cmp_pos: np.ndarray,
        dt: float,
        vs_model: np.ndarray,
        z_max: float,
        dz: float,
        velocity_mode: str = "rms",
        v_const: float | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        pad_t: int = 2,
        pad_x: int = 2,
        mute_evanescent: bool = True,
        jacobian_weight: bool = True,
    ):
        """
        SH-wave Post-stack FK（Stolt）深度マイグレーション (2D, zero-offset).

        Parameters
        ----------
        stack_horiz : ndarray, shape (nx, nt)
            NMO 補正後のゼロオフセット断面。
            横軸 = CMP 位置インデックス, 縦軸 = two-way time サンプル。
        cmp_pos : ndarray, shape (nx,)
            CMP の水平位置 [m]。
        dt : float
            時間サンプリング [s]。
        vs_model : ndarray, shape (nz_vs,)
            深度方向の SH 波速度モデル [m/s]。1D。
        z_max : float
            出力イメージの最大深度 [m]。
        dz : float
            出力深度サンプリング [m]。
        velocity_mode : str
            FK に使う代表速度の算出方法。
            'rms' | 'mean' | 'median' | 'const'
        v_const : float | None
            velocity_mode='const' のときの速度 [m/s]。
        fmin : float | None
            周波数下限 [Hz]。None → DC (0 Hz)。
        fmax : float | None
            周波数上限 [Hz]。None → Nyquist。
        pad_t, pad_x : int
            ゼロパディング倍率 (2^n に丸め)。
        mute_evanescent : bool
            True のとき evanescent 領域（ω < v|kx|）をゼロに。
        jacobian_weight : bool
            True のとき Stolt Jacobian v²kz/ω で振幅補正。

        Returns
        -------
        image : ndarray, shape (nx, nz_img)
            深度マイグレーション後のイメージ。
        z_img : ndarray, shape (nz_img,)
            深度軸 [m]。

        Notes
        -----
        アルゴリズム概要（Stolt 1978）:

          (1)  D(x, t)  --2D-FFT-->  D̃(kx, ω)
          (2)  分散関係 ω = v√(kx² + kz²) を使って
               (kx, ω) → (kx, kz) スペクトルマッピング（線形補間）
          (3)  Jacobian  J = v²kz / ω（obliquity factor）で振幅補正
          (4)  Ĩ(kx, kz) --2D-IFFT-->  I(x, z)

        定速度近似を使っているため、vs_model の代表値（RMS など）を
        velocity_mode で選択する。勾配の大きい速度構造には Stolt stretch
        や V(z) FK 拡張が必要（将来実装）。
        """
        stack_horiz = np.asarray(stack_horiz, dtype=float)
        cmp_pos = np.asarray(cmp_pos, dtype=float)
        vs_model = np.asarray(vs_model, dtype=float)

        nx, nt = stack_horiz.shape

        # --- 代表マイグレーション速度の決定 ---
        if velocity_mode == "const":
            if v_const is None:
                raise ValueError(
                    "velocity_mode='const' のとき v_const を指定してください。"
                )
            v_mig = float(v_const)
        elif velocity_mode == "rms":
            v_mig = float(np.sqrt(np.mean(vs_model**2)))
        elif velocity_mode == "mean":
            v_mig = float(np.mean(vs_model))
        elif velocity_mode == "median":
            v_mig = float(np.median(vs_model))
        else:
            raise ValueError(
                f"velocity_mode='{velocity_mode}' は未対応です。"
                " 'rms' / 'mean' / 'median' / 'const' から選んでください。"
            )

        # --- CMP 間隔の推定 ---
        if nx < 2:
            raise ValueError(
                "エラー。NMO補正には最低でも対応するCMP点が2つ以上必要です。"
            )
        dx = float(np.median(np.abs(np.diff(cmp_pos))))

        # --- 周波数軸 ---
        fs = 1.0 / dt
        nfft_t = int(2 ** np.ceil(np.log2(nt * pad_t)))
        nfft_x = int(2 ** np.ceil(np.log2(nx * pad_x)))

        omega_axis = 2.0 * np.pi * np.fft.rfftfreq(nfft_t, d=dt)  # shape (nfft_t//2+1,)
        kx_axis = 2.0 * np.pi * np.fft.fftfreq(nfft_x, d=dx)  # shape (nfft_x,)

        f_axis = np.fft.rfftfreq(nfft_t, d=dt)  # [Hz]（フィルタ用）
        fmin_used = 0.0 if fmin is None else float(fmin)
        fmax_used = f_axis[-1] if fmax is None else min(float(fmax), f_axis[-1])

        # --- 深度軸 (kz > 0 のみ使用) ---
        nz_img = int(np.floor(z_max / dz)) + 1
        z_img = np.arange(nz_img, dtype=float) * dz
        kz_pos = 2.0 * np.pi / (2.0 * dz) * np.linspace(0.0, 1.0, nz_img)
        # ↑ kz を 0 ≤ kz ≤ π/dz の等間隔グリッドとして定義

        # --- 2D FFT ---
        data_pad = np.zeros((nfft_x, nfft_t), dtype=float)
        data_pad[:nx, :nt] = stack_horiz
        FK = np.fft.rfft2(data_pad, axes=(0, 1))  # shape (nfft_x, nfft_t//2+1)

        # 周波数バンドパス
        band_mask = (f_axis >= fmin_used) & (f_axis <= fmax_used)
        FK[:, ~band_mask] = 0.0

        # --- Stolt マッピング: (kx, ω) → (kx, kz) ---
        # FK_mapped[ikx, ikz] = FK(kx, ω(kx, kz)) × Jacobian
        FK_mapped = np.zeros((nfft_x, nz_img), dtype=np.complex128)

        for ikx, kx_i in enumerate(kx_axis):
            # kz グリッドに対応する ω を計算
            omega_mapped = v_mig * np.sqrt(kx_i**2 + kz_pos**2)

            # evanescent 領域のマスク（ω < v|kx| の成分はゼロ）
            if mute_evanescent:
                prop_mask = omega_mapped >= v_mig * abs(kx_i) - 1e-12
            else:
                prop_mask = np.ones(nz_img, dtype=bool)

            # Nyquist を超える成分は補間できないのでマスク
            prop_mask &= omega_mapped <= omega_axis[-1]

            if not np.any(prop_mask):
                continue

            omega_tgt = omega_mapped[prop_mask]

            # スペクトルを ω 軸上で線形補間（実部・虚部を独立に）
            re_interp = np.interp(
                omega_tgt, omega_axis, FK[ikx, :].real, left=0.0, right=0.0
            )

            im_interp = np.interp(
                omega_tgt, omega_axis, FK[ikx, :].imag, left=0.0, right=0.0
            )
            spec_mapped = re_interp + 1j * im_interp

            # Jacobian（obliquity factor）: J = v² kz / ω
            if jacobian_weight:
                kz_tgt = kz_pos[prop_mask]
                om_safe = np.maximum(omega_tgt, 1e-12)
                jac = (v_mig**2) * kz_tgt / om_safe
                spec_mapped *= jac

            FK_mapped[ikx, prop_mask] = spec_mapped

        # --- 2D 逆 FFT → 深度イメージ ---
        # FK_mapped: (nfft_x, nz_img) を ifft2 で (x, z) へ
        image_xz = np.fft.ifft2(FK_mapped, axes=(0, 1)).real  # shape (nfft_x, nz_img)
        image = image_xz[:nx, :]  # shape (nx, nz_img)

        return image, z_img
