"""
DispersionPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「dispersion image のプレゼンテーション既定値の
canonical owner」と位置付け、`title` の安定既定値の宣言をここで一元化する。
描画ボディ自体は引き続き backend (`_dispersion_image_impl`) 側に残す。

互換性方針:
- 公開シグネチャ (positional 引数列) は据え置き
- wrapper 側の既定値・物理量計算 (nyquist_k / d_maxdiff 再構築) には触れない
- backend `_dispersion_image_impl` の本体・mode 分岐ロジックは動かさない
- backend 固有オプション (cmap / figsize / aspect / interpolation 等) は
  引き続き `**kw` 経由で受け渡す

Step 3 対象外（risky / 純粋表示既定値ではない）:
- `mode` は backend 内で 3 通りの分岐 (xlabel / ylabel / 補助線方程式) を
  制御する mode selection logic であり、presentation default ではないため対象外
- `nyquist_k` / `d_maxdiff` は wrapper が物理量から算出して渡す data input
  (補助線位置の決定に直接使用) であり対象外
- `ax_x` / `ax_y` は mode 依存の物理座標境界 (extent 構築用) であり対象外
- `xlabel` / `ylabel` / `colorbar` は backend 描画ボディ内でハードコード
  されており (xlabel/ylabel は mode 依存)、シグネチャ昇格には backend ボディの
  変更が必要なため対象外
- `cmap` / `figsize` / `aspect` / `interpolation` は backend 固有
  (Matplotlib `_default_imshow_kw` で解決) なので `**kw` に温存
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_TITLE = None


class DispersionPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        input: np.ndarray,
        ax_x: list[float],
        ax_y: list[float],
        axis: str,
        show: bool,
        save_name: str,
        nyquist_k: float,
        d_maxdiff: float,
        mode: str,
        *,
        title: str | None = _DEFAULT_TITLE,
        **kw,
    ):
        return self._backend._dispersion_image_impl(
            input,
            ax_x=ax_x,
            ax_y=ax_y,
            axis=axis,
            show=show,
            save_name=save_name,
            nyquist_k=nyquist_k,
            d_maxdiff=d_maxdiff,
            mode=mode,
            title=title,
            **kw,
        )
