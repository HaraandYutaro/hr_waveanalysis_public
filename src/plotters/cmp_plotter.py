"""
CmpPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「CMP gather プロットのプレゼンテーション既定値の
canonical owner」と位置付け、color / plot_mode など見た目に関する安定既定値の
宣言をここで一元化する。描画ボディ自体は引き続き backend 側に残す。

互換性方針:
- 公開シグネチャ・既定値は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `_cmp_impl` の本体は動かさない
- backend 固有オプション (figsize / spacing 等) は backend 本体内で
  ハードコードされており `**kw` 経由ではないため対象外

注意: 対象ループ・save_name接尾辞生成・mode互換性処理等の
オーケストレーションは PlotterWrapperMixin.cmp_image 側に残す。
本クラスは1回の対象に対する描画呼び出しのみを担当する。

Step 3 対象外（backend 本体内ハードコード）:
- xlabel / ylabel / title — _cmp_impl 内でハードコードされており、
  昇格には backend ボディ変更が必要なため対象外
- spacing / figsize — 計算パラメータまたは matplotlib 固有の描画設定であり、
  プレゼンテーション既定値ではないため対象外
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_COLOR = "black"
_DEFAULT_PLOT_MODE = "fill"


class CmpPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        show,
        fs,
        data: np.ndarray,
        off,
        target_cmp,
        t: list[float],
        *,
        save_name: str | None = None,
        color: str = _DEFAULT_COLOR,
        plot_mode: str = _DEFAULT_PLOT_MODE,
    ):
        return self._backend._cmp_impl(
            show,
            fs,
            data,
            off,
            target_cmp,
            t,
            save_name=save_name,
            color=color,
            plot_mode=plot_mode,
        )