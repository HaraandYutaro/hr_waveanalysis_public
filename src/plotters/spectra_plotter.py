"""
SpectraPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「周波数スペクトル画像プロットのプレゼンテーション既定値の
canonical owner」と位置付け、title など見た目に関する安定既定値の宣言を
ここで一元化する。描画ボディ自体は引き続き backend 側に残す。

互換性方針:
- 公開シグネチャ・既定値は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `_spectra_image_impl` の本体は動かさない
- backend 固有オプション (cmap / interpolation / aspect / figsize 等) は
  引き続き `**kw` 経由で受け渡す

注意: 現在 PlotterBase.spectra_image() (公開メソッド) 経由で委譲しており、
cmap / interpolation のデフォルト値注入は PlotterBase 側で行われる。
そのため _impl ではなく spectra_image() を呼ぶ。

Step 3 対象外（backend 本体内ハードコード / データ変換フラグ）:
- xlabel / ylabel は _spectra_image_impl 内でハードコードされており、
  昇格には backend ボディ変更が必要なため対象外
- log はデータ変換フラグであり、プレゼンテーション既定値ではないため対象外
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_TITLE = None


class SpectraPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        res: np.ndarray,
        freq: list[float],
        axis: str,
        title: str | None = _DEFAULT_TITLE,
        show: bool | None = False,
        save_name: str | None = None,
        log: bool | None = False,
        **kw,
    ):
        return self._backend.spectra_image(
            res,
            freq=freq,
            axis=axis,
            title=title,
            show=show,
            save_name=save_name,
            log=log,
            **kw,
        )