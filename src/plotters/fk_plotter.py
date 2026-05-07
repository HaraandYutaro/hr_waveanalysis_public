"""
FkPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「f-k スペクトル画像のプレゼンテーション既定値の
canonical owner」と位置付け、`title` の安定既定値の宣言をここで一元化する。
描画ボディ自体は引き続き backend (`fk_image_impl`) 側に残す。

互換性方針:
- 公開シグネチャ (positional 引数列) は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `fk_image_impl` の本体は動かさない
- backend 固有オプション (cmap / figsize / aspect / interpolation 等) は
  引き続き `**kw` 経由で受け渡す

Step 3 対象外（risky / 純粋表示既定値ではない）:
- `kxmin` / `kxmax` / `fmin` / `fmax` は backend 内で extent 構築と
  軸範囲適用 (xlim/ylim) の二重役割を担う物理座標境界であり、
  presentation default ではないため定数化しない
- `xlabel` / `ylabel` / `colorbar` は backend 描画ボディ内でハードコード
  されており、シグネチャ昇格には backend ボディの変更が必要なため対象外
- `cmap` / `figsize` / `aspect` / `interpolation` は backend 固有
  (Matplotlib `_default_imshow_kw` で解決) なので `**kw` に温存
"""

from src.plotting.backend_base import PlotterBase

_DEFAULT_TITLE = None


class FkPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        axis,
        res,
        kxmin,
        kxmax,
        fmin,
        fmax,
        analysis,
        name,
        show,
        save_name,
        *,
        title: str | None = _DEFAULT_TITLE,
        **kw,
    ):
        return self._backend.fk_image_impl(
            axis,
            res,
            kxmin,
            kxmax,
            fmin,
            fmax,
            analysis,
            name,
            show,
            save_name,
            title=title,
            **kw,
        )
