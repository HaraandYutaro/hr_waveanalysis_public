"""ToyForwardSolver — engine / misfit 単体検証専用の placeholder。

NOTE
----
**これは正規の Rayleigh 波 forward ではない。**
Step ?-1a の責務は engine と misfit の数値挙動を依存ゼロで検証可能にすることに
限定されており、Toy はそのための疑似 forward である。

物理的妥当性が必要な解析には Step ?-1b で導入される ThomsonHaskellSolver を
利用すること。Toy はフィールドデータ解析には絶対に使用しない。

挙動の概要 (deterministic, smooth, Vs-monotonic):
  各周波数 f で、深さ z における重み
      w_j(f) = exp(-k_eff * z_top_j) - exp(-k_eff * z_bot_j),
      k_eff  = 2*pi*f / vs_avg
  を用いて Vs_j の重み付き平均を取り、c(f) = 0.92 * weighted_mean(vs) を返す。

  - 高周波 → 浅層が支配的
  - 低周波 → 深層まで波及
  - 単純で滑らか、Vs に対し単調
  - 中心差分 Jacobian と damped LSQ で安定して逆解析できる
"""

from __future__ import annotations

import numpy as np

from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.model import LayeredEarthModel

_RAYLEIGH_VS_RATIO = 0.92  # ν=0.25 の半空間で c_R / vs ≈ 0.9194


class ToyForwardSolver(RayleighForwardSolver):
    """engine / misfit の単体検証専用 forward (NOT for production)."""

    def forward(
        self,
        model: LayeredEarthModel,
        freqs: np.ndarray,
        mode: int = 0,
    ) -> np.ndarray:
        if mode != 0:
            raise NotImplementedError(
                "ToyForwardSolver は基本モード (mode=0) のみサポート"
            )
        freqs = np.asarray(freqs, dtype=float)
        vs = model.vs
        h = model.h

        # 各層の上端深さと下端深さ。最下層は半空間 → 下端を +inf に。
        z_top = np.concatenate([[0.0], np.cumsum(h[:-1])])
        z_bot = z_top + h
        # 半空間の inf により exp(-k*inf) = 0 となるよう、numpy の inf 挙動に依拠。

        vs_avg = float(np.mean(vs))
        k_eff = 2.0 * np.pi * freqs / vs_avg  # shape=(m,)

        # w[j, k] = exp(-k_eff[k] * z_top[j]) - exp(-k_eff[k] * z_bot[j])
        # shape=(n_layers, m)
        w = np.exp(-np.outer(z_top, k_eff)) - np.exp(-np.outer(z_bot, k_eff))
        w_sum = w.sum(axis=0)
        # 数値的に w_sum が 0 に潰れる (極端な低周波 + 浅い構造) ケースのみ守る。
        w_sum = np.where(w_sum <= 0.0, 1.0, w_sum)

        weighted_vs = (vs[:, None] * w).sum(axis=0) / w_sum  # shape=(m,)
        return _RAYLEIGH_VS_RATIO * weighted_vs
