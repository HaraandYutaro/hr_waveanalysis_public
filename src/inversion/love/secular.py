"""Love-wave secular (characteristic) function for the determinant misfit.

Niching PSO (Zhang et al. 2023, DOI:10.1093/gji/ggac380) requires a
``D(c, f, model)`` evaluator whose zero-crossings are the roots of the
dispersion equation. The function VALUE itself (not just its roots) is
needed because the determinant misfit (Eq. 10 of PDF1) sums the magnitude
of D at the observed (f_i, v_i^obs) points without performing root search.

For SH-Love waves in a layered medium with N finite layers + halfspace,
the standard Thomson-Haskell propagator yields a scalar secular function:

    D(c, omega, model) = T_zy_bot + mu_N * gamma_N * v_bot

where (v_bot, T_zy_bot) is the displacement-stress state at the top of the
halfspace, propagated from the surface BC (T_zy = 0 at z=0), and
``mu_N * gamma_N * v_bot`` is the downward-radiating SH impedance of the
halfspace (Haskell 1953; Aki & Richards 2002 §7.2).

Vertical wavenumber per layer:

    k_z^2 = omega^2 * (1/beta_m^2 - 1/c^2)

  - If c > beta_m (homogeneous, propagating SH within the layer):
        E_m = [[ cos(k_z h),         sin(k_z h) / (mu_m k_z) ],
               [-mu_m k_z sin(k_z h), cos(k_z h)              ]]
  - If c < beta_m (evanescent within the layer):
        E_m = [[ cosh(kappa h),         sinh(kappa h) / (mu_m kappa) ],
               [ mu_m kappa sinh(kappa h), cosh(kappa h)              ]]
        with kappa = sqrt(-k_z^2).

Halfspace radiation condition (only downward-decaying SH for proper Love
modes; requires c < beta_N):

    T_zy + mu_N * kappa_N * v = 0  at z = z_top_of_halfspace

so the secular function is:

    D = M[1,0] * v0 + mu_N * kappa_N * M[0,0] * v0
      = (M[1,0] + mu_N * kappa_N * M[0,0]) * v0

with v0 the unknown surface displacement (set to 1 for evaluation).

If c >= beta_N (leaky regime), no proper Love mode exists; we return
``np.nan`` so that the determinant misfit naturally penalizes that region.

References
----------
- Haskell, N. A. (1953). "The dispersion of surface waves on multilayered
  media." Bull. Seismol. Soc. Am., 43, 17-34.
- Aki, K. & Richards, P. G. (2002). "Quantitative Seismology", 2nd ed.,
  University Science Books. §7.2.
  [出典注意: 本教科書は library/refs に未収録 (refs は PDF 論文のみ)。
   本 secular 定式化は refs 収録の Haskell (1953) からも導出可能であり、
   出典確保まで式は変更しない (保留)。Plan.md P2-G 参照。]
- Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380. (Determinant misfit
  formulation for surface wave inversion.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.inversion.rayleigh.model import LayeredEarthModel


def _layer_propagator_love(
    omega: float,
    c: float,
    beta: float,
    mu: float,
    h: float,
) -> Optional[np.ndarray]:
    """単層の Love propagator 行列 E_m を返す。

    Returns
    -------
    ndarray, shape=(2, 2) or None
        h が +inf または非有限の場合は None (halfspace は別処理)。

    Notes
    -----
    数値オーバーフロー回避のため、evanescent regime (c < beta) でも
    cosh/sinh を直接使用する。深部 / 高周波では非常に大きな値になり得るが、
    本関数は単純な PSO 評価用であり、PDF1 の modified Thomson-Haskell
    (stiffness matrix recursion) の数値安定化までは導入しない。
    厚い層 + 高周波 + 高 evanescent ratio の組合せで overflow した場合は
    numpy が +inf / NaN を返す。呼び出し側は ``np.isfinite`` で判定すること。
    """
    if not np.isfinite(h):
        return None
    inv_c2 = 1.0 / (c * c)
    inv_beta2 = 1.0 / (beta * beta)
    kz2 = (omega * omega) * (inv_beta2 - inv_c2)
    if kz2 >= 0.0:
        # propagating within layer (c > beta_m)
        kz = float(np.sqrt(kz2))
        if kz * h < 1e-30:
            # near-zero argument; fall back to identity-like
            return np.array(
                [[1.0, h / mu if mu > 0 else 0.0], [0.0, 1.0]],
                dtype=float,
            )
        cs = float(np.cos(kz * h))
        sn = float(np.sin(kz * h))
        denom = mu * kz
        return np.array(
            [[cs, sn / denom], [-denom * sn, cs]],
            dtype=float,
        )
    # evanescent within layer (c < beta_m)
    kappa = float(np.sqrt(-kz2))
    arg = kappa * h
    # clip for numerical safety (cosh(700) ~ 1e304)
    if arg > 700.0:
        # huge evanescent attenuation — surface decouples from halfspace.
        # Returning an array with large values; caller treats NaN/inf gracefully.
        ch = float(np.cosh(700.0))
        sh = float(np.sinh(700.0))
    else:
        ch = float(np.cosh(arg))
        sh = float(np.sinh(arg))
    denom = mu * kappa
    return np.array(
        [[ch, sh / denom], [denom * sh, ch]],
        dtype=float,
    )


def love_secular_value(
    c: float,
    f: float,
    model: LayeredEarthModel,
) -> float:
    """Love 波 secular 関数値 ``D(c, f, model)`` を返す。

    Parameters
    ----------
    c : float
        位相速度 [m/s], > 0.
    f : float
        周波数 [Hz], > 0.
    model : LayeredEarthModel
        層構造モデル。``model.vp`` は本関数では参照しない (Love は P 波非感応)。

    Returns
    -------
    float
        secular 関数値。物理単位は混合系 (応力 / 変位次元) だが、
        determinant misfit (PDF1 Eq. 10) は |D| の和を取るため、
        layered model 全体で一貫した正規化があれば misfit 比較として機能する。
        ``c >= halfspace Vs`` (leaky regime) の場合は ``np.nan`` を返す。
        数値オーバーフローで非有限値が出た場合も ``np.nan`` を返す。

    Notes
    -----
    - 規約: 表層 displacement v_0 = 1 を仮定。これにより D = M[1,0] + Z_h * M[0,0]
      (Z_h = mu_N * kappa_N) を返す。
    - PSO 内 loop で大量呼び出しされる想定で、numpy のスカラ演算で実装。
      ベクトル化は将来スライスで導入。
    """
    if not (np.isfinite(c) and np.isfinite(f) and c > 0.0 and f > 0.0):
        return float("nan")

    omega = 2.0 * np.pi * f
    n = int(model.n_layers)

    # Cumulative propagator from surface down to top-of-halfspace
    M = np.eye(2, dtype=float)
    for m in range(n - 1):  # finite layers only
        beta_m = float(model.vs[m])
        rho_m = float(model.rho[m])
        h_m = float(model.h[m])
        if beta_m <= 0.0 or rho_m <= 0.0 or h_m <= 0.0:
            return float("nan")
        mu_m = rho_m * beta_m * beta_m
        E_m = _layer_propagator_love(omega, c, beta_m, mu_m, h_m)
        if E_m is None:
            return float("nan")
        M = E_m @ M
        if not np.all(np.isfinite(M)):
            return float("nan")

    # Halfspace
    beta_N = float(model.vs[-1])
    rho_N = float(model.rho[-1])
    if beta_N <= 0.0 or rho_N <= 0.0:
        return float("nan")
    if c >= beta_N:
        # leaky regime: no proper trapped Love mode at this c
        return float("nan")
    mu_N = rho_N * beta_N * beta_N
    inv_c2 = 1.0 / (c * c)
    inv_betaN2 = 1.0 / (beta_N * beta_N)
    # For proper Love modes c < beta_N  =>  k_zN^2 = omega^2 * (1/beta_N^2 - 1/c^2) < 0
    # Evanescent decay rate in halfspace:  kappa_N = sqrt(-k_zN^2) = omega * sqrt(1/c^2 - 1/beta_N^2)
    # (sign convention: positive real kappa_N giving outgoing-decay condition
    #  T_zy + mu_N * kappa_N * v = 0 at z = z_top_of_halfspace).
    kappa_N = float(np.sqrt((omega * omega) * (inv_c2 - inv_betaN2)))

    # Surface BC: T_zy(0) = 0  =>  initial state vector [v0, 0] with v0 = 1
    # State at top-of-halfspace: [v_bot, T_bot] = M @ [1, 0] = [M[0,0], M[1,0]]
    # Radiation: T_bot + mu_N * kappa_N * v_bot = 0
    D = M[1, 0] + mu_N * kappa_N * M[0, 0]
    if not np.isfinite(D):
        return float("nan")
    return float(D)


def love_secular_values(
    cs: np.ndarray,
    fs: np.ndarray,
    model: LayeredEarthModel,
) -> np.ndarray:
    """``love_secular_value`` のベクトル版 (要素ごと評価)。

    Parameters
    ----------
    cs : ndarray, shape=(m,)
        位相速度 [m/s]。
    fs : ndarray, shape=(m,)
        周波数 [Hz]。``cs`` と同じ長さ。

    Returns
    -------
    ndarray, shape=(m,)
        各点での secular 関数値。leaky / 失敗点は NaN。
    """
    cs = np.asarray(cs, dtype=float)
    fs = np.asarray(fs, dtype=float)
    if cs.shape != fs.shape or cs.ndim != 1:
        raise ValueError(
            f"cs and fs must be 1D and share shape; got {cs.shape}, {fs.shape}"
        )
    out = np.empty(cs.shape, dtype=float)
    for i in range(cs.size):
        out[i] = love_secular_value(float(cs[i]), float(fs[i]), model)
    return out
