"""Forward solvers for Rayleigh-wave dispersion.

``ThomsonHaskellSolver`` は disba 依存のため、**遅延 import** で公開する。
これにより:
  - disba 未インストール環境でも本パッケージの import 自体は成功する
    (``from src.inversion.rayleigh.forward import ToyForwardSolver`` は通る)
  - ``ThomsonHaskellSolver`` を実際にアクセスしたときだけ disba を import し、
    無い場合のみ ImportError を案内付きで送出する
"""

from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.forward.toy import ToyForwardSolver

__all__ = ["RayleighForwardSolver", "ToyForwardSolver", "ThomsonHaskellSolver"]


def __getattr__(name):
    if name == "ThomsonHaskellSolver":
        from src.inversion.rayleigh.forward.thomson_haskell import ThomsonHaskellSolver
        return ThomsonHaskellSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
