"""verify_env.py — uv 環境構築の検証スクリプト.

`uv run python verify_env.py` で実行する。

以下を検証する:
  1. Python / プラットフォーム情報
  2. 核心依存 (numpy, scipy, matplotlib, pandas, openpyxl, numba, plotly,
     disba, PyQt6) と optional 依存 (dash) のインポートとバージョン
  3. スモークテスト (FFT / scipy.signal / numba JIT / disba 分散曲線)
  4. 自作モジュールのインポート (SingleProcesser, GroupProcesser,
     PlotProcesser, SEG2Reader, utils)
  5. GPU (cupy) の有無確認 — 本プロジェクトの必須依存ではないため任意

終了コード: すべて成功で 0、いずれか失敗で 1。
"""

import importlib
import os
import platform
import sys
import traceback

# --- ヘッドレス環境対策 -----------------------------------------------------
# 本番コード (src/plotting/backends/matplotlib_backend.py) は import 時に
# matplotlib.use("qtagg") を呼ぶ。ディスプレイの無い環境でも import 検証が
# できるよう、先に Agg を固定して use() を無効化する (conftest.py と同じ手法)。
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.use = lambda *a, **kw: None  # freeze backend

_failures = []


def _ok(msg):
    print(f"  [OK]   {msg}")


def _fail(msg, exc=None):
    print(f"  [FAIL] {msg}")
    if exc is not None:
        traceback.print_exc()
    _failures.append(msg)


def section(title):
    print(f"\n=== {title} ===")


# --- 1. 環境情報 ------------------------------------------------------------
section("1. Environment")
print(f"  Python      : {sys.version.splitlines()[0]}")
print(f"  Executable  : {sys.executable}")
print(f"  Platform    : {platform.platform()}")


# --- 2. 依存ライブラリのインポート ------------------------------------------
section("2. Core dependency imports")
core_mods = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("numba", "numba"),
    ("plotly", "plotly"),
    ("disba", "disba"),
    ("PyQt6.QtCore", "PyQt6"),
]
for import_name, label in core_mods:
    try:
        m = importlib.import_module(import_name)
        ver = getattr(m, "__version__", None)
        if ver is None and import_name == "PyQt6.QtCore":
            ver = getattr(m, "PYQT_VERSION_STR", "?")
        _ok(f"{label:<12} {ver if ver else ''}")
    except Exception as e:  # noqa: BLE001
        _fail(f"import {label}", e)

section("2b. Optional (viz) dependency imports")
for import_name, label in [("dash", "dash")]:
    try:
        m = importlib.import_module(import_name)
        _ok(f"{label:<12} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        _fail(f"import {label} (optional)", e)


# --- 3. スモークテスト ------------------------------------------------------
section("3. Smoke tests")

try:
    import numpy as np

    fs = 1000.0
    t = np.arange(0, 1.0, 1 / fs)
    sig = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(sig.size, 1 / fs)
    peak = freqs[np.argmax(np.abs(spec))]
    assert 45 < peak < 55, f"unexpected FFT peak: {peak} Hz"
    _ok(f"numpy FFT: dominant frequency = {peak:.1f} Hz (expected ~50)")
except Exception as e:  # noqa: BLE001
    _fail("numpy FFT smoke test", e)

try:
    from scipy import signal

    b, a = signal.butter(4, 0.2, btype="low")
    filtered = signal.filtfilt(b, a, sig)
    assert filtered.shape == sig.shape
    _ok("scipy.signal Butterworth filter")
except Exception as e:  # noqa: BLE001
    _fail("scipy.signal smoke test", e)

try:
    from numba import njit

    @njit(cache=False)
    def _sum_sq(arr):
        acc = 0.0
        for v in arr:
            acc += v * v
        return acc

    val = _sum_sq(np.arange(10, dtype=np.float64))
    assert abs(val - 285.0) < 1e-9
    _ok("numba njit compile & execute")
except Exception as e:  # noqa: BLE001
    _fail("numba JIT smoke test", e)

try:
    import numpy as np
    from disba import PhaseDispersion

    # 2 層モデル: thickness(km), vp(km/s), vs(km/s), rho(g/cm3)
    velocity_model = np.array(
        [
            [0.01, 1.00, 0.50, 1.8],
            [0.05, 1.80, 0.90, 1.9],
            [0.00, 2.50, 1.30, 2.0],
        ]
    )
    periods = np.linspace(0.05, 0.5, 10)
    pd = PhaseDispersion(*velocity_model.T)
    cp = pd(periods, mode=0, wave="rayleigh")
    assert np.all(np.isfinite(cp.velocity))
    _ok(
        "disba Rayleigh phase dispersion: "
        f"c(T={periods[0]:.2f}s) = {cp.velocity[0]:.3f} km/s"
    )
except Exception as e:  # noqa: BLE001
    _fail("disba dispersion smoke test", e)


# --- 4. 自作モジュールのインポート ------------------------------------------
section("4. Project module imports")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
project_mods = [
    "src.processor.single_processor",
    "src.processor.group_processor",
    "src.processor.plot_processor",
    "src.converter.seg2",
    "src.utils.utils",
    "src.inversion.rayleigh.forward.thomson_haskell",
    "src.inversion.love.forward.thomson_haskell_love",
]
for name in project_mods:
    try:
        importlib.import_module(name)
        _ok(name)
    except Exception as e:  # noqa: BLE001
        _fail(f"import {name}", e)

# 主要クラスが取り出せるか
try:
    from src.processor.single_processor import SingleProcesser  # noqa: F401
    from src.processor.group_processor import GroupProcesser  # noqa: F401
    from src.converter.seg2 import SEG2Reader  # noqa: F401

    _ok("SingleProcesser / GroupProcesser / SEG2Reader symbols accessible")
except Exception as e:  # noqa: BLE001
    _fail("project class symbols", e)


# --- 5. GPU (任意) ----------------------------------------------------------
section("5. GPU (cupy) — optional, not a project dependency")
try:
    import cupy  # noqa: F401

    n = cupy.cuda.runtime.getDeviceCount()
    _ok(f"cupy present, CUDA device count = {n}")
except Exception:  # noqa: BLE001
    print("  [SKIP] cupy not installed / no CUDA — CPU-only run is expected here")


# --- 結果 -------------------------------------------------------------------
section("Result")
if _failures:
    print(f"  ❌ {len(_failures)} check(s) FAILED:")
    for f in _failures:
        print(f"     - {f}")
    sys.exit(1)
else:
    print("  ✅ All environment checks passed.")
    sys.exit(0)
