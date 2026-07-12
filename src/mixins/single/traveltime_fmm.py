"""
src/mixins/single/traveltime_fmm.py

2D first-arrival traveltime solver based on the Fast Marching Method (FMM).

This module is the **forward-solver counterpart** of the Dijkstra / Shortest-
Path-Method (SPM) implementation already provided by
``src/mixins/single/traveltime_tomography.py`` (specifically the
``_solve_spm_2d`` and ``_solve_forward`` methods of the
``TraveltimeTomography`` mixin).

Purpose
-------
- Provide a self-contained ``TraveltimeFmm2D`` class that numerically solves
  the 2D Eikonal equation ``|∇T(x,z)|^2 = s(x,z)^2`` for SH body-wave
  first-arrival traveltimes on a Cartesian grid in physical ``[x, z]``
  coordinates (z positive downward, depth-below-surface).
- Coexist with — not replace — the existing SPM forward solver. Future
  integration into ``TraveltimeTomography.traveltime_tomography(...)`` is
  expected to be a single-keyword dispatch (``forward_method="dijkstra"``
  vs ``"fmm"``), preserving full backward compatibility by default.

Specification
-------------
The full theory, equations, node-state conventions, integration plan, and
phased roadmap are documented in:

    private/docs/tomography/fmm2d.md

Phase A (this file) implements:
- standard first-order 4-connected FMM on a flat free surface,
- Godunov upwind quadratic update with robust 1D fallback,
- heap-based narrow-band march with lazy stale-entry deletion,
- bilinear interpolation to sample the traveltime field at arbitrary receivers.

Phasing (mirrors fmm2d.md §5.2)
--------------------------------
- **Phase A** (implemented): flat free surface, first-order FMM (4-connected
  upwind). Returns the traveltime field only; sensitivity matrix raises
  ``NotImplementedError``.
- **Phase B** (TODO): heterogeneous (layered) velocity; same flat-surface.
- **Phase C** (implemented): undulating free surface via the column-shift
  convention used by ``_solve_spm_2d`` (``z_phys = topo_z[ix] + iz*dz``).
- **Phase D** (implemented): ray back-tracing along ``-∇T`` and assembly
  of the Fréchet (sensitivity) matrix as a ``scipy.sparse.csr_matrix``.
  Pipeline integration through ``_solve_forward`` dispatch is left as
  follow-up work.
- **Phase E** (TODO): MSFM (4-axis + 4-diagonal stencil), coarse-grid +
  hybrid (bilinear / cubic spline) interpolation (Sun 2025).
- **Phase F** (TODO): second-order MSFM upwind correction.

Coordinate and indexing conventions (must match the SPM solver)
---------------------------------------------------------------
- Slowness array shape: ``(nz, nx)``, dtype ``float64``, units ``s/m``.
- Row index ``iz=0`` is the topmost grid row (free surface for the flat
  case); ``iz`` increases downward.
- Column index ``ix=0`` is the leftmost column.
- Flat node id: ``node_id = iz * nx + ix``.
- Topography (when supplied in Phase C): ``topo_z[ix]`` is a per-column
  elevation offset in metres, **positive downward**, such that the physical
  z of grid node ``(iz, ix)`` is ``z_phys = topo_z[ix] + iz * dz``. This is
  the *column-shift* convention used by
  ``TraveltimeTomography._solve_spm_2d`` and is **not** the cut-cell
  surface representation of Sun 2025 §2.3.

References
----------
- Sun, H. et al. (2025). Topographic Seismic Travel-Time Calculation
  Based on Coarse-Grid Interpolation. Lithosphere 2025/158.
- Sethian, J. A. (1996). A fast marching level set method for
  monotonically advancing fronts. PNAS 93(4), 1591–1595.
- Hassouna, M. S. & Farag, A. A. (2007). MultiStencils fast marching
  methods. IEEE TPAMI 29(9), 1563–1574.
- Lan, H. & Zhang, Z. (2013). Topography-dependent eikonal equation and
  its solver for calculating first-arrival traveltimes with an irregular
  surface. GJI 193(2), 1010–1026.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple

import numpy as np
import scipy.sparse


# =============================================================================
# Module-level constants
# =============================================================================

#: Node state encoding. Stored in a ``(nz, nx) uint8`` array on the solver.
FAR: int = 0
TRIAL: int = 1
KNOWN: int = 2
MASKED: int = 3   # cell above the free surface (cut-cell mode; unused in Phase A–C)

#: Supported numerical schemes. Phase A implements ``"fmm"`` only.
_SCHEMES = ("fmm", "msfm")

#: Sentinel for unvisited / unreachable traveltimes.
_T_FAR: float = np.inf


# =============================================================================
# Lightweight grid container
# =============================================================================

@dataclass(frozen=True)
class Grid2D:
    """
    Minimal 2D Cartesian grid descriptor shared by the FMM solver and any
    future helper utilities.

    This dataclass intentionally mirrors a subset of the ``grid`` dict
    produced by ``TraveltimeTomography._make_grid`` (see
    ``src/mixins/single/traveltime_tomography.py``), so adapters between
    the two representations are trivial.

    Attributes
    ----------
    nx, nz : int
        Number of cells along x (horizontal) and z (depth-below-surface).
    dx, dz : float
        Cell spacings in metres along x and z.
    x0 : float, default 0.0
        Physical x-coordinate of the leftmost column (cell-center) in metres.
    z0 : float, default 0.0
        Physical z-coordinate of the topmost row (cell-center) in metres,
        ignoring topography. With topography (Phase C), the physical z of
        cell ``(iz, ix)`` is ``z0 + topo_z[ix] + iz*dz``.

    Notes
    -----
    - All consistency checks (positivity, minimum size) are performed in
      ``__post_init__`` so a malformed grid never reaches the solver.
    - ``frozen=True`` keeps grid instances hashable and prevents accidental
      mutation by the march routine.
    """

    nx: int
    nz: int
    dx: float
    dz: float
    x0: float = 0.0
    z0: float = 0.0

    def __post_init__(self) -> None:
        if int(self.nx) < 2 or int(self.nz) < 2:
            raise ValueError(
                f"Grid2D requires nx>=2 and nz>=2 (got nx={self.nx}, nz={self.nz})"
            )
        if float(self.dx) <= 0.0 or float(self.dz) <= 0.0:
            raise ValueError(
                f"Grid2D requires dx>0 and dz>0 (got dx={self.dx}, dz={self.dz})"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """``(nz, nx)`` — matches the convention used everywhere in the
        tomography pipeline."""
        return (int(self.nz), int(self.nx))

    @property
    def n_cells(self) -> int:
        return int(self.nx) * int(self.nz)

    @classmethod
    def from_dict(cls, grid: dict) -> "Grid2D":
        """
        Build a ``Grid2D`` from the ``grid`` dict produced by
        ``TraveltimeTomography._make_grid``.

        Parameters
        ----------
        grid : dict
            Must contain ``nx, nz, dx, dz`` and optionally ``x0, z0``.
        """
        return cls(
            nx=int(grid["nx"]),
            nz=int(grid["nz"]),
            dx=float(grid["dx"]),
            dz=float(grid["dz"]),
            x0=float(grid.get("x0", 0.0)),
            z0=float(grid.get("z0", 0.0)),
        )


# =============================================================================
# Main solver class
# =============================================================================

class TraveltimeFmm2D:
    """
    2D first-arrival traveltime solver based on the Fast Marching Method
    (Phase A: flat free surface, first-order 4-connected FMM).

    The solver discretizes the Eikonal equation

    .. math::
        \\left(\\frac{\\partial T}{\\partial x}\\right)^2
        + \\left(\\frac{\\partial T}{\\partial z}\\right)^2
        = s(x, z)^2

    on a regular Cartesian grid (constant ``dx``, ``dz``) and propagates
    the wavefront from a point source using a monotone min-heap, in the
    style of Sethian (1996). The resulting field ``T(iz, ix)`` is the
    first-arrival traveltime in seconds.

    The full algorithm, theoretical background, and integration plan are
    documented in ``private/docs/tomography/fmm2d.md``.

    Parameters
    ----------
    slowness : np.ndarray, shape=(nz, nx)
        Slowness model :math:`s = 1/v` in s/m. Must be strictly positive
        and finite. Row index ``iz=0`` is the topmost grid row.
    dx, dz : float
        Grid spacings in metres along x (column) and z (row).
    topo_z : np.ndarray of shape (nx,) or None, default None
        Free-surface elevation profile in metres (column-shift convention).
        Accepted by the constructor for forward compatibility but **not
        used in Phase A** (flat surface only). Phase C will activate this.
    scheme : {"fmm"}, default "fmm"
        Numerical scheme. Only ``"fmm"`` (standard 4-connected first-order
        Sethian FMM) is implemented in Phase A. ``"msfm"`` raises
        ``NotImplementedError`` until Phase E.
    second_order : bool, default False
        Must be ``False`` in Phase A. ``True`` raises ``NotImplementedError``
        until Phase F (Sun 2025 Eq. (9)–(11)).
    coarse_factor : int, default 1
        Must be ``1`` in Phase A. Values > 1 raise ``NotImplementedError``
        until Phase E (Sun 2025 coarse-grid interpolation).

    Attributes
    ----------
    slowness : np.ndarray, shape=(nz, nx)
        Stored slowness model (float64, read-only by convention).
    grid : Grid2D
        Grid descriptor deduced from ``slowness.shape``, ``dx``, ``dz``.
    topo_z : np.ndarray, shape=(nx,)
        Internally always populated (zeros when the user passed ``None``).
    scheme, second_order, coarse_factor :
        As passed to the constructor.

    Notes
    -----
    - **Phase C implemented: undulating topography** is supported via the
      column-shift convention — pass ``topo_z`` to the constructor.
      The flat-surface default (``topo_z=None``) is fully backward-compatible.
    - **Phase A limitation: first-order accuracy.** The first-order FMM
      exhibits a characteristic "rosette" error pattern along the grid
      diagonals (Sun 2025 Figure 5(a)). For higher accuracy, Phase E will
      add the MSFM diagonal stencil.
    - This class is designed to be **imported by**
      ``src/mixins/single/traveltime_tomography.py`` behind a keyword
      dispatch (``forward_method="fmm"`` in
      ``TraveltimeTomography.traveltime_tomography``). It is *not* a mixin;
      the existing SPM solver remains the default and is not modified.
    - Each call to ``compute_traveltime`` resets internal state and
      overwrites ``_T``. Copy the return value to retain the field across
      multiple source calls.

    Examples
    --------
    Basic Phase A usage (uniform velocity, flat surface)::

        import numpy as np
        v = 2000.0  # m/s
        nz, nx = 32, 64
        dz, dx = 2.0, 2.0  # m
        s = np.full((nz, nx), 1.0 / v)

        solver = TraveltimeFmm2D(slowness=s, dx=dx, dz=dz)
        T = solver.compute_traveltime(source_xz=(60.0, 0.0))  # shape (32, 64)

        # Sample at two surface receivers.
        tt = solver.traveltime_at(np.array([[0.0, 0.0], [120.0, 0.0]]))
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        slowness: np.ndarray,
        dx: float,
        dz: float,
        topo_z: Optional[np.ndarray] = None,
        scheme: Literal["fmm", "msfm"] = "fmm",
        second_order: bool = False,
        coarse_factor: int = 1,
    ) -> None:
        # --- validate all inputs before any allocation ---
        self._validate_inputs(
            slowness=slowness,
            dx=dx,
            dz=dz,
            topo_z=topo_z,
            scheme=scheme,
            second_order=second_order,
            coarse_factor=coarse_factor,
        )

        # Convert to a contiguous float64 copy so the march loop can index
        # into it cheaply and the caller's array is not mutated.
        self.slowness: np.ndarray = np.ascontiguousarray(slowness, dtype=np.float64)
        nz, nx = self.slowness.shape

        self.grid: Grid2D = Grid2D(nx=int(nx), nz=int(nz), dx=float(dx), dz=float(dz))

        if topo_z is None:
            self.topo_z: np.ndarray = np.zeros(int(nx), dtype=np.float64)
            self._has_topography: bool = False
        else:
            self.topo_z = np.ascontiguousarray(topo_z, dtype=np.float64).ravel()
            self._has_topography = bool(np.any(self.topo_z != 0.0))

        self.scheme: str = str(scheme)
        self.second_order: bool = bool(second_order)
        self.coarse_factor: int = int(coarse_factor)

        # --- preallocate state arrays (reset on each compute_traveltime call) ---
        # Traveltime field; _T_FAR (np.inf) marks unvisited / unreachable cells.
        self._T: np.ndarray = np.full(self.grid.shape, _T_FAR, dtype=np.float64)
        # Node states (FAR / TRIAL / KNOWN / MASKED).
        self._state: np.ndarray = np.full(self.grid.shape, FAR, dtype=np.uint8)
        # Lazy-deletion min-heap: entries are (t, counter, flat_node_id).
        self._heap: list = []
        # Monotonically increasing tie-breaker for deterministic heap ordering.
        self._push_counter: int = 0
        # Cached source grid indices, set on each compute_traveltime call.
        self._last_source_iz: Optional[int] = None
        self._last_source_ix: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute_traveltime(
        self,
        source_xz: Tuple[float, float],
    ) -> np.ndarray:
        """
        Compute the first-arrival traveltime field from a single point source.

        Solves the 2D Eikonal equation using the standard Sethian FMM
        (4-connected upwind, Godunov quadratic, lazy-deletion min-heap).

        Parameters
        ----------
        source_xz : (float, float)
            Source position ``(xs, zs)`` in physical metres. ``zs`` is the
            physical depth (positive downward). For Phase A (flat surface),
            a surface source has ``zs = 0.0``.

        Returns
        -------
        T : np.ndarray, shape=(nz, nx), dtype=float64
            Traveltime field in seconds. Cells that are unreachable from the
            source (e.g. disconnected regions) remain as ``np.inf``. The
            returned array is a reference to the internal buffer; copy it if
            you need to retain the field across multiple ``compute_traveltime``
            calls on the same solver instance.

        Raises
        ------
        ValueError
            If the projected source cell is outside the grid.

        Notes
        -----
        Phase A uses single-cell source initialization (``T[src] = 0``).
        Sun2025 page 2 mentions a wavefront-construction patch for higher
        accuracy near the source; this is a Phase B improvement.

        TODO (Phase E): when ``self.scheme == "msfm"``, also include the
        four diagonal neighbors via ``_update_node_msfm`` and take the
        minimum candidate per node (see fmm2d.md §1.3).

        TODO (Phase E): when ``self.coarse_factor > 1``, build a coarse
        slowness grid, run FMM on it, then interpolate to the fine grid
        using the hybrid bilinear/cubic-spline procedure of Sun 2025 §2.2.
        """
        # Reset all state arrays for a clean march.
        self._initialize_states()

        # Project source physical coordinates → nearest grid node.
        iz_src, ix_src = self._project_source_to_grid(source_xz)
        self._last_source_iz = int(iz_src)
        self._last_source_ix = int(ix_src)

        # Seed: source traveltime = 0, state immediately KNOWN.
        self._T[iz_src, ix_src] = 0.0
        self._state[iz_src, ix_src] = KNOWN

        # Run the FMM march and fill self._T.
        self._march(iz_src=int(iz_src), ix_src=int(ix_src))

        return self._T

    def traveltime_at(
        self,
        receiver_xz: np.ndarray,
    ) -> np.ndarray:
        """
        Sample the most recently computed traveltime field at receiver
        positions using bilinear interpolation.

        This is the canonical public method for extracting per-receiver
        traveltimes after ``compute_traveltime`` has been called.

        Parameters
        ----------
        receiver_xz : np.ndarray, shape=(n_rec, 2) or (2,), dtype=float
            Per-receiver ``(x, z)`` in physical metres. For Phase A
            (flat free surface), surface receivers have ``z = 0.0``.
            A single receiver may be supplied as a 1D array of shape ``(2,)``.

        Returns
        -------
        tt : np.ndarray, shape=(n_rec,), dtype=float64
            First-arrival traveltime at each receiver in seconds.
            Receivers that fall outside the grid or on unreachable cells
            are returned as ``np.inf``.

        Raises
        ------
        RuntimeError
            If called before ``compute_traveltime``.
        ValueError
            If ``receiver_xz`` has an unexpected shape.

        Notes
        -----
        Bilinear interpolation is applied within each grid cell using the
        column-shift convention (Phase A–C). For a surface receiver at column
        ``ix``, pass ``z = topo_z[ix]`` (0.0 for the flat-surface Phase A
        default).
        """
        return self.sample_traveltimes(receiver_xz)

    def sample_traveltimes(
        self,
        receiver_xz: np.ndarray,
    ) -> np.ndarray:
        """
        Sample the traveltime field at receiver positions (bilinear
        interpolation). Alias for ``traveltime_at``; kept for internal use
        and backward compatibility.

        Parameters
        ----------
        receiver_xz : np.ndarray, shape=(n_rec, 2) or (2,), dtype=float
            Per-receiver ``(x, z)`` in physical metres.

        Returns
        -------
        tt : np.ndarray, shape=(n_rec,), dtype=float64

        Raises
        ------
        RuntimeError
            If called before ``compute_traveltime``.
        ValueError
            If ``receiver_xz`` has an unexpected shape.
        """
        if self._last_source_ix is None:
            raise RuntimeError(
                "compute_traveltime must be called before sample_traveltimes."
            )

        rec = np.asarray(receiver_xz, dtype=np.float64)
        # Accept a single receiver as a 1D array of shape (2,).
        if rec.ndim == 1:
            if rec.shape[0] != 2:
                raise ValueError(
                    f"A single receiver must have shape (2,); got {rec.shape}"
                )
            rec = rec.reshape(1, 2)
        if rec.ndim != 2 or rec.shape[1] != 2:
            raise ValueError(
                f"receiver_xz must have shape (n_rec, 2) or (2,); got {rec.shape}"
            )

        n_rec = rec.shape[0]
        tt = np.empty(n_rec, dtype=np.float64)

        nz = self.grid.nz
        nx = self.grid.nx
        dx = self.grid.dx
        dz = self.grid.dz
        x0 = self.grid.x0
        z0 = self.grid.z0

        for k in range(n_rec):
            xr, zr = float(rec[k, 0]), float(rec[k, 1])

            # x-axis: fractional column index, clamped to [0, nx-2] so that
            # both topo_z[ix0] and topo_z[ix1] are always valid reads.
            xi_f = (xr - x0) / dx
            ix0 = int(np.floor(xi_f))
            ix0 = max(0, min(nx - 2, ix0))
            ix1 = ix0 + 1
            xi_w = max(0.0, min(1.0, xi_f - float(ix0)))

            # z-axis: column-shift convention (Phase A–C).
            # Physical z of grid node (iz, ix) = z0 + topo_z[ix] + iz*dz.
            # Linearly interpolate topo_z between the two bracketing columns
            # so that the fractional z-index is consistent with xi_w.
            topo_at_xr = (1.0 - xi_w) * self.topo_z[ix0] + xi_w * self.topo_z[ix1]
            zi_f = (zr - z0 - topo_at_xr) / dz
            iz0 = int(np.floor(zi_f))
            iz0 = max(0, min(nz - 2, iz0))
            iz1 = iz0 + 1
            zi_w = max(0.0, min(1.0, zi_f - float(iz0)))

            tt[k] = (
                (1.0 - zi_w) * (1.0 - xi_w) * self._T[iz0, ix0]
                + (1.0 - zi_w) * xi_w        * self._T[iz0, ix1]
                + zi_w         * (1.0 - xi_w) * self._T[iz1, ix0]
                + zi_w         * xi_w         * self._T[iz1, ix1]
            )

        return tt

    def sensitivity_matrix(
        self,
        receiver_xz: np.ndarray,
    ) -> scipy.sparse.csr_matrix:
        """
        Fréchet derivative matrix ``L``, shape ``(n_rec, nz*nx)``, such
        that ``L[r, iz*nx+ix]`` is the path length (in metres) of ray
        ``r`` through grid cell ``(iz, ix)``.

        Obtained a posteriori by ray back-tracing along ``-∇T`` from each
        receiver to the source, accumulating per-cell segment lengths.
        This satisfies the linearized traveltime equation

        .. math::
            \\delta t_r \\;=\\; \\sum_{\\text{cells}} L_{r,\\,c}\\,\\delta s_c .

        Parameters
        ----------
        receiver_xz : np.ndarray, shape=(n_rec, 2) or (2,)
            Per-receiver ``(x, z)`` in physical metres. A single receiver
            may be supplied as a 1D array of shape ``(2,)``.

        Returns
        -------
        L : scipy.sparse.csr_matrix, shape=(n_rec, nz*nx), dtype=float64
            Sparse sensitivity matrix in row-major (CSR) layout.

        Raises
        ------
        RuntimeError
            If called before ``compute_traveltime``.
        ValueError
            If ``receiver_xz`` has an unexpected shape.

        Notes
        -----
        Implementation outline (fmm2d.md §2.5 and §5.2 Phase D):
        1. Compute the upwind gradient field ``∇T`` once via
           ``_compute_gradient``.
        2. For each receiver, back-trace a ray to the source along
           ``-∇T / |∇T|`` using ``_trace_ray``.
        3. For each consecutive segment of the ray path, attribute its
           Euclidean length to the grid cell containing the segment
           midpoint (column-shift convention).
        4. Assemble the triplet list into a ``scipy.sparse.csr_matrix``.

        Unlike the SPM/Dijkstra solver, FMM does not store predecessor
        pointers — the ray geometry is reconstructed from ``∇T`` instead.
        """
        if self._last_source_ix is None:
            raise RuntimeError(
                "compute_traveltime must be called before sensitivity_matrix."
            )

        rec = np.asarray(receiver_xz, dtype=np.float64)
        if rec.ndim == 1:
            if rec.shape[0] != 2:
                raise ValueError(
                    f"A single receiver must have shape (2,); got {rec.shape}"
                )
            rec = rec.reshape(1, 2)
        if rec.ndim != 2 or rec.shape[1] != 2:
            raise ValueError(
                f"receiver_xz must have shape (n_rec, 2) or (2,); got {rec.shape}"
            )

        nrec = int(rec.shape[0])
        nz = self.grid.nz
        nx = self.grid.nx
        dx = self.grid.dx
        dz = self.grid.dz
        x0 = self.grid.x0
        z0 = self.grid.z0

        # Compute gradient once and cache for the inner _trace_ray calls.
        self._grad_T = self._compute_gradient()

        rows: list = []
        cols: list = []
        data: list = []

        try:
            for r in range(nrec):
                path = self._trace_ray((float(rec[r, 0]), float(rec[r, 1])))
                if path.shape[0] < 2:
                    continue

                # Walk consecutive segments; attribute length to midpoint cell.
                for k in range(path.shape[0] - 1):
                    x_a, z_a = float(path[k, 0]), float(path[k, 1])
                    x_b, z_b = float(path[k + 1, 0]), float(path[k + 1, 1])
                    seg_len = float(np.hypot(x_b - x_a, z_b - z_a))
                    if seg_len <= 0.0:
                        continue

                    mid_x = 0.5 * (x_a + x_b)
                    mid_z = 0.5 * (z_a + z_b)

                    # Midpoint → grid cell (column-shift convention).
                    xi_f = (mid_x - x0) / dx
                    ix_m = int(round(xi_f))
                    ix_m = max(0, min(nx - 1, ix_m))

                    # Linear topo interpolation between bracketing columns.
                    ix0 = int(np.floor(xi_f))
                    ix0 = max(0, min(nx - 2, ix0))
                    ix1 = ix0 + 1
                    xi_w = max(0.0, min(1.0, xi_f - float(ix0)))
                    topo_mid = (
                        (1.0 - xi_w) * self.topo_z[ix0]
                        + xi_w * self.topo_z[ix1]
                    )
                    zi_f = (mid_z - z0 - topo_mid) / dz
                    iz_m = int(round(zi_f))
                    iz_m = max(0, min(nz - 1, iz_m))

                    rows.append(r)
                    cols.append(iz_m * nx + ix_m)
                    data.append(seg_len)
        finally:
            # Drop the gradient cache so a subsequent ``compute_traveltime``
            # cannot leak a stale field into later ``_trace_ray`` calls.
            self._grad_T = None

        L = scipy.sparse.coo_matrix(
            (np.asarray(data, dtype=np.float64),
             (np.asarray(rows, dtype=np.int64),
              np.asarray(cols, dtype=np.int64))),
            shape=(nrec, nz * nx),
            dtype=np.float64,
        ).tocsr()
        return L

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_inputs(
        slowness: np.ndarray,
        dx: float,
        dz: float,
        topo_z: Optional[np.ndarray],
        scheme: str,
        second_order: bool,
        coarse_factor: int,
    ) -> None:
        """
        Validate constructor inputs.

        Raises ``ValueError`` for invalid physical inputs and
        ``NotImplementedError`` for valid but not-yet-implemented features.
        """
        s = np.asarray(slowness)
        if s.ndim != 2:
            raise ValueError(
                f"slowness must be 2D (nz, nx); got ndim={s.ndim}, shape={s.shape}"
            )
        nz, nx = s.shape
        if nz < 2 or nx < 2:
            raise ValueError(
                f"slowness must have nz>=2 and nx>=2; got shape={s.shape}"
            )
        if not np.all(np.isfinite(s)):
            raise ValueError("slowness must be finite everywhere.")
        if np.any(s <= 0.0):
            raise ValueError("slowness must be strictly positive everywhere.")

        if not (np.isfinite(dx) and dx > 0.0):
            raise ValueError(f"dx must be positive and finite; got dx={dx}")
        if not (np.isfinite(dz) and dz > 0.0):
            raise ValueError(f"dz must be positive and finite; got dz={dz}")

        if topo_z is not None:
            tz = np.asarray(topo_z)
            if tz.shape != (nx,):
                raise ValueError(
                    f"topo_z must have shape ({nx},); got {tz.shape}"
                )
            if not np.all(np.isfinite(tz)):
                raise ValueError("topo_z must be finite everywhere.")

        if scheme not in _SCHEMES:
            raise ValueError(
                f"scheme must be one of {_SCHEMES}; got {scheme!r}"
            )
        # Phase A: only the standard FMM stencil is implemented.
        if scheme == "msfm":
            raise NotImplementedError(
                "scheme='msfm' (MSFM diagonal stencil) is a Phase E TODO; "
                "see private/docs/tomography/fmm2d.md §1.3."
            )

        if not isinstance(second_order, (bool, np.bool_)):
            raise ValueError(
                f"second_order must be a bool; got {type(second_order).__name__}"
            )
        # Phase A/B/C: second-order upwind correction is not implemented.
        if second_order:
            raise NotImplementedError(
                "second_order=True (Sun 2025 Eq. (9)–(11)) is a Phase F TODO; "
                "see private/docs/tomography/fmm2d.md §1.3."
            )

        if not (isinstance(coarse_factor, (int, np.integer)) and int(coarse_factor) >= 1):
            raise ValueError(
                f"coarse_factor must be an integer >= 1; got {coarse_factor!r}"
            )
        if int(coarse_factor) != 1:
            # Phase A–D: only direct fine-grid solve is supported.
            raise NotImplementedError(
                "coarse_factor > 1 (Sun 2025 coarse-grid acceleration) is a "
                "Phase E TODO; see private/docs/tomography/fmm2d.md §1.6."
            )

    def _initialize_states(self) -> None:
        """
        Reset per-march arrays (``_T``, ``_state``, ``_heap``,
        ``_push_counter``) before a fresh ``compute_traveltime`` call.

        Notes
        -----
        TODO (Phase C / future cut-cell mode): when a cut-cell topography
        representation is introduced, populate ``self._state`` with MASKED
        for cells that lie above the free surface. The column-shift
        convention used through Phase C requires no masking here.
        See private/docs/tomography/fmm2d.md §1.5.
        """
        self._T.fill(_T_FAR)
        self._state.fill(FAR)
        self._heap.clear()
        self._push_counter = 0

    def _project_source_to_grid(
        self,
        source_xz: Tuple[float, float],
    ) -> Tuple[int, int]:
        """
        Project a physical source position to the nearest valid grid node.

        Parameters
        ----------
        source_xz : (xs, zs)
            Source in physical metres. ``zs`` positive downward.

        Returns
        -------
        (iz_src, ix_src) : (int, int)
            Grid indices, clipped to [0, nz-1] × [0, nx-1].

        Notes
        -----
        Mirrors ``TraveltimeTomography._project_to_grid_node`` but returns
        a (iz, ix) pair rather than the flat node id, for readability inside
        the march loop.
        """
        xs, zs = float(source_xz[0]), float(source_xz[1])

        ix = int(round((xs - self.grid.x0) / self.grid.dx))
        ix = max(0, min(self.grid.nx - 1, ix))

        # Column-shift convention (fmm2d.md §1.5):
        # physical z of (iz, ix) = topo_z[ix] + iz*dz + z0.
        local_z0 = float(self.grid.z0) + float(self.topo_z[ix])
        iz = int(round((zs - local_z0) / self.grid.dz))
        iz = max(0, min(self.grid.nz - 1, iz))

        return int(iz), int(ix)

    def _classify_nodes(self) -> None:
        """
        Classify grid nodes for topography-aware solvers (no-op in Phase A–C).

        Reserved for a future cut-cell topography representation (Sun 2025
        §2.3.1, Figure 3) in which surface-boundary cells need a distinct
        state from interior cells.

        TODO (future cut-cell mode): identify surface-boundary cells and
        mark them in ``self._state`` with a dedicated constant
        (e.g. ``SURFACE = 4``).
        """
        # Column-shift convention: iz=0 is always the free surface.
        # No nodes are above the surface; MASKED state is unused in Phase A–C.
        # Reserved for future cut-cell mode (Phase C-2 / fmm2d.md §1.5).
        return

    def _march(self, iz_src: int, ix_src: int) -> None:
        """
        Main Fast Marching loop: propagate traveltime from the source and
        finalize ``self._T`` for all reachable cells.

        Implements the standard Sethian FMM (Sethian 1996) with:
        - 4-connected (axis-aligned) stencil,
        - binary min-heap with lazy stale-entry deletion,
        - monotonic push counter as a heap tie-breaker.

        Parameters
        ----------
        iz_src, ix_src : int
            Grid indices of the already-initialized source node
            (``self._T[iz_src, ix_src] = 0``,
            ``self._state[iz_src, ix_src] = KNOWN``).

        Notes
        -----
        Algorithm (fmm2d.md §1.2):
        1. Seed the narrow band: compute initial traveltime estimates for
           the 4-connected neighbors of the source via ``_update_node``.
        2. While the heap is non-empty:
           a. Pop the minimum-traveltime node.
           b. Skip if already KNOWN (stale heap entry).
           c. Freeze as KNOWN.
           d. For each FAR or TRIAL neighbor, recompute via ``_update_node``
              and push if the new estimate improves the stored value.

        TODO (Phase E): when ``self.scheme == "msfm"``, also call
        ``_update_node_msfm`` for each neighbor and use the minimum of the
        two stencil candidates; see fmm2d.md §1.3.
        """
        nx = self.grid.nx

        # ---- Step 1: seed the narrow band from the source's neighbors ----
        for jz, jx in self._neighbors(iz_src, ix_src):
            if self._state[jz, jx] == MASKED:
                continue
            t_new = self._update_node(jz, jx)
            if t_new < self._T[jz, jx]:
                self._T[jz, jx] = t_new
                self._state[jz, jx] = TRIAL
                self._push_counter += 1
                heapq.heappush(
                    self._heap,
                    (t_new, self._push_counter, jz * nx + jx),
                )

        # ---- Step 2: drain the heap in traveltime order ----
        while self._heap:
            _, _, node_id = heapq.heappop(self._heap)
            iz = node_id // nx
            ix = node_id % nx

            # Lazy deletion: skip stale entries for already-frozen nodes.
            if self._state[iz, ix] == KNOWN:
                continue

            # Freeze this node: its traveltime is now final (causality).
            self._state[iz, ix] = KNOWN

            # Propagate to eligible 4-connected neighbors.
            for jz, jx in self._neighbors(iz, ix):
                if self._state[jz, jx] == KNOWN:
                    continue
                if self._state[jz, jx] == MASKED:
                    continue
                t_new = self._update_node(jz, jx)
                if t_new < self._T[jz, jx]:
                    self._T[jz, jx] = t_new
                    self._state[jz, jx] = TRIAL
                    self._push_counter += 1
                    heapq.heappush(
                        self._heap,
                        (t_new, self._push_counter, jz * nx + jx),
                    )

    def _update_node(self, iz: int, ix: int) -> float:
        """
        Compute a candidate first-arrival traveltime at grid node ``(iz, ix)``
        using the first-order Godunov upwind quadratic on the 4-connected
        stencil.

        Reads the current ``self._T`` field (which includes both KNOWN and
        TRIAL values) and delegates to ``_solve_quadratic``.

        Parameters
        ----------
        iz, ix : int
            Grid indices of the node to update.

        Returns
        -------
        t_new : float
            Candidate traveltime in seconds. The caller is responsible for
            accepting the value only if it improves ``self._T[iz, ix]``.

        Notes
        -----
        Out-of-bounds neighbors contribute ``np.inf``, which is handled
        naturally by the ``min`` in ``_solve_quadratic`` — they simply
        do not participate in the upwind update.
        """
        nz = self.grid.nz
        nx = self.grid.nx
        T = self._T

        # x-axis: minimum of left and right neighbors.
        a_left  = T[iz, ix - 1] if ix > 0     else _T_FAR
        a_right = T[iz, ix + 1] if ix < nx - 1 else _T_FAR
        a = min(a_left, a_right)

        # z-axis: minimum of up and down neighbors.
        b_up   = T[iz - 1, ix] if iz > 0     else _T_FAR
        b_down = T[iz + 1, ix] if iz < nz - 1 else _T_FAR
        b = min(b_up, b_down)

        return self._solve_quadratic(
            a, b, self.grid.dx, self.grid.dz, float(self.slowness[iz, ix])
        )

    def _update_node_msfm(self, iz: int, ix: int) -> float:
        """
        Candidate traveltime from the rotated (diagonal) MSFM stencil.

        Same quadratic form as ``_update_node`` but using the four
        diagonal neighbors ``(iz±1, ix±1)`` with effective spacing
        ``sqrt(dx**2 + dz**2)``.

        Notes
        -----
        TODO (Phase E): implement following Sun 2025 Eq. (7)–(8) and
        fmm2d.md §1.3. The MSFM final update per node is
        ``min(_update_node(iz, ix), _update_node_msfm(iz, ix))``.
        """
        # TODO (Phase E): see fmm2d.md §1.3.
        raise NotImplementedError(
            "_update_node_msfm is a Phase E TODO; "
            "see private/docs/tomography/fmm2d.md §1.3."
        )

    @staticmethod
    def _solve_quadratic(
        a: float,
        b: float,
        hx: float,
        hz: float,
        s_local: float,
    ) -> float:
        """
        Solve the 2D Godunov upwind quadratic for a single FMM node update.

        Given the minimum upwind x-axis traveltime ``a`` and z-axis
        traveltime ``b``, returns the causal (larger) root of

        .. math::
            \\left(\\frac{T - a}{h_x}\\right)^2 +
            \\left(\\frac{T - b}{h_z}\\right)^2 = s_{\\text{local}}^2 ,

        or falls back to the 1D update when only one axis has a finite
        upwind neighbor or when the discriminant is negative.

        Parameters
        ----------
        a : float
            Minimum upwind x-axis neighbor traveltime [s]. ``np.inf`` when
            no valid x-neighbor exists.
        b : float
            Minimum upwind z-axis neighbor traveltime [s]. ``np.inf`` when
            no valid z-neighbor exists.
        hx, hz : float
            Grid spacings [m] along the x and z axes.
        s_local : float
            Local slowness [s/m] at the node being updated.

        Returns
        -------
        t_new : float
            Candidate traveltime [s]. Returns ``np.inf`` if neither axis
            has a finite upwind neighbor.

        Notes
        -----
        **Quadratic setup** (fmm2d.md §1.1):
        Let ``A = 1/hx² + 1/hz²``, ``P = a/hx² + b/hz²``.
        The quadratic ``A·T² − 2P·T + (a²/hx² + b²/hz² − s²) = 0`` has
        discriminant ``Δ = P² − A·(a²/hx² + b²/hz² − s²)``.
        The larger root is ``T = (P + √Δ) / A``.

        Equivalently, ``Δ = s²·A − (a−b)²/(hx·hz)²``, so ``Δ ≥ 0`` iff
        ``|a−b| ≤ s·√(hx²+hz²)``. When ``Δ < 0``, the two-axis update is
        not causal and we fall back to the 1D formula.

        **1D fallback** (fmm2d.md §1.1, pitfall 5):
        ``T = min(a, b) + s · h_perp`` where ``h_perp = hx`` if ``a ≤ b``,
        else ``h_perp = hz``.

        This function is a pure helper with no side effects and is unit-
        testable in isolation (fmm2d.md §5.2 Phase A).
        """
        a_fin = np.isfinite(a)
        b_fin = np.isfinite(b)

        # ---- handle degenerate cases first ----
        if not a_fin and not b_fin:
            return _T_FAR

        if not a_fin:
            # Only z-axis available: 1D update.
            return float(b + s_local * hz)

        if not b_fin:
            # Only x-axis available: 1D update.
            return float(a + s_local * hx)

        # ---- both axes available: attempt 2D quadratic ----
        inv_hx2 = 1.0 / (hx * hx)
        inv_hz2 = 1.0 / (hz * hz)

        A = inv_hx2 + inv_hz2
        P = a * inv_hx2 + b * inv_hz2                     # centre of quadratic
        C = a * a * inv_hx2 + b * b * inv_hz2 - s_local * s_local

        disc = P * P - A * C                               # = s²·A − (a−b)²/(hx·hz)²

        if disc >= 0.0:
            t_new = (P + np.sqrt(disc)) / A
            # Causality check: result must not precede the upstream neighbors.
            # A tiny tolerance absorbs floating-point rounding.
            if t_new >= max(a, b) - 1e-10 * max(abs(a), abs(b), 1.0):
                return float(t_new)
            # If the 2D root fails the causality check (rare numerical glitch),
            # fall through to the 1D fallback below.

        # ---- 1D fallback: use the axis with the smaller upwind value ----
        if a <= b:
            return float(a + s_local * hx)
        return float(b + s_local * hz)

    def _neighbors(
        self,
        iz: int,
        ix: int,
        include_diagonals: bool = False,
    ) -> Iterator[Tuple[int, int]]:
        """
        Yield in-bounds neighbors of grid node ``(iz, ix)``.

        Parameters
        ----------
        iz, ix : int
            Node indices.
        include_diagonals : bool, default False
            If True, also yield the 4 diagonal neighbors. Used by the
            MSFM stencil in Phase E; must be ``False`` in Phase A.

        Yields
        ------
        (jz, jx) : (int, int)
            Each neighbor that lies inside the grid. Out-of-bounds neighbors
            are silently omitted (boundary cells simply have fewer neighbors,
            which the ``_solve_quadratic`` handles via ``np.inf``).
        """
        nz, nx = self.grid.shape
        # 4-connected (axis-aligned) neighbors — Phase A stencil.
        for djz, djx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            jz, jx = iz + djz, ix + djx
            if 0 <= jz < nz and 0 <= jx < nx:
                yield jz, jx
        # Diagonal neighbors — Phase E (MSFM) only.
        if include_diagonals:
            for djz, djx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                jz, jx = iz + djz, ix + djx
                if 0 <= jz < nz and 0 <= jx < nx:
                    yield jz, jx

    # ------------------------------------------------------------------ #
    # Phase D — ray back-tracing and sensitivity matrix
    # ------------------------------------------------------------------ #
    #
    # Numerical scheme
    # ----------------
    # The traveltime gradient is estimated by **first-order upwind**
    # one-sided differences on the four axis-aligned neighbours, choosing
    # the side with the smaller absolute slope. This is the side that
    # points *toward* the source in a monotone-from-source field, and is
    # the natural companion to the Godunov upwind quadratic used by
    # ``_solve_quadratic`` during the march.
    #
    # Pitfall addressed
    # -----------------
    # Central differences mix information from the upwind side (causal,
    # accurate) with information from the downwind side (just-extrapolated,
    # noisy). On a first-order FMM field this would amplify the
    # characteristic "rosette" diagonal error and break ray directions
    # near sharp wavefront curvature. Upwind selection avoids that.
    #
    # Column-shift in bilinear interpolation
    # --------------------------------------
    # The two bracketing columns ``ix0`` and ``ix1`` carry independent
    # surface offsets ``topo_z[ix0]`` and ``topo_z[ix1]``. The four
    # corner cells used for bilinear interpolation must therefore be
    # selected from each column's *own* fractional z-index — they do
    # not share a single ``iz0``. See ``private/docs/tomography/fmm2d.md``
    # §1.5 and §2.5.

    def _compute_gradient(self) -> np.ndarray:
        """
        Upwind first-order ∇T on the current traveltime field.

        Returns
        -------
        grad : np.ndarray, shape=(nz, nx, 2), dtype=float64
            ``grad[..., 0] = ∂T/∂x`` and ``grad[..., 1] = ∂T/∂z`` at each
            grid node. The source node, MASKED nodes, and unreachable
            (``T == inf``) nodes are returned as ``(0.0, 0.0)``.

        Notes
        -----
        For each interior node the forward and backward one-sided
        differences are computed and the one with the smaller absolute
        value is retained (upwind direction in a monotone field). At
        boundary nodes only the single in-bounds one-sided difference
        is available.
        """
        T = self._T
        nz = self.grid.nz
        nx = self.grid.nx
        dx = self.grid.dx
        dz = self.grid.dz

        finite_T = np.isfinite(T)

        # ---- x-component: forward / backward one-sided differences ----
        fwd_x = np.full((nz, nx), np.inf, dtype=np.float64)
        bwd_x = np.full((nz, nx), np.inf, dtype=np.float64)

        # Only compute where both endpoints are finite (suppress inf−inf).
        both_x = finite_T[:, :-1] & finite_T[:, 1:]
        with np.errstate(invalid="ignore"):
            diff_x = (T[:, 1:] - T[:, :-1]) / dx
        diff_x = np.where(both_x, diff_x, np.inf)
        fwd_x[:, :-1] = diff_x          # forward at column j uses j+1 − j
        bwd_x[:, 1:] = diff_x           # backward at column j+1 uses (j+1) − j

        # Upwind choice: smaller |·|. Boundaries fall back to the only
        # available one-sided estimate.
        use_fwd_x = np.abs(fwd_x) <= np.abs(bwd_x)
        g_x = np.where(use_fwd_x, fwd_x, bwd_x)
        g_x[:, 0] = fwd_x[:, 0]
        g_x[:, -1] = bwd_x[:, -1]
        g_x = np.where(np.isfinite(g_x), g_x, 0.0)
        g_x = np.where(finite_T, g_x, 0.0)

        # ---- z-component: same logic along the row axis ----
        fwd_z = np.full((nz, nx), np.inf, dtype=np.float64)
        bwd_z = np.full((nz, nx), np.inf, dtype=np.float64)

        both_z = finite_T[:-1, :] & finite_T[1:, :]
        with np.errstate(invalid="ignore"):
            diff_z = (T[1:, :] - T[:-1, :]) / dz
        diff_z = np.where(both_z, diff_z, np.inf)
        fwd_z[:-1, :] = diff_z
        bwd_z[1:, :] = diff_z

        use_fwd_z = np.abs(fwd_z) <= np.abs(bwd_z)
        g_z = np.where(use_fwd_z, fwd_z, bwd_z)
        g_z[0, :] = fwd_z[0, :]
        g_z[-1, :] = bwd_z[-1, :]
        g_z = np.where(np.isfinite(g_z), g_z, 0.0)
        g_z = np.where(finite_T, g_z, 0.0)

        grad = np.empty((nz, nx, 2), dtype=np.float64)
        grad[..., 0] = g_x
        grad[..., 1] = g_z

        # Source node: gradient is undefined — set to zero so that any
        # ray landing in the source cell terminates rather than wandering.
        if (self._last_source_iz is not None
                and self._last_source_ix is not None):
            grad[self._last_source_iz, self._last_source_ix, :] = 0.0

        # MASKED nodes (unused in Phase A–C, reserved for future cut-cell):
        # report a zero gradient so back-tracing cannot enter them.
        mask_m = (self._state == MASKED)
        if mask_m.any():
            grad[mask_m, :] = 0.0

        return grad

    def _trace_ray(
        self,
        receiver_xz: Tuple[float, float],
    ) -> np.ndarray:
        """
        Back-trace a single ray from a receiver to the source along ``−∇T``.

        Parameters
        ----------
        receiver_xz : (float, float)
            Receiver position ``(x, z)`` in physical metres.

        Returns
        -------
        path : np.ndarray, shape=(npts, 2), dtype=float64
            Sequence of physical ``(x, z)`` points from the receiver to
            the source. The first row is the receiver position; the last
            row is the source's physical position.

        Raises
        ------
        RuntimeError
            If called before ``compute_traveltime``.
        ValueError
            If ``receiver_xz`` cannot be coerced to a 2-tuple.

        Notes
        -----
        Step length is ``min(dx, dz) / 2``. Termination occurs when any
        of the following holds:

        * the Euclidean distance to the source is below one step,
        * the interpolated traveltime is below one step of slowness-weighted
          travel,
        * the maximum step count ``3 * (nx*dx + nz*dz) / Δℓ`` is exceeded,
        * the current position leaves the grid domain,
        * the interpolated gradient magnitude is below 1e−10.

        When ``self._grad_T`` has been pre-computed (typically by
        ``sensitivity_matrix``), it is reused for performance; otherwise
        ``_compute_gradient`` is called once per invocation.
        """
        if self._last_source_ix is None:
            raise RuntimeError(
                "compute_traveltime must be called before _trace_ray."
            )

        rxz = np.asarray(receiver_xz, dtype=np.float64).ravel()
        if rxz.shape[0] != 2:
            raise ValueError(
                f"receiver_xz must be a 2-tuple; got shape {rxz.shape}"
            )

        # Reuse cached gradient if sensitivity_matrix is driving us;
        # otherwise compute on demand.
        grad = getattr(self, "_grad_T", None)
        if grad is None:
            grad = self._compute_gradient()

        T = self._T
        s = self.slowness
        nz = self.grid.nz
        nx = self.grid.nx
        dx = self.grid.dx
        dz = self.grid.dz
        x0 = self.grid.x0
        z0 = self.grid.z0

        step = 0.5 * min(dx, dz)
        nmax = int(3.0 * (nx * dx + nz * dz) / step)

        # Source physical position (column-shift convention).
        src_ix = int(self._last_source_ix)
        src_iz = int(self._last_source_iz)
        src_x = x0 + src_ix * dx
        src_z = z0 + float(self.topo_z[src_ix]) + src_iz * dz

        # Grid domain in x; z-domain depends on column so we check it
        # implicitly through the column-shift conversion.
        x_min = x0
        x_max = x0 + (nx - 1) * dx

        x = float(rxz[0])
        z = float(rxz[1])
        path = [(x, z)]

        for _ in range(nmax):
            # Out-of-bounds in x: stop (clamp would distort the geometry).
            if x < x_min or x > x_max:
                break

            # ---- fractional column index and bracketing columns ----
            xi_f = (x - x0) / dx
            ix0 = int(np.floor(xi_f))
            if ix0 < 0:
                ix0 = 0
            elif ix0 > nx - 2:
                ix0 = nx - 2
            ix1 = ix0 + 1
            xi_w = xi_f - float(ix0)
            if xi_w < 0.0:
                xi_w = 0.0
            elif xi_w > 1.0:
                xi_w = 1.0

            # ---- column-shift: each column has its own fractional z ----
            # Pitfall 3: ix0 and ix1 carry independent topo offsets, so
            # the four bilinear corners come from per-column iz indices.
            zi0_f = (z - z0 - float(self.topo_z[ix0])) / dz
            iz0_c0 = int(np.floor(zi0_f))
            if iz0_c0 < 0:
                iz0_c0 = 0
            elif iz0_c0 > nz - 2:
                iz0_c0 = nz - 2
            iz1_c0 = iz0_c0 + 1
            zw_c0 = zi0_f - float(iz0_c0)
            if zw_c0 < 0.0:
                zw_c0 = 0.0
            elif zw_c0 > 1.0:
                zw_c0 = 1.0

            zi1_f = (z - z0 - float(self.topo_z[ix1])) / dz
            iz0_c1 = int(np.floor(zi1_f))
            if iz0_c1 < 0:
                iz0_c1 = 0
            elif iz0_c1 > nz - 2:
                iz0_c1 = nz - 2
            iz1_c1 = iz0_c1 + 1
            zw_c1 = zi1_f - float(iz0_c1)
            if zw_c1 < 0.0:
                zw_c1 = 0.0
            elif zw_c1 > 1.0:
                zw_c1 = 1.0

            # ---- bilinear interpolation of grad, slowness, traveltime ----
            w_left = 1.0 - xi_w
            w_right = xi_w

            gx = (
                w_left  * ((1.0 - zw_c0) * grad[iz0_c0, ix0, 0]
                           + zw_c0       * grad[iz1_c0, ix0, 0])
                + w_right * ((1.0 - zw_c1) * grad[iz0_c1, ix1, 0]
                             + zw_c1       * grad[iz1_c1, ix1, 0])
            )
            gz = (
                w_left  * ((1.0 - zw_c0) * grad[iz0_c0, ix0, 1]
                           + zw_c0       * grad[iz1_c0, ix0, 1])
                + w_right * ((1.0 - zw_c1) * grad[iz0_c1, ix1, 1]
                             + zw_c1       * grad[iz1_c1, ix1, 1])
            )
            s_interp = (
                w_left  * ((1.0 - zw_c0) * s[iz0_c0, ix0]
                           + zw_c0       * s[iz1_c0, ix0])
                + w_right * ((1.0 - zw_c1) * s[iz0_c1, ix1]
                             + zw_c1       * s[iz1_c1, ix1])
            )
            t_interp = (
                w_left  * ((1.0 - zw_c0) * T[iz0_c0, ix0]
                           + zw_c0       * T[iz1_c0, ix0])
                + w_right * ((1.0 - zw_c1) * T[iz0_c1, ix1]
                             + zw_c1       * T[iz1_c1, ix1])
            )

            # ---- termination tests ----
            if np.hypot(x - src_x, z - src_z) < step:
                break
            if (not np.isfinite(t_interp)) or t_interp < step * s_interp:
                break

            grad_mag = float(np.hypot(gx, gz))
            if grad_mag < 1e-10:
                break

            dir_x = -gx / grad_mag
            dir_z = -gz / grad_mag

            x = x + dir_x * step
            z = z + dir_z * step
            path.append((x, z))

        # Anchor the path at the exact source position so segment lengths
        # near the source are not under-counted by the last finite step.
        path.append((src_x, src_z))

        return np.asarray(path, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"TraveltimeFmm2D(nz={self.grid.nz}, nx={self.grid.nx}, "
            f"dx={self.grid.dx:g}, dz={self.grid.dz:g}, "
            f"scheme={self.scheme!r}, has_topography={self._has_topography}, "
            f"coarse_factor={self.coarse_factor})"
        )


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "TraveltimeFmm2D",
    "Grid2D",
    "FAR",
    "TRIAL",
    "KNOWN",
    "MASKED",
]
