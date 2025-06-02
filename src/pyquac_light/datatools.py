"Base class for data manipulation"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Iterable, List, Tuple
import threading

import numpy as np
import numba as nb

__all__ = [
    "peak_detection",
    "Spectroscopy",
]

################################################################################
#                            Generic peak detector                             #
################################################################################


def peak_detection(
    x: np.ndarray,
    k: float = 3.0,
    tail_metric: str = "count",
    on_no_peaks: str = "empty",
) -> np.ndarray:
    """Boolean mask of peak samples"""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma_r = 1.4826 * mad
    diff = x - med

    hi_tail = diff > k * sigma_r
    lo_tail = diff < -k * sigma_r

    if not hi_tail.any() and not lo_tail.any():
        if on_no_peaks == "empty":
            return np.zeros_like(x, dtype=bool)
        raise RuntimeError("No peaks detected")

    if tail_metric == "count":
        choose_hi = hi_tail.sum() >= lo_tail.sum()
    elif tail_metric == "energy":
        choose_hi = (x[hi_tail] ** 2).sum() >= (x[lo_tail] ** 2).sum()
    else:
        raise ValueError("tail_metric must be 'count' or 'energy'")

    return hi_tail if choose_hi else lo_tail


################################################################################
#                         Helpers & Numba-accelerated bits                     #
################################################################################


@nb.njit(cache=True)
def _xy_to_index(
    x_val: float, y_val: float, x0: float, dx: float, y0: float, dy: float, ny: int
) -> int:
    """Map *(x, y)* to the flattened index in column-major order."""
    ix = int(round((x_val - x0) / dx))
    iy = int(round((y_val - y0) / dy))
    return ix * ny + iy


# force first‐call compile now
_xy_to_index(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1)

################################################################################
#                                Main class                                    #
################################################################################


@dataclass
class Spectroscopy:
    """Container + post-processing for a regular 2-D *(x, y) → z* measurement."""

    x_arr: np.ndarray
    y_arr: np.ndarray
    _rtol: float = field(init=False, repr=False)
    _atol: float = field(init=False, repr=False)
    _z_flat: np.ndarray = field(init=False, repr=False)
    _mask_written: np.ndarray = field(init=False, repr=False)
    _x_list: np.ndarray = field(init=False, repr=False)
    _y_list: np.ndarray = field(init=False, repr=False)

    # raw lists for quick append; converted to NumPy on demand
    _raw_x: List[float] = field(default_factory=list, init=False, repr=False)
    _raw_y: List[float] = field(default_factory=list, init=False, repr=False)
    _raw_z: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # tolerances
        self._rtol = 1e-05

        # enforce arrays
        self._x_list = np.asarray(self.x_arr, dtype=float)
        self._y_list = np.asarray(self.y_arr, dtype=float)

        # --- uniformity checks with tolerance -----------------------------------
        dxs = np.diff(self._x_list)
        dy_s = np.diff(self._y_list)
        # nominal steps
        nominal_dx = float(np.median(dxs)) if dxs.size > 0 else 0.0
        nominal_dy = float(np.median(dy_s)) if dy_s.size > 0 else 0.0
        # absolute tolerance ~1/2 ulp of step
        self._atol = min(abs(nominal_dx), abs(nominal_dy), 1.0) * 1e-3
        # check closeness
        if dxs.size > 0 and not np.allclose(
            dxs, nominal_dx, rtol=self._rtol, atol=self._atol
        ):
            raise ValueError("x_arr must form a uniform grid within tolerance")
        if dy_s.size > 0 and not np.allclose(
            dy_s, nominal_dy, rtol=self._rtol, atol=self._atol
        ):
            raise ValueError("y_arr must form a uniform grid within tolerance")
        # assign
        self._dx = nominal_dx
        self._dy = nominal_dy

        # initialize data containers
        ny, nx = len(self._y_list), len(self._x_list)
        self._z_flat = np.full(nx * ny, np.nan, dtype=float)
        self._mask_written = np.zeros_like(self._z_flat, dtype=bool)

    # --------------------------------------------------------------------- IO
    def save_csv(self, path: str | Path, header: bool = True) -> None:
        """Save current data (only measured points) to ``x,y,z`` CSV."""
        data = np.column_stack((self._raw_x, self._raw_y, self._raw_z))
        fmt = ["%.10g", "%.10g", "%.10g"]
        np.savetxt(
            path,
            data,
            delimiter=",",
            header="x_value,y_value,z_value" if header else "",
            comments="",
            fmt=fmt,
        )

    @classmethod
    def load_csv(
        cls,
        path: str | Path,
        x_arr: np.ndarray | None = None,
        y_arr: np.ndarray | None = None,
        has_header: bool = True,
    ) -> "Spectroscopy":
        """Create a *Spectroscopy* object and populate it from CSV.

        If ``x_arr`` and/or ``y_arr`` are not supplied, they are derived from
        the data file (unique values). Grid spacing is validated within tolerance.
        """
        raw = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        x_vals, y_vals, z_vals = raw[:, 0], raw[:, 1], raw[:, 2]

        if x_arr is None:
            x_arr = np.unique(x_vals)
        if y_arr is None:
            y_arr = np.unique(y_vals)

        spec = cls(x_arr=x_arr, y_arr=y_arr)
        for x, y, z in zip(x_vals, y_vals, z_vals):
            spec.write(float(x), float(y), float(z))
        return spec

    # ------------------------------------------------------------------ writing
    def write(self, x: float, y: float, z: float) -> None:
        """Record a single sample (x, y, z)."""
        ix_flat = _xy_to_index(
            x,
            y,
            self._x_list[0],
            self._dx,
            self._y_list[0],
            self._dy,
            len(self._y_list),
        )
        self._z_flat[ix_flat] = z
        self._mask_written[ix_flat] = True

        self._raw_x.append(x)
        self._raw_y.append(y)
        self._raw_z.append(z)

    # -------------------------------------------------------------- data views
    @property
    def z_matrix(self) -> np.ndarray:
        """Return Z as (ny, nx) array (column-major order)."""
        return self._z_flat.reshape(len(self._y_list), len(self._x_list), order="F")

    def to_points(self) -> np.ndarray:
        """Return (N, 3) array of measured points."""
        return np.column_stack((self._raw_x, self._raw_y, self._raw_z))

    # ----------------------------------------------------------- peak & ridge
    def highest_peaks(
        self,
        peak_kwargs: dict | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays (x, y_peak, z_peak) (one per x)."""
        peak_kwargs = peak_kwargs or {}
        xs, ys, zs = [], [], []
        med_cache: dict[int, float] = {}
        for col, x_val in enumerate(self._x_list):
            z_line = self.z_matrix[:, col]
            if np.isnan(z_line).all():
                continue
            mask = peak_detection(z_line, **peak_kwargs)
            if not mask.any():
                continue
            # compute baseline
            if col not in med_cache:
                med_cache[col] = np.median(z_line)
            baseline = med_cache[col]
            # candidate peaks indices
            peaks = np.where(mask)[0]
            offsets = np.abs(z_line[peaks] - baseline)
            y_vals = self._y_list[peaks]
            # select the peak maximizing (offset, y value)
            # lexsort: primary key offsets, secondary key y_vals
            order = np.lexsort((y_vals, offsets))
            best = order[-1]
            peak_idx = peaks[best]
            # append results
            xs.append(x_val)
            ys.append(self._y_list[peak_idx])
            zs.append(z_line[peak_idx])
        return np.asarray(xs), np.asarray(ys), np.asarray(zs)

    def fit_ridge(
        self,
        deg: int = 2,
        peak_kwargs: dict | None = None,
        x_subset: Iterable[float] | None = None,
        manual_peaks: Iterable[Tuple[float, float]] | None = None,
    ) -> np.poly1d:
        """Fit a polynomial ridge and return poly1d.

        Parameters:
        - deg: polynomial degree.
        - peak_kwargs: passed to highest_peaks() if manual_peaks is None.
        - x_subset: optional list of x values to include.
        - manual_peaks: optional iterable of (x, y) to use instead of auto peaks.
        """
        if manual_peaks is not None:
            # use user-supplied points
            arr = np.asarray(manual_peaks, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("manual_peaks must be an iterable of (x, y) pairs")
            x_p = arr[:, 0]
            y_p = arr[:, 1]
        else:
            # detect all peaks automatically
            peak_kwargs = peak_kwargs or {}
            x_p, y_p, _ = self.highest_peaks(peak_kwargs)
        # optionally filter by user-provided x values
        if x_subset is not None:
            x_subset_arr = np.asarray(x_subset, dtype=float)
            mask = np.isclose(
                x_p[:, None], x_subset_arr[None, :], rtol=self._rtol, atol=self._atol
            ).any(axis=1)
            x_p = x_p[mask]
            y_p = y_p[mask]
        if x_p.size < deg + 1:
            raise RuntimeError("Not enough points to fit the requested polynomial")
        coeff = np.polyfit(x_p, y_p, deg)
        return np.poly1d(coeff)

    # --------------------------------------------------------- acquisition aid
    def corridor_mask(self, ridge: np.poly1d, width_frac: float = 0.2) -> np.ndarray:
        """Boolean mask (ny, nx) selecting cells within a vertical corridor."""
        half_span = int(round(width_frac * len(self._y_list) / 2))
        ny, nx = len(self._y_list), len(self._x_list)
        mask = np.zeros((ny, nx), dtype=bool)
        for col, x_val in enumerate(self._x_list):
            y_center = ridge(x_val)
            y_idx = int(round((y_center - self._y_list[0]) / self._dy))
            lo = max(0, y_idx - half_span)
            hi = min(ny, y_idx + half_span + 1)
            mask[lo:hi, col] = True
        return mask

    def next_unmeasured_points(
        self,
        mask: np.ndarray | None = None,
        x_subset: Iterable[float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return x, y arrays of points still missing.

        Optionally restrict by *mask* and/or by specific *x_subset* values.
        If an element of *x_subset* is not exactly in `x_arr`, the nearest grid x is used.
        """
        # base missing mask
        missing = (
            (~self._mask_written)
            if mask is None
            else ((~self._mask_written) & mask.ravel(order="F"))
        )

        # if x_subset provided, restrict to those columns
        if x_subset is not None:
            # map requested x's to nearest grid indices
            xs_req = np.asarray(x_subset, dtype=float)
            # find nearest index in x_list for each requested x
            col_idxs = []
            for xv in xs_req:
                idx = np.abs(self._x_list - xv).argmin()
                col_idxs.append(idx)
            col_idxs = np.unique(col_idxs)
            # build a column mask
            ny = len(self._y_list)
            col_mask = np.zeros(len(self._x_list), dtype=bool)
            col_mask[col_idxs] = True
            # broadcast to flat
            missing = missing & np.repeat(col_mask[None, :], ny, axis=0).ravel(
                order="F"
            )

        # extract flat indices
        flat_idx = np.where(missing)[0]
        ny = len(self._y_list)
        ix = flat_idx // ny
        iy = flat_idx % ny
        return self._x_list[ix], self._y_list[iy]

    # ----------------------------------------------------------- housekeeping
    def drop(
        self, *, x: Iterable[float] | None = None, y: Iterable[float] | None = None
    ) -> None:
        """Erase measured samples whose coordinates match x and/or y."""
        if x is None and y is None:
            return
        drop_mask = np.ones_like(self._mask_written, dtype=bool)
        ny = len(self._y_list)
        if x is not None:
            ix = np.flatnonzero(
                np.isclose(
                    self._x_list[:, None], x, rtol=self._rtol, atol=self._atol
                ).any(axis=1)
            )
            col_mask = np.zeros_like(self._x_list, dtype=bool)
            col_mask[ix] = True
            drop_mask &= np.repeat(col_mask[None, :], ny, axis=0).ravel(order="F")
        if y is not None:
            iy = np.flatnonzero(
                np.isclose(
                    self._y_list[:, None], y, rtol=self._rtol, atol=self._atol
                ).any(axis=1)
            )
            row_mask = np.zeros_like(self._y_list, dtype=bool)
            row_mask[iy] = True
            drop_mask &= np.repeat(row_mask[:, None], len(self._x_list), axis=1).ravel(
                order="F"
            )
        self._mask_written[drop_mask] = False
        self._z_flat[drop_mask] = np.nan

    def clear(self) -> None:
        """Remove all measured samples."""
        self._mask_written.fill(False)
        self._z_flat.fill(np.nan)
        self._raw_x.clear()
        self._raw_y.clear()
        self._raw_z.clear()

    def clean_up(self, ridge: np.poly1d, width_frac: float = 0.2) -> None:
        """
        Drop all measured data _except_ the corridor around the given ridge.
        Points outside the vertical corridor of width_frac are erased.
        """
        # compute corridor mask and flatten it
        corridor = self.corridor_mask(ridge, width_frac=width_frac)
        flat_mask = corridor.ravel(order="F")
        # apply mask: keep only within corridor
        inv = ~flat_mask
        self._mask_written[inv] = False
        self._z_flat[inv] = np.nan
        # rebuild raw history lists from remaining mask
        idxs = np.where(self._mask_written)[0]
        if idxs.size:
            ny = len(self._y_list)
            ix = idxs // ny
            iy = idxs % ny
            zs = self._z_flat[idxs]
            xs = self._x_list[ix]
            ys = self._y_list[iy]
            self._raw_x = list(xs)
            self._raw_y = list(ys)
            self._raw_z = list(zs)
        else:
            # no points remain
            self._raw_x.clear()
            self._raw_y.clear()
            self._raw_z.clear()


class RandomSpectroscopy(Spectroscopy):
    def run_full_scan(
        self,
        sleep: float = 0.001,
        x_subset: Iterable[float] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        Measure every (x,y) once, or only for x in `x_subset`.
        Off-grid x's in `x_subset` are snapped to the nearest grid point.
        Can be stopped early by setting stop_event.
        """
        xs, ys = self.next_unmeasured_points(x_subset=x_subset)
        for x, y in zip(xs, ys):
            # Check if we should stop
            if stop_event is not None and stop_event.is_set():
                break
            z = np.random.random()
            self.write(x, y, z)
            time.sleep(sleep)

    def run_corridor_scan(
        self,
        ridge: np.poly1d,
        width_frac: float = 0.2,
        sleep: float = 0.001,
        x_subset: Iterable[float] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        Measure only points within ±width_frac corridor around `ridge`,
        and only for x in `x_subset` if provided.
        Can be stopped early by setting stop_event.
        """
        mask = self.corridor_mask(ridge, width_frac=width_frac)
        xs, ys = self.next_unmeasured_points(mask=mask, x_subset=x_subset)
        for x, y in zip(xs, ys):
            # Check if we should stop
            if stop_event is not None and stop_event.is_set():
                break
            z = np.random.random()
            self.write(x, y, z)
            time.sleep(sleep)


class SkeletonSpectroscopy(Spectroscopy, ABC):
    """
    A Template-Method version of Spectroscopy that
    implements run_full_scan and run_corridor_scan
    by calling four user-supplied hooks.
    """

    @abstractmethod
    def pre_scan(self) -> None:
        """One-time setup before any measurements."""

    @abstractmethod
    def pre_column(self, x: float) -> None:
        """Setup before each new x-column."""

    @abstractmethod
    def measure_point(self, x: float, y: float) -> float:
        """
        Do the actual acquisition for (x,y);
        return the measured z-value.
        """

    @abstractmethod
    def post_scan(self) -> None:
        """Clean up after the entire scan."""

    def _scan_loop(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        sleep: float,
        stop_event: threading.Event | None,
    ) -> None:
        last_x = None
        for x, y in zip(xs, ys):
            if stop_event is not None and stop_event.is_set():
                break
            if x != last_x:
                self.pre_column(x)
                last_x = x
            z = self.measure_point(x, y)
            self.write(x, y, z)
            time.sleep(sleep)

    def run_full_scan(
        self,
        sleep: float = 0.001,
        x_subset: Iterable[float] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        Default full-grid scan.  Users only need to implement the four hooks.
        """
        self.pre_scan()
        xs, ys = self.next_unmeasured_points(x_subset=x_subset)
        self._scan_loop(xs, ys, sleep, stop_event)
        self.post_scan()

    def run_corridor_scan(
        self,
        ridge: np.poly1d,
        width_frac: float = 0.2,
        sleep: float = 0.001,
        x_subset: Iterable[float] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        Default corridor scan around `ridge`.  Again, all you need
        to do is implement the four hooks.
        """
        self.pre_scan()
        mask = self.corridor_mask(ridge, width_frac=width_frac)
        xs, ys = self.next_unmeasured_points(mask=mask, x_subset=x_subset)
        self._scan_loop(xs, ys, sleep, stop_event)
        self.post_scan()


class ColumnSkeletonSpectroscopy(Spectroscopy, ABC):
    """
    A Template-Method version of Spectroscopy for “column-at-once” acquisition.

    Subclasses should implement:
      • pre_scan(self) → None
      • pre_column(self, x: float) → None
      • measure_column(self, x: float) → 1D np.ndarray of length len(self._y_list)
      • post_scan(self) → None

    run_full_scan() will:
      - call pre_scan()
      - for each x-column that still has missing data:
          • call pre_column(x)
          • call measure_column(x) to fetch the entire z-column
          • if stop_event was set before or after measure_column, skip writing that column
          • otherwise write all (x, y, z) pairs and mark them as measured
      - call post_scan()

    run_corridor_scan(...) is not supported and will just print a warning.
    """

    @abstractmethod
    def pre_scan(self) -> None:
        """One-time setup before any measurements (e.g. power on, init hardware)."""

    @abstractmethod
    def pre_column(self, x: float) -> None:
        """Setup before measuring an entire column at x (e.g. step device to x)."""

    @abstractmethod
    def measure_column(self, x: float) -> np.ndarray:
        """
        Acquire and return a full column of z-values at this x.
        Must return a 1D array of length len(self._y_list), ordered so that
        index i corresponds to y = self._y_list[i]. If this raises or if
        stop_event is set during measurement, the column will be skipped.
        """

    @abstractmethod
    def post_scan(self) -> None:
        """Clean up after the entire scan (e.g. power off, close connections)."""

    def run_full_scan(
        self,
        sleep: float = 0.001,
        x_subset: Iterable[float] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        Acquire every “missing” (x,y) by fetching each column at once.

        If `x_subset` is given, only columns nearest those x-values (in the grid) are measured.
        If `stop_event.is_set()` becomes True before or after calling measure_column(x),
        that column is skipped (no data is written), and scanning stops immediately.
        """
        # One‐time setup
        self.pre_scan()

        # Find all (x, y) still missing, possibly restricted to x_subset
        xs_missing, _ = self.next_unmeasured_points(x_subset=x_subset)
        if xs_missing.size == 0:
            # Nothing to do
            self.post_scan()
            return

        # Which unique x‐columns need measurement?  Keep them in grid order.
        unique_x_missing = np.unique(xs_missing)
        xs_to_scan = [
            x_val
            for x_val in self._x_list
            if np.isclose(
                unique_x_missing, x_val, rtol=self._rtol, atol=self._atol
            ).any()
        ]

        for x in xs_to_scan:
            # If the user set stop_event before starting this column, abort the scan
            if stop_event is not None and stop_event.is_set():
                break

            # One‐time per‐column hookup
            self.pre_column(x)

            try:
                # Fetch the entire column of z‐values at once
                z_col = self.measure_column(x)
            except Exception:
                # If measurement failed or was aborted mid‐column, skip this column
                # and stop the entire scan
                break

            # If stop_event was set during measure_column, skip writing and break
            if stop_event is not None and stop_event.is_set():
                break

            # Write every (x, y[i], z_col[i]) into the grid and mark as measured
            for i, y in enumerate(self._y_list):
                self.write(x, y, float(z_col[i]))
                time.sleep(sleep)

        # Final cleanup
        self.post_scan()

    def run_corridor_scan(
        self,
        *args,
        **kwargs,
    ) -> None:
        print("Corridor scan is not supported for column-at-once acquisition.")
