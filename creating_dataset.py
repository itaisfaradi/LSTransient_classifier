import numpy as np
import torch
from torch.utils.data import Dataset

def make_time_grid(t_all, T=200.0, L=128):
    """Build a uniform time grid anchored at the earliest observation.

    Args:
        t_all: Array of all observation times across both bands.
        T:     Total time span of the grid in days.
        L:     Number of equally-spaced grid points.

    Returns:
        t0:   Reference time (minimum of t_all).
        grid: Array of shape (L,) with times in [0, T].
    """

    t0 = np.min(t_all)
    grid = np.linspace(0.0, T, L)
    return t0, grid


def interp_with_mask(t, y, e, t0, grid):
    """Interpolate flux (and errors) onto a uniform time grid with a validity mask.

    Args:
        t:    Observed times for this band.
        y:    Observed flux values.
        e:    Observed flux errors, or None if unavailable.
        t0:   Reference time; observations are shifted to relative time t - t0.
        grid: Uniform time grid to interpolate onto.

    Returns:
        y_grid: Interpolated flux at each grid point, shape (L,).
        e_grid: Interpolated errors at each grid point, shape (L,).
                Filled with ones if e is None.
        mask:   Float32 array of shape (L,); 1 where grid falls within the
                observed time range, 0 outside (model should ignore these).
    """

    if len(t) == 0:
        y_grid = np.zeros_like(grid)
        e_grid = np.ones_like(grid)
        mask = np.zeros_like(grid)
        return y_grid, e_grid, mask

    dt = t - t0
    # sort by time
    order = np.argsort(dt)
    dt = dt[order]
    y = y[order]
    e = e[order] if e is not None else None

    # valid only inside [dt.min, dt.max]
    mask = ((grid >= dt.min()) & (grid <= dt.max())).astype(np.float32)

    # interpolate; outside range we'll still get values, but mask=0 will tell model to ignore
    y_grid = np.interp(grid, dt, y).astype(np.float32)

    if e is None:
        e_grid = np.ones_like(grid, dtype=np.float32)
    else:
        e_grid = np.interp(grid, dt, e).astype(np.float32)

    return y_grid, e_grid, mask

def log_flux_transform(f, scale=1.0):
    """Apply a sign-preserving log1p transform to compress flux dynamic range.

    Computes sign(f) * log1p(|f| / scale), which handles negative fluxes
    and avoids large values from bright sources.

    Args:
        f:     Input flux array.
        scale: Normalisation scale applied before log1p (default 1.0).

    Returns:
        Transformed array with the same shape as f.
    """

    return np.sign(f) * np.log1p(np.abs(f) / (scale + 1e-8))

def _clip_bands_to_peak(tg, yg, eg, tr, yr, er):
    """Clip both band light curves to include only observations up to peak flux.

    Peak time is defined as the time of maximum flux across both bands combined.
    If both bands are empty, data is returned unchanged.

    Args:
        tg, yg, eg: Time, flux, and error arrays for the g band.
        tr, yr, er: Time, flux, and error arrays for the r band.

    Returns:
        tg, yg, eg, tr, yr, er: Clipped arrays (same structure as input).
    """

    all_flux = []
    all_time = []

    if len(tg):
        all_flux.append(yg)
        all_time.append(tg)

    if len(tr):
        all_flux.append(yr)
        all_time.append(tr)

    if len(all_flux) > 0:
        all_flux = np.concatenate(all_flux)
        all_time = np.concatenate(all_time)

        t_peak = all_time[np.argmax(all_flux)]

        # cut both bands at t_peak (keep points with t <= t_peak)
        g_keep = tg <= t_peak
        r_keep = tr <= t_peak

        tg, yg = tg[g_keep], yg[g_keep]
        if eg is not None:
            eg = eg[g_keep]

        tr, yr = tr[r_keep], yr[r_keep]
        if er is not None:
            er = er[r_keep]
    return tg, yg, eg, tr, yr, er

class TwoBandLC(Dataset):
    """PyTorch Dataset for two-band (g, r) astronomical light curves.

    Each item is a feature tensor X of shape (L, 6):
        [g_flux, r_flux, g_err, r_err, g_mask, r_mask]
    where mask is 1 where the interpolation is valid, 0 otherwise.

    Args:
        objects:            List of dicts, each with keys 'g', 'r', and 'label'.
                            Each band dict contains 't' (times), 'y' (fluxes),
                            and optionally 'e' (errors).
        T:                    Total time span of the grid in days.
        L:                    Number of time grid points.
        use_err:              If True, include flux errors as features.
        normalize_per_object: If True, scale fluxes by per-object median flux.
        up_to_peak:           If True, truncate light curves before returning them.
    """

    def __init__(self, objects, T=300.0, L=128, use_err=True, normalize_per_object=True, up_to_peak=False):
        self.objects = objects
        self.T = float(T)
        self.L = int(L)
        self.use_err = bool(use_err) # whether to use errors of data points
        self.up_to_peak = bool(up_to_peak) # whether to truncate data up to peak time
        self.normalize_per_object = bool(normalize_per_object)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]

        # extract data for g band
        tg = np.asarray(obj["g"]["t"], dtype=np.float32)
        yg = np.asarray(obj["g"]["y"], dtype=np.float32)
        eg = obj["g"].get("e", None)
        eg = np.asarray(eg, dtype=np.float32) if self.use_err and eg is not None else None

        # extract data for r band
        tr = np.asarray(obj["r"]["t"], dtype=np.float32)
        yr = np.asarray(obj["r"]["y"], dtype=np.float32)
        er = obj["r"].get("e", None)
        er = np.asarray(er, dtype=np.float32) if self.use_err and er is not None else None
        
        # optionally clip data to pre-peak only
        if self.up_to_peak:
            tg, yg, eg, tr, yr, er = _clip_bands_to_peak(tg, yg, eg, tr, yr, er)

        # create common time grid
        t_all = np.concatenate([tg, tr]) if (len(tg) and len(tr)) else (tg if len(tg) else tr)
        t0, grid = make_time_grid(t_all, T=self.T, L=self.L)

        # interpolate both bands onto grid
        g_grid, g_err, g_mask = interp_with_mask(tg, yg, eg, t0, grid)
        r_grid, r_err, r_mask = interp_with_mask(tr, yr, er, t0, grid)

        # per-object robust scale (median of positive observed fluxes across both bands)
        if self.normalize_per_object:
            obs = []
            if len(yg):
                obs.append(yg)
            if len(yr): 
                obs.append(yr)
            obs = np.concatenate(obs) if len(obs) else np.array([1.0], dtype=np.float32)
            pos = obs[obs > 0]
            scale = np.median(pos) if len(pos) else (np.median(np.abs(obs)) + 1e-6)
            scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
        else:
            scale = 1.0

        # log-flux transform and scale to reduce dynamic range, feat is for "feature"
        g_feat = log_flux_transform(g_grid, scale=scale).astype(np.float32)
        r_feat = log_flux_transform(r_grid, scale=scale).astype(np.float32)

        # Optionally also scale errors into the same space (roughly)
        if self.use_err:
            g_err_feat = (g_err / (scale + 1e-8)).astype(np.float32)
            r_err_feat = (r_err / (scale + 1e-8)).astype(np.float32)
        else:
            g_err_feat = np.ones_like(g_feat, dtype=np.float32)
            r_err_feat = np.ones_like(r_feat, dtype=np.float32)

        # Features: (L, F)
        X = np.stack([g_feat, r_feat, g_err_feat, r_err_feat, g_mask, r_mask], axis=1).astype(np.float32)

        y = float(obj["label"])  # 0.0 or 1.0
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)