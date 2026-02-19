"""
This creates a tensor X of shape (L, F) per object, where:
F = 6 right now: [g, r, g_err, r_err, g_mask, r_mask]
masks are 1 if observed/interpolated, 0 if missing
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

def make_time_grid(t_all, T=200.0, L=128):
    # relative time grid from 0..T
    t0 = np.min(t_all) # reference time
    grid = np.linspace(0.0, T, L) # time grid
    return t0, grid


def interp_with_mask(t, y, e, t0, grid):
    # """
    # Interpolate y (and e if given) onto grid in relative-time coordinates.
    # Returns (y_grid, e_grid, mask_grid).
    # Mask is 1 where interpolation is valid within data range, else 0.
    # """
    if len(t) == 0:
        y_grid = np.zeros_like(grid) # (L,)
        e_grid = np.ones_like(grid) # (L,)
        mask = np.zeros_like(grid) # (L,)
        return y_grid, e_grid, mask

    dt = t - t0 # relative times
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
    # """
    # Sign-safe log1p transform:
    # sign(f) * log1p(|f|/scale)
    # """
    return np.sign(f) * np.log1p(np.abs(f) / (scale + 1e-8))

class TwoBandLC(Dataset):
    def __init__(self, objects, T=300.0, L=128, use_err=True, normalize_per_object=True, up_to_peak=False):
        self.objects = objects # list of dicts, each with 'g' and 'r' keys for bands
        self.T = float(T) # total time span
        self.L = int(L) # length of time grid
        self.use_err = bool(use_err) # whether to use errors
        self.up_to_peak = bool(up_to_peak) # whether to truncate data up to peak time
        self.normalize_per_object = bool(normalize_per_object) # whether to normalize per object

    def __len__(self):
        return len(self.objects) # number of objects

    def __getitem__(self, idx):
        obj = self.objects[idx] # get object dict

        # extract data for g band
        tg = np.asarray(obj["g"]["t"], dtype=np.float32)
        yg = np.asarray(obj["g"]["y"], dtype=np.float32)
        eg = obj["g"].get("e", None)
        eg = np.asarray(eg, dtype=np.float32) if self.use_err and eg is not None else None

        # extract data for g band
        tr = np.asarray(obj["r"]["t"], dtype=np.float32)
        yr = np.asarray(obj["r"]["y"], dtype=np.float32)
        er = obj["r"].get("e", None)
        er = np.asarray(er, dtype=np.float32) if self.use_err and er is not None else None

        # -------------------------------
        # Optionally truncate light curves up to the peak flux time
        # Define peak time as the time of maximum flux across BOTH bands
        # -------------------------------
        if self.up_to_peak:
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