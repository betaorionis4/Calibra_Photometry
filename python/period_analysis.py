"""
Period analysis of light curves (v4.21).

Methods for unevenly sampled data:
  * Lomb-Scargle periodogram (astropy), optionally multi-harmonic (nterms > 1)
  * Phase Dispersion Minimisation (PDM, Stellingwerf 1978; 1 - theta is plotted)
  * Box Least Squares (BLS, astropy) for eclipses / transits
  * spectral window (periodogram of the sampling times alone): shows the aliases
    that the gaps in the data produce
  * prewhitening: fit the strongest frequency (with harmonics), subtract all
    fitted frequencies, search the residuals again; a final non-linear
    least-squares fit of all frequencies together (as in Period04)

Units: time in days (HJD), frequency in cycles/day (c/d). The model is
    y(t) = c + sum_k sum_h [ a_kh sin(2 pi h f_k (t - t0)) + b_kh cos(2 pi h f_k (t - t0)) ]
amplitude A_kh = sqrt(a^2 + b^2) (semi-amplitude, in the unit of the data).
Uncertainties of a single sinusoid follow Montgomery & O'Donoghue (1999):
    sigma_f = sqrt(6/N) * sigma / (pi * T * A),  sigma_A = sqrt(2/N) * sigma,
    sigma_phi = sigma_A / A  (phase in rad)
(they assume uncorrelated noise and are therefore lower limits).
"""
import csv
import os
import numpy as np

TIME_COLS = ('hjd', 'bjd', 'jd', 'time', 'date', 'mjd', 'hjd_utc', 'bjd_tdb', 't')
ERR_SUFFIX = ('_err', 'err', '_error', 'error', 'e_')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def read_light_curve(path):
    """
    Read a light-curve CSV (Calibra output or any CSV with a time column).
    Returns dict: name, path, columns (numeric column names), rows (list of dicts),
    time_col, default value column ('mag' for calibrated Calibra curves, 'flux'
    otherwise), lc_mode.
    """
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith('#')]   # v4.24: comment lines
    sample = "".join(lines[:40])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(lines, dialect=dialect))
    if not rows:
        raise ValueError("file contains no data rows")
    cols = [c for c in rows[0].keys() if c is not None]
    low = {c.strip().lower(): c for c in cols}
    time_col = next((low[c] for c in TIME_COLS if c in low), None)
    if time_col is None:
        raise ValueError(f"no time column found (expected one of {', '.join(TIME_COLS)})")

    def numeric(c):
        n = ok = 0
        for r in rows[:50]:
            v = str(r.get(c, '')).strip()
            if v == '':
                continue
            n += 1
            try:
                float(v)
                ok += 1
            except ValueError:
                pass
        return n > 0 and ok >= 0.8 * n
    num_cols = [c for c in cols if c != time_col and numeric(c)]
    mode = str(rows[0].get('lc_mode', '') or '').strip().lower() or None
    if mode == 'calibrated' or (mode is None and 'mag' in num_cols):
        default = 'mag'
    elif 'flux' in num_cols:
        default = 'flux'
    else:
        default = next((c for c in num_cols if 'mag' in c.lower()), num_cols[0] if num_cols else None)
    return {'name': os.path.splitext(os.path.basename(path))[0], 'path': path, 'rows': rows,
            'columns': num_cols, 'time_col': time_col, 'value_col': default, 'lc_mode': mode}


def error_column(columns, value_col):
    """Best guess of the uncertainty column belonging to value_col."""
    cand = [value_col + '_err', value_col + 'err', value_col + '_error', 'e_' + value_col, 'err_' + value_col]
    for c in cand:
        if c in columns:
            return c
    lowmap = {c.lower(): c for c in columns}
    for c in cand:
        if c.lower() in lowmap:
            return lowmap[c.lower()]
    return None


def extract_series(lc, value_col=None, err_col='auto', only_ok=True):
    """(t, y, e) arrays of one light curve; rows with non-finite values or (if
    only_ok) flag != OK are dropped. e is None if there is no error column."""
    value_col = value_col or lc['value_col']
    if err_col == 'auto':
        err_col = error_column(lc['columns'], value_col)
    t, y, e = [], [], []
    for r in lc['rows']:
        if only_ok and str(r.get('flag', 'OK') or 'OK').strip().upper() not in ('OK', ''):
            continue
        try:
            tv, yv = float(r[lc['time_col']]), float(r[value_col])
        except (TypeError, ValueError, KeyError):
            continue
        ev = np.nan
        if err_col:
            try:
                ev = float(r.get(err_col, 'nan'))
            except (TypeError, ValueError):
                ev = np.nan
        if np.isfinite(tv) and np.isfinite(yv):
            t.append(tv); y.append(yv); e.append(ev)
    t, y, e = np.array(t), np.array(y), np.array(e)
    if not np.any(np.isfinite(e) & (e > 0)):
        e = None
    else:
        med = np.nanmedian(e[np.isfinite(e) & (e > 0)])
        e = np.where(np.isfinite(e) & (e > 0), e, med)
    order = np.argsort(t)
    return t[order], y[order], (e[order] if e is not None else None)


def combine_series(series, subtract_mean=False):
    """Concatenate several (t, y, e) series; optionally remove each series' mean
    (data sets with different zero points, e.g. different comparison stars or
    relative flux from different nights). Returns t, y, e, set_index."""
    ts, ys, es, idx = [], [], [], []
    any_err = any(s[2] is not None for s in series)
    for k, (t, y, e) in enumerate(series):
        if t.size == 0:
            continue
        yy = y - np.mean(y) if subtract_mean else y
        ts.append(t); ys.append(yy)
        if any_err:
            es.append(e if e is not None else np.full(t.size, np.nan))
        idx.append(np.full(t.size, k))
    if not ts:
        return np.array([]), np.array([]), None, np.array([], int)
    t, y, idx = np.concatenate(ts), np.concatenate(ys), np.concatenate(idx)
    e = None
    if any_err:
        e = np.concatenate(es)
        good = np.isfinite(e) & (e > 0)
        e = np.where(good, e, np.median(e[good]) if np.any(good) else 1.0)
    o = np.argsort(t)
    return t[o], y[o], (e[o] if e is not None else None), idx[o]


def data_summary(t):
    """N, time span T, number of nights, median spacing within nights,
    frequency resolution 1/T and pseudo-Nyquist frequency 0.5/median spacing."""
    t = np.sort(np.asarray(t, float))
    if t.size < 2:
        return {'n': int(t.size), 'span': 0.0, 'nights': int(t.size), 'dt_med': np.nan, 'resolution': np.nan, 'nyquist': np.nan}
    d = np.diff(t)
    within = d[d < 0.3]
    dt_med = float(np.median(within)) if within.size else float(np.median(d))
    span = float(t[-1] - t[0])
    return {'n': int(t.size), 'span': span, 'nights': int(1 + np.sum(d > 0.3)), 'dt_med': dt_med,
            'resolution': 1.0 / span if span > 0 else np.nan,
            'nyquist': 0.5 / dt_med if dt_med > 0 else np.nan}


# --------------------------------------------------------------------------
# Frequency grid and periodograms
# --------------------------------------------------------------------------
def frequency_grid(t, fmin=None, fmax=None, oversample=10, max_points=2_000_000):
    """Evenly spaced frequency grid [c/d]. Default fmax = pseudo-Nyquist (max 100 c/d),
    fmin = 0.5/T (at least 1/(10 T)); step = 1/(oversample*T)."""
    s = data_summary(t)
    T = s['span'] if s['span'] > 0 else 1.0
    if fmax is None or not np.isfinite(fmax) or fmax <= 0:
        fmax = min(s['nyquist'] if np.isfinite(s['nyquist']) else 50.0, 100.0)
    if fmin is None or not np.isfinite(fmin) or fmin <= 0:
        fmin = 0.5 / T
    fmin = max(fmin, 0.1 / T)
    if fmax <= fmin:
        raise ValueError(f"maximum frequency ({fmax:.4g} c/d) must be larger than minimum ({fmin:.4g} c/d)")
    df = 1.0 / (oversample * T)
    n = int(np.ceil((fmax - fmin) / df)) + 1
    if n > max_points:
        df = (fmax - fmin) / (max_points - 1)
        n = max_points
    return fmin + df * np.arange(n)


def amplitude_spectrum(t, y, freq, e=None, nterms=1):
    """Lomb-Scargle as an amplitude spectrum [unit of y]: semi-amplitude of the
    best-fitting sinusoid at each frequency (for nterms > 1: of the fundamental
    is not separated - the 'power' is converted with the single-sine relation)."""
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, dy=None, nterms=nterms, normalization='psd', center_data=True, fit_mean=True)
    p = ls.power(freq, method='fast' if nterms == 1 else 'auto')
    return np.sqrt(4.0 * np.maximum(p, 0) / len(t))


def lomb_scargle(t, y, freq, e=None, nterms=1):
    """Normalised Lomb-Scargle power (0..1, 'standard') and the LombScargle object."""
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, dy=e, nterms=nterms, normalization='standard')
    method = 'fast' if nterms == 1 else 'auto'
    try:
        p = ls.power(freq, method=method)
    except Exception:
        p = ls.power(freq)
    return p, ls


def false_alarm_probability(ls, power_max, freq):
    """FAP of the highest peak (Baluev 2008 approximation; only for nterms = 1)."""
    try:
        return float(ls.false_alarm_probability(power_max, minimum_frequency=freq[0],
                                                maximum_frequency=freq[-1], method='baluev'))
    except Exception:
        return np.nan


def spectral_window(t, freq):
    """Spectral window |sum exp(2 pi i f t)| / N (1 at f = 0). Peaks show the
    aliases produced by the sampling (e.g. 1 c/d for single-site nightly data)."""
    t = np.asarray(t, float) - np.mean(t)
    out = np.empty(freq.size)
    step = max(1, int(2e7 // max(t.size, 1)))
    for i in range(0, freq.size, step):
        ph = 2 * np.pi * np.outer(freq[i:i + step], t)
        out[i:i + step] = np.hypot(np.cos(ph).sum(1), np.sin(ph).sum(1)) / t.size
    return out


def pdm(t, y, freq, nbins=10, progress=None, cancel=None):
    """Phase Dispersion Minimisation. Returns theta (pooled bin variance / total
    variance; deep minima = good periods). Bins with < 2 points are ignored."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    var = np.var(y, ddof=1)
    n = y.size
    theta = np.ones(freq.size)
    t0 = t.min()
    y2 = y * y
    for i, f in enumerate(freq):
        if cancel is not None and cancel.is_set():
            break
        b = (((t - t0) * f) % 1.0 * nbins).astype(int)
        cnt = np.bincount(b, minlength=nbins)
        s1 = np.bincount(b, weights=y, minlength=nbins)
        s2 = np.bincount(b, weights=y2, minlength=nbins)
        ok = cnt > 1
        ss = s2[ok] - s1[ok] ** 2 / cnt[ok]
        dof = cnt[ok].sum() - ok.sum()
        theta[i] = (ss.sum() / dof) / var if dof > 0 and var > 0 else 1.0
        if progress is not None and i % 2000 == 0:
            progress(100.0 * i / freq.size)
    return theta


def bls(t, y, freq, e=None, durations=None, magnitudes=False):
    """Box Least Squares power (astropy) - searches for periodic DIPS in flux.
    magnitudes=True: y are magnitudes (an eclipse is an increase), the sign is
    reversed internally. durations in days (default 1-10 % of the shortest
    period, at most 0.3 d). Returns (power, result)."""
    from astropy.timeseries import BoxLeastSquares
    if magnitudes:
        y = -np.asarray(y, float)
    periods = 1.0 / freq[::-1]
    if durations is None:
        pmin = periods.min()
        durations = np.unique(np.clip(np.array([0.01, 0.02, 0.04, 0.07, 0.1]) * pmin, 0.005, 0.3))
    model = BoxLeastSquares(t, y, dy=e)
    res = model.power(periods, durations, objective='snr')
    return np.asarray(res.power)[::-1], res


# --------------------------------------------------------------------------
# Multi-frequency fit and prewhitening
# --------------------------------------------------------------------------
def _design(t, freqs, nharm, t0):
    cols = [np.ones_like(t)]
    for f, h in zip(freqs, nharm):
        for k in range(1, int(h) + 1):
            ph = 2 * np.pi * k * f * (t - t0)
            cols += [np.sin(ph), np.cos(ph)]
    return np.column_stack(cols)


def fit_frequencies(t, y, freqs, nharm=None, e=None, t0=None, refine=True):
    """
    Least-squares fit of several frequencies (each with nharm harmonics).
    refine=True also optimises the frequencies (non-linear fit, scipy).
    Returns dict: freqs, freq_err, nharm, components (per frequency: list of
    (harmonic, amplitude, amplitude_err, phase [cycles], phase_err)), offset,
    model (at t), residuals, rms, t0.
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    freqs = [float(f) for f in freqs]
    nharm = list(nharm) if nharm is not None else [1] * len(freqs)
    if t0 is None:
        t0 = float(np.round(np.mean(t), 1))
    w = 1.0 / e if e is not None else np.ones_like(t)

    def linfit(fr):
        A = _design(t, fr, nharm, t0)
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        return coef, A @ coef

    coef, model = linfit(freqs)
    if refine and freqs:
        from scipy.optimize import least_squares
        def resid(p):
            fr = p[:len(freqs)]
            A = _design(t, fr, nharm, t0)
            return (A @ p[len(freqs):] - y) * w
        p0 = np.concatenate([freqs, coef])
        T = max(t.max() - t.min(), 1e-3)
        lo = np.concatenate([np.array(freqs) - 0.5 / T, np.full(coef.size, -np.inf)])
        hi = np.concatenate([np.array(freqs) + 0.5 / T, np.full(coef.size, np.inf)])
        try:
            sol = least_squares(resid, p0, bounds=(lo, hi), x_scale='jac')
            freqs = list(sol.x[:len(freqs)])
            coef, model = linfit(freqs)
        except Exception:
            pass
    resid = y - model
    n = t.size
    npar = 1 + 2 * sum(nharm) + len(freqs)
    rms = float(np.sqrt(np.sum(resid ** 2) / max(n - npar, 1)))
    T = float(t.max() - t.min()) if n > 1 else np.nan
    comps, ferr = [], []
    j = 1
    for f, h in zip(freqs, nharm):
        lst = []
        for k in range(1, int(h) + 1):
            a, b = coef[j], coef[j + 1]; j += 2
            amp = float(np.hypot(a, b))
            phase = float((np.arctan2(b, a) / (2 * np.pi)) % 1.0)    # y ~ A sin(2pi(k f (t-t0) + phase))
            s_amp = float(np.sqrt(2.0 / n) * rms)
            lst.append((k, amp, s_amp, phase, float(s_amp / amp / (2 * np.pi)) if amp > 0 else np.nan))
        comps.append(lst)
        a1 = lst[0][1]
        ferr.append(float(np.sqrt(6.0 / n) * rms / (np.pi * T * a1)) if a1 > 0 and T > 0 else np.nan)
    return {'freqs': [float(f) for f in freqs], 'freq_err': ferr, 'nharm': nharm, 'components': comps,
            'offset': float(coef[0]), 'model': model, 'residuals': resid, 'rms': rms, 't0': t0}


def model_curve(fit, t):
    """Evaluate a fit (from fit_frequencies) at times t."""
    t = np.asarray(t, float)
    y = np.full(t.shape, fit['offset'])
    for f, comps in zip(fit['freqs'], fit['components']):
        for k, amp, _, phase, _ in comps:
            y = y + amp * np.sin(2 * np.pi * (k * f * (t - fit['t0']) + phase))
    return y


def peak_snr(freq, amp_spec, f0, amp, half_width=None):
    """S/N of a peak: amplitude / mean residual amplitude spectrum in a box around
    f0 (Breger et al. 1993: S/N >= 4 for a significant peak). half_width default
    1 c/d (or 10 % of the frequency range if that is smaller)."""
    if half_width is None:
        half_width = min(1.0, 0.1 * (freq[-1] - freq[0]))
    m = np.abs(freq - f0) <= half_width
    noise = float(np.mean(amp_spec[m])) if np.any(m) else float(np.mean(amp_spec))
    return amp / noise if noise > 0 else np.nan


def find_peak(freq, power, exclude=(), resolution=None):
    """Frequency of the highest peak, ignoring +/- resolution around excluded ones."""
    p = np.array(power, float)
    if resolution:
        for f in exclude:
            p[np.abs(freq - f) < resolution] = -np.inf
    i = int(np.nanargmax(p))
    # parabolic interpolation of the maximum
    if 0 < i < len(freq) - 1 and np.isfinite(p[i - 1]) and np.isfinite(p[i + 1]):
        d = p[i - 1] - 2 * p[i] + p[i + 1]
        if d < 0:
            return float(freq[i] + 0.5 * (p[i - 1] - p[i + 1]) / d * (freq[1] - freq[0]))
    return float(freq[i])


def identify(f, known, resolution, tol_factor=1.0, max_order=6):
    """Notes on a frequency: harmonic/combination of known frequencies, or a
    1 c/d (daily) or 1/yr alias of one; for an independent frequency the period
    ratio with f1 if it is typical of radial double-mode pulsation. Returns a string.
    Combinations are only formed from independent known frequencies (not from
    frequencies that are themselves harmonics)."""
    tol = tol_factor * resolution
    known = list(known)
    # which known frequencies are harmonics of an earlier one?
    indep = []
    for j, fj in enumerate(known):
        if not any(abs(fj - n * known[i]) < tol for i in range(j) for n in range(2, max_order + 1)):
            indep.append(j)
    notes = []
    for i in indep:
        fk = known[i]
        for n in range(2, max_order + 1):
            if abs(f - n * fk) < tol:
                notes.append(f"{n}f{i + 1}")
    for i in indep:
        fk = known[i]
        for d in (1.0, 2.0):
            if abs(abs(f - fk) - d) < tol:
                notes.append(f"f{i + 1}{'+' if f > fk else '-'}{d:g} c/d alias?")
        if abs(abs(f - fk) - 1 / 365.25) < tol and abs(f - fk) > 1e-9:
            notes.append(f"f{i + 1} 1/yr alias?")
    if not any(n_[0].isdigit() for n_ in notes):
        for a_i, i in enumerate(indep):
            for j in indep[a_i + 1:]:
                for a, b in ((1, 1), (1, -1), (-1, 1), (2, -1), (-1, 2), (2, 1), (1, 2)):
                    c = a * known[i] + b * known[j]
                    if c > 0 and abs(f - c) < tol:
                        notes.append(f"{'' if a == 1 else ('-' if a == -1 else a)}f{i + 1}{'+' if b > 0 else '-'}"
                                     f"{'' if abs(b) == 1 else abs(b)}f{j + 1}")
    if f < 2 * resolution:
        notes.append("< 2/T: trend or night offsets?")
    if not notes and known:
        r = period_ratio_note(f, known[0])
        if r:
            notes.append(f"with f1: {r}")
    return ", ".join(dict.fromkeys(notes))


def phase_fold(t, f, t0):
    return ((np.asarray(t) - t0) * f) % 1.0


def save_results(path, fit, info):
    """Frequency table as CSV (+ header lines with the data description)."""
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        for k, v in info.items():
            fh.write(f"# {k}: {v}\n")
        w = csv.writer(fh)
        w.writerow(['id', 'harmonic', 'frequency_c_per_d', 'freq_err', 'period_d', 'period_err',
                    'amplitude', 'amp_err', 'phase_cycles', 'phase_err', 'snr', 'note'])
        for i, (f, fe, comps) in enumerate(zip(fit['freqs'], fit['freq_err'], fit['components'])):
            snr = fit.get('snr', [np.nan] * len(fit['freqs']))[i]
            note = fit.get('notes', [''] * len(fit['freqs']))[i]
            for k, amp, sa, ph, sph in comps:
                fk = k * f
                w.writerow([f"f{i + 1}", k, f"{fk:.7f}", f"{k * fe:.7f}", f"{1 / fk:.7f}",
                            f"{k * fe / fk ** 2:.7f}", f"{amp:.5f}", f"{sa:.5f}", f"{ph:.4f}", f"{sph:.4f}",
                            f"{snr:.1f}" if k == 1 and np.isfinite(snr) else "", note if k == 1 else ""])
        fh.write(f"# zero point (offset): {fit['offset']:.5f}\n# residual rms: {fit['rms']:.5f}\n# t0 (HJD): {fit['t0']}\n")


def snr_all(t, fit, freq, half_width=None):
    """S/N of every fitted frequency: amplitude / mean amplitude spectrum of the
    RESIDUALS (after subtracting all fitted frequencies) within +/- half_width
    (default 1 c/d) - the usual criterion S/N >= 4 (Breger et al. 1993)."""
    res_spec = amplitude_spectrum(t, fit['residuals'], freq)
    if half_width is None:
        half_width = min(1.0, 0.1 * (freq[-1] - freq[0]))
    out = []
    for f, comps in zip(fit['freqs'], fit['components']):
        m = np.abs(freq - f) <= half_width
        noise = float(np.mean(res_spec[m])) if np.any(m) else float(np.mean(res_spec))
        out.append(comps[0][1] / noise if noise > 0 else np.nan)
    return out, res_spec


# --------------------------------------------------------------------------
# v4.24: quality of a fit, night-to-night residuals, period ratios,
# times of maximum and O-C diagram
# --------------------------------------------------------------------------
def nights_of(t, gap_days=0.3):
    """Index of the observing night for every time (new night after a gap > gap_days)."""
    t = np.asarray(t, float)
    o = np.argsort(t)
    idx = np.empty(t.size, int)
    idx[o] = np.concatenate([[0], np.cumsum(np.diff(t[o]) > gap_days)]) if t.size else []
    return idx


def fit_quality(fit, t, e=None):
    """
    How well do the fitted frequencies describe the data?
      rms        residual standard deviation (unit of the data)
      med_err    median photometric error (None without errors)
      ratio      rms / med_err  (~1: fully described; >1.3: more signal or errors underestimated)
      chi2_red   sum((res/e)^2) / (N - n_par)
      nights     per night: (night index, N, first time, mean residual, rms)
      night_scatter  standard deviation of the night means (zero-point problems or slow variations)
    """
    res = np.asarray(fit['residuals'], float)
    n = res.size
    npar = 1 + 2 * sum(fit['nharm']) + len(fit['freqs'])
    out = {'rms': float(fit['rms']), 'n': n, 'n_par': npar, 'med_err': None, 'ratio': np.nan, 'chi2_red': np.nan}
    if e is not None:
        e = np.asarray(e, float)
        out['med_err'] = float(np.median(e))
        out['ratio'] = out['rms'] / out['med_err'] if out['med_err'] > 0 else np.nan
        out['chi2_red'] = float(np.sum((res / e) ** 2) / max(n - npar, 1))
    ni = nights_of(t)
    nights = []
    for k in np.unique(ni):
        m = ni == k
        nights.append((int(k), int(m.sum()), float(np.min(np.asarray(t)[m])), float(np.mean(res[m])),
                       float(np.std(res[m], ddof=1)) if m.sum() > 1 else np.nan))
    out['nights'] = nights
    means = np.array([x[3] for x in nights if x[1] >= 5])
    out['night_scatter'] = float(np.std(means, ddof=1)) if means.size > 1 else np.nan
    return out


# Typical period ratios of radial double-mode pulsators (shorter / longer period).
PERIOD_RATIOS = [
    (0.740, 0.750, "RRd: F + 1O (RR Lyrae)"),
    (0.756, 0.790, "F + 1O of delta Sct / HADS"),
    (0.695, 0.725, "F + 1O of a Cepheid"),
    (0.795, 0.815, "1O + 2O (delta Sct / Cepheid)"),
]


def period_ratio_note(f_a, f_b):
    """Note on the period ratio of two independent frequencies, e.g. 'P2/P1 = 0.773: F + 1O ...'."""
    lo, hi = sorted((f_a, f_b))
    r = lo / hi           # = P(short)/P(long)
    for a, b, txt in PERIOD_RATIOS:
        if a <= r <= b:
            return f"period ratio {r:.4f}: typical of {txt}"
    return ""


def _phase_of_max(fit, i, magnitudes=True, n=2000):
    """Phase (0..1, relative to fit['t0']) of maximum brightness of frequency i,
    including its own harmonics and harmonic frequencies of it."""
    f = fit['freqs'][i]
    keep = harmonic_family(fit, i)
    sub = dict(fit, freqs=[fit['freqs'][j] for j in keep], components=[fit['components'][j] for j in keep],
               nharm=[fit['nharm'][j] for j in keep], offset=0.0)
    ph = np.linspace(0, 1, n, endpoint=False)
    y = model_curve(sub, fit['t0'] + ph / f)
    k = int(np.argmin(y) if magnitudes else np.argmax(y))
    return float(ph[k]), sub


def harmonic_family(fit, i, resolution=None):
    """Indices of frequency i and of fitted frequencies that are its harmonics (2f, 3f ...)."""
    f = fit['freqs'][i]
    if resolution is None:
        resolution = 0.5 * min(abs(a - b) for a in fit['freqs'] for b in fit['freqs'] if a != b) if len(fit['freqs']) > 1 else 1.0
        resolution = min(resolution, 0.05 * f)
    out = [i]
    for j, fj in enumerate(fit['freqs']):
        if j != i and fj > f:
            n = round(fj / f)
            if n >= 2 and abs(fj - n * f) < resolution:
                out.append(j)
    return out


def ephemeris(fit, i, magnitudes=True):
    """Ephemeris of maximum light for frequency i: (T0 [HJD], P [d], sigma_P). T0 is the
    maximum closest to the middle of the data (from the fitted light-curve shape)."""
    f = fit['freqs'][i]
    ph_max, _ = _phase_of_max(fit, i, magnitudes)
    T0 = fit['t0'] + ph_max / f
    sP = fit['freq_err'][i] / f ** 2 if np.isfinite(fit['freq_err'][i]) else np.nan
    return float(T0), float(1.0 / f), float(sP)


def times_of_maximum(t, y, fit, i, e=None, magnitudes=True, method='template', window=0.3,
                     min_points=8, subtract_others=True):
    """
    Times of maximum light, one per covered cycle of frequency i.
    The other (non-harmonic) frequencies of the fit are subtracted first.
      method 'template': the fitted light-curve shape of frequency i (with its harmonics) is
             shifted in time (and in brightness) to the points of one cycle within +/- window
             (in phase) of the expected maximum - robust for any shape and partial coverage;
      method 'poly':     parabola fitted to the points within +/- window/2 of the brightest
             point of the cycle (classical method; needs a well-covered, sharp maximum).
    A maximum is only accepted if there are >= min_points points and the expected maximum
    lies inside the observed part of the cycle (points before AND after it).
    Returns list of dicts: E (cycle number), tmax, err, n, method, rms.
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    f = fit['freqs'][i]
    P = 1.0 / f
    T0, _, _ = ephemeris(fit, i, magnitudes)
    _, sub = _phase_of_max(fit, i, magnitudes)
    if subtract_others:
        fam = harmonic_family(fit, i)
        others = [j for j in range(len(fit['freqs'])) if j not in fam]
        if others:
            oth = dict(fit, freqs=[fit['freqs'][j] for j in others], components=[fit['components'][j] for j in others],
                       nharm=[fit['nharm'][j] for j in others], offset=0.0)
            y = y - model_curve(oth, t)
    w = 1.0 / e ** 2 if e is not None else np.ones_like(t)
    E_all = np.round((t - T0) / P).astype(int)
    out = []
    for E in np.unique(E_all):
        tc = T0 + E * P
        m = np.abs(t - tc) <= window * P
        if m.sum() < min_points or not (np.any(t[m] < tc) and np.any(t[m] > tc)):
            continue
        tt, yy, ww = t[m], y[m], w[m]
        if method == 'poly':
            k = int(np.argmin(yy) if magnitudes else np.argmax(yy))
            mm = np.abs(tt - tt[k]) <= 0.5 * window * P
            if mm.sum() < max(5, min_points // 2):
                continue
            x = tt[mm] - tt[k]
            try:
                c, cov = np.polyfit(x, yy[mm], 2, w=np.sqrt(ww[mm]), cov='unscaled' if e is not None else True)
            except Exception:
                continue
            a, b = c[0], c[1]
            if (magnitudes and a <= 0) or (not magnitudes and a >= 0):
                continue
            dx = -b / (2 * a)
            if abs(dx) > 0.5 * window * P:
                continue
            g = np.array([b / (2 * a * a), -1 / (2 * a), 0.0])     # d(dx)/d(a,b,c)
            err = float(np.sqrt(max(g @ cov @ g, 0.0)))
            resid = yy[mm] - np.polyval(c, x)
            out.append({'E': int(E), 'tmax': float(tt[k] + dx), 'err': err, 'n': int(mm.sum()), 'method': 'poly',
                        'rms': float(np.std(resid))})
        else:
            # template fit: y = s(t - dt) + c ; Gauss-Newton on dt, linear in c
            dt = 0.0
            ok = True
            for _ in range(20):
                s0 = model_curve(sub, tt - dt)
                ds = (model_curve(sub, tt - dt + 1e-5 * P) - model_curve(sub, tt - dt - 1e-5 * P)) / (2e-5 * P)
                A = np.column_stack([-ds, np.ones_like(tt)])
                r = yy - s0
                try:
                    sol, *_ = np.linalg.lstsq(A * np.sqrt(ww)[:, None], r * np.sqrt(ww), rcond=None)
                except Exception:
                    ok = False
                    break
                dt += sol[0]
                if abs(dt) > window * P:
                    ok = False
                    break
                if abs(sol[0]) < 1e-7 * P:
                    break
            if not ok:
                continue
            s0 = model_curve(sub, tt - dt)
            ds = (model_curve(sub, tt - dt + 1e-5 * P) - model_curve(sub, tt - dt - 1e-5 * P)) / (2e-5 * P)
            A = np.column_stack([-ds, np.ones_like(tt)])
            c_ = np.sum(ww * (yy - s0)) / np.sum(ww)
            resid = yy - s0 - c_
            s2 = np.sum(ww * resid ** 2) / max(tt.size - 2, 1)
            try:
                cov = np.linalg.inv((A * ww[:, None]).T @ A) * (s2 if e is None else max(s2, 1.0))
                err = float(np.sqrt(cov[0, 0]))
            except np.linalg.LinAlgError:
                continue
            out.append({'E': int(E), 'tmax': float(tc + dt), 'err': err, 'n': int(tt.size), 'method': 'template',
                        'rms': float(np.sqrt(np.mean(resid ** 2)))})
    return out


def oc_analysis(maxima, T0, P, quadratic=False):
    """
    O-C of times of maximum against the ephemeris T0 + E*P. A weighted straight line
    O-C = a + b*E gives corrected T0' = T0 + a and P' = P + b; a parabola
    (quadratic=True) O-C = a + b*E + c*E^2 gives the period change dP/dE = 2c
    (dP/dt = 2c/P). Returns dict with E, oc, err, coefficients and errors.
    """
    E = np.array([m['E'] for m in maxima], float)
    T = np.array([m['tmax'] for m in maxima], float)
    s = np.array([m['err'] for m in maxima], float)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s[np.isfinite(s) & (s > 0)]) if np.any(s > 0) else 1e-3)
    oc = T - (T0 + E * P)
    res = {'E': E, 'oc': oc, 'err': s, 'T0': T0, 'P': P, 'n': int(E.size)}
    deg = 2 if quadratic else 1
    if E.size < deg + 2 or np.ptp(E) == 0:
        return res
    A = np.vander(E, deg + 1, increasing=True)              # 1, E, E^2
    W = 1.0 / s ** 2
    cov = np.linalg.inv((A * W[:, None]).T @ A)
    coef = cov @ (A * W[:, None]).T @ oc
    r = oc - A @ coef
    chi2 = float(np.sum(W * r ** 2) / max(E.size - deg - 1, 1))
    cov = cov * max(chi2, 1.0)            # scale errors if the scatter is larger than the error bars
    err = np.sqrt(np.diag(cov))
    res.update(coef=coef, coef_err=err, chi2_red=chi2, rms=float(np.std(r)),
               T0_new=T0 + coef[0], T0_new_err=err[0], P_new=P + coef[1], P_new_err=err[1])
    if quadratic:
        res['dP_dE'] = 2 * coef[2]; res['dP_dE_err'] = 2 * err[2]
        res['dP_dt'] = 2 * coef[2] / P; res['dP_dt_err'] = 2 * err[2] / P   # days per day
    return res
