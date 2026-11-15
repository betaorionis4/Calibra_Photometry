"""
Faint-target light curves (v4.44): forced photometry at a fixed sky position and
combining of frames.

Why
---
A target near the detection limit (S/N ~ 1-3 per frame) cannot be centroided: the
centroid wanders to noise peaks and the aperture "finds" flux that is not there.
Longer exposures would lose time resolution. Instead every frame is measured
at the SAME sky position with the SAME aperture, and frames are combined
afterwards:

1. Position: WCS position of the target, corrected by the mean offset
   (centroid - WCS) of the reference stars in that frame. The reference stars
   are bright, so this offset is precise; it also checks the WCS (offsets are
   reported per frame).
2. Aperture: one radius for ALL frames and all stars (target, reference and
   check stars) - by default 1.5 x the median FWHM of the selected frames
   (optimal S/N for faint stars). Because target and comparison stars use the
   same aperture in the same frame, the light outside the aperture cancels in
   the ratio.
3. Every frame gives net fluxes (ADU) of target, reference and check stars,
   kept even when the target flux is zero or negative (dropping them would
   bias the faint points).
4. Combining a group of frames = summing the fluxes of every star over the
   group. This is exactly what an aligned stack would give, without
   resampling. The magnitude is computed only after summing; below a chosen
   S/N the point becomes an upper limit ("fainter than").

Groups are chosen by the user: single frames, groups of X consecutive frames
(never across a gap in time, e.g. another night), or any set of frames marked
in the result table.
"""
import os
import numpy as np


# ------------------------------------------------------------------ measuring
def _read_frame(fpath):
    from astropy.io import fits
    from astropy.wcs import WCS
    from photometry.fastphot import as_float_image
    with fits.open(fpath) as hdul:
        header, data = hdul[0].header, hdul[0].data
        if data is None and len(hdul) > 1:
            header, data = hdul[1].header, hdul[1].data
        if data is None:
            return None, None, None
        if data.ndim == 3:
            data = data[0]
        data = as_float_image(data)
        header = header.copy()
    try:
        from photometry.master_frames import apply_on_the_fly
        data, _ = apply_on_the_fly(data, header, fpath)
    except Exception:
        pass
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = WCS(header)
    return data, header, wcs


def _jd_mid(header):
    exptime = float(header.get('EXPTIME', 0.0) or 0.0)
    jd_raw = header.get('JD') or (header.get('MJD') + 2400000.5 if header.get('MJD') else None)
    if not jd_raw and header.get('DATE-OBS'):
        try:
            from astropy.time import Time
            jd_raw = Time(header['DATE-OBS']).jd
        except Exception:
            jd_raw = None
    return (float(jd_raw) + exptime / 2.0 / 86400.0) if jd_raw else np.nan, exptime


def _pix(wcs, ra, dec):
    x, y = wcs.world_to_pixel_values(float(ra), float(dec))
    return float(x), float(y)


def _forced_flux(data, x, y, r_ap, r_in, r_out, g_eff, sky_factor=1.0):
    """Aperture flux at a FIXED position (no centroiding). None if not fully on the image."""
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from photometry.noise import sky_stats, flux_variance
    ny, nx = data.shape
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if x - r_out < -0.5 or y - r_out < -0.5 or x + r_out > nx - 0.5 or y + r_out > ny - 0.5:
        return None
    ap = CircularAperture((x, y), r=r_ap)
    an = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
    raw = float(aperture_photometry(data, ap, method='exact')['aperture_sum'][0])
    sky_med, sky_std, n_sky = sky_stats(an.to_mask(method='center').get_values(data), sigma=3.0)
    if not n_sky:
        return None
    net = raw - sky_med * ap.area
    err = float(np.sqrt(flux_variance(net, ap.area, sky_std, n_sky, g_eff, sky_factor)))
    return {'flux': float(net), 'err': err, 'x': x, 'y': y, 'sky': float(sky_med)}


def frame_fwhm(data, wcs, stars, box=15):
    """Median FWHM [px] from the (bright) reference stars of a frame (None if not measurable)."""
    try:
        from photometry.psf_fitting import refine_coordinates_psf
        lst = []
        for k, s in enumerate(stars):
            x, y = _pix(wcs, s['ra'], s['dec'])
            lst.append({'id': f"Ref_{k}", 'x': x + 1.0, 'y': y + 1.0})
        f = refine_coordinates_psf(data, lst, box, 5.0, 60000, 0, display_plots=False)
        return float(f) if f and np.isfinite(f) and f > 0 else None
    except Exception:
        return None


def measure_frames(fits_files, target, ensemble, check=None,
                   aperture_radius=5.0, annulus_inner=8.0, annulus_outer=13.0,
                   fwhm_factor=1.5, annulus_inner_gap=2.0, annulus_width=5.0,
                   gain=1.0, stack_mode='average', sky_noise_factor=1.0,
                   max_shift=4.0, initial_radius=15.0,
                   site_lat=0.0, site_long=0.0,
                   cancel_event=None, update_progress=None, log=print):
    """
    Forced photometry of `target` in every frame.

    target   : {'ra','dec'} (deg)
    ensemble : list of {'name','ra','dec', ...} - reference stars (needed for the
               position correction; may be empty -> pure WCS position)
    check    : optional {'name','ra','dec'} - measured like the target (forced)
    fwhm_factor > 0 : aperture radius = fwhm_factor x median FWHM of all frames,
                      annulus = radius + annulus_inner_gap ... + annulus_width;
    fwhm_factor <= 0: aperture_radius / annulus_inner / annulus_outer as given.

    Returns (frames, info). frames: list (time order) of dicts with
      file, jd, hjd, exptime, airmass, fwhm, dx, dy (ref offset px), n_off_refs,
      target {'flux','err',...} or None, refs {name: {...}}, check {...} or None
    info: {'r_ap','r_in','r_out','median_fwhm','n_files','n_used','offsets_rms'}
    """
    from photometry.noise import effective_gain, stack_count
    from photometry.fastphot import track_centroid
    from photometry.time_series import get_hjd
    ens = [s for s in (ensemble or []) if s.get('ra') is not None and s.get('dec') is not None]
    n = len(fits_files)

    # pass 1: FWHM of every frame (only when the aperture follows the seeing)
    fwhms = {}
    if fwhm_factor and fwhm_factor > 0:
        for i, f in enumerate(fits_files):
            if cancel_event is not None and cancel_event.is_set():
                return [], {}
            if update_progress:
                update_progress(30.0 * i / max(n, 1))
            data, hdr, wcs = _read_frame(f)
            if data is None:
                continue
            fw = frame_fwhm(data, wcs, ens) if ens else None
            if fw is None:
                try:
                    fw = float(hdr.get('FWHM')) if hdr.get('FWHM') else None
                except (TypeError, ValueError):
                    fw = None
            if fw:
                fwhms[f] = fw
        med_fwhm = float(np.median(list(fwhms.values()))) if fwhms else None
        if med_fwhm:
            r_ap = fwhm_factor * med_fwhm
            r_in = r_ap + annulus_inner_gap
            r_out = r_in + annulus_width
        else:
            log("Faint target: FWHM not measurable - aperture from Settings used.")
            r_ap, r_in, r_out = aperture_radius, annulus_inner, annulus_outer
    else:
        med_fwhm = None
        r_ap, r_in, r_out = aperture_radius, annulus_inner, annulus_outer
    log(f"Faint target: forced photometry, aperture r = {r_ap:.2f} px, annulus {r_in:.2f}-{r_out:.2f} px"
        + (f" (median FWHM {med_fwhm:.2f} px x {fwhm_factor:g})" if med_fwhm else ""))

    frames = []
    ref_locked = {}          # name -> last good offset (for tracking)
    p0 = 30.0 if fwhms else 0.0
    for i, f in enumerate(fits_files):
        if cancel_event is not None and cancel_event.is_set():
            log("Operation cancelled by user.")
            break
        if update_progress:
            update_progress(p0 + (100.0 - p0) * i / max(n, 1))
        try:
            data, hdr, wcs = _read_frame(f)
            if data is None:
                continue
            if not wcs.has_celestial:
                log(f"  - {os.path.basename(f)}: no WCS (plate-solve first) - skipped")
                continue
            jd, exptime = _jd_mid(hdr)
            g_eff, _ = effective_gain(gain, {'STACKCNT': stack_count(hdr)}, stack_mode)
            sig_w = max((fwhms.get(f) or med_fwhm or 3.0) / 2.355, 1.0)
            half_box = max(5, int(np.ceil(3 * sig_w)) + 1)
            # reference stars: centroid, offset to WCS, flux at the centroid
            refs, offs, wts = {}, [], []
            for s in ens:
                xw, yw = _pix(wcs, s['ra'], s['dec'])
                ms = max_shift if s['name'] in ref_locked else initial_radius
                xp, yp = (xw + ref_locked[s['name']][0], yw + ref_locked[s['name']][1]) if s['name'] in ref_locked else (xw, yw)
                try:
                    xc, yc, ok = track_centroid(data, xp, yp, ms, sigma_w=sig_w, half_box=half_box)
                except Exception:
                    xc, yc, ok = xp, yp, False
                m = _forced_flux(data, xc, yc, r_ap, r_in, r_out, g_eff, sky_noise_factor)
                if m is None:
                    continue
                m['ok_centroid'] = bool(ok)
                m['dx'], m['dy'] = xc - xw, yc - yw
                refs[s['name']] = m
                if ok and m['flux'] > 0 and np.hypot(m['dx'], m['dy']) < max(initial_radius, 1.0):
                    ref_locked[s['name']] = (m['dx'], m['dy'])
                    offs.append((m['dx'], m['dy']))
                    wts.append(max(m['flux'] / max(m['err'], 1e-9), 0.0) ** 2)
            if offs:
                w = np.asarray(wts) if np.sum(wts) > 0 else np.ones(len(offs))
                dx = float(np.sum(w * np.array([o[0] for o in offs])) / np.sum(w))
                dy = float(np.sum(w * np.array([o[1] for o in offs])) / np.sum(w))
            else:
                dx = dy = 0.0
            xt, yt = _pix(wcs, target['ra'], target['dec'])
            tm = _forced_flux(data, xt + dx, yt + dy, r_ap, r_in, r_out, g_eff, sky_noise_factor)
            if tm is None:
                log(f"  - {os.path.basename(f)}: target (or its sky annulus) not on the image - skipped")
                continue
            cm = None
            if check and check.get('ra') is not None:
                xc_, yc_ = _pix(wcs, check['ra'], check['dec'])
                cm = _forced_flux(data, xc_ + dx, yc_ + dy, r_ap, r_in, r_out, g_eff, sky_noise_factor)
            lat = hdr.get('SITELAT', site_lat)
            lon = hdr.get('SITELONG', site_long)
            hjd = get_hjd(None, target['ra'], target['dec'], hdr, lat, lon) if np.isfinite(jd) else None
            try:
                airmass = float(hdr.get('AIRMASS', np.nan))
            except (TypeError, ValueError):
                airmass = np.nan
            frames.append({'file': os.path.basename(f), 'path': f, 'jd': jd, 'hjd': hjd if hjd else jd,
                           'exptime': exptime, 'airmass': airmass, 'fwhm': fwhms.get(f, med_fwhm or np.nan),
                           'dx': dx, 'dy': dy, 'n_off_refs': len(offs),
                           'target': tm, 'refs': refs, 'check': cm})
        except Exception as e:
            log(f"  - {os.path.basename(f)}: {e}")
    if update_progress:
        update_progress(100)
    frames.sort(key=lambda fr: fr['hjd'])
    offs = np.array([[fr['dx'], fr['dy']] for fr in frames if fr['n_off_refs']]) if frames else np.zeros((0, 2))
    info = {'r_ap': r_ap, 'r_in': r_in, 'r_out': r_out, 'median_fwhm': med_fwhm, 'n_files': n,
            'n_used': len(frames),
            'offset_mean': tuple(np.mean(offs, axis=0)) if len(offs) else (np.nan, np.nan),
            'offset_scatter': tuple(np.std(offs, axis=0)) if len(offs) > 1 else (np.nan, np.nan)}
    return frames, info


# ------------------------------------------------------------------ grouping
def groups_single(frames):
    return [[i] for i in range(len(frames))]


def groups_consecutive(frames, size, max_gap_factor=5.0):
    """Groups of `size` consecutive frames (time order). A group is closed early at a gap in
    time larger than max_gap_factor x the median frame spacing (e.g. another night), so a
    group never spans a gap. The last group of a run may have fewer frames."""
    size = max(int(size), 1)
    n = len(frames)
    if n == 0:
        return []
    t = np.array([fr['jd'] for fr in frames], dtype=float)
    dt = np.diff(t)
    med = float(np.median(dt)) if dt.size else 0.0
    gap = (dt > max_gap_factor * med) if med > 0 else np.zeros(dt.size, bool)
    groups, cur = [], [0]
    for i in range(1, n):
        if gap[i - 1] or len(cur) >= size:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    return groups


def merge_groups(groups, rows):
    """Merge the groups at positions `rows` (e.g. rows selected in the table) into one group."""
    rows = sorted({int(r) for r in rows if 0 <= int(r) < len(groups)})
    if len(rows) < 2:
        return [list(g) for g in groups]
    merged = sorted(i for r in rows for i in groups[r])
    out = [list(g) for k, g in enumerate(groups) if k not in rows]
    out.append(merged)
    return sorted(out, key=lambda g: g[0])


def split_groups(groups, rows):
    """Split the groups at positions `rows` back into single frames."""
    rows = {int(r) for r in rows}
    out = []
    for k, g in enumerate(groups):
        if k in rows:
            out.extend([[i] for i in g])
        else:
            out.append(list(g))
    return sorted(out, key=lambda g: g[0])


# ------------------------------------------------------------------ combining
def _m(f):
    return -2.5 * np.log10(f) if (f is not None and np.isfinite(f) and f > 0) else np.nan


def combine(frames, groups, lc_mode='calibrated', ensemble=None, check=None,
            target_bv=0.0, coeff_term=0.0, flux_per_second=False, snr_limit=3.0):
    """
    One light-curve point per group (list of frame indices) - fluxes of every star are
    summed over the frames of the group. Result dicts are compatible with
    run_time_series_photometry() (plot, table, CSV, AAVSO report) plus:
      n_frames, t_span_min, frames ('first .. last'), upper_limit, mag_limit,
      wcs_dx/wcs_dy (mean reference-star offset, px).
    A reference star is used in a group only if it was measured in ALL frames of the group.
    """
    from photometry.time_series import ut_date, ut_hms
    ens = {s['name']: s for s in (ensemble or [])}
    out = []
    for g in groups:
        frs = [frames[i] for i in g]
        nfr = len(frs)
        exp = np.array([float(fr['exptime'] or 0.0) for fr in frs])
        w = exp if exp.sum() > 0 else np.ones(nfr)
        jd = float(np.sum(w * np.array([fr['jd'] for fr in frs])) / np.sum(w))
        hjd = float(np.sum(w * np.array([fr['hjd'] for fr in frs])) / np.sum(w))
        span = (max(fr['jd'] + fr['exptime'] / 172800.0 for fr in frs)
                - min(fr['jd'] - fr['exptime'] / 172800.0 for fr in frs)) * 1440.0
        Ft = float(sum(fr['target']['flux'] for fr in frs))
        sFt = float(np.sqrt(sum(fr['target']['err'] ** 2 for fr in frs)))
        snr = Ft / sFt if sFt > 0 else 0.0
        names = [n for n in ens if all(n in fr['refs'] for fr in frs)] if ens else []
        R = {n: (float(sum(fr['refs'][n]['flux'] for fr in frs)),
                 float(np.sqrt(sum(fr['refs'][n]['err'] ** 2 for fr in frs)))) for n in names}
        R = {n: v for n, v in R.items() if v[0] > 0}
        chk = None
        if check is not None and all(fr.get('check') for fr in frs):
            chk = (float(sum(fr['check']['flux'] for fr in frs)),
                   float(np.sqrt(sum(fr['check']['err'] ** 2 for fr in frs))))
        am = [fr['airmass'] for fr in frs if np.isfinite(fr.get('airmass', np.nan))]
        rec = {'file': frs[0]['file'] if nfr == 1 else f"{frs[0]['file']} .. {frs[-1]['file']}",
               'date_ut': ut_date(jd), 'ut_mid': ut_hms(jd), 'jd': jd, 'hjd': hjd,
               'n_frames': nfr, 't_span_min': round(span, 2),
               'airmass': float(np.mean(am)) if am else np.nan,
               'exptime': float(exp.sum()), 'lc_mode': lc_mode,
               'target_adu': Ft, 'target_adu_err': sFt, 'snr': snr,
               'fwhm': float(np.nanmean([fr['fwhm'] for fr in frs])),
               'wcs_dx': float(np.mean([fr['dx'] for fr in frs])), 'wcs_dy': float(np.mean([fr['dy'] for fr in frs])),
               'n_ensemble': len(R), 'upper_limit': False, 'mag_limit': np.nan, 'flag': 'OK'}
        lim_flux = max(Ft, 0.0) + snr_limit * sFt        # flux of the "fainter than" limit
        is_lim = snr < snr_limit

        if lc_mode == 'calibrated':
            if not R:
                rec.update({'mag': np.nan, 'mag_err': np.nan, 'check_mag': np.nan, 'zp_avg': np.nan,
                            'zp_err': np.nan, 'ref_mags': {}, 'flag': 'FAILED'})
                out.append(rec)
                continue
            bv_t = target_bv if (target_bv is not None and np.isfinite(target_bv)) else 0.0
            zv = {}
            for n, (F, _) in R.items():
                s = ens[n]
                bv = s.get('bv_std')
                bv = bv if (bv is not None and np.isfinite(bv)) else 0.0
                zv[n] = float(s['mag_std']) - (_m(F) + coeff_term * bv)
            zvals = np.array(list(zv.values()))
            zp = float(np.mean(zvals))
            zp_err = float(np.std(zvals, ddof=1) / np.sqrt(len(zvals))) if len(zvals) > 1 else 0.01
            mag = _m(Ft) + coeff_term * bv_t + zp
            mag_err = float(np.sqrt((1.0857 / snr) ** 2 + zp_err ** 2)) if snr > 0 else np.nan
            rec.update({'zp_avg': zp, 'zp_err': zp_err,
                        'ref_mags': {n: float(ens[n]['mag_std']) - zv[n] + zp for n in zv}})
            cm = np.nan
            if chk and chk[0] > 0:
                cs = check if isinstance(check, dict) else {}
                cbv = cs.get('bv_std')
                cbv = cbv if (cbv is not None and np.isfinite(cbv)) else 0.0
                cm = _m(chk[0]) + coeff_term * cbv + zp
            rec['check_mag'] = cm
            if is_lim:
                rec.update({'mag': np.nan, 'mag_err': np.nan, 'upper_limit': True,
                            'mag_limit': _m(lim_flux) + coeff_term * bv_t + zp, 'flag': 'LIMIT'})
            else:
                rec.update({'mag': mag, 'mag_err': mag_err})
        else:
            if lc_mode == 'relative':
                if not R:
                    rec.update({'flux': np.nan, 'flux_err': np.nan, 'mag': np.nan, 'mag_err': np.nan,
                                'check_flux': np.nan, 'check_mag': np.nan, 'flag': 'FAILED'})
                    out.append(rec)
                    continue
                S = sum(v[0] for v in R.values())
                sS = float(np.sqrt(sum(v[1] ** 2 for v in R.values())))
                y = Ft / S
                sy = float(np.sqrt((sFt / S) ** 2 + (Ft * sS / S ** 2) ** 2))
                rec.update({'flux_unit': 'ratio', 'ens_flux': S / nfr, 'ens_flux_err': sS / nfr})
                if chk:
                    rec['check_flux'] = chk[0] / S
                    rec['check_flux_err'] = float(np.sqrt((chk[1] / S) ** 2 + (chk[0] * sS / S ** 2) ** 2))
                lim = lim_flux / S
                if len(R) >= 2:
                    rec['ref_fluxes'] = {n: R[n][0] / (S - R[n][0]) if S - R[n][0] > 0 else np.nan for n in R}
                else:
                    rec['ref_fluxes'] = {}
            else:   # raw: mean flux per frame, or per second
                if flux_per_second and exp.sum() > 0:
                    scale, unit = 1.0 / exp.sum(), 'ADU/s'
                else:
                    scale, unit = 1.0 / nfr, 'ADU'
                y, sy = Ft * scale, sFt * scale
                lim = lim_flux * scale
                rec['flux_unit'] = unit
                if chk:
                    rec['check_flux'], rec['check_flux_err'] = chk[0] * scale, chk[1] * scale
                rec['ref_fluxes'] = {n: R[n][0] * scale for n in R}
            rec.setdefault('check_flux', np.nan)
            rec.setdefault('check_flux_err', np.nan)
            rec['check_mag'] = _m(rec['check_flux'])
            rec['ref_mags'] = {n: _m(v) for n, v in rec['ref_fluxes'].items()}
            rec.update({'flux': y, 'flux_err': sy})
            if is_lim:
                rec.update({'mag': np.nan, 'mag_err': np.nan, 'upper_limit': True, 'mag_limit': _m(lim), 'flag': 'LIMIT'})
            else:
                rec.update({'mag': _m(y), 'mag_err': 1.0857 * sy / y if y > 0 else np.nan})
        out.append(rec)
    return out
