import os
from python import paths
from python.version import __version__
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_2dg
import json

# Session-based caches to avoid re-calculating photometry/PSF when roles change
# Now persisted to disk to survive restarts
_SESSION_PHOT_CACHE = {} # (fpath, ra, dec, ap, ann_in, ann_out) -> measure_star dict
_SESSION_FWHM_CACHE = {} # (fpath, aperture_radius) -> median_fwhm
_SESSION_HEADER_CACHE = {} # fpath -> {jd, airmass, lat, lon}

# v4.16: new cache file name - measurements cached by older versions used the old
# error model and are therefore not reused.

def load_session_cache():
    global _SESSION_PHOT_CACHE, _SESSION_FWHM_CACHE
    if not os.path.exists(paths.TIME_SERIES_CACHE_FILE):
        return
    try:
        with open(paths.TIME_SERIES_CACHE_FILE, 'r') as f:
            data = json.load(f)
            # JSON doesn't support tuple keys, so we convert back
            for k, v in data.get('phot', {}).items():
                parts = k.split('|')
                if len(parts) == 6:
                    key = (parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                    _SESSION_PHOT_CACHE[key] = v
            for k, v in data.get('fwhm', {}).items():
                parts = k.split('|')
                if len(parts) == 2:
                    key = (parts[0], float(parts[1]))
                    _SESSION_FWHM_CACHE[key] = v
            for k, v in data.get('header', {}).items():
                _SESSION_HEADER_CACHE[k] = v
            print(f"Photometry cache loaded: {len(_SESSION_PHOT_CACHE)} measurements, {len(_SESSION_FWHM_CACHE)} FWHM entries.")
    except Exception as e:
        print(f"Warning: Could not load photometry cache: {e}")

def save_session_cache():
    try:
        os.makedirs(os.path.dirname(paths.TIME_SERIES_CACHE_FILE), exist_ok=True)
        phot_export = {"|".join(map(str, k)): v for k, v in _SESSION_PHOT_CACHE.items()}
        fwhm_export = {"|".join(map(str, k)): v for k, v in _SESSION_FWHM_CACHE.items()}
        with open(paths.TIME_SERIES_CACHE_FILE, 'w') as f:
            json.dump({'phot': phot_export, 'fwhm': fwhm_export, 'header': _SESSION_HEADER_CACHE}, f)
    except Exception as e:
        print(f"Warning: Could not save photometry cache: {e}")

def clear_session_cache():
    """ Wipes all in-memory and on-disk photometry caches for a fresh start. """
    global _SESSION_PHOT_CACHE, _SESSION_FWHM_CACHE, _SESSION_HEADER_CACHE
    _SESSION_PHOT_CACHE = {}
    _SESSION_FWHM_CACHE = {}
    _SESSION_HEADER_CACHE = {}
    if os.path.exists(paths.TIME_SERIES_CACHE_FILE):
        try:
            os.remove(paths.TIME_SERIES_CACHE_FILE)
        except:
            pass
    print("Photometry cache cleared.")

def ut_hms(jd):
    """v4.42: UT of a (mid-exposure, UTC) Julian date as 'HH:MM:SS' ('' if unknown)."""
    try:
        jd = float(jd)
        if not np.isfinite(jd) or jd <= 0:
            return ''
        frac = (jd + 0.5) % 1.0
        s = int(round(frac * 86400.0)) % 86400
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    except (TypeError, ValueError):
        return ''


def ut_date(jd):
    """v4.43: UT date of a (mid-exposure, UTC) Julian date as 'YYYY-MM-DD' ('' if unknown)."""
    try:
        jd = float(jd)
        if not np.isfinite(jd) or jd <= 0:
            return ''
        frac = (jd + 0.5) % 1.0
        if int(round(frac * 86400.0)) >= 86400:      # rounds up to 00:00:00 of the next day (consistent with ut_hms)
            jd += 0.5 / 86400.0
        from astropy.time import Time
        return Time(jd, format='jd', scale='utc').strftime('%Y-%m-%d')
    except (TypeError, ValueError):
        return ''


# columns of the light-curve CSV files that are kept internally only (v4.42: 'jd' replaced by 'ut_mid')
CSV_HIDDEN = ('jd',)


def csv_fields(results):
    """Field names of a light-curve CSV: union of all keys in first-seen order, without CSV_HIDDEN."""
    keys = []
    for r in results:
        keys += [k for k in r if k not in keys and k not in CSV_HIDDEN]
    return keys


def _site_deg(header, keys, default=0.0, plain=False):
    """Site coordinate from the first present header keyword: number, or sexagesimal text
    ('38 27 33', '-01:04:57', '38d27m33s'); default if missing or unreadable."""
    for k in keys:
        v = header.get(k)
        if v is None or str(v).strip() == '':
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
        if plain:
            continue
        try:
            from astropy.coordinates import Angle
            return float(Angle(str(v).strip(), unit=u.deg).deg)
        except Exception:
            continue
    try:
        return float(default or 0.0)
    except (TypeError, ValueError):
        try:
            from astropy.coordinates import Angle
            return float(Angle(str(default).strip(), unit=u.deg).deg)
        except Exception:
            return 0.0


def get_hjd(time_str, ra, dec, header, site_lat=0.0, site_long=0.0):
    """ Calculates Heliocentric Julian Date (HJD) from DATE-OBS and coordinates. """
    try:
        # 1. Parse Time & Offset to Mid-Exposure
        exptime = header.get('EXPTIME', 0.0)
        t_offset_days = (float(exptime) / 2.0) / 86400.0
        
        t_val = header.get('JD') or header.get('JD_SOBJ')
        if t_val:
            t = Time(float(t_val) + t_offset_days, format='jd', scale='utc')
        else:
            # Fallback to DATE-OBS
            t_str = header.get('DATE-OBS')
            if not t_str: return None
            t = Time(t_str, format='isot', scale='utc') + t_offset_days * u.day
        
        # 2. Get Observer Location
        # Note: the session header cache stores missing keywords as None, so a
        # plain .get(key, default) is not enough (would give None -> HJD failed
        # and silently fell back to JD on cached re-runs).
        # v4.41: sexagesimal values (MaxIm/ACP: SITELAT = '38 27 33', SITELONG = '-01 04 57') are
        # understood; before, they raised an error and the plain JD (no heliocentric correction)
        # was used as "HJD". The site changes the HJD by < 0.03 s - an unreadable site is
        # replaced by the geocentre instead of dropping the correction.
        lon = _site_deg(header, ('SITELONG', 'LONG-OBS', 'OBSLONG'), site_long)
        lat = _site_deg(header, ('SITELAT', 'LAT-OBS', 'OBSLAT'), site_lat)
        height = _site_deg(header, ('SITEELEV', 'ALT-OBS', 'OBSALT'), 0.0, plain=True)
        loc = EarthLocation(lon=lon*u.deg, lat=lat*u.deg, height=height*u.m)
        
        # 3. Target Coordinates
        target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        
        # 4. Calculate Light Travel Time to Sun
        ltt_helio = t.light_travel_time(target, 'heliocentric', location=loc)
        t_helio = t + ltt_helio
        
        return t_helio.jd
    except Exception as e:
        print(f"Warning: Could not calculate HJD: {e}")
        return None

def measure_star(image_data, wcs, ra, dec, aperture_radius, annulus_inner, annulus_outer, gain=1.0, max_shift=4.0,
                 sky_noise_factor=1.0, sigma_w=1.5):
    """ Performs targeted aperture photometry on a specific RA/Dec.

    v4.16: `gain` is the EFFECTIVE gain (see python/noise.py); the error uses
    the net flux, the clipped sky scatter and the number of sky pixels used.
    Stars whose aperture is not fully on the image return None.
    """
    from python.noise import flux_variance, sky_stats
    try:
        # 1. Convert RA/Dec to pixel coordinates using WCS
        x_init, y_init = wcs.world_to_pixel(SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs'))
        
        # 2. Refine Centroid
        # v4.17: coarse peak search within max_shift + fast windowed centroid
        # (python/fastphot.py; same accuracy as the former 2-D Gaussian fit
        # centroid_2dg, ~100x faster). Falls back to the WCS position if the star
        # is not found within max_shift (protection against snapping to a neighbour).
        from python.fastphot import track_centroid
        size = 15
        x_int, y_int = int(np.round(x_init)), int(np.round(y_init))
        if x_int < size or y_int < size or x_int > image_data.shape[1]-size or y_int > image_data.shape[0]-size:
            return None
        half_box = max(5, int(np.ceil(3 * sigma_w)) + 1)
        x_refined, y_refined, _ = track_centroid(image_data, float(x_init), float(y_init), max_shift,
                                                 sigma_w=sigma_w, half_box=half_box)
        
        # 3. Aperture Photometry
        ny_img, nx_img = image_data.shape
        if (x_refined - aperture_radius < -0.5 or y_refined - aperture_radius < -0.5 or
                x_refined + aperture_radius > nx_img - 0.5 or y_refined + aperture_radius > ny_img - 0.5):
            return None
        pos = (x_refined, y_refined)
        aperture = CircularAperture(pos, r=aperture_radius)
        annulus = CircularAnnulus(pos, r_in=annulus_inner, r_out=annulus_outer)
        
        phot_table = aperture_photometry(image_data, aperture, method='exact')
        raw_flux = phot_table['aperture_sum'][0]
        
        # Background subtraction
        annulus_mask = annulus.to_mask(method='center')
        bkg_pixels = annulus_mask.get_values(image_data)
        bkg_median, bkg_std, n_sky = sky_stats(bkg_pixels, sigma=3.0)
        if n_sky == 0:
            return None
        
        net_flux = raw_flux - (bkg_median * aperture.area)
        
        # Error calculation (v4.16 noise model)
        variance = flux_variance(net_flux, aperture.area, bkg_std, n_sky, gain, sky_noise_factor)
        flux_err = np.sqrt(variance)
        snr = net_flux / flux_err if flux_err > 0 and net_flux > 0 else 0
        mag_err = 1.0857 / snr if snr > 0 else np.nan
        
        return {
            'x': x_refined, 'y': y_refined,
            'net_flux': net_flux,
            'flux_err': flux_err,
            'mag_inst': -2.5 * np.log10(net_flux) if net_flux > 0 else np.nan,
            'mag_err': mag_err,
            'snr': snr
        }
    except Exception as e:
        print(f"Error measuring star at {ra}, {dec}: {e}")
        return None

def run_time_series_photometry(fits_files, target_ra, target_dec, 
                               ensemble_stars, # List of dicts: {'ra', 'dec', 'mag_std', 'bv_std', 'name'}
                               check_star,     # Dict: {'ra', 'dec', 'bv_std', 'name'} or None
                               target_bv,
                               coeff_term, coeff_color, 
                               aperture_radius, annulus_inner, annulus_outer,
                               gain=1.0, k_coeff=0.15, filter_name='V',
                               observer_name="Calibra User",
                               site_lat=0.0, site_long=0.0,
                               cancel_event=None, update_progress=None,
                               use_flexible_aperture=False, 
                               aperture_fwhm_factor=2.0, annulus_inner_gap=2.0, annulus_width=5.0,
                               print_psf_fitting=False,
                               centroid_initial_radius=15.0, centroid_tracking_shift=4.0,
                               lc_mode='calibrated', flux_per_second=False,
                               stack_mode='average', sky_noise_factor=1.0):
    """
    Main loop for time-series photometry of ONE target.

    lc_mode selects how the target brightness is expressed:
      'calibrated' : (default, unchanged behaviour) ensemble differential photometry
                     using the catalogue magnitudes of the reference stars ->
                     standard magnitudes, AAVSO-ready.
      'relative'   : ensemble relative flux. Only the measured ADU of the
                     reference stars are used (catalogue magnitudes ignored):
                     rel_flux = F_target / sum(F_ref). Requires >= 1 ref star.
      'raw'        : raw net target flux in ADU (background subtracted) with
                     CCD-equation uncertainties. No reference or check star
                     needed. If reference/check stars are supplied they are
                     measured for display only and NOT used for the target.
    flux_per_second : in 'raw' mode, divide all fluxes by EXPTIME (ADU/s).

    v4.19: implemented as a one-target call of run_multi_target_photometry().
    """
    target = {'name': 'Target', 'ra': target_ra, 'dec': target_dec, 'ensemble': ensemble_stars,
              'check': check_star, 'target_bv': target_bv, 'lc_mode': lc_mode}
    out = run_multi_target_photometry(
        fits_files, [target], coeff_term=coeff_term, coeff_color=coeff_color,
        aperture_radius=aperture_radius, annulus_inner=annulus_inner, annulus_outer=annulus_outer,
        gain=gain, k_coeff=k_coeff, filter_name=filter_name, observer_name=observer_name,
        site_lat=site_lat, site_long=site_long, cancel_event=cancel_event, update_progress=update_progress,
        use_flexible_aperture=use_flexible_aperture, aperture_fwhm_factor=aperture_fwhm_factor,
        annulus_inner_gap=annulus_inner_gap, annulus_width=annulus_width, print_psf_fitting=print_psf_fitting,
        centroid_initial_radius=centroid_initial_radius, centroid_tracking_shift=centroid_tracking_shift,
        flux_per_second=flux_per_second, stack_mode=stack_mode, sky_noise_factor=sky_noise_factor,
        verbose=True)
    return out[0]


def run_multi_target_photometry(fits_files, targets,
                                coeff_term=0.0, coeff_color=0.0,
                                aperture_radius=5.0, annulus_inner=7.0, annulus_outer=13.0,
                                gain=1.0, k_coeff=0.15, filter_name='V',
                                observer_name="Calibra User",
                                site_lat=0.0, site_long=0.0,
                                cancel_event=None, update_progress=None,
                                use_flexible_aperture=False,
                                aperture_fwhm_factor=2.0, annulus_inner_gap=2.0, annulus_width=5.0,
                                print_psf_fitting=False,
                                centroid_initial_radius=15.0, centroid_tracking_shift=4.0,
                                flux_per_second=False,
                                stack_mode='average', sky_noise_factor=1.0, verbose=None):
    """
    v4.19: light curves of SEVERAL targets from the same image sequence.

    Every FITS file is read and converted only once; all targets (each with its
    own ensemble, check star and light-curve mode) are measured from it.

    targets: list of dicts with keys
        'name', 'ra', 'dec' (deg), 'ensemble' (list of {'ra','dec','mag_std','bv_std','name'}),
        'check' (dict or None), 'target_bv' (float), 'lc_mode' ('calibrated'|'relative'|'raw')
    Returns a list of (results, message) in the order of `targets`
    (results is None if nothing could be measured for that target).
    """
    if verbose is None:
        verbose = (len(targets) == 1)
    cache_hits = 0
    
    from astropy.wcs import WCS
    from python.psf_fitting import refine_coordinates_psf
    import astropy.units as u
    import warnings
    from astropy.utils.exceptions import AstropyWarning
    warnings.simplefilter('ignore', category=AstropyWarning)

    # Per-target state
    states = []
    for t in targets:
        mode = (t.get('lc_mode') or 'calibrated').lower()
        if mode not in ('calibrated', 'relative', 'raw'):
            raise ValueError(f"Unknown lc_mode '{mode}'")
        ens = t.get('ensemble') or []
        chk = t.get('check')
        msg = None
        if mode == 'calibrated' and not ens:
            msg = "Calibrated mode requires at least one reference star."
        elif mode == 'relative' and not ens:
            msg = "Relative-flux mode requires at least one reference star."
        if chk: chk['locked'] = False
        for s in ens: s['locked'] = False
        states.append({'name': t.get('name', 'Target'), 'ra': float(t['ra']), 'dec': float(t['dec']),
                       'locked': False, 'ens': ens, 'check': chk, 'bv': t.get('target_bv', 0.0),
                       'mode': mode, 'results': [], 'error': msg})
    active = [st for st in states if st['error'] is None]

    modes = sorted({st['mode'] for st in active})
    print(f"Starting time-series analysis on {len(fits_files)} files... "
          f"({len(active)} target(s); mode: {', '.join(modes) if modes else '-'})")
    if use_flexible_aperture:
        print(f"  -> Flexible Aperture Enabled (Factor: {aperture_fwhm_factor}x FWHM)")
    
    # v4.27: measurements with / without on-the-fly calibration are cached separately
    try:
        from python.master_frames import state_tag as _cal_state_tag
        _caltag = _cal_state_tag()
    except Exception:
        _caltag = ''
    if _caltag:
        print("Bias/flat calibration on the fly is ON (Pre-processing).")

    for i, fpath in enumerate(fits_files):
        if not active:
            break
        if cancel_event and cancel_event.is_set():
            print("Operation cancelled by user.")
            break

        if update_progress:
            update_progress((i / len(fits_files)) * 100)
            
        try:
            # --- FAST PATH: Check if we can skip FITS open entirely ---
            header_info = _SESSION_HEADER_CACHE.get(fpath)
            fwhm_cache_key = (fpath, aperture_radius)
            cached_fwhm = _SESSION_FWHM_CACHE.get(fwhm_cache_key)
            
            can_skip = False
            if header_info and cached_fwhm is not None:
                if use_flexible_aperture:
                    f_ap = cached_fwhm * aperture_fwhm_factor
                    f_ann_in = f_ap + annulus_inner_gap
                    f_ann_out = f_ann_in + annulus_width
                else:
                    f_ap, f_ann_in, f_ann_out = aperture_radius, annulus_inner, annulus_outer
                
                from python.noise import effective_gain as _eg
                _g_fast, _ = _eg(gain, {'STACKCNT': header_info.get('STACKCNT', 1) or 1}, stack_mode)
                _pref = f"{fpath}#g={_g_fast:.5g};k={float(sky_noise_factor):.3g}" + (f";{_caltag}" if _caltag else "")
                def is_cached(ra, dec):
                    k = (_pref, round(ra, 6), round(dec, 6), round(f_ap, 2), round(f_ann_in, 2), round(f_ann_out, 2))
                    return k in _SESSION_PHOT_CACHE
                
                can_skip = True
                for st in active:
                    if not (is_cached(st['ra'], st['dec']) and all(is_cached(s['ra'], s['dec']) for s in st['ens'])
                            and (not st['check'] or is_cached(st['check']['ra'], st['check']['dec']))):
                        can_skip = False
                        break

            if can_skip:
                jd = header_info['jd']
                airmass = header_info['airmass']
                hdr_for_hjd, hjd_lat, hjd_lon = header_info, header_info['lat'], header_info['lon']
                median_fwhm = cached_fwhm
                curr_ap, curr_ann_in, curr_ann_out = f_ap, f_ann_in, f_ann_out
                data, wcs = None, None
                is_fwhm_cached = True
                frame_exptime = header_info.get('EXPTIME', 0.0)
                frame_stackcnt = header_info.get('STACKCNT', 1) or 1
            else:
                with fits.open(fpath) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data
                    if data is None and len(hdul) > 1:
                        data = hdul[1].data
                        header = hdul[1].header
                    if data is None: continue
                    if data.ndim == 3: data = data[0]
                    # v4.17: convert ONCE per frame (photutils otherwise converts the
                    # whole integer image to float for every star: ~0.1 s per star)
                    from python.fastphot import as_float_image
                    data = as_float_image(data)
                    # v4.27: bias/dark/flat calibration on the fly (Pre-processing option)
                    from python.master_frames import apply_on_the_fly
                    data, _ = apply_on_the_fly(data, header, fpath)

                    wcs = WCS(header)
                    lat = header.get('SITELAT', site_lat)
                    lon = header.get('SITELONG', site_long)
                    
                    exptime = header.get('EXPTIME', 0.0)
                    frame_exptime = exptime
                    from python.noise import stack_count
                    frame_stackcnt = stack_count(header)
                    t_offset_days = (float(exptime) / 2.0) / 86400.0
                    
                    jd_raw = header.get('JD') or (header.get('MJD') + 2400000.5 if header.get('MJD') else None)
                    if not jd_raw:
                        t_obs = header.get('DATE-OBS')
                        if t_obs:
                            try: from astropy.time import Time; jd_raw = Time(t_obs).jd
                            except: jd_raw = 0
                        else:
                            jd_raw = 0
                    
                    jd = jd_raw + t_offset_days
                    
                    airmass = header.get('AIRMASS', 1.0)
                    hdr_for_hjd, hjd_lat, hjd_lon = header, lat, lon
                    
                    _SESSION_HEADER_CACHE[fpath] = {
                        'jd': jd, 'airmass': airmass, 'lat': lat, 'lon': lon,
                        'JD': header.get('JD'), 'DATE-OBS': header.get('DATE-OBS'),
                        'EXPTIME': header.get('EXPTIME', 0.0),
                        'STACKCNT': frame_stackcnt,
                        'SITELONG': header.get('SITELONG'), 'SITELAT': header.get('SITELAT'), 'SITEELEV': header.get('SITEELEV')
                    }
                    
                    median_fwhm = None
                    is_fwhm_cached = False
                    if fwhm_cache_key in _SESSION_FWHM_CACHE:
                        median_fwhm = _SESSION_FWHM_CACHE[fwhm_cache_key]
                        is_fwhm_cached = True
                    else:
                        try:
                            # FWHM from the first target and its comparison stars
                            st0 = active[0]
                            stars_to_fit = []
                            tx, ty = wcs.world_to_pixel(SkyCoord(ra=st0['ra']*u.deg, dec=st0['dec']*u.deg, frame='icrs'))
                            stars_to_fit.append({'id': 'Target', 'x': tx + 1.0, 'y': ty + 1.0})
                            for idx, s in enumerate(st0['ens']):
                                sx, sy = wcs.world_to_pixel(SkyCoord(ra=s['ra']*u.deg, dec=s['dec']*u.deg, frame='icrs'))
                                stars_to_fit.append({'id': f"Ref_{idx}", 'x': sx + 1.0, 'y': sy + 1.0})
                            median_fwhm = refine_coordinates_psf(data, stars_to_fit, 15, aperture_radius, 60000, 0, display_plots=False, print_psf_fitting=print_psf_fitting)
                            _SESSION_FWHM_CACHE[fwhm_cache_key] = median_fwhm
                        except: median_fwhm = None

                    if use_flexible_aperture and median_fwhm:
                        curr_ap = median_fwhm * aperture_fwhm_factor
                        curr_ann_in = curr_ap + annulus_inner_gap
                        curr_ann_out = curr_ann_in + annulus_width
                    else:
                        curr_ap, curr_ann_in, curr_ann_out = aperture_radius, annulus_inner, annulus_outer

            # v4.16: effective gain of this frame (gain x N for averaged stacks)
            from python.noise import effective_gain
            g_eff, _ = effective_gain(gain, {'STACKCNT': frame_stackcnt}, stack_mode)
            # cache key includes the noise settings so changed gain/stack mode/factor are re-measured
            key_prefix = f"{fpath}#g={g_eff:.5g};k={float(sky_noise_factor):.3g}" + (f";{_caltag}" if _caltag else "")

            def get_cached_measurement(fpath, ra, dec, ap, ann_in, ann_out, data, wcs, gain, max_shift=4.0):
                nonlocal cache_hits
                key = (key_prefix, round(ra, 6), round(dec, 6), round(ap, 2), round(ann_in, 2), round(ann_out, 2))
                if key in _SESSION_PHOT_CACHE:
                    cache_hits += 1
                    return _SESSION_PHOT_CACHE[key]
                if data is None: return None
                sig_w = max(median_fwhm / 2.355, 1.0) if (median_fwhm and np.isfinite(median_fwhm)) else 1.5
                res = measure_star(data, wcs, ra, dec, ap, ann_in, ann_out, g_eff, max_shift, sky_noise_factor, sig_w)
                _SESSION_PHOT_CACHE[key] = res
                return res

            def process_target(st):
                """Measure one target (with its ensemble/check star) in the current frame."""
                target_ra, target_dec, target_locked = st['ra'], st['dec'], st['locked']
                ensemble_stars, check_star, target_bv = st['ens'], st['check'], st['bv']
                lc_mode, results = st['mode'], st['results']
                hjd = get_hjd(None, target_ra, target_dec, hdr_for_hjd, hjd_lat, hjd_lon) or jd
                try:
                    _process(target_ra, target_dec, target_locked, ensemble_stars, check_star,
                             target_bv, lc_mode, results, hjd, st)
                except Exception as e:
                    print(f"  - {os.path.basename(fpath)} / {st['name']}: {e}")

            def _process(target_ra, target_dec, target_locked, ensemble_stars, check_star,
                         target_bv, lc_mode, results, hjd, st):
                try:
                    # v4.20: target not on this image (other pointing / other night) -> skip quietly
                    if wcs is not None and data is not None:
                        try:
                            _tx, _ty = wcs.world_to_pixel(SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg, frame='icrs'))
                            _ny, _nx = data.shape
                            if not (np.isfinite(_tx) and np.isfinite(_ty)) or _tx < 0 or _ty < 0 or _tx > _nx - 1 or _ty > _ny - 1:
                                st['n_off'] = st.get('n_off', 0) + 1
                                return
                        except Exception:
                            pass
                    # Measure Target
                    t_max_shift = centroid_tracking_shift if target_locked else centroid_initial_radius
                    target_res = get_cached_measurement(fpath, target_ra, target_dec, curr_ap, curr_ann_in, curr_ann_out, data, wcs, gain, max_shift=t_max_shift)
                    if target_res and not np.isnan(target_res['mag_inst']) and not target_locked and data is not None:
                        c = wcs.pixel_to_world(target_res['x'], target_res['y'])
                        target_ra, target_dec = float(c.ra.deg), float(c.dec.deg)
                        target_locked = True
                
                    # Measure Ensemble
                    ensemble_res = []
                    for s in ensemble_stars:
                        s_max_shift = centroid_tracking_shift if s.get('locked') else centroid_initial_radius
                        res = get_cached_measurement(fpath, s['ra'], s['dec'], curr_ap, curr_ann_in, curr_ann_out, data, wcs, gain, max_shift=s_max_shift)
                        if res and not np.isnan(res['mag_inst']):
                            if not s.get('locked') and data is not None:
                                c = wcs.pixel_to_world(res['x'], res['y'])
                                s['ra'], s['dec'] = float(c.ra.deg), float(c.dec.deg)
                                s['locked'] = True
                        
                            m_inst_corr = res['mag_inst'] - (k_coeff * airmass)
                            safe_ens_bv = s['bv_std'] if s.get('bv_std') is not None and not np.isnan(s.get('bv_std', 0.0)) else 0.0
                            zv_i = s['mag_std'] - (m_inst_corr + coeff_term * safe_ens_bv)
                            ensemble_res.append({
                                'name': s['name'],
                                'mag_std': s['mag_std'],
                                'zv': zv_i,
                                'snr': res['snr'],
                                'mag_err': res['mag_err'],
                                'flux': res['net_flux'],
                                'flux_err': res['flux_err']
                            })

                    if lc_mode != 'calibrated':
                        # ---------------------------------------------------------
                        # UNCALIBRATED MODES: 'raw' (target ADU) / 'relative' (ADU ratio)
                        # Catalogue magnitudes, B-V, colour terms and extinction are
                        # NOT used here.
                        # ---------------------------------------------------------
                        if not target_res or target_res.get('net_flux') is None or not np.isfinite(target_res['net_flux']):
                            print(f"  - {os.path.basename(fpath)}: Target star could not be measured.")
                        elif lc_mode == 'relative' and not ensemble_res:
                            print(f"  - {os.path.basename(fpath)}: No ensemble stars could be measured.")
                        else:
                            # Check star (optional)
                            check_res = None
                            if check_star:
                                c_max_shift = centroid_tracking_shift if check_star.get('locked') else centroid_initial_radius
                                check_res = get_cached_measurement(fpath, check_star['ra'], check_star['dec'], curr_ap, curr_ann_in, curr_ann_out, data, wcs, gain, max_shift=c_max_shift)
                                if check_res and not np.isnan(check_res['mag_inst']) and not check_star.get('locked') and data is not None:
                                    c = wcs.pixel_to_world(check_res['x'], check_res['y'])
                                    check_star['ra'], check_star['dec'] = float(c.ra.deg), float(c.dec.deg)
                                    check_star['locked'] = True
                                if check_res and not np.isfinite(check_res.get('net_flux', np.nan)):
                                    check_res = None

                            F_t = float(target_res['net_flux'])
                            sF_t = float(target_res['flux_err'])

                            if lc_mode == 'raw':
                                scale = 1.0
                                if flux_per_second:
                                    try:
                                        et = float(frame_exptime)
                                    except (TypeError, ValueError):
                                        et = 0.0
                                    if et > 0:
                                        scale = 1.0 / et
                                    else:
                                        print(f"  - {os.path.basename(fpath)}: EXPTIME missing/zero, flux left in ADU.")
                                y, sy = F_t * scale, sF_t * scale
                                ens_flux, ens_flux_err = np.nan, np.nan
                                check_flux = check_res['net_flux'] * scale if check_res else np.nan
                                check_flux_err = check_res['flux_err'] * scale if check_res else np.nan
                                # Reference stars are measured for display only
                                ref_fluxes = {r['name']: r['flux'] * scale for r in ensemble_res}
                            else:  # relative
                                S = float(np.sum([r['flux'] for r in ensemble_res]))
                                sS = float(np.sqrt(np.sum([r['flux_err']**2 for r in ensemble_res])))
                                if S <= 0:
                                    print(f"  - {os.path.basename(fpath)}: Ensemble flux <= 0, frame skipped.")
                                    return
                                y = F_t / S
                                # Uncorrelated error propagation for a ratio
                                sy = np.sqrt((sF_t / S)**2 + (F_t * sS / S**2)**2)
                                ens_flux, ens_flux_err = S, sS
                                if check_res:
                                    Fc = check_res['net_flux']
                                    check_flux = Fc / S
                                    check_flux_err = np.sqrt((check_res['flux_err'] / S)**2 + (Fc * sS / S**2)**2)
                                else:
                                    check_flux, check_flux_err = np.nan, np.nan
                                # Diagnostics for each reference star: flux relative to the
                                # sum of the OTHER reference stars (leave-one-out).
                                ref_fluxes = {}
                                if len(ensemble_res) >= 2:
                                    for r in ensemble_res:
                                        S_other = S - r['flux']
                                        ref_fluxes[r['name']] = r['flux'] / S_other if S_other > 0 else np.nan

                            def _to_mag(v):
                                return -2.5 * np.log10(v) if (v is not None and np.isfinite(v) and v > 0) else np.nan

                            mag_val = _to_mag(y)
                            mag_err_val = 1.0857 * sy / y if (np.isfinite(mag_val) and y > 0) else np.nan
                            ref_mags = {k: _to_mag(v) for k, v in ref_fluxes.items()}

                            if verbose and i < 3:
                                unit = ('ADU/s' if (lc_mode == 'raw' and flux_per_second) else 'ADU') if lc_mode == 'raw' else 'ratio'
                                print(f"\n--- Diagnostic (Image {i+1}: {os.path.basename(fpath)}) [{lc_mode}] ---")
                                print(f"  Target net flux: {F_t:.1f} +/- {sF_t:.1f} ADU (SNR={target_res['snr']:.1f})")
                                if lc_mode == 'relative':
                                    print(f"  Ensemble ({len(ensemble_res)} stars): sum flux = {ens_flux:.1f} +/- {ens_flux_err:.1f} ADU")
                                print(f"  RESULT: {y:.6g} +/- {sy:.3g} {unit}")
                                print("------------------------------------------------------\n")

                            results.append({
                                'file': os.path.basename(fpath),
                                'date_ut': ut_date(jd), 'ut_mid': ut_hms(jd),
                                'jd': jd,
                                'hjd': hjd,
                                'airmass': airmass,
                                'exptime': frame_exptime,
                                'lc_mode': lc_mode,
                                'flux_unit': ('ADU/s' if flux_per_second else 'ADU') if lc_mode == 'raw' else 'ratio',
                                'flux': y,
                                'flux_err': sy,
                                'target_adu': F_t,
                                'target_adu_err': sF_t,
                                'mag': mag_val,          # instrumental (raw) or differential instrumental (relative)
                                'mag_err': mag_err_val,
                                'check_flux': check_flux,
                                'check_flux_err': check_flux_err,
                                'check_mag': _to_mag(check_flux),
                                'snr': target_res['snr'],
                                'fwhm': median_fwhm if median_fwhm else np.nan,
                                'ens_flux': ens_flux,
                                'ens_flux_err': ens_flux_err,
                                'zp_avg': np.nan,
                                'zp_err': np.nan,
                                'n_ensemble': len(ensemble_res) if lc_mode == 'relative' else 0,
                                '_ens_members': {r['name']: (r['flux'], r['flux_err']) for r in ensemble_res},
                                'ref_fluxes': ref_fluxes,
                                'ref_mags': ref_mags
                            })

                    elif target_res and ensemble_res:
                        if np.isnan(target_res['mag_inst']):
                            print(f"  - {os.path.basename(fpath)}: Target measurement returned NaN")
                            return
                    
                        # 4. Ensemble Calibration Logic
                        # We average the zero points from all valid ensemble stars
                        zvs = [r['zv'] for r in ensemble_res]
                        zv_avg = np.mean(zvs)
                        zv_std = np.std(zvs, ddof=1) if len(zvs) > 1 else 0.0
                        zv_err = zv_std / np.sqrt(len(zvs)) if len(zvs) > 1 else 0.01 # Fallback 0.01 if only 1 star
                
                        # Safely handle NaN in B-V values which would propagate even if coeff_term is 0
                        safe_tbv = target_bv if target_bv is not None and not np.isnan(target_bv) else 0.0
                
                        # Target Calibration
                        v_target_corr = target_res['mag_inst'] - (k_coeff * airmass)
                        v_target_std = v_target_corr + coeff_term * safe_tbv + zv_avg
                
                        # Total uncertainty
                        total_mag_err = np.sqrt(target_res['mag_err']**2 + zv_err**2)

                        if verbose and i < 3:
                            print(f"\n--- Diagnostic (Image {i+1}: {os.path.basename(fpath)}) ---")
                            print(f"  Target Inst: {target_res['mag_inst']:.4f} | Corr: {v_target_corr:.4f}")
                            print(f"  Ensemble ({len(ensemble_res)} stars): ZP_avg = {zv_avg:.4f} +/- {zv_err:.4f} (std={zv_std:.4f})")
                            for r in ensemble_res:
                                print(f"    * {r['name']}: ZP={r['zv']:.4f}, SNR={r['snr']:.1f}")
                            print(f"  FINAL MAG:   {v_target_std:.4f} +/- {total_mag_err:.4f}")
                            print("------------------------------------------------------\n")
                
                        # 5. Check Star Calibration
                        check_mag_std = np.nan
                        if check_star:
                            c_max_shift = centroid_tracking_shift if check_star.get('locked') else centroid_initial_radius
                            check_res = get_cached_measurement(fpath, check_star['ra'], check_star['dec'], curr_ap, curr_ann_in, curr_ann_out, data, wcs, gain, max_shift=c_max_shift)
                            if check_res and not np.isnan(check_res['mag_inst']):
                                if not check_star.get('locked') and data is not None:
                                    c = wcs.pixel_to_world(check_res['x'], check_res['y'])
                                    check_star['ra'], check_star['dec'] = float(c.ra.deg), float(c.dec.deg)
                                    check_star['locked'] = True
                            
                                c_inst_corr = check_res['mag_inst'] - (k_coeff * airmass)
                                safe_check_bv = check_star['bv_std'] if check_star.get('bv_std') is not None and not np.isnan(check_star.get('bv_std', 0.0)) else 0.0
                                check_mag_std = c_inst_corr + coeff_term * safe_check_bv + zv_avg

                        # 6. Reference Stars Calibration (for stats)
                        ref_mags = {}
                        for r in ensemble_res:
                            ref_mags[r['name']] = r['mag_std'] - r['zv'] + zv_avg

                        results.append({
                            'file': os.path.basename(fpath),
                            'date_ut': ut_date(jd), 'ut_mid': ut_hms(jd),
                            'jd': jd,
                            'hjd': hjd,
                            'airmass': airmass,
                            'mag': v_target_std,
                            'mag_err': total_mag_err,
                            'check_mag': check_mag_std,
                            'snr': target_res['snr'],
                            'fwhm': median_fwhm if median_fwhm else np.nan,
                            'zp_avg': zv_avg,
                            'zp_err': zv_err,
                            'n_ensemble': len(ensemble_res),
                            'ref_mags': ref_mags
                        })
                    else:
                        if not target_res: print(f"  - {os.path.basename(fpath)}: Target star could not be centroided.")
                        if not ensemble_res: print(f"  - {os.path.basename(fpath)}: No ensemble stars could be measured.")
                    
                finally:
                    # keep refined (locked) position for the next frame
                    st['ra'], st['dec'], st['locked'] = target_ra, target_dec, target_locked

            for st in active:
                process_target(st)
                    
            if (i+1) % 10 == 0:
                cache_msg = f" ({cache_hits} star measurements retrieved from session cache)" if cache_hits > 0 else ""
                print(f"Processed {i+1}/{len(fits_files)} images...{cache_msg}")
                
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    # Save cache to disk for future runs
    save_session_cache()

    if update_progress:
        update_progress(100)

    def _finalise(results, lc_mode):
        if not results:
            status = "Cancelled." if cancel_event is not None and cancel_event.is_set() else "No data successfully processed."
            return None, status
        
        # Sort by time FIRST so we can do temporal outlier rejection
        results.sort(key=lambda x: x['hjd'])
    
        # v4.20 relative mode: frames in which not all comparison stars could be measured
        # (e.g. a star outside the image on another night) would make the ensemble sum
        # smaller and the ratio jump. Their ensemble sum is scaled up by the share the
        # missing stars have in the complete ensemble (median over complete frames).
        if lc_mode == 'relative':
            names = sorted({n for r in results for n in r.get('_ens_members', {})})
            complete = [r for r in results if len(r.get('_ens_members', {})) == len(names)]
            if names and complete and len(complete) < len(results):
                share = {}
                for n in names:
                    share[n] = float(np.median([r['_ens_members'][n][0] / r['ens_flux'] for r in complete
                                                if r['ens_flux'] > 0]))
                for r in results:
                    mem = r.get('_ens_members', {})
                    if len(mem) == len(names):
                        r['ens_complete'] = True
                        continue
                    frac = sum(share[n] for n in mem)
                    r['ens_complete'] = False
                    if not (frac > 0.05):
                        r['flux'] = r['flux_err'] = np.nan
                        continue
                    S, sS = r['ens_flux'] / frac, r['ens_flux_err'] / frac
                    F_t, sF_t = r['target_adu'], r['target_adu_err']
                    r['flux'] = F_t / S
                    r['flux_err'] = np.sqrt((sF_t / S) ** 2 + (F_t * sS / S ** 2) ** 2)
                    r['ens_flux'], r['ens_flux_err'] = S, sS
                    if np.isfinite(r.get('check_flux', np.nan)):
                        r['check_flux'] *= frac
                        r['check_flux_err'] *= frac
                        r['check_mag'] = -2.5 * np.log10(r['check_flux']) if r['check_flux'] > 0 else np.nan
                    r['mag'] = -2.5 * np.log10(r['flux']) if r['flux'] > 0 else np.nan
                    r['mag_err'] = 1.0857 * r['flux_err'] / r['flux'] if r['flux'] > 0 else np.nan
            for r in results:
                r.pop('_ens_members', None)

        for r in results:          # internal bookkeeping, not for the CSV (all modes)
            r.pop('_ens_members', None)

        # Relative mode: also provide a transparency-corrected target flux in ADU,
        # i.e. the ratio rescaled by the mean ensemble flux of the whole run.
        if lc_mode == 'relative':
            ens_vals = np.array([r['ens_flux'] for r in results], dtype=float)
            mean_S = np.nanmean(ens_vals) if np.any(np.isfinite(ens_vals)) else np.nan
            for r in results:
                r['flux_corr_adu'] = r['flux'] * mean_S
                r['flux_corr_adu_err'] = r['flux_err'] * mean_S

        # Outlier Flagging (Rolling Median of 5 to preserve true stellar variations)
        # Calibrated mode works in magnitudes (threshold >= 0.05 mag).
        # Raw/relative modes work in flux (threshold >= 5 % of the median flux),
        # so that frames with zero/negative net flux are still handled.
        if lc_mode == 'calibrated':
            vals = np.array([r['mag'] for r in results], dtype=float)
            min_jump = 0.05
            min_std = 0.005
        else:
            vals = np.array([r['flux'] for r in results], dtype=float)
            med_abs = abs(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else 0.0
            min_jump = 0.05 * med_abs
            min_std = 0.005 * med_abs

        finite = np.isfinite(vals)
        if len(vals) > 5 and np.sum(finite) > 5:
            # Fill non-finite values with the median so the rolling window works
            work = np.where(finite, vals, np.nanmedian(vals))
            # Pad with edge values for the rolling window
            padded = np.pad(work, (2, 2), mode='edge')
            rolling_median = np.array([np.median(padded[i:i+5]) for i in range(len(work))])
            diffs = work - rolling_median

            # Calculate robust standard deviation using Median Absolute Deviation (MAD)
            mad = np.median(np.abs(diffs - np.median(diffs)))
            robust_std = 1.4826 * mad

            # Fallback to standard deviation if MAD is too small (e.g., perfectly flat light curve)
            if robust_std < min_std:
                robust_std = np.std(diffs, ddof=1)

            # Minimum threshold to avoid over-flagging very clean data
            # We require at least a 3-sigma deviation AND a minimum absolute jump
            threshold = max(3 * robust_std, min_jump)

            for i, r in enumerate(results):
                if not finite[i] or abs(diffs[i]) > threshold:
                    r['flag'] = 'SUSPECT'
                else:
                    r['flag'] = 'OK'
        else:
            for i, r in enumerate(results):
                r['flag'] = 'OK' if finite[i] else 'SUSPECT'

        return results, "Success"

    out = []
    for st in states:
        if st['error']:
            out.append((None, st['error']))
        else:
            res, msg = _finalise(st['results'], st['mode'])
            n_off = st.get('n_off', 0)
            if n_off:
                if res is None:
                    msg = f"target outside the image in all {n_off} measured frame(s)"
                else:
                    print(f"  {st['name']}: outside the image in {n_off} frame(s) - skipped there.")
            out.append((res, msg))
    return out

def count_nights(jds, gap_days=0.3):
    """Number of observing nights (groups of observations separated by > gap_days)."""
    t = np.sort(np.asarray([j for j in jds if j and np.isfinite(j)], dtype=float))
    return int(1 + np.sum(np.diff(t) > gap_days)) if t.size else 0


def date_label(jds):
    """UT date of the first observation ('2026-04-13'); for data spanning more than
    one night 'first_to_last' ('2026-04-13_to_2026-05-02'). v4.20"""
    from astropy.time import Time as _T
    t = [j for j in jds if j and np.isfinite(j)]
    if not t:
        return "nodate"
    d0 = _T(min(t), format='jd').datetime.strftime('%Y-%m-%d')
    if count_nights(t) <= 1:
        return d0
    return d0 + "_to_" + _T(max(t), format='jd').datetime.strftime('%Y-%m-%d')


def save_aavso_report(results, output_path, target_name, filter_name, obs_code, 
                      comp_name="ENSEMBLE", comp_mag="na", check_name="na", check_mag="na",
                      trans="NO", software_version=None, comments="na", notes="na", chart="na"):
    """ Generates a report in AAVSO Extended Format. """
    try:
        def fmt_mag(val, precision=3):
            if val is None or val == "" or str(val).lower() in ("none", "nan"):
                return "na"
            if isinstance(val, (int, float)) and not np.isnan(val):
                return f"{val:.{precision}f}"
            return str(val)

        with open(output_path, 'w', newline='') as f:
            f.write("#TYPE=Extended\n")
            f.write(f"#OBSCODE={obs_code}\n")
            f.write(f"#SOFTWARE=Calibra {software_version or __version__}\n")
            f.write(f"#DELIM=,\n")
            f.write(f"#DATE=HJD\n")
            f.write(f"#OBSTYPE=CCD\n")
            f.write(f"#TRANS={trans}\n")
            f.write(f"#COMMENTS={comments}\n")
            f.write("#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES\n")
            
            chart_val = chart if chart and str(chart).strip() else "na"

            for r in results:
                # v4.44: upper limit ("fainter than") of a faint-target point -> "<mag", MERR na
                if r.get('upper_limit'):
                    lim = r.get('mag_limit')
                    if not isinstance(lim, (int, float)) or not np.isfinite(lim):
                        continue
                    r = dict(r, mag=lim, mag_err=np.nan)
                elif not isinstance(r.get('mag'), (int, float)) or not np.isfinite(r['mag']):
                    continue
                # Format: Name, Date, Mag, Merr, Filt, Trans, Mtype, Cname, Cmag, Kname, Kmag, Amass, Group, Chart, Notes
                # Date: 5 digits, Mag: 3 digits, Merr: 4 digits
                mag_str = ("<" if r.get('upper_limit') else "") + f"{r['mag']:.3f}"
                err_str = f"{r['mag_err']:.4f}" if np.isfinite(r['mag_err']) else "na"
                date_str = f"{r['hjd']:.5f}"
                c_mag_str = fmt_mag(comp_mag, 3)
                
                # Use frame-specific measured check magnitude if available, otherwise catalog mag fallback
                k_val = r.get('check_mag', np.nan)
                if k_val is None or not isinstance(k_val, (int, float)) or np.isnan(k_val):
                    k_val = check_mag
                k_mag_str = fmt_mag(k_val, 3)
                
                line = f"{target_name},{date_str},{mag_str},{err_str},{filter_name},{trans},STD,"
                am = r.get('airmass', np.nan)
                am_str = f"{am:.3f}" if isinstance(am, (int, float)) and np.isfinite(am) else "na"
                nf = int(r.get('n_frames', 1) or 1)
                line_notes = notes + (f" | {nf} frames combined" if nf > 1 else "")
                line += f"{comp_name},{c_mag_str},{check_name},{k_mag_str},{am_str},na,{chart_val},{line_notes}\n"
                f.write(line)
        return True
    except Exception as e:
        print(f"Error saving AAVSO report: {e}")
        return False

def plot_light_curve(results, target_name, output_path, ax=None, visibility_state=None, plot_options=None):
    """ Plots the light curve. If ax is provided, plots to that axes.

    Works for all light-curve modes (see run_time_series_photometry):
      - calibrated : magnitudes (default y-axis inverted)
      - raw        : net flux in ADU (or ADU/s), or instrumental magnitude
      - relative   : flux ratio target / ensemble, or differential instr. magnitude
    plot_options['y_quantity'] = 'flux' | 'mag' selects the quantity for the
    raw/relative modes (default 'flux'). Calibrated mode always plots magnitudes.
    """
    if not results: return
    if visibility_state is None: visibility_state = {}
    if plot_options is None: plot_options = {}
    
    lc_mode = results[0].get('lc_mode', 'calibrated')
    if lc_mode == 'calibrated':
        y_quantity = 'mag'
    else:
        y_quantity = plot_options.get('y_quantity', 'flux')
    use_flux = (y_quantity == 'flux')
    
    if ax is None:
        # Use an object-oriented Figure (no pyplot) so this is safe to call
        # from worker threads while the Tk GUI is running.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        is_standalone = True
    else:
        fig = ax.figure
        fig.clear()
        ax = fig.add_subplot(111)
        is_standalone = False
    
    val_key, err_key = ('flux', 'flux_err') if use_flux else ('mag', 'mag_err')
    chk_key = 'check_flux' if use_flux else 'check_mag'
    chk_err_key = 'check_flux_err' if use_flux else None
    ref_key = 'ref_fluxes' if use_flux else 'ref_mags'
    
    def _arr(key, default=np.nan):
        out = []
        for r in results:
            v = r.get(key, default) if key else default
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out, dtype=float)
    
    times = np.array([r['hjd'] for r in results])
    # v4.44: combined frames (faint target) -> horizontal bar = time span of the frames
    spans = np.array([float(r.get('t_span_min', 0.0) or 0.0) / 2880.0 if (r.get('n_frames', 1) or 1) > 1 else 0.0
                      for r in results])
    xerr_all = spans if np.any(spans > 0) else None
    mags = _arr(val_key)
    errs = _arr(err_key)
    flags = np.array([r.get('flag', 'OK') for r in results])
    finite = np.isfinite(mags)
    errs_plot = np.where(np.isfinite(errs), errs, 0.0)
    
    # JD Offset for readability
    t0 = int(min(times))
    times_rel = times - t0
    
    show_target = visibility_state.get("Target", True)
    show_check = visibility_state.get("Check Star", True)
    custom_labels = plot_options.get("custom_labels", {})
    custom_colors = plot_options.get("custom_colors", {})
    custom_symbols = plot_options.get("custom_symbols", {})
    
    if show_target:
        # Plot OK points
        ok_mask = ((flags == 'OK') | (flags == 'LIMIT')) & finite   # v4.44: in flux plots limits are ordinary points
        t_label = custom_labels.get("Target", "Target")
        t_color = custom_colors.get("Target", "darkblue")
        t_sym = custom_symbols.get("Target", "o")
        if np.any(ok_mask):
            ax.errorbar(times_rel[ok_mask], mags[ok_mask], yerr=errs_plot[ok_mask],
                         xerr=xerr_all[ok_mask] if xerr_all is not None else None,
                         fmt=t_sym, color=t_color, ecolor='gray', capsize=2, markersize=4, label=t_label)
        # v4.44: upper limits (fainter than) in magnitude plots
        if not use_flux:
            lim = np.array([float(r['mag_limit']) if r.get('upper_limit') and r.get('mag_limit') is not None else np.nan
                            for r in results], dtype=float)
            lm = np.isfinite(lim)
            if np.any(lm):
                ax.errorbar(times_rel[lm], lim[lm], xerr=xerr_all[lm] if xerr_all is not None else None,
                            fmt='v', color=t_color, mfc='none', ecolor='lightgray', markersize=7,
                            label=f"{t_label} (fainter than)")
        
        # Plot SUSPECT points
        suspect_mask = (flags == 'SUSPECT') & finite
        if np.any(suspect_mask):
            ax.errorbar(times_rel[suspect_mask], mags[suspect_mask], yerr=errs_plot[suspect_mask], 
                         fmt=t_sym, color='none', markeredgecolor='red', ecolor='red', capsize=2, markersize=6, label=f"{t_label} (SUSPECT)")

    if show_check:
        # Plot Check Star
        check_mags = _arr(chk_key)
        valid_check = np.isfinite(check_mags)
        if np.any(valid_check):
            c_label = custom_labels.get("Check Star", "Check Star")
            c_color = custom_colors.get("Check Star", "green")
            c_sym = custom_symbols.get("Check Star", "o")
            c_err = _arr(chk_err_key) if chk_err_key else None
            yerr = None
            if c_err is not None and np.any(np.isfinite(c_err[valid_check])):
                yerr = np.where(np.isfinite(c_err), c_err, 0.0)[valid_check]
            ax.errorbar(times_rel[valid_check], check_mags[valid_check], yerr=yerr,
                         fmt=c_sym, color=c_color, alpha=0.6, markersize=4, label=c_label)
                         
    # Plot Reference Stars
    all_refs = set()
    for r in results:
        if ref_key in r and isinstance(r[ref_key], dict):
            all_refs.update(r[ref_key].keys())

    def_colors = ['orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']
    for idx, ref_name in enumerate(sorted(all_refs)):
        key = f"Ref: {ref_name}"
        if visibility_state.get(key, False):
            ref_mags = np.array([r.get(ref_key, {}).get(ref_name, np.nan) for r in results], dtype=float)
            valid_ref = np.isfinite(ref_mags)
            if np.any(valid_ref):
                r_label = custom_labels.get(key, key)
                r_color = custom_colors.get(key, def_colors[idx % len(def_colors)])
                r_sym = custom_symbols.get(key, "x")
                ax.errorbar(times_rel[valid_ref], ref_mags[valid_ref], 
                             fmt=r_sym, color=r_color, alpha=0.5, markersize=4, label=r_label)

    # Magnitudes are plotted inverted by default, fluxes are not.
    invert_y = plot_options.get('invert_y', not use_flux)
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()
    elif not invert_y and ax.yaxis_inverted():
        ax.invert_yaxis()

    y_min_val = plot_options.get('y_min', "")
    y_max_val = plot_options.get('y_max', "")
    if str(y_min_val).strip() and str(y_max_val).strip():
        try:
            y_min, y_max = float(y_min_val), float(y_max_val)
            if invert_y:
                ax.set_ylim(max(y_min, y_max), min(y_min, y_max))
            else:
                ax.set_ylim(min(y_min, y_max), max(y_min, y_max))
        except ValueError:
            pass
    
    # Default y-axis label depends on mode/quantity
    unit = results[0].get('flux_unit', 'ADU')
    if lc_mode == 'calibrated':
        default_ylabel = "Magnitude"
    elif lc_mode == 'raw':
        default_ylabel = f"Net flux [{unit}]" if use_flux else "Instrumental magnitude"
    else:
        default_ylabel = "Relative flux (target / \u03a3 ref)" if use_flux else "Differential instrumental magnitude"
    ylabel = plot_options.get('ylabel', default_ylabel)
    if not str(ylabel).strip() or (ylabel == "Magnitude" and lc_mode != 'calibrated'):
        ylabel = default_ylabel
    
    ax.set_title(plot_options.get('title', f"Light Curve: {target_name}"))
    ax.set_xlabel(plot_options.get('xlabel', f"HJD - {t0}"))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.0), framealpha=1.0, edgecolor='black')
    fig.tight_layout()
    
    # Only write a file when a path is given (previously an empty path wrote a
    # stray ".png" into the working directory on every redraw).
    if output_path:
        fig.savefig(output_path)
