import numpy as np
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.utils.exceptions import AstropyWarning
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from python import paths

# =============================================================================
# DATA MINING PIPELINE
# =============================================================================
# The Data Mining module is responsible for analyzing entire fields of view across
# an entire timeseries of FITS files. It detects all stars in a reference frame,
# forces aperture photometry at those coordinates across all frames, and performs
# global differential calibration to extract intrinsic stellar variability,
# identifying candidate variable stars while rejecting artifacts and noise.
# =============================================================================

DM_CACHE_VERSION = 1


def _catalog_hash(master_catalog):
    import hashlib
    h = hashlib.sha1()
    for st in master_catalog:
        h.update(f"{st['id']}|{st['ra_deg']:.7f}|{st['dec_deg']:.7f};".encode())
    return h.hexdigest()


def _frame_cache_path(fits_path, cat_hash, settings, cache_dir):
    """Cache file for one frame; the key changes when the file, the star list or any
    measurement setting changes, so stale results are never reused."""
    import hashlib, json
    try:
        st = os.stat(fits_path)
        fid = f"{os.path.abspath(fits_path)}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return None
    key = json.dumps({'v': DM_CACHE_VERSION, 'f': fid, 'c': cat_hash, 's': settings}, sort_keys=True)
    return os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest() + ".npz")


def _measure_frame(data, x_arr, y_arr, n_stars, aperture_radius, annulus_inner, annulus_outer,
                   saturation_limit, edge_buffer, centroid_tracking_shift, g_eff, sky_noise_factor, sigma_w):
    """
    v4.17: vectorised measurement of all stars in one frame.
    Returns a dict of arrays (length n_stars): net_flux, mag_inst, mag_err, snr, bkg_std, x, y.
    """
    from python.fastphot import track_centroid, fast_sky_stats
    from python.noise import flux_variance
    out = {k: np.full(n_stars, np.nan) for k in ('net_flux', 'mag_inst', 'mag_err', 'snr', 'bkg_std', 'x', 'y')}
    y_max, x_max = data.shape
    half_box = max(5, int(np.ceil(3 * sigma_w)) + 1)
    idx_ok, xs, ys = [], [], []
    for idx in range(n_stars):
        x_init, y_init = x_arr[idx], y_arr[idx]
        if not (np.isfinite(x_init) and np.isfinite(y_init)):
            continue
        if (x_init < edge_buffer or x_init > (x_max - edge_buffer) or
                y_init < edge_buffer or y_init > (y_max - edge_buffer)):
            continue
        # Centroid refinement (fast windowed centroid; falls back to the WCS position
        # if it fails or moves further than the tracking limit)
        xr, yr, _ = track_centroid(data, x_init, y_init, centroid_tracking_shift, sigma_w=sigma_w, half_box=half_box)
        if (xr - aperture_radius < -0.5 or yr - aperture_radius < -0.5 or
                xr + aperture_radius > x_max - 0.5 or yr + aperture_radius > y_max - 0.5):
            continue
        idx_ok.append(idx); xs.append(xr); ys.append(yr)
    if not idx_ok:
        return out
    pos = np.column_stack([xs, ys])
    apers = CircularAperture(pos, r=aperture_radius)
    annuli = CircularAnnulus(pos, r_in=annulus_inner, r_out=annulus_outer)
    sums = np.asarray(aperture_photometry(data, apers, method='exact')['aperture_sum'], dtype=float)
    ap_masks = apers.to_mask(method='center')
    an_masks = annuli.to_mask(method='center')
    area = apers.area
    for j, idx in enumerate(idx_ok):
        ap_vals = ap_masks[j].get_values(data)
        if ap_vals.size == 0 or np.nanmax(ap_vals) > saturation_limit:
            continue                                   # saturated / non-linear
        bkg_median, bkg_std, n_sky = fast_sky_stats(an_masks[j].get_values(data), sigma=3.0)
        if n_sky == 0:
            continue
        net = sums[j] - bkg_median * area
        if not np.isfinite(net) or net <= 0:
            continue
        err = np.sqrt(flux_variance(net, area, bkg_std, n_sky, g_eff, sky_noise_factor))
        snr = net / err if err > 0 else 0.0
        out['net_flux'][idx] = net
        out['mag_inst'][idx] = -2.5 * np.log10(net)
        out['mag_err'][idx] = 1.0857 / snr if snr > 0 else np.nan
        out['snr'][idx] = snr
        out['bkg_std'][idx] = bkg_std
        out['x'][idx] = xs[j]
        out['y'][idx] = ys[j]
    return out


def track_and_measure(fits_files, master_catalog, aperture_radius=5.0, 
                      annulus_inner=7.0, annulus_outer=13.0, 
                      saturation_limit=55000, edge_buffer=50, 
                      centroid_tracking_shift=4.0,
                      cancel_event=None, progress_callback=None,
                      gain=1.0, stack_mode='average', sky_noise_factor=1.0,
                      fwhm=3.5, use_cache=True, cache_dir=None):
    """
    Tracks and measures all stars in the master_catalog across all FITS files.
    Returns a list of dicts, one per frame, containing the measurements for all stars.

    v4.17 (speed): each frame is converted to float once and all stars are
    measured with one vectorised aperture call and a fast windowed centroid
    (was: whole-image conversion + 2-D Gaussian fit for every star, ~65 ms per
    star and frame; now well below 1 ms). With use_cache=True the measurements of
    every frame are stored in <user folder>/cache/dm_cache/ and reused when the same
    frame is analysed again with the same star list and settings, so re-running
    the analysis (other thresholds, ensemble size, ...) does not re-read the FITS files.
    """
    from python.fastphot import as_float_image
    from python.noise import effective_gain
    results = []
    warnings_list = []
    
    if not fits_files or not master_catalog:
        return results, warnings_list
        
    base_filter = None
    total_files = len(fits_files)
    n_stars = len(master_catalog)
    ids = [star['id'] for star in master_catalog]
    cache_dir = cache_dir or paths.DM_CACHE_DIR
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
    cat_hash = _catalog_hash(master_catalog)
    settings = {'ap': round(float(aperture_radius), 3), 'ai': round(float(annulus_inner), 3), 'ao': round(float(annulus_outer), 3),
                'sat': float(saturation_limit), 'edge': float(edge_buffer), 'shift': round(float(centroid_tracking_shift), 3),
                'gain': round(float(gain), 5), 'stack': str(stack_mode), 'k': round(float(sky_noise_factor), 4),
                'fwhm': round(float(fwhm), 3)}
    try:                                   # v4.27: on-the-fly calibration state
        from python.master_frames import state_tag as _cal_state_tag, apply_on_the_fly as _cal_apply
        settings['cal'] = _cal_state_tag()
    except Exception:
        _cal_apply = None
    sigma_w = max(float(fwhm) / 2.355, 1.0)
    n_cached = 0
    
    # Pre-create SkyCoord array for all stars
    ra_list = [star['ra_deg'] for star in master_catalog]
    dec_list = [star['dec_deg'] for star in master_catalog]
    coords = SkyCoord(ra=ra_list*u.deg, dec=dec_list*u.deg, frame='icrs')
    
    for i, fits_path in enumerate(fits_files):
        if cancel_event and cancel_event.is_set():
            break
            
        frame_result = {
            'file': os.path.basename(fits_path),
            'measurements': {} # star_id -> {net_flux, mag_inst, mag_err, snr, bkg_std}
        }
        
        try:
            cpath = _frame_cache_path(fits_path, cat_hash, settings, cache_dir) if use_cache else None
            arrays = None
            if cpath and os.path.exists(cpath):
                try:
                    with np.load(cpath, allow_pickle=False) as z:
                        arrays = {k: z[k] for k in z.files}
                    curr_filter = str(arrays.pop('filter')) if 'filter' in arrays else 'UNKNOWN'
                    frame_result['img_std'] = float(arrays.pop('img_std'))
                    n_cached += 1
                except Exception:
                    arrays = None

            if arrays is None:
                with fits.open(fits_path) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data
                    if data is None and len(hdul) > 1:
                        data = hdul[1].data
                        header = hdul[1].header
                    if data is None:
                        continue
                    if data.ndim == 3:
                        data = data[0]
                    data = as_float_image(data)   # ONE conversion per frame
                    if _cal_apply is not None:
                        data, _ = _cal_apply(data, header, fits_path)
                curr_filter = header.get('FILTER', 'UNKNOWN')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyWarning)
                    wcs = WCS(header)
                if not wcs.has_celestial:
                    continue
                g_eff, _ = effective_gain(gain, header, stack_mode)

                # Global image background scatter (to flag stars with a disturbed local
                # background). v4.17: from a 1/16 pixel subsample (full image: ~3 s/frame).
                _, _, img_std = sigma_clipped_stats(data[::4, ::4], sigma=3.0, maxiters=3)
                frame_result['img_std'] = float(img_std)

                x_arr, y_arr = wcs.world_to_pixel(coords)
                arrays = _measure_frame(data, np.atleast_1d(x_arr), np.atleast_1d(y_arr), n_stars,
                                        aperture_radius, annulus_inner, annulus_outer,
                                        saturation_limit, edge_buffer, centroid_tracking_shift,
                                        g_eff, sky_noise_factor, sigma_w)
                del data
                if cpath:
                    try:
                        np.savez_compressed(cpath, filter=np.array(str(curr_filter)), img_std=np.array(img_std), **arrays)
                    except Exception as e:
                        print(f"Note: could not write Data Mining cache: {e}")

            # Check filter
            if base_filter is None:
                base_filter = curr_filter
            elif curr_filter != base_filter:
                msg = f"Warning: Mixed filters detected! {os.path.basename(fits_path)} uses {curr_filter}, expected {base_filter}."
                if msg not in warnings_list:
                    warnings_list.append(msg)

            meas = frame_result['measurements']
            nf, mi, me, sn, bs = arrays['net_flux'], arrays['mag_inst'], arrays['mag_err'], arrays['snr'], arrays['bkg_std']
            for k, sid in enumerate(ids):
                if np.isfinite(mi[k]):
                    meas[sid] = {'net_flux': float(nf[k]), 'mag_inst': float(mi[k]), 'mag_err': float(me[k]),
                                 'snr': float(sn[k]), 'bkg_std': float(bs[k])}
                else:
                    meas[sid] = {'net_flux': np.nan, 'mag_inst': np.nan, 'snr': np.nan}
                
        except Exception as e:
            print(f"Error measuring frame {fits_path}: {e} "
                  f"(stars not measured in this frame are treated as missing)")
            
        results.append(frame_result)
        
        if progress_callback:
            progress_callback(((i + 1) / total_files) * 100)

    if n_cached:
        print(f"Data Mining: {n_cached} of {total_files} frame(s) taken from the measurement cache.")
            
    return results, warnings_list

def detect_all_stars(fits_path, detect_sigma=5.0, saturation_limit=55000, edge_buffer=50,
                     fwhm=3.5, sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0, use_cache=True, cache_dir=None):
    """
    Detects all stars in a reference frame, rejecting saturated stars and stars near the edge.
    Returns a list of dictionaries containing star coordinates and initial fluxes.

    v4.17: the result is cached (<user folder>/cache/dm_cache/) per reference file and
    detection settings, so repeated runs on the same sequence skip the detection.
    """
    import json, hashlib
    cpath = None
    if use_cache:
        try:
            st = os.stat(fits_path)
            key = json.dumps({'v': DM_CACHE_VERSION, 'f': f"{os.path.abspath(fits_path)}|{st.st_mtime_ns}|{st.st_size}",
                              'p': [detect_sigma, saturation_limit, edge_buffer, fwhm, sharplo, sharphi, roundlo, roundhi]},
                             sort_keys=True, default=str)
            cdir = cache_dir or paths.DM_CACHE_DIR
            os.makedirs(cdir, exist_ok=True)
            cpath = os.path.join(cdir, "detect_" + hashlib.sha1(key.encode()).hexdigest() + ".json")
            if os.path.exists(cpath):
                with open(cpath) as f:
                    cached = json.load(f)
                print(f"Data Mining: reference-frame detection taken from cache ({len(cached)} stars).")
                return cached
        except Exception:
            cpath = None
    stars = _detect_all_stars_uncached(fits_path, detect_sigma, saturation_limit, edge_buffer,
                                       fwhm, sharplo, sharphi, roundlo, roundhi)
    if cpath and stars:
        try:
            with open(cpath, 'w') as f:
                json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in st_.items()} for st_ in stars], f)
        except Exception as e:
            print(f"Note: could not write detection cache: {e}")
    return stars


def _detect_all_stars_uncached(fits_path, detect_sigma, saturation_limit, edge_buffer,
                               fwhm, sharplo, sharphi, roundlo, roundhi):
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data
            if data is None and len(hdul) > 1:
                data = hdul[1].data
                header = hdul[1].header
            if data is None:
                return []
            if data.ndim == 3:
                data = data[0]
                
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            wcs = WCS(header)
            
        if not wcs.has_celestial:
            print(f"Error: {fits_path} missing valid WCS.")
            return []

        # ---------------------------------------------------------
        # Background Estimation & Source Detection
        # ---------------------------------------------------------
        # Calculate the global background median and standard deviation.
        mean, median_bg, std_bg = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        
        # Initialize DAOStarFinder (using the robust fallback for different photutils versions)
        # We detect sources that are `detect_sigma` times brighter than the background std.
        # FWHM=3.5 is a generic starting point. Sharpness and roundness help reject cosmic rays.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=AstropyWarning)
            try:
                daofind = DAOStarFinder(fwhm=fwhm, threshold=detect_sigma * std_bg,
                                        sharpness_range=(sharplo, sharphi), 
                                        roundness_range=(roundlo, roundhi))
            except TypeError:
                daofind = DAOStarFinder(fwhm=fwhm, threshold=detect_sigma * std_bg,
                                        sharplo=sharplo, sharphi=sharphi, 
                                        roundlo=roundlo, roundhi=roundhi)
                                        
        sources = daofind(data - median_bg)
        if sources is None:
            return []
            
        # Sort by flux descending
        sources.sort('flux')
        sources.reverse()
        
        y_max, x_max = data.shape
        results = []
        
        x_col = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
        y_col = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
        
        valid_count = 0
        for row in sources:
            x_pix = row[x_col]
            y_pix = row[y_col]
            peak_val = row['peak']
            
            # 1. Edge rejection
            if (x_pix < edge_buffer or x_pix > (x_max - edge_buffer) or 
                y_pix < edge_buffer or y_pix > (y_max - edge_buffer)):
                continue
                
            # 2. Saturation rejection
            if peak_val > saturation_limit:
                continue
                
            # 3. WCS Conversion
            ra_deg, dec_deg = wcs.all_pix2world(x_pix + 1.0, y_pix + 1.0, 1)
            ra_deg, dec_deg = float(ra_deg), float(dec_deg)
            
            valid_count += 1
            star_id = f"DM_{valid_count:04d}"
            
            results.append({
                'id': star_id,
                'x_ref': x_pix + 1.0,
                'y_ref': y_pix + 1.0,
                'ra_deg': ra_deg,
                'dec_deg': dec_deg,
                'peak_adu': peak_val,
                'dao_flux': row['flux']
            })
            
        return results

    except Exception as e:
        print(f"Error in detect_all_stars for {fits_path}: {e}")
        return []

def calibrate_and_analyze(track_results, master_catalog, out_csv_path, min_valid_fraction=0.7, ensemble_size=100, 
                          center_ra=None, center_dec=None, suspect_sigma=3.0, min_excess=0.03):
    """
    Performs the 2-Pass Global Differential Calibration and Statistical Analysis.
    
    This function takes raw instrumental magnitudes for thousands of stars across the time series,
    constructs an artificial 'super-reference' ensemble of the most stable stars, and aligns every 
    frame's zero-point so that instrumental drops in brightness (due to clouds, airmass, etc) are 
    perfectly subtracted out.
    
    Returns final_stats list and expected_rms_func for plotting.
    """
    if not track_results or not master_catalog:
        return [], None
        
    num_frames = len(track_results)
    min_valid_frames = int(num_frames * min_valid_fraction)
    
    # 1. Reorganize data: star_id -> array of mag_inst indexed by FRAME.
    # v4.15: pre-filled with NaN and written by frame index. Previously values were
    # appended, so a star missing in one frame (e.g. an exception part-way through a
    # frame) shifted all its later points to the wrong frames / raised IndexError.
    star_mags = {star['id']: np.full(num_frames, np.nan) for star in master_catalog}
    for fi, frame in enumerate(track_results):
        for star_id, m_dict in frame.get('measurements', {}).items():
            if star_id in star_mags:
                star_mags[star_id][fi] = m_dict.get('mag_inst', np.nan)
        
    # Filter stars with enough valid frames
    valid_stars = {}
    for star in master_catalog:
        star_id = star['id']
        mags = star_mags[star_id]
        valid_mask = ~np.isnan(mags)
        if np.sum(valid_mask) >= min_valid_frames:
            # Pre-calculate mean magnitude for sorting
            mean_mag = np.nanmean(mags)
            valid_stars[star_id] = {'mean_mag': mean_mag, 'mags': mags, 'valid_mask': valid_mask, 'star_info': star}
            
    if not valid_stars:
        print("No stars met the minimum valid frame threshold.")
        return [], None
        
    # Sort valid stars by mean instrumental magnitude (brightest to faintest)
    sorted_star_ids = sorted(valid_stars.keys(), key=lambda sid: valid_stars[sid]['mean_mag'])
    
    # =========================================================================
    # 2-PASS DIFFERENTIAL CALIBRATION
    # =========================================================================
    # PASS 1: Select an initial ensemble of the brightest available stars.
    # We assume brightest = highest signal-to-noise = most stable.
    ensemble_ids = sorted_star_ids[:ensemble_size]
    if len(ensemble_ids) < 5:
        print("Warning: Very few stars available for ensemble calibration.")
        
    def compute_zero_points(ens_ids):
        """
        Computes the frame-by-frame zero-point offsets.
        For every frame, we compute how much each ensemble star shifted from its 
        own average magnitude, and take the MEDIAN shift across the ensemble.
        Using the median makes it highly robust against individual variable stars 
        in the ensemble.
        """
        zps = np.zeros(num_frames)
        for f in range(num_frames):
            shifts = []
            for sid in ens_ids:
                mag_f = valid_stars[sid]['mags'][f]
                if not np.isnan(mag_f):
                    shifts.append(valid_stars[sid]['mean_mag'] - mag_f)
            if shifts:
                zps[f] = np.median(shifts)
            else:
                zps[f] = 0.0
        return zps
        
    # Calculate Pass 1 zero-points
    zp_pass1 = compute_zero_points(ensemble_ids)
    
    # Calculate preliminary RMS for the initial ensemble
    ens_rms = {}
    for sid in ensemble_ids:
        # Apply the Pass 1 correction to this star
        corrected_mags = valid_stars[sid]['mags'] + zp_pass1
        clean_mags = corrected_mags[np.isfinite(corrected_mags)]
        if len(clean_mags) > 0:
            _, _, rms = sigma_clipped_stats(clean_mags, sigma=4.0, maxiters=2, cenfunc='median', stdfunc='std')
        else:
            rms = np.nan
        ens_rms[sid] = rms
        
    median_ens_rms = np.median(list(ens_rms.values()))
    
    # PASS 2: Refine ensemble
    # Kick out any "bad" stars from the initial ensemble. If a bright star was actually 
    # highly variable (e.g. an eclipsing binary) or noisy, its RMS will be much higher 
    # than the median RMS of the ensemble. By kicking them out, our Pass 2 ensemble is 
    # strictly composed of constant, stable stars.
    refined_ensemble_ids = [sid for sid in ensemble_ids if ens_rms[sid] <= 2.0 * median_ens_rms]
    if len(refined_ensemble_ids) < 3:
        refined_ensemble_ids = ensemble_ids
        
    # Calculate Pass 2 zero-points using the purified ensemble
    zp_pass2 = compute_zero_points(refined_ensemble_ids)
    
    # Apply Pass 2 zero-points to ALL valid stars and compute final stats
    final_stats = []
    for sid, data in valid_stars.items():
        corrected_mags = data['mags'] + zp_pass2
        clean_mags = corrected_mags[np.isfinite(corrected_mags)]
        if len(clean_mags) > 0:
            final_mean, _, final_rms = sigma_clipped_stats(clean_mags, sigma=4.0, maxiters=2, cenfunc='median', stdfunc='std')
            _, _, tight_rms = sigma_clipped_stats(clean_mags, sigma=2.5, maxiters=5, cenfunc='median', stdfunc='std')
            mad_rms = mad_std(clean_mags, ignore_nan=True)
        else:
            final_mean, final_rms, tight_rms, mad_rms = np.nan, np.nan, np.nan, np.nan
            
        # ---------------------------------------------------------
        # High Background Variance Check
        # ---------------------------------------------------------
        # Some coordinates fall on the boundary of a very bright star's glare. As the 
        # tracking drifts, the bright star's light spills into the local background annulus,
        # causing the background subtraction to fluctuate wildly, which corrupts the target flux.
        # We flag stars whose local background standard deviation is consistently much higher 
        # (> 2x) than the overall image background standard deviation.
        star_meas = [track_results[i]['measurements'].get(sid, {}) for i in range(num_frames) if data['valid_mask'][i]]
        bkg_stds = [m.get('bkg_std', np.nan) for m in star_meas]
        img_stds = [track_results[i].get('img_std', np.nan) for i in range(num_frames) if data['valid_mask'][i]]
        bkg_stds = [v for v in bkg_stds if not np.isnan(v)]
        img_stds = [v for v in img_stds if not np.isnan(v)]
        
        high_bg_var = False
        if len(bkg_stds) > 0 and len(img_stds) > 0:
            if np.median(bkg_stds) > 2.0 * np.median(img_stds):
                high_bg_var = True

        n_valid = np.sum(data['valid_mask'])

        # v4.16: the star's OWN expected photometric noise (median formal error of
        # its measurements) and the ratio RMS / expected noise (~1 for a constant star)
        errs = np.array([m.get('mag_err', np.nan) for m in star_meas], dtype=float)
        errs = errs[np.isfinite(errs)]
        phot_err = float(np.median(errs)) if errs.size else np.nan
        noise_ratio = final_rms / phot_err if (np.isfinite(phot_err) and phot_err > 0) else np.nan
        
        final_stats.append({
            'phot_err': phot_err,
            'noise_ratio': noise_ratio,
            'id': sid,
            'ra_deg': data['star_info']['ra_deg'],
            'dec_deg': data['star_info']['dec_deg'],
            'mean_mag': final_mean,
            'rms': final_rms,
            'tight_rms': tight_rms,
            'mad_rms': mad_rms,
            'n_valid': n_valid,
            'corrected_mags': list(corrected_mags),
            'high_bg_var': high_bg_var
        })
        
    # =========================================================================
    # OUTLIER DETECTION (Noise Floor Model)
    # =========================================================================
    # In order to identify true variable stars, we must establish a baseline for
    # what "normal" photometric scatter (noise) looks like at any given magnitude.
    # Fainter stars naturally have higher RMS scatter due to lower signal-to-noise.
    
    mags_all = np.array([s['mean_mag'] for s in final_stats])
    rms_all = np.array([s['rms'] for s in final_stats])
    
    min_mag = np.min(mags_all)
    max_mag = np.max(mags_all)
    # Divide the magnitude range into dynamic bins (up to 20 bins)
    bins = np.linspace(min_mag, max_mag, min(20, len(final_stats)//10 + 2))
    
    bin_centers = []
    bin_rms_env = []
    
    inds = np.digitize(mags_all, bins)
    for i in range(1, len(bins)):
        in_bin = rms_all[inds == i]
        if len(in_bin) > 0:
            bin_centers.append(0.5 * (bins[i] + bins[i-1]))
            # Use the 15th percentile of RMS in this bin to track the dense lower 
            # envelope of the noise floor. We use a low percentile because true 
            # variables (outliers) will pull the mean/median upwards.
            bin_rms_env.append(np.percentile(in_bin, 15))
            
    if len(bin_centers) > 3:
        # Fit a linear model in log-space: log10(RMS) = A * mag + B
        # This models the exponential relationship between magnitude and noise.
        poly_coeffs = np.polyfit(bin_centers, np.log10(bin_rms_env), 1)
        expected_rms_func = lambda m: 10**(np.polyval(poly_coeffs, m))
    else:
        # Fallback to a flat median if there are too few stars to fit a curve
        med_rms = float(np.median(rms_all))
        # v4.16: return an array for array input (the plot passes a magnitude grid;
        # a scalar here raised "x and y must have same first dimension")
        expected_rms_func = lambda m: np.full(np.shape(m), med_rms) if np.ndim(m) else med_rms
        
    # =========================================================================
    # CALCULATE EXCESS RMS & CLASSIFY STARS (Suspect vs Noisy)
    # =========================================================================
    suspects = []
    for s in final_stats:
        # Determine the expected baseline noise for a star of this brightness
        exp_rms = expected_rms_func(s['mean_mag'])
        exp_rms = max(exp_rms, 0.001) # Prevent division by zero / negative noise
        
        # Excess RMS: scatter in excess of the noise floor, added in quadrature
        # (v4.16; was a linear difference rms - expected)
        excess_rms = float(np.sqrt(max(s['rms']**2 - exp_rms**2, 0.0))) if np.isfinite(s['rms']) else np.nan
        s['expected_rms'] = exp_rms
        s['excess_rms'] = excess_rms
        
        # Initial Flagging: 
        # 1. The scatter must be at least 3x the expected noise floor (Sigma detection).
        # 2. The absolute scatter must be > 0.03 mag to avoid flagging micro-variations 
        #    in ultra-bright stars where the theoretical noise floor approaches zero.
        is_suspect = s['rms'] > suspect_sigma * exp_rms and excess_rms > min_excess
        # v4.16: additionally require that the scatter clearly exceeds the star's
        # OWN photometric error (protects bright stars where the empirical noise
        # floor is unreliable). Only applied when the error is known.
        nr = s.get('noise_ratio', np.nan)
        if is_suspect and np.isfinite(nr) and nr < 2.0:
            is_suspect = False
        s['is_noisy'] = False
        
        if is_suspect:
            # We have a candidate! But is it a true variable or just a bad measurement?
            # 'tight_rms': Standard deviation calculated after heavily aggressively clipping outliers (2.5 sigma).
            # 'mad_rms': Median Absolute Deviation, which is intrinsically robust to single-point spikes.
            tight_val = s.get('tight_rms', np.nan)
            mad_val = s.get('mad_rms', np.nan)
            
            # Reclassify as "Noisy" if the robust scatter metrics are low.
            # Why? If 'rms' is high but 'tight_rms' or 'mad_rms' is low, it means the 
            # high scatter is driven entirely by a few extreme outlier frames (e.g. 
            # a cosmic ray hit, bad tracking frame, or satellite trail) rather than 
            # intrinsic, continuous stellar variability.
            is_flat = False
            if not np.isnan(tight_val) and tight_val < 1.5 * exp_rms:
                is_flat = True
            if not np.isnan(mad_val) and mad_val < 1.5 * exp_rms:
                is_flat = True
                
            if is_flat:
                # Discard as a true variable suspect, mark as a "noisy" artifact
                s['is_suspect'] = False
                s['is_noisy'] = True
            else:
                # The variability is robust and continuous. Mark as a true suspect!
                s['is_suspect'] = True
                suspects.append(s)
        else:
            s['is_suspect'] = False
            
    # Sort ALL results by excess RMS descending
    final_stats.sort(key=lambda x: x['excess_rms'] if np.isfinite(x['excess_rms']) else -1.0, reverse=True)
    
    # Initialize VSX name and type fields for all stars to prevent KeyError
    for s in final_stats:
        s['vsx_name'] = ''
        s['vsx_type'] = ''
        
    # Cross-match suspects with AAVSO VSX catalog before writing CSV
    if center_ra is None or center_dec is None:
        if master_catalog:
            try:
                center_ra = float(np.median([s['ra_deg'] for s in master_catalog]))
                center_dec = float(np.median([s['dec_deg'] for s in master_catalog]))
            except Exception:
                pass
                
    if center_ra is not None and center_dec is not None:
        try:
            cross_match_vsx_suspects(final_stats, center_ra, center_dec, verbose=False)
            cross_match_deep_catalog(final_stats, center_ra, center_dec, catalog_name="GAIA_DR3", verbose=False)
        except Exception as e:
            print(f"Error performing cross-match in calibrate_and_analyze: {e}")
            
    # Write CSV
    if out_csv_path:
        import csv
        try:
            with open(out_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'RA (deg)', 'Dec (deg)', 'Mean Mag', 'RMS', 'Expected RMS', 'Excess RMS', 'Valid Frames', 'Suspect', 'VSX Name', 'VSX Type', 'Catalog ID', 'Phot. Error (median)', 'RMS / Phot. Error'])
                for s in final_stats:
                    writer.writerow([
                        s['id'], 
                        f"{s['ra_deg']:.6f}", 
                        f"{s['dec_deg']:.6f}", 
                        f"{s['mean_mag']:.4f}", 
                        f"{s['rms']:.5f}", 
                        f"{s['expected_rms']:.5f}", 
                        f"{s['excess_rms']:.5f}", 
                        s['n_valid'],
                        'SUSPECT' if s['is_suspect'] else ('NOISY' if s.get('is_noisy') else 'NO'),
                        s.get('vsx_name', ''),
                        s.get('vsx_type', ''),
                        s.get('catalog_id', ''),
                        f"{s['phot_err']:.5f}" if np.isfinite(s.get('phot_err', np.nan)) else '',
                        f"{s['noise_ratio']:.2f}" if np.isfinite(s.get('noise_ratio', np.nan)) else ''
                    ])
        except Exception as e:
            print(f"Error writing CSV: {e}")
            
    return final_stats, expected_rms_func


def cross_match_vsx_suspects(final_stats, center_ra, center_dec, radius_arcmin=30.0, verbose=True):
    """
    Cross-matches all stars in final_stats against the AAVSO Variable Star Index (VSX).
    Adds 'vsx_name' and 'vsx_type' fields to each star dict.
    Returns the number of known matches found.
    """
    from python.calibration import get_vsx_stars
    
    if not final_stats or center_ra is None or center_dec is None:
        for s in final_stats:
            s['vsx_name'] = s.get('vsx_name', '')
            s['vsx_type'] = s.get('vsx_type', '')
        return 0
        
    # =========================================================================
    # DYNAMIC SEARCH RADIUS
    # =========================================================================
    # Calculate how wide the search radius needs to be to encompass the 
    # furthest detected star in the field of view. 
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        center_coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
        star_coords = SkyCoord(
            ra=[s['ra_deg'] for s in final_stats] * u.deg,
            dec=[s['dec_deg'] for s in final_stats] * u.deg
        )
        seps = center_coord.separation(star_coords)
        max_sep = float(seps.arcmin.max())
        dynamic_radius = min(max_sep + 5.0, 90.0)
        if radius_arcmin is None:
            radius_arcmin = max(30.0, dynamic_radius)
        else:
            radius_arcmin = max(radius_arcmin, dynamic_radius)
    except Exception as e:
        if verbose:
            print(f"Error computing dynamic VSX radius: {e}")
        if radius_arcmin is None:
            radius_arcmin = 30.0
    
    # Query VSX locally or fetch from online catalog
    vsx_stars = get_vsx_stars(center_ra, center_dec, radius_arcmin=radius_arcmin, verbose=verbose)
    
    if not vsx_stars:
        for s in final_stats:
            s['vsx_name'] = s.get('vsx_name', '')
            s['vsx_type'] = s.get('vsx_type', '')
        return 0
    
    # Cross-match using astropy's fast match_to_catalog_sky function
    vsx_coords = SkyCoord(
        ra=[v['ra_deg'] for v in vsx_stars] * u.deg,
        dec=[v['dec_deg'] for v in vsx_stars] * u.deg
    )
    
    star_coords = SkyCoord(
        ra=[s['ra_deg'] for s in final_stats] * u.deg,
        dec=[s['dec_deg'] for s in final_stats] * u.deg
    )
    
    idx, d2d, _ = star_coords.match_to_catalog_sky(vsx_coords)
    
    known_count = 0
    for i, s in enumerate(final_stats):
        if d2d[i].arcsec < 10.0:  # Relaxed from 5.0 to 10.0 to handle boundary WCS distortions
            matched = vsx_stars[idx[i]]
            vsx_type = matched.get('Type', '')
            # Don't label constant stars as known variables
            if vsx_type and 'CST' in str(vsx_type).upper():
                s['vsx_name'] = matched.get('id', '')
                s['vsx_type'] = f"{vsx_type} (constant)"
            else:
                s['vsx_name'] = matched.get('id', '')
                s['vsx_type'] = vsx_type
                if s.get('is_suspect'):
                    known_count += 1
        else:
            s['vsx_name'] = s.get('vsx_name', '')
            s['vsx_type'] = s.get('vsx_type', '')
    
    return known_count

def cross_match_deep_catalog(final_stats, center_ra, center_dec, radius_arcmin=None, catalog_name="GAIA_DR3", verbose=False):
    """
    Cross-matches the pipeline's detections with a deep catalog (e.g., GAIA)
    to confirm if the detected source exists in reality, helping filter out
    hot pixels or cosmic rays.
    """
    if not final_stats or center_ra is None or center_dec is None:
        for s in final_stats:
            s['catalog_match'] = False
        return 0
        
    try:
        from python.calibration import get_cached_catalog, fetch_online_catalog, save_to_cache
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError as e:
        if verbose: print(f"Error importing catalog modules: {e}")
        return 0
        
    # Compute dynamic radius
    try:
        center_coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
        star_coords = SkyCoord(
            ra=[s['ra_deg'] for s in final_stats] * u.deg,
            dec=[s['dec_deg'] for s in final_stats] * u.deg
        )
        seps = center_coord.separation(star_coords)
        max_sep = float(seps.arcmin.max())
        dynamic_radius = min(max_sep + 5.0, 90.0)
        if radius_arcmin is None:
            radius_arcmin = max(30.0, dynamic_radius)
        else:
            radius_arcmin = max(radius_arcmin, dynamic_radius)
    except Exception as e:
        if radius_arcmin is None: radius_arcmin = 30.0

    # Fetch deep catalog for FOV
    # v4.46: local copy -> cache -> local RefCat2 (Gaia-based) -> VizieR, inside fetch_online_catalog
    if verbose: print(f"Fetching {catalog_name} for deep cross-match...")
    cat_stars = fetch_online_catalog(center_ra, center_dec, radius_arcmin=radius_arcmin, catalog_name=catalog_name, verbose=verbose)
            
    if not cat_stars:
        if verbose: print("No deep catalog stars found for cross-match.")
        for s in final_stats:
            s['catalog_match'] = False
        return 0
        
    # Cross match
    cat_coords = SkyCoord(
        ra=[v['ra_deg'] for v in cat_stars] * u.deg,
        dec=[v['dec_deg'] for v in cat_stars] * u.deg
    )
    
    idx, d2d, _ = star_coords.match_to_catalog_sky(cat_coords)
    
    match_count = 0
    for i, s in enumerate(final_stats):
        if d2d[i].arcsec < 10.0:  # 10 arcseconds tolerance
            s['catalog_match'] = True
            s['catalog_id'] = cat_stars[idx[i]].get('cat_id', '')
            match_count += 1
        else:
            s['catalog_match'] = False
            s['catalog_id'] = ''
            
    return match_count
