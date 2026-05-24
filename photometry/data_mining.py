import numpy as np
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.utils.exceptions import AstropyWarning
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_2dg
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# =============================================================================
# DATA MINING PIPELINE
# =============================================================================
# The Data Mining module is responsible for analyzing entire fields of view across
# an entire timeseries of FITS files. It detects all stars in a reference frame,
# forces aperture photometry at those coordinates across all frames, and performs
# global differential calibration to extract intrinsic stellar variability,
# identifying candidate variable stars while rejecting artifacts and noise.
# =============================================================================

def track_and_measure(fits_files, master_catalog, aperture_radius=5.0, 
                      annulus_inner=7.0, annulus_outer=13.0, 
                      saturation_limit=55000, edge_buffer=50, 
                      cancel_event=None, progress_callback=None):
    """
    Tracks and measures all stars in the master_catalog across all FITS files.
    Returns a list of dicts, one per frame, containing the measurements for all stars.
    """
    results = []
    warnings_list = []
    
    if not fits_files or not master_catalog:
        return results, warnings_list
        
    base_filter = None
    total_files = len(fits_files)
    
    # Pre-create SkyCoord array for all stars
    ra_list = [star['ra_deg'] for star in master_catalog]
    dec_list = [star['dec_deg'] for star in master_catalog]
    coords = SkyCoord(ra=ra_list*u.deg, dec=dec_list*u.deg, frame='icrs')
    
    for i, fits_path in enumerate(fits_files):
        if cancel_event and cancel_event.is_set():
            break
            
        frame_result = {
            'file': os.path.basename(fits_path),
            'measurements': {} # star_id -> {net_flux, mag_inst, snr}
        }
        
        try:
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
                    
            # Check filter
            curr_filter = header.get('FILTER', 'UNKNOWN')
            if base_filter is None:
                base_filter = curr_filter
            elif curr_filter != base_filter:
                msg = f"Warning: Mixed filters detected! {os.path.basename(fits_path)} uses {curr_filter}, expected {base_filter}."
                if msg not in warnings_list:
                    warnings_list.append(msg)
                    
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                wcs = WCS(header)
                
            if not wcs.has_celestial:
                continue
                
            # Calculate global image background std to compare against local annulus std
            # This helps us identify if a star's local background is unusually noisy 
            # (e.g. it sits on the diffraction spike of a very bright neighbor).
            _, _, img_std = sigma_clipped_stats(data, sigma=3.0, maxiters=3)
            frame_result['img_std'] = img_std
                
            y_max, x_max = data.shape
            
            # Fast bulk conversion: WCS world coordinates -> image pixel coordinates
            x_arr, y_arr = wcs.world_to_pixel(coords)
            
            # Now loop over stars and measure flux using aperture photometry
            for idx, star in enumerate(master_catalog):
                star_id = star['id']
                x_init = x_arr[idx]
                y_init = y_arr[idx]
                
                # Check NaNs from WCS
                if np.isnan(x_init) or np.isnan(y_init):
                    frame_result['measurements'][star_id] = {'net_flux': np.nan, 'mag_inst': np.nan, 'snr': np.nan}
                    continue
                    
                # Check Edge
                if (x_init < edge_buffer or x_init > (x_max - edge_buffer) or
                    y_init < edge_buffer or y_init > (y_max - edge_buffer)):
                    frame_result['measurements'][star_id] = {'net_flux': np.nan, 'mag_inst': np.nan, 'snr': np.nan}
                    continue
                    
                # ---------------------------------------------------------
                # Centroid Refinement
                # ---------------------------------------------------------
                # Since telescope tracking drifts slightly and WCS solutions aren't perfect,
                # we attempt to find the true center of the star within a 15x15 pixel box.
                size = 15
                x_int, y_int = int(np.round(x_init)), int(np.round(y_init))
                
                # Make sure the box is fully inside the image boundaries
                if (x_int - size < 0 or x_int + size > x_max or
                    y_int - size < 0 or y_int + size > y_max):
                    frame_result['measurements'][star_id] = {'net_flux': np.nan, 'mag_inst': np.nan, 'snr': np.nan}
                    continue

                cutout = data[y_int-size:y_int+size, x_int-size:x_int+size]
                
                try:
                    _, median_bg_cutout, _ = sigma_clipped_stats(cutout)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        warnings.simplefilter("ignore", category=AstropyWarning)
                        # centroid_2dg fits a 2D Gaussian to find the sub-pixel center
                        x_rel, y_rel = centroid_2dg(cutout - median_bg_cutout)
                    
                    # If Gaussian fitting fails (star is too faint or a hot pixel), 
                    # fallback to the original exact WCS coordinate to force measurement.
                    if np.isnan(x_rel) or np.isnan(y_rel):
                        x_ref, y_ref = x_init, y_init
                    else:
                        x_ref = x_int - size + x_rel
                        y_ref = y_int - size + y_rel
                except Exception:
                    x_ref, y_ref = x_init, y_init
                    
                # ---------------------------------------------------------
                # Measurement: Saturation Check & Photometry
                # ---------------------------------------------------------
                # Check for saturation (non-linear detector response) within the aperture
                aperture = CircularAperture((x_ref, y_ref), r=aperture_radius)
                ap_mask = aperture.to_mask(method='center')
                ap_data = ap_mask.get_values(data)
                
                if len(ap_data) == 0 or np.max(ap_data) > saturation_limit:
                    frame_result['measurements'][star_id] = {'net_flux': np.nan, 'mag_inst': np.nan, 'snr': np.nan}
                    continue
                    
                # Photometry
                annulus = CircularAnnulus((x_ref, y_ref), r_in=annulus_inner, r_out=annulus_outer)
                phot_table = aperture_photometry(data, aperture, method='exact')
                raw_flux = phot_table['aperture_sum'][0]
                
                ann_mask = annulus.to_mask(method='center')
                bkg_pixels = ann_mask.get_values(data)
                bkg_pixels = bkg_pixels[~np.isnan(bkg_pixels)]
                
                if len(bkg_pixels) == 0:
                    frame_result['measurements'][star_id] = {'net_flux': np.nan, 'mag_inst': np.nan, 'snr': np.nan}
                    continue
                    
                _, bkg_median, bkg_std = sigma_clipped_stats(bkg_pixels, sigma=3.0)
                
                net_flux = raw_flux - (bkg_median * aperture.area)
                
                # Negative flux guard
                if net_flux <= 0:
                    frame_result['measurements'][star_id] = {'net_flux': np.nan, 'mag_inst': np.nan, 'snr': np.nan}
                    continue
                    
                mag_inst = -2.5 * np.log10(net_flux)
                
                # Basic SNR
                variance = net_flux + aperture.area * (bkg_std**2) + (aperture.area**2 * bkg_std**2 / len(bkg_pixels))
                flux_err = np.sqrt(variance)
                snr = net_flux / flux_err if flux_err > 0 else 0
                
                frame_result['measurements'][star_id] = {
                    'net_flux': net_flux,
                    'mag_inst': mag_inst,
                    'snr': snr,
                    'bkg_std': bkg_std
                }
                
        except Exception as e:
            print(f"Error measuring frame {fits_path}: {e}")
            
        results.append(frame_result)
        
        if progress_callback:
            progress_callback(((i + 1) / total_files) * 100)
            
    return results, warnings_list

def detect_all_stars(fits_path, detect_sigma=5.0, saturation_limit=55000, edge_buffer=50):
    """
    Detects all stars in a reference frame, rejecting saturated stars and stars near the edge.
    Returns a list of dictionaries containing star coordinates and initial fluxes.
    """
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
                daofind = DAOStarFinder(fwhm=3.5, threshold=detect_sigma * std_bg,
                                        sharpness_range=(0.2, 1.0), 
                                        roundness_range=(-1.0, 1.0))
            except TypeError:
                daofind = DAOStarFinder(fwhm=3.5, threshold=detect_sigma * std_bg,
                                        sharplo=0.2, sharphi=1.0, 
                                        roundlo=-1.0, roundhi=1.0)
                                        
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

def calibrate_and_analyze(track_results, master_catalog, out_csv_path, min_valid_fraction=0.7, ensemble_size=100, center_ra=None, center_dec=None):
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
    
    # 1. Reorganize data: star_id -> list of mag_inst across frames
    star_mags = {star['id']: [] for star in master_catalog}
    for frame in track_results:
        for star_id, m_dict in frame['measurements'].items():
            star_mags[star_id].append(m_dict['mag_inst'])
            
    # Convert lists to numpy arrays
    for star_id in star_mags:
        star_mags[star_id] = np.array(star_mags[star_id])
        
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
        
        final_stats.append({
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
        med_rms = np.median(rms_all)
        expected_rms_func = lambda m: float(med_rms)
        
    # =========================================================================
    # CALCULATE EXCESS RMS & CLASSIFY STARS (Suspect vs Noisy)
    # =========================================================================
    suspects = []
    for s in final_stats:
        # Determine the expected baseline noise for a star of this brightness
        exp_rms = expected_rms_func(s['mean_mag'])
        exp_rms = max(exp_rms, 0.001) # Prevent division by zero / negative noise
        
        # Excess RMS is how much more scatter the star has compared to the noise floor
        excess_rms = s['rms'] - exp_rms
        s['expected_rms'] = exp_rms
        s['excess_rms'] = excess_rms
        
        # Initial Flagging: 
        # 1. The scatter must be at least 3x the expected noise floor (Sigma detection).
        # 2. The absolute scatter must be > 0.03 mag to avoid flagging micro-variations 
        #    in ultra-bright stars where the theoretical noise floor approaches zero.
        is_suspect = s['rms'] > 3.0 * exp_rms and excess_rms > 0.03
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
    final_stats.sort(key=lambda x: x['excess_rms'], reverse=True)
    
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
                writer.writerow(['ID', 'RA (deg)', 'Dec (deg)', 'Mean Mag', 'RMS', 'Expected RMS', 'Excess RMS', 'Valid Frames', 'Suspect', 'VSX Name', 'VSX Type', 'Catalog ID'])
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
                        s.get('catalog_id', '')
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
    from photometry.calibration import get_vsx_stars
    
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
        from photometry.calibration import get_cached_catalog, fetch_online_catalog, save_to_cache
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
    cat_stars = get_cached_catalog(center_ra, center_dec, radius_arcmin, catalog_name, verbose=verbose)
    if not cat_stars:
        if verbose: print(f"Fetching {catalog_name} for deep cross-match...")
        cat_stars = fetch_online_catalog(center_ra, center_dec, radius_arcmin=radius_arcmin, catalog_name=catalog_name, verbose=verbose)
        if cat_stars:
            save_to_cache(cat_stars, center_ra, center_dec, radius_arcmin, catalog_name, verbose=verbose)
            
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
        if d2d[i].arcsec < 5.0:  # 5 arcseconds tolerance
            s['catalog_match'] = True
            s['catalog_id'] = cat_stars[idx[i]].get('cat_id', '')
            match_count += 1
        else:
            s['catalog_match'] = False
            s['catalog_id'] = ''
            
    return match_count
