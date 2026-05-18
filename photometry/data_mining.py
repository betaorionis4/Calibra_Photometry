import numpy as np
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_2dg
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

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
                
            y_max, x_max = data.shape
            
            # WCS -> pixel for all stars
            x_arr, y_arr = wcs.world_to_pixel(coords)
            
            # Now loop over stars and measure
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
                    
                # Re-centroid in a 15x15 box
                size = 15
                x_int, y_int = int(np.round(x_init)), int(np.round(y_init))
                
                # Make sure the box is fully inside the image
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
                        x_rel, y_rel = centroid_2dg(cutout - median_bg_cutout)
                    if np.isnan(x_rel) or np.isnan(y_rel):
                        x_ref, y_ref = x_init, y_init
                    else:
                        x_ref = x_int - size + x_rel
                        y_ref = y_int - size + y_rel
                except Exception:
                    x_ref, y_ref = x_init, y_init
                    
                # Saturation check inside aperture
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
                    'snr': snr
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

        mean, median_bg, std_bg = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        
        # Initialize DAOStarFinder (using the robust fallback from earlier)
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

def calibrate_and_analyze(track_results, master_catalog, out_csv_path, min_valid_fraction=0.7, ensemble_size=100):
    """
    Performs the 2-Pass Global Differential Calibration and Statistical Analysis.
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
    
    # PASS 1: Select initial ensemble
    ensemble_ids = sorted_star_ids[:ensemble_size]
    if len(ensemble_ids) < 5:
        print("Warning: Very few stars available for ensemble calibration.")
        
    def compute_zero_points(ens_ids):
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
    
    # Calculate preliminary RMS for ensemble
    ens_rms = {}
    for sid in ensemble_ids:
        corrected_mags = valid_stars[sid]['mags'] + zp_pass1
        _, _, rms = sigma_clipped_stats(corrected_mags, sigma=4.0, maxiters=2, cenfunc='median', stdfunc='mad_std')
        ens_rms[sid] = rms
        
    median_ens_rms = np.median(list(ens_rms.values()))
    
    # PASS 2: Refine ensemble
    refined_ensemble_ids = [sid for sid in ensemble_ids if ens_rms[sid] <= 2.0 * median_ens_rms]
    if len(refined_ensemble_ids) < 3:
        refined_ensemble_ids = ensemble_ids
        
    # Calculate Pass 2 zero-points
    zp_pass2 = compute_zero_points(refined_ensemble_ids)
    
    # Apply Pass 2 zero-points to ALL valid stars and compute final stats
    final_stats = []
    for sid, data in valid_stars.items():
        corrected_mags = data['mags'] + zp_pass2
        final_mean, _, final_rms = sigma_clipped_stats(corrected_mags, sigma=4.0, maxiters=2, cenfunc='median', stdfunc='mad_std')
        n_valid = np.sum(data['valid_mask'])
        
        final_stats.append({
            'id': sid,
            'ra_deg': data['star_info']['ra_deg'],
            'dec_deg': data['star_info']['dec_deg'],
            'mean_mag': final_mean,
            'rms': final_rms,
            'n_valid': n_valid
        })
        
    # OUTLIER DETECTION (Noise Floor Model)
    mags_all = np.array([s['mean_mag'] for s in final_stats])
    rms_all = np.array([s['rms'] for s in final_stats])
    
    min_mag = np.min(mags_all)
    max_mag = np.max(mags_all)
    bins = np.linspace(min_mag, max_mag, min(20, len(final_stats)//10 + 2))
    
    bin_centers = []
    bin_rms_env = []
    
    inds = np.digitize(mags_all, bins)
    for i in range(1, len(bins)):
        in_bin = rms_all[inds == i]
        if len(in_bin) > 0:
            bin_centers.append(0.5 * (bins[i] + bins[i-1]))
            # Use 15th percentile to track the dense lower envelope of noise
            bin_rms_env.append(np.percentile(in_bin, 15))
            
    if len(bin_centers) > 3:
        # Linear fit in log-space: log10(RMS) = A * mag + B
        poly_coeffs = np.polyfit(bin_centers, np.log10(bin_rms_env), 1)
        expected_rms_func = lambda m: 10**(np.polyval(poly_coeffs, m))
    else:
        med_rms = np.median(rms_all)
        expected_rms_func = lambda m: float(med_rms)
        
    # Calculate excess RMS
    suspects = []
    for s in final_stats:
        exp_rms = expected_rms_func(s['mean_mag'])
        exp_rms = max(exp_rms, 0.001)
        
        excess_rms = s['rms'] - exp_rms
        s['expected_rms'] = exp_rms
        s['excess_rms'] = excess_rms
        
        # Flagging suspects: RMS must be 3x expected noise floor AND > 0.03 magnitude deviation
        if s['rms'] > 3.0 * exp_rms and excess_rms > 0.03:
            s['is_suspect'] = True
            suspects.append(s)
        else:
            s['is_suspect'] = False
            
    # Sort ALL results by excess RMS descending
    final_stats.sort(key=lambda x: x['excess_rms'], reverse=True)
    
    # Write CSV
    if out_csv_path:
        import csv
        try:
            with open(out_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'RA (deg)', 'Dec (deg)', 'Mean Mag', 'RMS', 'Expected RMS', 'Excess RMS', 'Valid Frames', 'Suspect'])
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
                        'YES' if s['is_suspect'] else 'NO'
                    ])
        except Exception as e:
            print(f"Error writing CSV: {e}")
            
    return final_stats, expected_rms_func
