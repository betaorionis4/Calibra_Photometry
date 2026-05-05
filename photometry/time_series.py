import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_2dg

def get_hjd(time_str, ra, dec, header, site_lat=0.0, site_long=0.0):
    """ Calculates Heliocentric Julian Date (HJD) from DATE-OBS and coordinates. """
    try:
        # 1. Parse Time
        # Try different possible header keywords for time
        t_val = header.get('JD') or header.get('JD_SOBJ')
        if t_val:
            t = Time(float(t_val), format='jd', scale='utc')
        else:
            # Fallback to DATE-OBS
            t_str = header.get('DATE-OBS')
            if not t_str: return None
            t = Time(t_str, format='isot', scale='utc')
        
        # 2. Get Observer Location
        lon = header.get('SITELONG', site_long)
        lat = header.get('SITELAT', site_lat)
        height = header.get('SITEELEV', 0.0)
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

def measure_star(image_data, wcs, ra, dec, aperture_radius, annulus_inner, annulus_outer, gain=1.0):
    """ Performs targeted aperture photometry on a specific RA/Dec. """
    try:
        # 1. Convert RA/Dec to pixel coordinates using WCS
        x_init, y_init = wcs.world_to_pixel(SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs'))
        
        # 2. Refine Centroid (in a small 15x15 box)
        size = 15
        x_int, y_int = int(np.round(x_init)), int(np.round(y_init))
        # Bounds check
        if x_int < size or y_int < size or x_int > image_data.shape[1]-size or y_int > image_data.shape[0]-size:
            return None
            
        cutout = image_data[y_int-size:y_int+size, x_int-size:x_int+size]
        # Subtract local background for centroiding
        _, median, _ = sigma_clipped_stats(cutout)
        x_rel, y_rel = centroid_2dg(cutout - median)
        
        x_refined = x_int - size + x_rel
        y_refined = y_int - size + y_rel
        
        # 3. Aperture Photometry
        pos = (x_refined, y_refined)
        aperture = CircularAperture(pos, r=aperture_radius)
        annulus = CircularAnnulus(pos, r_in=annulus_inner, r_out=annulus_outer)
        
        phot_table = aperture_photometry(image_data, aperture, method='exact')
        raw_flux = phot_table['aperture_sum'][0]
        
        # Background subtraction
        annulus_mask = annulus.to_mask(method='center')
        bkg_pixels = annulus_mask.get_values(image_data)
        bkg_pixels = bkg_pixels[~np.isnan(bkg_pixels)]
        _, bkg_median, bkg_std = sigma_clipped_stats(bkg_pixels, sigma=3.0)
        
        net_flux = raw_flux - (bkg_median * aperture.area)
        
        # Error calculation
        variance = (max(net_flux, 0) / gain) + aperture.area * (bkg_std**2) + (aperture.area**2 * bkg_std**2 / len(bkg_pixels))
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

def run_time_series_photometry(fits_files, target_ra, target_dec, ref_ra, ref_dec,
                               ref_mag_std, ref_bv, target_bv,
                               coeff_term, coeff_color, 
                               aperture_radius, annulus_inner, annulus_outer,
                               gain=1.0, k_coeff=0.15, filter_name='V',
                               observer_name="Calibra User",
                               site_lat=0.0, site_long=0.0):
    """
    Main loop for time-series photometry.
    """
    results = []
    
    from astropy.wcs import WCS
    import warnings
    from astropy.utils.exceptions import AstropyWarning
    warnings.simplefilter('ignore', category=AstropyWarning)

    print(f"Starting time-series analysis on {len(fits_files)} files...")
    
    for i, fpath in enumerate(fits_files):
        try:
            with fits.open(fpath) as hdul:
                header = hdul[0].header
                data = hdul[0].data
                if data is None and len(hdul) > 1:
                    data = hdul[1].data
                    header = hdul[1].header
                
                if data is None:
                    continue

                if data.ndim == 3:
                    data = data[0]

                wcs = WCS(header)
                
                # Use header site info if available, else use manual site info
                lat = header.get('SITELAT', site_lat)
                lon = header.get('SITELONG', site_long)
                
                # 1. Timing and Airmass
                jd = header.get('JD') or header.get('MJD', 0) + 2400000.5
                if not jd:
                    t_obs = header.get('DATE-OBS')
                    if t_obs:
                        try: jd = Time(t_obs).jd
                        except: jd = 0
                
                hjd = get_hjd(None, target_ra, target_dec, header, lat, lon) or jd
                airmass = header.get('AIRMASS', 1.0)
                
                # 2. Measure Target and Reference
                target_res = measure_star(data, wcs, target_ra, target_dec, aperture_radius, annulus_inner, annulus_outer, gain)
                ref_res = measure_star(data, wcs, ref_ra, ref_dec, aperture_radius, annulus_inner, annulus_outer, gain)
                
                if target_res and ref_res:
                    if np.isnan(target_res['mag_inst']) or np.isnan(ref_res['mag_inst']):
                        print(f"  - {os.path.basename(fpath)}: Measurement returned NaN (too faint or saturated?)")
                        continue
                    # 3. Calibration Logic
                    # m_corr = m_inst - k*X
                    v_target_corr = target_res['mag_inst'] - (k_coeff * airmass)
                    v_ref_corr = ref_res['mag_inst'] - (k_coeff * airmass)
                    
                    # ZV = V_ref_std - (v_ref_corr + Tv_bv * (B-V)_ref)
                    zv = ref_mag_std - (v_ref_corr + coeff_term * ref_bv)
                    
                    # V_target_std = v_target_corr + Tv_bv * (B-V)_target + ZV
                    v_target_std = v_target_corr + coeff_term * target_bv + zv

                    if i < 5:
                        print(f"\n--- Diagnostic (Image {i+1}: {os.path.basename(fpath)}) ---")
                        print(f"  Target Inst: {target_res['mag_inst']:.4f} | Corr: {v_target_corr:.4f}")
                        print(f"  Ref Inst:    {ref_res['mag_inst']:.4f} | Corr: {v_ref_corr:.4f}")
                        print(f"  Ref Std:     {ref_mag_std:.4f} | B-V: {ref_bv:.4f}")
                        print(f"  Zero Point:  {zv:.4f}")
                        print(f"  Color Term:  {coeff_term:.4f} * {target_bv:.4f} = {coeff_term * target_bv:.4f}")
                        print(f"  FINAL MAG:   {v_target_std:.4f}")
                        print("------------------------------------------------------\n")
                    
                    results.append({
                        'file': os.path.basename(fpath),
                        'jd': jd,
                        'hjd': hjd,
                        'airmass': airmass,
                        'mag': v_target_std,
                        'mag_err': target_res['mag_err'],
                        'snr': target_res['snr']
                    })
                else:
                    if not target_res: print(f"  - {os.path.basename(fpath)}: Target star could not be centroided or is off-image.")
                    if not ref_res: print(f"  - {os.path.basename(fpath)}: Reference star could not be centroided or is off-image.")
                    
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(fits_files)} images...")
                
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    if not results:
        return None, "No data successfully processed."
        
    # Sort by time
    results.sort(key=lambda x: x['hjd'])
    
    return results, "Success"

def save_aavso_report(results, output_path, target_name, filter_name, obs_code, site_id=""):
    """ Generates a report in AAVSO Extended Format. """
    try:
        with open(output_path, 'w', newline='') as f:
            f.write("#TYPE=Extended\n")
            f.write(f"#OBSCODE={obs_code}\n")
            f.write(f"#SOFTWARE=Calibra 1.5\n")
            f.write(f"#DELIM=,\n")
            f.write(f"#DATE=HJD\n")
            f.write(f"#OBSTYPE=CCD\n")
            f.write("NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,NOTES\n")
            
            for r in results:
                # Format: Name, Date, Mag, Merr, Filt, Trans, Mtype, Cname, Cmag, Kname, Kmag, Amass, Notes
                f.write(f"{target_name},{r['hjd']:.6f},{r['mag']:.4f},{r['mag_err']:.4f},{filter_name},NO,STD,na,na,na,na,{r['airmass']:.3f},na\n")
        return True
    except Exception as e:
        print(f"Error saving AAVSO report: {e}")
        return False

def plot_light_curve(results, target_name, output_path):
    """ Plots the light curve. """
    if not results: return
    
    times = [r['hjd'] for r in results]
    mags = [r['mag'] for r in results]
    errs = [r['mag_err'] for r in results]
    
    # JD Offset for readability
    t0 = int(min(times))
    times_rel = [t - t0 for t in times]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(times_rel, mags, yerr=errs, fmt='o', color='darkblue', ecolor='gray', capsize=2, markersize=4)
    plt.gca().invert_yaxis() # Magnitudes are inverted
    plt.title(f"Light Curve: {target_name}")
    plt.xlabel(f"HJD - {t0}")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
