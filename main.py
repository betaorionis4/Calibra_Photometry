import glob
import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from photometry.star_detection import detect_stars
from photometry.psf_fitting import refine_coordinates_psf
from photometry.aperture_phot import perform_aperture_photometry
from photometry.calibration import match_and_calibrate
from photometry.shift_analysis import generate_shift_report
from photometry.image_calibration import calibrate_image
import csv
import sys

from gui import run_config_gui

def process_file(fits_filename, config):
    # Determine Log Filename
    os.makedirs('photometry_output/logs', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(fits_filename))[0]
    
    # We'll refine the log name after reading the header for the filter
    # For now, a temporary log name or just wait
    
    print(f"\n=================================================================")
    print(f"PROCESSING FILE: {fits_filename}")
    print(f"=================================================================")

    # Load Image Data
    try:
        print(f"Reading {fits_filename}...")
        with fits.open(fits_filename) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header

        # Check CCD Settings
        exptime = header.get('EXPTIME', 1.0)
        hdr_gain = header.get('GAIN', 'Unknown')
        hdr_offset = header.get('OFFSET', header.get('BLKLEVEL', 'Unknown'))
        
        # Extract RA/Dec for online catalog query
        ra_val = header.get('RA')
        dec_val = header.get('DEC')
        
        # Fallback to OBJCTRA/DEC if RA/DEC not found
        if ra_val is None: ra_val = header.get('OBJCTRA')
        if dec_val is None: dec_val = header.get('OBJCTDEC')
        
        # Convert to float if they are strings (e.g. HH MM SS)
        center_ra = None
        center_dec = None
        if ra_val is not None and dec_val is not None:
            try:
                if isinstance(ra_val, str) and (':' in ra_val or ' ' in ra_val):
                    c = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg))
                    center_ra = c.ra.deg
                    center_dec = c.dec.deg
                else:
                    center_ra = float(ra_val)
                    center_dec = float(dec_val)
            except Exception as e:
                print(f"Warning: Could not parse RA/Dec from header: {e}")

        print(f"FITS Header Check -> EXPTIME: {exptime}s | GAIN: {hdr_gain} | OFFSET: {hdr_offset}")
        if center_ra is not None:
            print(f"Center Coordinates: RA={center_ra:.4f}, Dec={center_dec:.4f}")

        # Update Log file with Filter and Catalog info
        cat_name = os.path.basename(config['reference_catalog']).split('.')[0]
        filt = header.get('FILTER', 'NoFilt').replace('/', '_')
        log_name = f"log_{base_name}_{filt}_{cat_name}.txt"
        log_path = os.path.join('photometry_output', 'logs', log_name)
        
        if hasattr(sys.stdout, 'set_log_file'):
            sys.stdout.set_log_file(log_path)
            print(f"Session log initiated: {log_path}")

        # NEW: FITS Calibration Step
        cal_cfg = config.get('calibration_settings', {})
        if cal_cfg.get('enable', False):
            bias_path = cal_cfg.get('bias_path')
            # Determine flat based on filter
            filter_name = header.get('FILTER', 'V').upper()
            if 'B' in filter_name:
                flat_path = cal_cfg.get('flat_b_path')
            else:
                flat_path = cal_cfg.get('flat_v_path')
                
            # Save calibrated files in a 'calibrated' subfolder of the input directory
            input_dir = os.path.dirname(fits_filename)
            out_dir = os.path.join(input_dir, 'calibrated')
            image_data, header = calibrate_image(image_data, header, bias_path, flat_path, out_dir=out_dir)

    except FileNotFoundError:
        print(f"Error: {fits_filename} not found.")
        return

    # Ensure output directories exist
    os.makedirs('photometry_output', exist_ok=True)
    os.makedirs('photometry_plots', exist_ok=True)

    # Define a unique output CSV per fits file
    base_name = os.path.splitext(os.path.basename(fits_filename))[0]
    output_csv = os.path.join('photometry_output', f'targets_auto_{base_name}.csv')

    # 1. Star Detection
    results = detect_stars(
        image_data, header, config['detect_sigma'],
        sharplo=config['dao_sharplo'], sharphi=config['dao_sharphi'], roundlo=config['dao_roundlo'], roundhi=config['dao_roundhi'],
        filter_mode=config['filter_mode'], xy_bounds=config['xy_bounds'], radec_bounds=config['radec_bounds']
    )
    if not results:
        return

    # 2. PSF Fitting
    refine_coordinates_psf(
        image_data, results, config['box_size'], config['aperture_radius'], 
        config['saturation_limit'], config['max_plots_to_show_per_file'],
        display_plots=config['display_plots'],
        plot_output_dir='photometry_plots',
        base_filename=base_name,
        print_psf_fitting=config['print_psf_fitting']
    )

    # 3. Aperture Photometry
    perform_aperture_photometry(
        image_data, results, config['aperture_radius'], config['annulus_inner'], config['annulus_outer'],
        print_table=config['print_star_detection_table'],
        gain=config['ccd_gain'], read_noise=config['ccd_read_noise'], dark_current=config['ccd_dark_current'], exptime=exptime
    )

    # Filter out questionable detections (e.g. net_flux <= 0 or snr < 3.0)
    original_count = len(results)
    results = [rs for rs in results if rs.get('net_flux', 0) > 0 and rs.get('snr', 0) >= 3.0]
    filtered_count = len(results)
    print(f"Filtered out {original_count - filtered_count} questionable/faint detections (SNR < 3 or negative flux). Remaining: {filtered_count}")

    # 4. Zero Point Calibration
    filter_name = header.get('FILTER', 'V')
    output_report = os.path.join('photometry_output', f'calibration_report_{base_name}.md')
    match_and_calibrate(results, config['reference_catalog'], filter_name, config['match_tolerance_arcsec'],
                        default_zp=config['default_zero_point'], run_new_calibration=config['run_new_calibration'],
                        output_report=output_report, center_ra=center_ra, center_dec=center_dec,
                        snr_threshold=config['calib_snr_threshold'],
                        print_to_console=config['print_detailed_calibration'],
                        header=header, radius_arcmin=config.get('catalog_search_radius', 15.0))


    # Calculate Detection Limits
    if len(results) > 0:
        snr_vals = [rs.get('snr', 0) for rs in results]
        # Find the star with the lowest SNR that is still >= 3.0
        min_snr_idx = np.argmin(snr_vals)
        faintest_star = results[min_snr_idx]
        print(f"\n--- Detection Limit (Faintest Star @ SNR={faintest_star.get('snr', 0):.1f}) ---")
        if 'mag_inst' in faintest_star and not np.isnan(faintest_star['mag_inst']):
            print(f"Instrumental Magnitude Limit: {faintest_star['mag_inst']:.2f}")
        if 'mag_calibrated' in faintest_star and not np.isnan(faintest_star['mag_calibrated']):
            print(f"Calibrated Magnitude Limit: {faintest_star['mag_calibrated']:.2f}")
        print("------------------------------------------------------------------\n")

    # 5. Export Results
    print(f"Saving results to {output_csv}...")
    with open(output_csv, mode='w', newline='') as f:
        fieldnames = [
            'id', 'raw_x', 'raw_y', 'refined_x', 'refined_y', 
            'ra_deg', 'dec_deg', 'ra_hms', 'dec_dms',
            'peak_adu', 'dao_flux', 'net_flux', 'flux_err', 'snr',
            'mag_inst', 'mag_inst_err', 'mag_calibrated', 'mag_calibrated_err', 'airmass', 'is_variable'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for rs in results:
            writer.writerow({
                'id': rs.get('id', ''),
                'raw_x': f"{rs.get('x', 0):.2f}",
                'raw_y': f"{rs.get('y', 0):.2f}",
                'refined_x': f"{rs.get('refined_x', 0):.2f}" if 'refined_x' in rs else "",
                'refined_y': f"{rs.get('refined_y', 0):.2f}" if 'refined_y' in rs else "",
                'ra_deg': rs.get('ra_deg', ''),
                'dec_deg': rs.get('dec_deg', ''),
                'ra_hms': rs.get('ra_hms', ''),
                'dec_dms': rs.get('dec_dms', ''),
                'peak_adu': f"{rs.get('peak_adu', 0):.1f}",
                'dao_flux': f"{rs['dao_flux']:.3f}",
                'net_flux': f"{rs.get('net_flux', 0):.3f}",
                'flux_err': f"{rs.get('flux_err', 0):.3f}",
                'snr': f"{rs.get('snr', 0):.2f}",
                'mag_inst': f"{rs.get('mag_inst', 0):.3f}" if 'mag_inst' in rs and not np.isnan(rs['mag_inst']) else "",
                'mag_inst_err': f"{rs.get('mag_inst_err', 0):.3f}" if 'mag_inst_err' in rs and not np.isnan(rs['mag_inst_err']) else "",
                'mag_calibrated': f"{rs.get('mag_calibrated', 0):.3f}" if 'mag_calibrated' in rs and not np.isnan(rs['mag_calibrated']) else "",
                'mag_calibrated_err': f"{rs.get('mag_calibrated_err', 0):.3f}" if 'mag_calibrated_err' in rs and not np.isnan(rs['mag_calibrated_err']) else "",
                'airmass': f"{header.get('AIRMASS', 1.0):.4f}",
                'is_variable': 'Yes' if rs.get('is_variable', False) else 'No'
            })

    # 6. Shift Analysis
    # 6. Shift Analysis
    if config['run_shift_analysis']:
        output_md = os.path.join('photometry_output', f'shift_analysis_{base_name}.md')
        print(f"Generating shift analysis report at {output_md}...")
        shift_stats = generate_shift_report(results, config['reference_catalog'], header, config['match_tolerance_arcsec'], output_md,
                                            center_ra=center_ra, center_dec=center_dec)
        
        if shift_stats:
            print("\n--- Positional Shift Summary (Detected - Reference) ---")
            print(f"Matched Stars: {shift_stats['count']}")
            print(f"Median Shift (Pixels): dX={shift_stats['med_dx']:+.2f} | dY={shift_stats['med_dy']:+.2f}")
            print(f"Median Shift (Arcsec): dRA={shift_stats['med_dra']:+.2f} | dDec={shift_stats['med_ddec']:+.2f}")
            print("-------------------------------------------------------\n")

    print("Done!\n")
    if hasattr(sys.stdout, 'set_log_file'):
        sys.stdout.set_log_file(None)
    return output_csv, filt

def run_pipeline(cfg):
    input_pattern = cfg['input_pattern']

    if isinstance(input_pattern, list):
        files_to_process = input_pattern
    elif os.path.isfile(input_pattern):
        files_to_process = [input_pattern]
    else:
        files_to_process = glob.glob(input_pattern)

    processed_results = []
    if not files_to_process:
        print(f"No files found matching pattern: {input_pattern}")
    else:
        print(f"Found {len(files_to_process)} file(s) to process.")
        for f in files_to_process:
            res = process_file(f, cfg)
            if res:
                processed_results.append(res)
    return processed_results

def main():
    # run_config_gui now handles its own loop and calls run_pipeline in a thread
    run_config_gui(run_pipeline)

if __name__ == '__main__':
    main()
