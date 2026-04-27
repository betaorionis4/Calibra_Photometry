import glob
import os
import numpy as np
from astropy.io import fits

from photometry.star_detection import detect_stars
from photometry.psf_fitting import refine_coordinates_psf
from photometry.aperture_phot import perform_aperture_photometry
from photometry.calibration import match_and_calibrate
from photometry.shift_analysis import generate_shift_report
import csv

from gui import run_config_gui

def process_file(fits_filename, config):
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
        print(f"FITS Header Check -> EXPTIME: {exptime}s | GAIN: {hdr_gain} | OFFSET: {hdr_offset}")

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
                        output_report=output_report)

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
            'mag_inst', 'mag_inst_err', 'mag_calibrated', 'mag_calibrated_err'
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
                'mag_calibrated_err': f"{rs.get('mag_calibrated_err', 0):.3f}" if 'mag_calibrated_err' in rs and not np.isnan(rs['mag_calibrated_err']) else ""
            })

    # 6. Shift Analysis
    # 6. Shift Analysis
    if config['run_shift_analysis']:
        output_md = os.path.join('photometry_output', f'shift_analysis_{base_name}.md')
        print(f"Generating shift analysis report at {output_md}...")
        generate_shift_report(results, config['reference_catalog'], header, config['match_tolerance_arcsec'], output_md)

    print("Done!\n")

def main():
    print("Launching Configuration GUI...")
    cfg = run_config_gui()
    
    if not cfg:
        print("Configuration cancelled. Exiting.")
        return

    print("Configuration Received. Finding matching files...")
    input_pattern = cfg['input_pattern']

    if isinstance(input_pattern, list):
        files_to_process = input_pattern
    elif os.path.isfile(input_pattern):
        files_to_process = [input_pattern]
    else:
        files_to_process = glob.glob(input_pattern)

    if not files_to_process:
        print(f"No files found matching pattern: {input_pattern}")
    else:
        print(f"Found {len(files_to_process)} file(s) to process.")
        for f in files_to_process:
            process_file(f, cfg)

if __name__ == '__main__':
    main()
