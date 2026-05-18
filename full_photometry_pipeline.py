import numpy as np
import csv
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# =================================================================
# GLOBAL SETTINGS
# =================================================================
fits_filename = 'AE_UMa_Bmag_corr_00003.fits'
output_csv = 'targets_auto.csv'

detect_sigma = 5.0
box_size = 15
aperture_radius = 5.0
annulus_inner = 7.0
annulus_outer = 13.0
saturation_limit = 60000

max_plots_to_show = 3

def main():
    print("=================================================================")
    print("--- 1. Automated Star Retrieval (DAOStarFinder) ---")
    print("=================================================================\n")

    # Load Image Data
    try:
        with fits.open(fits_filename) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header
    except FileNotFoundError:
        print(f"Error: {fits_filename} not found.")
        return

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        wcs = WCS(header)
    has_wcs = wcs.has_celestial

    mean, median_bg, std_bg = sigma_clipped_stats(image_data, sigma=3.0, maxiters=5)
    print(f"Global Image Stats: Median Bkg = {median_bg:.1f} ADU | Noise Sigma = {std_bg:.1f} ADU")
    
    # Initialize DAOStarFinder
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyWarning, message=".*sharplo.*")
        warnings.filterwarnings("ignore", category=AstropyWarning, message=".*roundlo.*")
        try:
            daofind = DAOStarFinder(fwhm=3.5, threshold=detect_sigma * std_bg,
                                    sharpness_range=(0.2, 1.0), roundness_range=(-1.0, 1.0))
        except TypeError:
            daofind = DAOStarFinder(fwhm=3.5, threshold=detect_sigma * std_bg,
                                    sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0)
    
    sources = daofind(image_data - median_bg)

    if sources is None:
        print("No targets found above the threshold!")
        return

    sources.sort('flux')
    sources.reverse()
    print(f"Found {len(sources)} stars.\n")

    csv_data = []
    results = []

    # Detect column names dynamically to handle different photutils versions
    x_col = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
    y_col = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'

    for i, row in enumerate(sources):
        star_id = f"Auto_{i+1:03d}"
        fits_x = row[x_col] + 1.0
        fits_y = row[y_col] + 1.0
        peak_val = row['peak']
        flux_val = row['flux']

        ra_deg, dec_deg, ra_hms, dec_dms = "", "", "", ""
        if has_wcs:
            ra_conv, dec_conv = wcs.all_pix2world(fits_x, fits_y, 1)
            ra_deg, dec_deg = float(ra_conv), float(dec_conv)
            coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
            ra_hms = coord.ra.to_string(unit=u.hourangle, sep='hms', precision=2, pad=True)
            dec_dms = coord.dec.to_string(unit=u.degree, sep='dms', precision=1, pad=True, alwayssign=True)

        csv_data.append({
            'id': star_id, 'coord_system': 'XY', 'val1': f"{fits_x:.2f}", 'val2': f"{fits_y:.2f}",
            'peak_adu': f"{peak_val:.1f}", 'est_flux': f"{flux_val:.1f}",
            'ra_deg': f"{ra_deg:.5f}" if ra_deg else "", 'dec_deg': f"{dec_deg:.5f}" if dec_deg else "",
            'ra_hms': ra_hms, 'dec_dms': dec_dms
        })
        
        results.append({'id': star_id, 'x': fits_x, 'y': fits_y})

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'coord_system', 'val1', 'val2', 'peak_adu', 'est_flux', 'ra_deg', 'dec_deg', 'ra_hms', 'dec_dms'])
        writer.writeheader()
        writer.writerows(csv_data)

    print("=================================================================")
    print("--- 2. PSF Modeling & Coordinate Refinement ---")
    print("=================================================================\n")
    fitter = fitting.LevMarLSQFitter()
    plots_shown = 0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        for rs in results:
            star_id = rs['id']
            x = rs['x'] - 1 if rs['x'] is not None else None
            y = rs['y'] - 1 if rs['y'] is not None else None
            if x is None or y is None: continue

            try:
                cutout = Cutout2D(image_data, (x, y), box_size)
            except Exception:
                continue

            stamp = cutout.data
            sy, sx = stamp.shape
            if sy < 4 or sx < 4:
                continue

            yy, xx = np.mgrid[:sy, :sx]
            cx, cy = cutout.to_cutout_position((x, y))

            bg_guess = np.median(stamp)
            amp_guess = np.max(stamp) - bg_guess
            peak_adu = np.max(stamp)
            sat_status = "SATURATED!" if peak_adu > saturation_limit else "OK"

            g_init = models.Gaussian2D(amplitude=amp_guess, x_mean=cx, y_mean=cy, x_stddev=2.0, y_stddev=2.0)
            data_to_fit = stamp - bg_guess

            try:
                g_fit = fitter(g_init, xx, yy, data_to_fit)
                if g_fit.x_stddev.value > box_size or g_fit.amplitude.value <= 0:
                    raise ValueError("Calculus Diverged")
            except Exception:
                print(f"[{star_id}] FAILED PSF FIT (Too faint/noisy)")
                continue

            fwhm_x = g_fit.x_stddev.value * 2.355
            fwhm_y = g_fit.y_stddev.value * 2.355
            avg_fwhm = abs((fwhm_x + fwhm_y) / 2.0)

            psf_integral_flux = 2 * np.pi * g_fit.amplitude.value * g_fit.x_stddev.value * g_fit.y_stddev.value
            pixel_flux_sum = np.sum(data_to_fit)
            diff_percent = ((psf_integral_flux - pixel_flux_sum) / pixel_flux_sum * 100) if pixel_flux_sum != 0 else 0.0

            fit_x_cutout = g_fit.x_mean.value
            fit_y_cutout = g_fit.y_mean.value
            fit_x_orig, fit_y_orig = cutout.to_original_position((fit_x_cutout, fit_y_cutout))
            
            fits_x = fit_x_orig + 1.0
            fits_y = fit_y_orig + 1.0

            rs['refined_x'] = fits_x
            rs['refined_y'] = fits_y

            print(f"[{star_id}] Peak: {peak_adu:,.0f} ({sat_status}) | FWHM: {avg_fwhm:.2f}px | Integral Diff: {diff_percent:+.1f}%")
            
            if plots_shown < max_plots_to_show:
                model_image = g_fit(xx, yy)
                residual_image = data_to_fit - model_image
                
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))
                
                im1 = ax1.imshow(data_to_fit, origin='lower', cmap='viridis')
                ax1.set_title('Raw Data (bg sub)')
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                circ = patches.Circle((fit_x_cutout, fit_y_cutout), radius=aperture_radius, edgecolor='red', facecolor='none', linewidth=2)
                ax1.add_patch(circ)
                
                im2 = ax2.imshow(model_image, origin='lower', cmap='viridis')
                ax2.set_title('Gaussian Model')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                
                im3 = ax3.imshow(residual_image, origin='lower', cmap='seismic') 
                ax3.set_title('Residuals')
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                
                distances = np.sqrt((xx - fit_x_cutout)**2 + (yy - fit_y_cutout)**2)
                rad_limit = aperture_radius + 2.0
                rad_mask = distances <= rad_limit
                
                ax4.scatter(distances[rad_mask].flatten(), data_to_fit[rad_mask].flatten(), color='royalblue', alpha=0.7, s=25, label='Physical Pixels')
                curve_dist = np.linspace(0, rad_limit, 100)
                ax4.plot(curve_dist, g_fit(fit_x_cutout + curve_dist, fit_y_cutout), color='darkorange', linewidth=2.5, label='Gaussian Curve')
                ax4.axvline(x=aperture_radius, color='red', linestyle='--', linewidth=2, label='Aperture')
                ax4.set_title(f'Radial Profile (r < {rad_limit})')
                ax4.set_xlim(0, rad_limit)
                ax4.legend(loc='upper right', fontsize=9)
                
                plt.tight_layout()
                plt.show()
                plots_shown += 1

    print("\n=================================================================")
    print("--- 3. Aperture Photometry with Local Background Subtraction ---")
    print("=================================================================\n")
    positions = []
    valid_ids = []

    for rs in results:
        # Use refined coordinates if available to perfectly center the aperture, else fallback
        rx = rs.get('refined_x', rs['x'])
        ry = rs.get('refined_y', rs['y'])
        if rx is not None and ry is not None:
            positions.append((rx - 1, ry - 1))
            valid_ids.append(rs['id'])

    star_apertures = CircularAperture(positions, r=aperture_radius)
    bg_annuli = CircularAnnulus(positions, r_in=annulus_inner, r_out=annulus_outer)

    phot_table = aperture_photometry(image_data, star_apertures, method='exact')
    area_aperture = star_apertures.area

    print(f"{'Target ID':<10} | {'Raw ADU':<10} | {'Bkg/Pixel':<10} | {'Polluted':<10} | {'NET FLUX (ADU)':<15}")
    print("-" * 80)

    for i in range(len(phot_table)):
        target_id = valid_ids[i]
        raw_flux = phot_table['aperture_sum'][i] 
        if np.isnan(raw_flux): continue

        annulus_mask = bg_annuli[i].to_mask(method='center')
        bkg_pixels = annulus_mask.get_values(image_data)
        bkg_pixels = bkg_pixels[~np.isnan(bkg_pixels)]

        if len(bkg_pixels) == 0: 
            print(f"{target_id:<10} | OFF SENSOR EDGE")
            continue

        bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(bkg_pixels, sigma=3.0, maxiters=5)
        bkg_std = max(bkg_std, 1e-6) 
        
        rejected_count = np.sum(np.abs(bkg_pixels - bkg_median) > (3.0 * bkg_std))
        pollution_pct = (rejected_count / len(bkg_pixels)) * 100
        
        pollution_str = f"{pollution_pct:.1f}%"
        if pollution_pct > 10.0: pollution_str += " (HIGH)"
        
        total_bkg_in_aperture = bkg_mean * area_aperture
        net_flux = raw_flux - total_bkg_in_aperture
        
        flag = "  <-- WARN: Target darker than bg" if net_flux < 0 else ""
        print(f"{target_id:<10} | {raw_flux:<10,.1f} | {bkg_mean:<10,.2f} | {pollution_str:<10} | {net_flux:,.2f}{flag}")

if __name__ == '__main__':
    main()
