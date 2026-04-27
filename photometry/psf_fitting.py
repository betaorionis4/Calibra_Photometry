import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyWarning

def refine_coordinates_psf(image_data, results, box_size, aperture_radius, saturation_limit, max_plots_to_show, display_plots=False, plot_output_dir=None, base_filename="", print_psf_fitting=False):
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

            if print_psf_fitting:
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
                if plot_output_dir:
                    plot_path = os.path.join(plot_output_dir, f"{base_filename}_star_{star_id}.png")
                    plt.savefig(plot_path)
                if display_plots:
                    plt.show()
                plt.close(fig)
                plots_shown += 1
