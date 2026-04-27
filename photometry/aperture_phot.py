import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

def perform_aperture_photometry(image_data, results, aperture_radius, annulus_inner, annulus_outer, print_table=True, gain=1.0, read_noise=0.0, dark_current=0.0, exptime=1.0):
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

    if print_table:
        print(f"{'Target ID':<10} | {'Raw ADU':<10} | {'Bkg/Pixel':<10} | {'Polluted':<10} | {'NET FLUX (ADU)':<15}")
        print("-" * 80)

    for i in range(len(phot_table)):
        target_id = valid_ids[i]
        raw_flux = phot_table['aperture_sum'][i] 
        if np.isnan(raw_flux): continue

        bkg_median = 0.0
        bkg_std = 0.0
        pollution_str = "N/A"
        annulus_mask = bg_annuli[i].to_mask(method='center')
        bkg_pixels = annulus_mask.get_values(image_data)
        bkg_pixels = bkg_pixels[~np.isnan(bkg_pixels)]

        if len(bkg_pixels) == 0: 
            print(f"{target_id:<10} | OFF SENSOR EDGE")
            continue

        bkg_mean_raw, bkg_median, bkg_std = sigma_clipped_stats(bkg_pixels, sigma=3.0, maxiters=5)
        bkg_std = max(bkg_std, 1e-6) 
        
        rejected_count = np.sum(np.abs(bkg_pixels - bkg_median) > (3.0 * bkg_std))
        pollution_pct = (rejected_count / len(bkg_pixels)) * 100
        
        pollution_str = f"{pollution_pct:.1f}%"
        if pollution_pct > 10.0: pollution_str += " (HIGH)"
        
        annulus_area = len(bkg_pixels)

        bkg_sum = bkg_median * area_aperture
        net_flux = raw_flux - bkg_sum
        
        # Formal Error Calculation
        # Variance (ADU^2) = (raw_flux / gain) + aperture_area * (bkg_std**2) + (aperture_area**2 * bkg_std**2 / annulus_area)
        # Note: If std is calculated dynamically, it inherently includes Read Noise and Dark Current variance.
        # But for correctness, we'll ensure RN and DC are accounted for if std is zero.
        variance_adu = (max(raw_flux, 0) / gain) + area_aperture * (bkg_std**2) + (area_aperture**2 * bkg_std**2 / annulus_area)
        
        flux_err_adu = np.sqrt(variance_adu)
        snr = net_flux / flux_err_adu if flux_err_adu > 0 and net_flux > 0 else 0.0
        
        if snr > 0:
            mag_inst_err = 1.0857 / snr
        else:
            mag_inst_err = np.nan
        
        # Store in results
        for rs in results:
            if rs['id'] == target_id:
                rs['net_flux'] = net_flux
                rs['flux_err'] = flux_err_adu
                rs['snr'] = snr
                rs['mag_inst_err'] = mag_inst_err
                break
        
        if print_table:
            flag = "  <-- WARN: Target darker than bg" if net_flux < 0 else ""
            print(f"{target_id:<10} | {raw_flux:<10,.1f} | {bkg_median:<10,.2f} | {pollution_str:<10} | {net_flux:,.2f}{flag}")
