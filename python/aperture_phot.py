import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from python.noise import effective_gain, flux_variance, sky_stats


def perform_aperture_photometry(image_data, results, aperture_radius, annulus_inner, annulus_outer, print_table=True,
                                gain=1.0, read_noise=0.0, dark_current=0.0, exptime=1.0,
                                header=None, stack_mode='average', sky_noise_factor=1.0):
    """
    Aperture photometry with local (annulus) background for all stars in `results`.

    v4.16:
      * Error model (see python/noise.py): Poisson term from the NET flux with
        the effective gain (gain x STACKCNT for averaged stacks), sky term from
        the clipped annulus scatter and the number of sky pixels actually used,
        optional correlated-noise factor. Previously the raw aperture sum (incl.
        sky) and the single-frame gain were used -> errors ~3x too large for
        faint stars in stacks.
      * Stars whose aperture is not completely on the image are flagged 'edge'
        and get no flux (previously only the on-chip part was summed while the
        full-area sky was subtracted -> strongly negative fluxes).
      * Each star gets 'peak_raw' (highest pixel in the aperture), 'blended'
        (another detection within the outer annulus radius) and 'n_sky'.
    """
    print("\n=================================================================")
    print("--- 3. Aperture Photometry with Local Background Subtraction ---")
    print("=================================================================\n")

    g_eff, n_stack = effective_gain(gain, header, stack_mode)
    if n_stack > 1:
        mode_txt = f"average of {n_stack} frames -> effective gain {g_eff:.2f} e-/ADU" if str(stack_mode).lower() == 'average' \
            else f"{n_stack} frames, stack mode '{stack_mode}' -> gain {g_eff:.2f} e-/ADU"
        print(f"Noise model: {mode_txt}; sky noise factor {sky_noise_factor:.2f}")
    else:
        print(f"Noise model: gain {g_eff:.2f} e-/ADU; sky noise factor {sky_noise_factor:.2f}")

    ny, nx = image_data.shape
    positions = []
    valid = []
    for rs in results:
        rx = rs.get('refined_x', rs['x'])
        ry = rs.get('refined_y', rs['y'])
        if rx is not None and ry is not None:
            positions.append((rx - 1, ry - 1))   # FITS 1-based -> numpy 0-based
            valid.append(rs)
    if not positions:
        return

    pos = np.array(positions, dtype=float)

    # Neighbour (blend) flag: any other detection inside the outer annulus radius
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=annulus_outer)
        blended = np.zeros(len(pos), dtype=bool)
        for a, b in pairs:
            blended[a] = blended[b] = True
    except Exception:
        blended = np.zeros(len(pos), dtype=bool)

    star_apertures = CircularAperture(pos, r=aperture_radius)
    bg_annuli = CircularAnnulus(pos, r_in=annulus_inner, r_out=annulus_outer)
    phot_table = aperture_photometry(image_data, star_apertures, method='exact')
    area_aperture = star_apertures.area
    ann_masks = bg_annuli.to_mask(method='center')
    ap_masks = star_apertures.to_mask(method='center')

    if print_table:
        print(f"{'Target ID':<10} | {'Raw ADU':<10} | {'Bkg/Pixel':<10} | {'Polluted':<10} | {'NET FLUX (ADU)':<15}")
        print("-" * 80)

    n_edge = 0
    for i, rs in enumerate(valid):
        target_id = rs['id']
        x0, y0 = pos[i]
        rs['blended'] = bool(blended[i])

        # Aperture must lie completely on the image
        if (x0 - aperture_radius < -0.5 or y0 - aperture_radius < -0.5 or
                x0 + aperture_radius > nx - 0.5 or y0 + aperture_radius > ny - 0.5):
            rs['edge'] = True
            rs['net_flux'] = np.nan
            rs['flux_err'] = np.nan
            rs['snr'] = 0.0
            rs['mag_inst_err'] = np.nan
            n_edge += 1
            if print_table:
                print(f"{target_id:<10} | APERTURE OFF IMAGE EDGE")
            continue
        rs['edge'] = False

        raw_flux = phot_table['aperture_sum'][i]
        if np.isnan(raw_flux):
            continue

        bkg_pixels = ann_masks[i].get_values(image_data)
        bkg_pixels = bkg_pixels[np.isfinite(bkg_pixels)]
        if len(bkg_pixels) == 0:
            if print_table:
                print(f"{target_id:<10} | OFF SENSOR EDGE")
            continue

        bkg_median, bkg_std, n_sky = sky_stats(bkg_pixels, sigma=3.0, maxiters=5)
        bkg_std = max(bkg_std, 1e-6)
        rejected_count = len(bkg_pixels) - n_sky
        pollution_pct = (rejected_count / len(bkg_pixels)) * 100
        pollution_str = f"{pollution_pct:.1f}%"
        if pollution_pct > 10.0:
            pollution_str += " (HIGH)"

        net_flux = raw_flux - bkg_median * area_aperture
        variance_adu = flux_variance(net_flux, area_aperture, bkg_std, n_sky, g_eff, sky_noise_factor)
        flux_err_adu = np.sqrt(variance_adu)
        snr = net_flux / flux_err_adu if flux_err_adu > 0 and net_flux > 0 else 0.0
        mag_inst_err = 1.0857 / snr if snr > 0 else np.nan

        ap_vals = ap_masks[i].get_values(image_data)
        rs['peak_raw'] = float(np.nanmax(ap_vals)) if ap_vals.size else np.nan
        rs['net_flux'] = net_flux
        rs['flux_err'] = flux_err_adu
        rs['snr'] = snr
        rs['mag_inst_err'] = mag_inst_err
        rs['n_sky'] = n_sky
        rs['bkg_median'] = bkg_median
        rs['bkg_std'] = bkg_std

        if print_table:
            flag = "  <-- WARN: Target darker than bg" if net_flux < 0 else ""
            print(f"{target_id:<10} | {raw_flux:<10,.1f} | {bkg_median:<10,.2f} | {pollution_str:<10} | {net_flux:,.2f}{flag}")

    if n_edge:
        print(f"{n_edge} star(s) skipped: aperture not completely on the image (flag 'edge').")
    n_bl = int(np.sum(blended))
    if n_bl:
        print(f"{n_bl} star(s) have another detection within {annulus_outer:.1f} px (flag 'blended'; excluded from calibration).")
