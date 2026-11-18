"""
Shared CCD/CMOS noise model for Calibra (v4.16).

Before v4.16 the photometric error was computed as

    var = raw_flux/gain + A*s^2 + A^2*s^2/n_ann

where raw_flux is the aperture SUM including sky and pedestal (so the sky was
counted twice: once as Poisson noise of raw_flux and once through the measured
sky scatter s), and 'gain' was the single-frame gain even for averaged stacks.
On the AE UMa V stack (49-frame average) this overestimated faint-star errors
by about a factor 3.

The model used now (all quantities in ADU):

    var = F_net / g_eff  +  k^2 * A * s^2 * (1 + A / n_sky)

    F_net  net (background-subtracted) flux in the aperture
    g_eff  effective gain [e-/ADU]: gain * N for an AVERAGE of N frames
           (N from STACKCNT / NCOMBINE / NSTACK), gain for a SUM or single frame
    A      aperture area [px]
    s      sigma-clipped standard deviation of the sky annulus [ADU/px]
           (already contains sky Poisson noise, read noise and dark noise)
    n_sky  number of annulus pixels actually used (after sigma clipping)
    k      'sky noise factor' (>= 1) for pixel-to-pixel correlated noise in
           registered/interpolated stacks. Default 1. It can be measured with
           blank_sky_noise_factor() below (the pipeline prints it).
"""
import numpy as np

STACK_KEYWORDS = ('STACKCNT', 'NCOMBINE', 'NSTACK', 'NSUBEXP', 'SNAPSHOT')


def stack_count(header):
    """Number of combined frames from the FITS header (1 if unknown)."""
    if header is None:
        return 1
    for key in STACK_KEYWORDS:
        try:
            n = int(header.get(key, 0) or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 1:
            return n
    return 1


def effective_gain(gain, header=None, stack_mode='average'):
    """
    Effective gain [e-/ADU] of an image.

    stack_mode: 'average' (pixel values are the mean of N frames: noise in ADU
    is reduced, so g_eff = gain * N), 'sum' (g_eff = gain) or 'none'
    (ignore stack keywords).
    Returns (g_eff, N).
    """
    try:
        g = float(gain)
    except (TypeError, ValueError):
        g = 1.0
    if not np.isfinite(g) or g <= 0:
        g = 1.0
    n = stack_count(header)
    if str(stack_mode).lower() == 'average' and n > 1:
        return g * n, n
    return g, n


def flux_variance(net_flux, area, sky_std, n_sky, g_eff, sky_factor=1.0):
    """Variance [ADU^2] of a background-subtracted aperture flux (see module doc)."""
    n_sky = max(int(n_sky), 1)
    k2 = float(sky_factor) ** 2 if sky_factor and sky_factor > 0 else 1.0
    return max(float(net_flux), 0.0) / g_eff + k2 * area * sky_std ** 2 * (1.0 + area / n_sky)


def sky_stats(pixels, sigma=3.0, maxiters=5):
    """Sigma-clipped (median, std, n_used) of sky pixels (same algorithm as
    astropy sigma_clip with median/std; v4.17: plain-numpy implementation,
    see python/fastphot.fast_sky_stats)."""
    from python.fastphot import fast_sky_stats
    return fast_sky_stats(pixels, sigma=sigma, maxiters=maxiters)


def blank_sky_noise_factor(image, star_xy, aperture_radius, annulus_inner, annulus_outer,
                           n_samples=1500, avoid_radius=None, rng_seed=1):
    """
    Empirical check of the noise model on empty sky.

    Places apertures at random positions at least `avoid_radius` px from every
    star in star_xy (0-based pixel coordinates), measures the background-
    subtracted aperture sums and compares their scatter with the prediction
    sqrt(A*s^2*(1 + A/n_sky)). Returns (factor, measured_rms, predicted_rms, n)
    or None. factor ~ 1 means the sky term is right; > 1 indicates correlated
    noise (typical for registered stacks).
    """
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from astropy.stats import mad_std
    img = np.asarray(image, dtype=float)
    ny, nx = img.shape
    if avoid_radius is None:
        avoid_radius = annulus_outer + 3 * aperture_radius
    margin = int(np.ceil(annulus_outer)) + 2
    rng = np.random.default_rng(rng_seed)
    cand = np.column_stack([rng.uniform(margin, nx - margin, n_samples * 4),
                            rng.uniform(margin, ny - margin, n_samples * 4)])
    if star_xy is not None and len(star_xy):
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(star_xy, dtype=float))
        d, _ = tree.query(cand)
        cand = cand[d > avoid_radius]
    cand = cand[:n_samples]
    if len(cand) < 50:
        return None
    ap = CircularAperture(cand, r=aperture_radius)
    an = CircularAnnulus(cand, r_in=annulus_inner, r_out=annulus_outer)
    sums = aperture_photometry(img, ap, method='exact')['aperture_sum']
    nets, preds = [], []
    for i, m in enumerate(an.to_mask(method='center')):
        med, std, n = sky_stats(m.get_values(img))
        if n < 10 or not np.isfinite(med):
            continue
        nets.append(float(sums[i]) - med * ap.area)
        preds.append(np.sqrt(ap.area * std ** 2 * (1 + ap.area / n)))
    if len(nets) < 50:
        return None
    measured = float(mad_std(nets))
    predicted = float(np.median(preds))
    if predicted <= 0:
        return None
    return measured / predicted, measured, predicted, len(nets)
