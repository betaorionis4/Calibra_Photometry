"""
Fast photometry helpers (v4.17) used by Light Curves and Data Mining.

Why this module exists (measured on a 6248 x 4176 uint16 frame):
  * photutils.aperture_photometry() converts the WHOLE image to float on every
    call when it is given integer FITS data: ~116 ms per call. Called once per
    star (as before), this dominated Light Curves and Data Mining. Converting
    the frame once (as_float_image) brings a call down to < 1 ms, and measuring
    all stars in one vectorised call takes ~0.01 s for 2000 stars.
  * photutils.centroid_2dg() (a 2-D Gaussian fit) costs ~34 ms per star.
    windowed_centroid() gives the same accuracy (tested in tests/) in ~0.1 ms.
  * astropy sigma_clipped_stats() costs ~0.25 ms per annulus; fast_sky_stats()
    reproduces the same algorithm with plain numpy.
"""
import numpy as np


def as_float_image(data):
    """Native-endian, C-contiguous float64 copy of an image (made once per frame)."""
    return np.ascontiguousarray(data, dtype=np.float64)


def fast_sky_stats(pixels, sigma=3.0, maxiters=5):
    """
    Sigma-clipped (median, std, n_used) of sky pixels - same algorithm as
    astropy.stats.sigma_clip with cenfunc='median', stdfunc='std' (ddof=0):
    values with |x - median| > sigma*std are rejected, iterated until nothing
    changes or maxiters is reached.
    """
    x = np.asarray(pixels, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, 0
    for _ in range(maxiters):
        med = np.median(x)
        std = x.std()
        keep = np.abs(x - med) <= sigma * std
        n_keep = int(keep.sum())
        if n_keep == x.size or n_keep == 0:
            break
        x = x[keep]
    return float(np.median(x)), float(x.std()), int(x.size)


def windowed_centroid(img, x0, y0, sigma_w=2.0, half_box=7, max_iter=12, tol=0.005):
    """
    Iterative Gaussian-windowed first moment ("XWIN"-type centroid).

    img: float image (0-based numpy coordinates, x = column, y = row).
    The local background (median of the box border) is subtracted, the pixels
    are weighted with a Gaussian of width sigma_w centred on the current
    estimate, and the weighted centroid is iterated. Returns (x, y) or
    (nan, nan) if the window leaves the image or the signal is not positive.
    """
    ny, nx = img.shape
    xc, yc = float(x0), float(y0)
    for _ in range(max_iter):
        xi, yi = int(round(xc)), int(round(yc))
        x_lo, x_hi = xi - half_box, xi + half_box + 1
        y_lo, y_hi = yi - half_box, yi + half_box + 1
        if x_lo < 0 or y_lo < 0 or x_hi > nx or y_hi > ny:
            return np.nan, np.nan
        cut = img[y_lo:y_hi, x_lo:x_hi]
        border = np.concatenate([cut[0, :], cut[-1, :], cut[1:-1, 0], cut[1:-1, -1]])
        sub = cut - np.median(border)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        w = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2.0 * sigma_w ** 2))
        ws = w * sub
        tot = ws.sum()
        if not np.isfinite(tot) or tot <= 0:
            return np.nan, np.nan
        # factor 2: unbiased windowed-centroid update (SExtractor XWIN scheme)
        xn = xc + 2.0 * (ws * (xx - xc)).sum() / tot
        yn = yc + 2.0 * (ws * (yy - yc)).sum() / tot
        if not (np.isfinite(xn) and np.isfinite(yn)):
            return np.nan, np.nan
        if abs(xn - xc) > half_box or abs(yn - yc) > half_box:
            return np.nan, np.nan
        done = abs(xn - xc) < tol and abs(yn - yc) < tol
        xc, yc = xn, yn
        if done:
            break
    return xc, yc


def coarse_peak(img, x_pred, y_pred, radius):
    """Brightest (3x3-smoothed, background-subtracted) pixel within `radius` px of the
    prediction - used to find the star before the fine centroid when the
    prediction may be off by several pixels. Returns (x, y) or (nan, nan)."""
    ny, nx = img.shape
    r = int(np.ceil(radius)) + 1
    xi, yi = int(round(x_pred)), int(round(y_pred))
    x_lo, x_hi = max(xi - r, 0), min(xi + r + 1, nx)
    y_lo, y_hi = max(yi - r, 0), min(yi + r + 1, ny)
    if x_hi - x_lo < 3 or y_hi - y_lo < 3:
        return np.nan, np.nan
    cut = img[y_lo:y_hi, x_lo:x_hi]
    from scipy.ndimage import uniform_filter
    sm = uniform_filter(cut - np.median(cut), size=3, mode='nearest')
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    sm = np.where((xx - x_pred) ** 2 + (yy - y_pred) ** 2 <= radius ** 2, sm, -np.inf)
    j = int(np.argmax(sm))
    if not np.isfinite(sm.flat[j]) or sm.flat[j] <= 0:
        return np.nan, np.nan
    return float(xx.flat[j]), float(yy.flat[j])


def track_centroid(img, x_pred, y_pred, max_shift, sigma_w=2.0, half_box=7):
    """
    Centroid near a predicted position: coarse peak search within max_shift,
    then the windowed centroid. If it fails or ends further than max_shift from
    the prediction (protection against snapping to a neighbour), the predicted
    position is returned. Returns (x, y, ok).
    """
    x0, y0 = x_pred, y_pred
    if max_shift > 1.0:
        xp, yp = coarse_peak(img, x_pred, y_pred, max_shift)
        if np.isfinite(xp):
            x0, y0 = xp, yp
    xc, yc = windowed_centroid(img, x0, y0, sigma_w=sigma_w, half_box=half_box)
    if not (np.isfinite(xc) and np.isfinite(yc)):
        return float(x_pred), float(y_pred), False
    if np.hypot(xc - x_pred, yc - y_pred) > max_shift:
        return float(x_pred), float(y_pred), False
    return float(xc), float(yc), True
