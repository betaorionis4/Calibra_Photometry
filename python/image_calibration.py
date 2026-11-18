import os
import numpy as np
from astropy.io import fits


def calibrate_image(data, header, bias_path, flat_path, out_dir="fitsfiles/calibrated", verbose=True,
                    input_path=None, flat_is_bias_subtracted=True, return_path=False):
    """
    Performs basic Bias subtraction and Flat-fielding.

        Data_cal = (Raw - Bias) / normalised(Flat - Bias)

    v4.15 changes:
      * The output file is named after the INPUT file (input_path) instead of the
        FILENAME header keyword (stacking software copies a sub-frame's FILENAME
        into the stack, so the calibrated file was not found and the GUI silently
        kept using the uncalibrated one; two stacks could overwrite each other).
      * The result is saved as 32-bit float: negative values (noise around a low
        sky) are no longer clipped to 0 and values are no longer truncated.
      * Optionally (flat_is_bias_subtracted=False) the bias is subtracted from the
        master flat before normalisation. The default (True) keeps the previous
        assumption that master flats are already bias-corrected.
      * A size mismatch between light, bias and flat is an error (frame skipped)
        instead of a warning followed by a crash.
      * Calibration files are closed after reading.

    Returns (data_cal, header) - or (data_cal, header, out_path) if return_path=True.
    On failure the input data/header are returned (and out_path is None).
    """
    def _fail():
        return (data, header, None) if return_path else (data, header)

    if verbose:
        print(f"--- FITS Calibration ---")
        print(f"Master Bias: {os.path.basename(str(bias_path))}")
        print(f"Master Flat: {os.path.basename(str(flat_path))}")

    # 1. Load Calibration Frames
    try:
        with fits.open(bias_path, memmap=False) as bias_hdu:
            bias_data = bias_hdu[0].data.astype(np.float64)
            bias_header = bias_hdu[0].header.copy()
        with fits.open(flat_path, memmap=False) as flat_hdu:
            flat_data = flat_hdu[0].data.astype(np.float64)
            flat_header = flat_hdu[0].header.copy()
    except Exception as e:
        print(f"Error loading calibration frames: {e}")
        return _fail()

    light = np.asarray(data, dtype=np.float64)
    light_is_adu = np.nanmax(light) > 1.1

    # Calibration frames stored normalised to 0..1 (e.g. Siril/PixInsight float
    # masters) are rescaled to ADU when the light frame is in ADU.
    if light_is_adu and np.nanmax(bias_data) <= 1.1:
        if verbose: print("  Note: Scaling Master Bias from [0,1] to 16-bit range.")
        bias_data *= 65535.0
    flat_normalised = np.nanmedian(flat_data) <= 1.1

    if verbose:
        print(f"  Bias Stats -> Mean: {np.nanmean(bias_data):.1f} | Median: {np.nanmedian(bias_data):.1f} | Zeros: {np.sum(bias_data <= 0)}")
        print(f"  Flat Stats -> Mean: {np.nanmean(flat_data):.4g} | Median: {np.nanmedian(flat_data):.4g} | Zeros: {np.sum(flat_data <= 0)}")

    # 2. Verify Metadata
    if light.shape != bias_data.shape or light.shape != flat_data.shape:
        print(f"ERROR: Dimension mismatch - frame NOT calibrated!")
        print(f"  Data: {light.shape} | Bias: {bias_data.shape} | Flat: {flat_data.shape}")
        print(f"  (Check binning / sub-frame / orientation of the master frames.)")
        return _fail()

    for key in ['XBINNING', 'YBINNING', 'GAIN']:
        val_data = header.get(key)
        val_bias = bias_header.get(key)
        val_flat = flat_header.get(key)
        if val_data is not None and val_bias is not None and val_data != val_bias:
            print(f"WARNING: {key} mismatch between Data ({val_data}) and Bias ({val_bias})")
        if val_data is not None and val_flat is not None and val_data != val_flat:
            print(f"WARNING: {key} mismatch between Data ({val_data}) and Flat ({val_flat})")

    # 3. Apply Calibration
    data_sub = light - bias_data

    if flat_normalised or flat_is_bias_subtracted:
        flat_corr = flat_data
        if verbose:
            why = "already normalised" if flat_normalised else "marked as bias-subtracted"
            print(f"  Flat is {why}: bias NOT subtracted from the flat.")
    else:
        flat_corr = flat_data - bias_data
        if verbose:
            print("  Bias subtracted from the master flat before normalisation.")

    flat_median = np.nanmedian(flat_corr)
    if not np.isfinite(flat_median) or flat_median <= 0:
        print("ERROR: Flat field median is <= 0 - frame NOT calibrated.")
        return _fail()

    norm_flat = flat_corr / flat_median
    bad = ~np.isfinite(norm_flat) | (norm_flat <= 0.05)
    if np.any(bad):
        if verbose: print(f"  {np.sum(bad)} flat pixels <= 5% of the median set to 1 (not corrected).")
        norm_flat[bad] = 1.0

    data_cal = (data_sub / norm_flat).astype(np.float32)

    # 4. Save Calibrated FITS (32-bit float)
    os.makedirs(out_dir, exist_ok=True)
    src_name = input_path if input_path else header.get('FILENAME', 'unnamed.fits')
    base_name = os.path.basename(str(src_name))
    out_path = os.path.join(out_dir, f"cal_{base_name}")

    new_header = header.copy()
    for k in ('BZERO', 'BSCALE', 'BLANK'):  # integer scaling keywords do not apply to float data
        if k in new_header:
            del new_header[k]
    new_header['HISTORY'] = "Calibrated with Calibra (bias + flat, 32-bit float)"
    new_header['BIASFILE'] = os.path.basename(str(bias_path))
    new_header['FLATFILE'] = os.path.basename(str(flat_path))
    new_header['CALSRC'] = (base_name[:68], 'Uncalibrated source file')

    try:
        hdu = fits.PrimaryHDU(data=data_cal, header=new_header)
        hdu.writeto(out_path, overwrite=True)
        if verbose:
            print(f"Saved calibrated file to: {out_path} (32-bit float)")
    except Exception as e:
        print(f"Error saving calibrated FITS: {e}")
        return (data_cal, new_header, None) if return_path else (data_cal, new_header)

    return (data_cal, new_header, out_path) if return_path else (data_cal, new_header)
