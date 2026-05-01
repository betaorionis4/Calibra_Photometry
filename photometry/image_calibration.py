import os
import numpy as np
from astropy.io import fits
import astropy.units as u

def calibrate_image(data, header, bias_path, flat_path, out_dir="fitsfiles/calibrated", verbose=True):
    """
    Performs basic Bias subtraction and Flat-fielding.
    Data_cal = (Raw - Bias) / (Flat / median(Flat))
    """
    if verbose:
        print(f"--- FITS Calibration ---")
        print(f"Master Bias: {os.path.basename(bias_path)}")
        print(f"Master Flat: {os.path.basename(flat_path)}")

    # 1. Load Calibration Frames
    try:
        bias_hdu = fits.open(bias_path)
        bias_data = bias_hdu[0].data.astype(float)
        bias_header = bias_hdu[0].header
        
        flat_hdu = fits.open(flat_path)
        flat_data = flat_hdu[0].data.astype(float)
        flat_header = flat_hdu[0].header
        
        # Detect if calibration frames are normalized (0-1 range) and scale to 16-bit if so
        if np.max(bias_data) <= 1.1: # Allow a small margin above 1.0 for noise
            if verbose: print("  Note: Scaling Master Bias from [0,1] to 16-bit range.")
            bias_data *= 65535.0
            
        if np.max(flat_data) <= 1.1:
            if verbose: print("  Note: Scaling Master Flat from [0,1] to 16-bit range.")
            flat_data *= 65535.0

        if verbose:
            # Bias stats
            b_mean = np.mean(bias_data)
            b_med = np.median(bias_data)
            b_zeros = np.sum(bias_data <= 0)
            print(f"  Bias Stats -> Mean: {int(b_mean)} | Median: {int(b_med)} | Zeros: {b_zeros}")
            
            # Flat stats
            f_mean = np.mean(flat_data)
            f_med = np.median(flat_data)
            f_zeros = np.sum(flat_data <= 0)
            print(f"  Flat Stats -> Mean: {int(f_mean)} | Median: {int(f_med)} | Zeros: {f_zeros}")

    except Exception as e:
        print(f"Error loading calibration frames: {e}")
        return data, header

    # 2. Verify Metadata
    # Check dimensions
    if data.shape != bias_data.shape or data.shape != flat_data.shape:
        print(f"WARNING: Dimension mismatch!")
        print(f"  Data: {data.shape} | Bias: {bias_data.shape} | Flat: {flat_data.shape}")
        # We continue but warn
    
    # Check Binning/Gain if available in headers
    for key in ['XBINNING', 'YBINNING', 'GAIN']:
        val_data = header.get(key)
        val_bias = bias_header.get(key)
        val_flat = flat_header.get(key)
        if val_data is not None and val_bias is not None and val_data != val_bias:
            print(f"WARNING: {key} mismatch between Data ({val_data}) and Bias ({val_bias})")
        if val_data is not None and val_flat is not None and val_data != val_flat:
            print(f"WARNING: {key} mismatch between Data ({val_data}) and Flat ({val_flat})")

    # 3. Apply Calibration
    # Subtraction
    data_sub = data - bias_data
    
    # Flat Field normalization
    flat_median = np.median(flat_data)
    if flat_median == 0:
        print("Error: Flat field median is zero. Skipping flat correction.")
        return data_sub, header
        
    norm_flat = flat_data / flat_median
    
    # Avoid division by zero in the flat
    norm_flat[norm_flat <= 0] = 1e-6 
    
    data_cal = data_sub / norm_flat
    
    # --- Convert to 16-bit unsigned integer ---
    # We clip to [0, 65535] to avoid overflow/underflow
    data_cal = np.clip(data_cal, 0, 65535).astype(np.uint16)
    
    # 4. Save Calibrated FITS
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    orig_name = header.get('FILENAME', 'unnamed.fits')
    base_name = os.path.basename(orig_name)
    out_path = os.path.join(out_dir, f"cal_{base_name}")
    
    # Update header to reflect calibration
    new_header = header.copy()
    new_header['HISTORY'] = f"Calibrated with Calibra"
    new_header['BIASFILE'] = os.path.basename(bias_path)
    new_header['FLATFILE'] = os.path.basename(flat_path)
    # Ensure BZERO/BSCALE are set for uint16 if needed, though PrimaryHDU handles uint16 well
    
    try:
        hdu = fits.PrimaryHDU(data=data_cal, header=new_header)
        hdu.writeto(out_path, overwrite=True)
        if verbose:
            print(f"Saved calibrated file to: {out_path} (16-bit UINT)")
    except Exception as e:
        print(f"Error saving calibrated FITS: {e}")

    return data_cal, new_header
