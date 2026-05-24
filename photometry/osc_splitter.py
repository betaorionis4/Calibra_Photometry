import os
import numpy as np
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning

def superpixel_debayer(bayer_img, pattern='RGGB'):
    """
    Performs superpixel debayering of a 2D monochrome Bayer image.
    Reduces resolution by 2x in both dimensions.
    Returns R, G, B arrays of shape (height//2, width//2).
    """
    height, width = bayer_img.shape
    h_half = height // 2
    w_half = width // 2
    
    # Crop to even dimensions if necessary
    img = bayer_img[:h_half*2, :w_half*2]
    
    # Normalize pattern to uppercase
    pattern = pattern.upper()
    
    if pattern == 'RGGB':
        # R  G1
        # G2 B
        r = img[0::2, 0::2]
        g1 = img[0::2, 1::2]
        g2 = img[1::2, 0::2]
        b = img[1::2, 1::2]
    elif pattern == 'BGGR':
        # B  G1
        # G2 R
        b = img[0::2, 0::2]
        g1 = img[0::2, 1::2]
        g2 = img[1::2, 0::2]
        r = img[1::2, 1::2]
    elif pattern == 'GRBG':
        # G1 R
        # B  G2
        g1 = img[0::2, 0::2]
        r = img[0::2, 1::2]
        b = img[1::2, 0::2]
        g2 = img[1::2, 1::2]
    elif pattern == 'GBRG':
        # G1 B
        # R  G2
        g1 = img[0::2, 0::2]
        b = img[0::2, 1::2]
        r = img[1::2, 0::2]
        g2 = img[1::2, 1::2]
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
        
    # Average the green channels
    g = (g1.astype(np.float32) + g2.astype(np.float32)) / 2.0
    
    return r, g, b

def simple_bilinear_debayer(bayer_img, pattern='RGGB'):
    """
    Performs a simple bilinear debayering of a 2D monochrome Bayer image.
    Returns R, G, B arrays of the same shape as bayer_img.
    """
    height, width = bayer_img.shape
    r = np.zeros_like(bayer_img, dtype=np.float32)
    g = np.zeros_like(bayer_img, dtype=np.float32)
    b = np.zeros_like(bayer_img, dtype=np.float32)
    
    # Coordinates grid
    y, x = np.indices((height, width))
    
    # Normalize pattern to uppercase
    pattern = pattern.upper()
    
    if pattern == 'RGGB':
        is_r_row = (y % 2 == 0)
        is_r_col = (x % 2 == 0)
        
        mask_r = is_r_row & is_r_col
        mask_g1 = is_r_row & ~is_r_col
        mask_g2 = ~is_r_row & is_r_col
        mask_b = ~is_r_row & ~is_col if 'is_col' in locals() else ~is_r_row & ~is_r_col
        mask_b = ~is_r_row & ~is_r_col
        
    elif pattern == 'BGGR':
        is_b_row = (y % 2 == 0)
        is_b_col = (x % 2 == 0)
        
        mask_b = is_b_row & is_b_col
        mask_g1 = is_b_row & ~is_b_col
        mask_g2 = ~is_b_row & is_b_col
        mask_r = ~is_b_row & ~is_b_col
        
    elif pattern == 'GRBG':
        is_g1_row = (y % 2 == 0)
        is_g1_col = (x % 2 == 0)
        
        mask_g1 = is_g1_row & is_g1_col
        mask_r = is_g1_row & ~is_g1_col
        mask_b = ~is_g1_row & is_g1_col
        mask_g2 = ~is_g1_row & ~is_g1_col
        
    elif pattern == 'GBRG':
        is_g1_row = (y % 2 == 0)
        is_g1_col = (x % 2 == 0)
        
        mask_g1 = is_g1_row & is_g1_col
        mask_b = is_g1_row & ~is_g1_col
        mask_r = ~is_g1_row & is_g1_col
        mask_g2 = ~is_g1_row & ~is_g1_col
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    # Assign raw pixels
    r[mask_r] = bayer_img[mask_r]
    b[mask_b] = bayer_img[mask_b]
    g[mask_g1 | mask_g2] = bayer_img[mask_g1 | mask_g2]
    
    from scipy.ndimage import convolve
    
    # Kernels
    k_g = np.array([[0, 1, 0],
                    [1, 4, 1],
                    [0, 1, 0]]) / 4.0
                    
    k_rb_cross = np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]]) / 4.0
                           
    k_rb_ortho = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]]) / 2.0

    # Interpolate G channel
    g_interp = convolve(g, k_g)
    g[~(mask_g1 | mask_g2)] = g_interp[~(mask_g1 | mask_g2)]
    
    # Interpolate R & B channel
    r_cross = convolve(r, k_rb_cross)
    r_ortho = convolve(r, k_rb_ortho)
    
    b_cross = convolve(b, k_rb_cross)
    b_ortho = convolve(b, k_rb_ortho)
    
    if pattern == 'RGGB':
        r[mask_b] = r_cross[mask_b]
        r[mask_g1 | mask_g2] = r_ortho[mask_g1 | mask_g2]
        
        b[mask_r] = b_cross[mask_r]
        b[mask_g1 | mask_g2] = b_ortho[mask_g1 | mask_g2]
    elif pattern == 'BGGR':
        b[mask_r] = b_cross[mask_r]
        b[mask_g1 | mask_g2] = b_ortho[mask_g1 | mask_g2]
        
        r[mask_b] = r_cross[mask_b]
        r[mask_g1 | mask_g2] = r_ortho[mask_g1 | mask_g2]
        
    return r, g, b

def split_osc_fits(fits_path, output_dir=None, pattern='RGGB', method='superpixel', channels=None):
    """
    Loads a FITS file, checks if it's 2D (Bayered), debayers it,
    and writes out split FITS files with WCS preserved.
    """
    if channels is None:
        channels = ['R', 'G', 'B']
        
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
        
    if output_dir is None:
        output_dir = os.path.dirname(fits_path)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        if data is None and len(hdul) > 1:
            data = hdul[1].data
            header = hdul[1].header
            
    if data is None:
        raise ValueError(f"FITS file contains no data: {fits_path}")
        
    # Handle color/monochrome
    if data.ndim == 3:
        if data.shape[0] == 3:
            r, g, b = data[0], data[1], data[2]
        elif data.shape[2] == 3:
            r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        else:
            raise ValueError(f"Unsupported 3D data shape: {data.shape}")
    elif data.ndim == 2:
        if method == 'superpixel':
            r, g, b = superpixel_debayer(data, pattern)
        else:
            r, g, b = simple_bilinear_debayer(data, pattern)
    else:
        raise ValueError(f"Unsupported dimensions: {data.ndim}")
        
    # Scale WCS if superpixel (halved resolution)
    wcs_header = header.copy()
    if data.ndim == 2 and method == 'superpixel':
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                wcs = WCS(header)
            if wcs.has_celestial:
                wcs.wcs.crpix = (wcs.wcs.crpix - 0.5) / 2.0 + 0.5
                if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
                    wcs.wcs.cd = wcs.wcs.cd * 2.0
                else:
                    wcs.wcs.cdelt = wcs.wcs.cdelt * 2.0
                
                # Write back scaled coordinates
                wcs_header.update(wcs.to_header(relax=True))
        except Exception as e:
            # Non-blocking warning
            print(f"Warning: WCS scaling failed: {e}")
            
    base_name = os.path.splitext(os.path.basename(fits_path))[0]
    suffix = "_SP" if method == 'superpixel' else ""
    
    extracted_paths = []
    
    channel_data = {'R': r, 'G': g, 'B': b}
    for chan in channels:
        if chan not in channel_data:
            continue
        img = channel_data[chan]
        out_name = f"{base_name}{suffix}_{chan}.fits"
        out_path = os.path.join(output_dir, out_name)
        
        # Cast to uint16 safely (keeping 16MB file size)
        img_uint16 = np.clip(img, 0, 65535).astype(np.uint16)
        
        new_hdr = wcs_header.copy()
        new_hdr['FILTER'] = chan
        new_hdr['COMMENT'] = f"Extracted {chan} channel from OSC via {method} ({pattern})"
        
        hdu = fits.PrimaryHDU(data=img_uint16, header=new_hdr)
        hdu.writeto(out_path, overwrite=True)
        extracted_paths.append(out_path)
        
    return extracted_paths
