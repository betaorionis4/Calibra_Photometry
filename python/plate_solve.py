import subprocess
import os
import glob
import shutil
import time
from astropy.io import fits
from astropy.wcs import WCS

def header_position(header):
    """
    v4.22: approximate pointing (ra_deg, dec_deg) from a FITS header, or None.
    Looks at RA/DEC (degrees, or sexagesimal text), OBJCTRA/OBJCTDEC (sexagesimal
    hours/degrees, as written by N.I.N.A., MaxIm, SGP ...) and CRVAL1/CRVAL2.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    def _num(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    for kra, kdec, unit_ra in (('RA', 'DEC', u.deg), ('OBJCTRA', 'OBJCTDEC', u.hourangle),
                               ('RA_OBJ', 'DEC_OBJ', u.deg), ('CRVAL1', 'CRVAL2', u.deg)):
        ra, dec = header.get(kra), header.get(kdec)
        if ra is None or dec is None or str(ra).strip() == '' or str(dec).strip() == '':
            continue
        fr, fd = _num(ra), _num(dec)
        try:
            if fr is not None and fd is not None:
                if kra == 'OBJCTRA':          # numeric OBJCTRA is in hours
                    fr *= 15.0
                if 0 <= fr < 360 and -90 <= fd <= 90:
                    return fr, fd
                continue
            c = SkyCoord(str(ra).strip(), str(dec).strip(),
                         unit=(u.hourangle if kra != 'RA_OBJ' else u.deg, u.deg))
            return float(c.ra.deg), float(c.dec.deg)
        except Exception:
            continue
    return None


def wcs_center(wcs, shape):
    """(ra_deg, dec_deg) of the image centre for a celestial WCS and shape (ny, nx)."""
    ny, nx = shape
    c = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    return float(c.ra.deg), float(c.dec.deg)


def solve_with_astap(fits_path, astap_exe="astap", search_radius=5.0, annotate=False,
                     ra_hint_deg=None, dec_hint_deg=None, fov=0.0, allow_blind=True):
    """
    Solves a FITS file using the ASTAP command-line interface.
    https://www.hnsky.org/astap.htm#astap_command_line

    v4.22: an explicit position guess (ra_hint_deg, dec_hint_deg in degrees - from
    an object name, a manual entry or the solution of the previous image) is given
    to ASTAP as -ra <hours> -spd <dec+90>; it overrides the header. Without a guess
    ASTAP uses the header position (RA/DEC/OBJCTRA/OBJCTDEC); if the header has
    none either, a blind solve (radius 180 deg) is made if allow_blind, otherwise
    the file is skipped. fov: image height in degrees (0 = automatic).
    Returns the WCS (celestial) or None.
    """
    
    # 1. Read existing header for RA/Dec hints
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            hdr_pos = header_position(header)
    except Exception as e:
        print(f"Error reading FITS header for {fits_path}: {e}")
        return None

    # 2. Build the command
    # ASTAP command line: -ra in HOURS, -spd = south pole distance (Dec + 90) in degrees,
    # -r search radius in degrees, -fov image height in degrees (0 = auto).
    
    # Normalize paths for Windows (converts / to \ and handles absolute paths)
    astap_exe = os.path.normpath(astap_exe)
    fits_path = os.path.normpath(fits_path)
    
    # We remove -wcs to avoid conflicts with our manual merge logic
    radius = float(search_radius) if search_radius else 5.0
    cmd = [astap_exe, "-f", fits_path]
    if annotate:
        cmd.append("-annotate")
    try:
        if fov and float(fov) > 0:
            cmd += ["-fov", f"{float(fov):.4f}"]
    except (TypeError, ValueError):
        pass

    if ra_hint_deg is not None and dec_hint_deg is not None:
        ra_h = (float(ra_hint_deg) % 360.0) / 15.0
        spd = float(dec_hint_deg) + 90.0
        cmd += ["-ra", f"{ra_h:.6f}", "-spd", f"{spd:.6f}"]
        print(f"Position guess RA={float(ra_hint_deg):.4f} deg, Dec={float(dec_hint_deg):+.4f} deg "
              f"(search radius {radius:g} deg).")
    elif hdr_pos is not None:
        print(f"RA/Dec found in header ({hdr_pos[0]:.4f}, {hdr_pos[1]:+.4f}). Letting ASTAP read them directly.")
    else:
        if not allow_blind:
            print("No position guess and no RA/Dec in the header - skipped (blind solving is switched off).")
            return None
        print("No RA/Dec hints found in header. Attempting blind solve (slow)...")
        radius = 180.0
    cmd += ["-r", f"{radius:g}"]

    # 3. Execute ASTAP
    # v4.15: the solution file is <name>.wcs for ANY extension (.fits/.fit/.FIT/.fts).
    # The old str.replace() gave the FITS file itself for e.g. '.FIT', so a failed
    # solve looked successful. A stale .wcs from an earlier run is removed first.
    wcs_file = os.path.splitext(fits_path)[0] + ".wcs"
    if os.path.exists(wcs_file):
        try:
            os.remove(wcs_file)
        except OSError as e:
            print(f"Warning: could not remove old solution file {wcs_file}: {e}")
    print(f"Running command: {' '.join(cmd)}")
    try:
        # We capture output to check for success/failure strings
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        
        # Give Windows a moment to release file locks
        time.sleep(1.0)
        
        if os.path.exists(wcs_file) or "Solved" in result.stdout:
            print(f"Successfully solved: {fits_path}")
            
            # Verify FITS file is readable BEFORE merging
            try:
                with fits.open(fits_path) as test_hdul:
                    pass
            except Exception as e:
                print(f"Warning: FITS file {fits_path} is unreadable BEFORE merging: {e}")

            # If the .wcs file exists, ensure it's merged into the FITS header manually
            if os.path.exists(wcs_file):
                print(f"Merging solution from {wcs_file} into {fits_path}...")
                try:
                    with open(wcs_file, 'r') as f:
                        wcs_lines = f.readlines()
                    
                    # Safely update the FITS header
                    # memmap=False: data must be fully in memory before the same
                    # file is overwritten below (memory-mapped data can fail on Windows)
                    with fits.open(fits_path, memmap=False) as hdul:
                        header = hdul[0].header.copy()
                        data = hdul[0].data.copy() if hdul[0].data is not None else None
                    
                    for line in wcs_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Handle COMMENT or HISTORY lines first (they might contain '=')
                        line_upper = line.upper()
                        if line_upper.startswith('COMMENT'):
                            header.add_comment(line[7:].strip())
                            continue
                        if line_upper.startswith('HISTORY'):
                            header.add_history(line[7:].strip())
                            continue
                            
                        if '=' not in line:
                            continue
                        
                        # Standard key = value
                        key, rest = line.split('=', 1)
                        key = key.strip().upper()
                        
                        # Skip structural keywords that should not be overridden
                        if key in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'END']:
                            continue
                            
                        if '/' in rest:
                            val, comment = rest.split('/', 1)
                            val = val.strip()
                            comment = comment.strip()
                        else:
                            val = rest.strip()
                            comment = ""
                        
                        # Type conversion for FITS values
                        if val.startswith("'") and val.endswith("'"):
                            val = val[1:-1].strip()
                        else:
                            try:
                                if '.' in val:
                                    val = float(val)
                                else:
                                    val = int(val)
                            except ValueError:
                                pass
                        
                        # Add or update the keyword (limit to 8 chars for standard FITS)
                        if len(key) <= 8:
                            header[key] = (val, comment)
                        else:
                            header[f'HIERARCH {key}'] = (val, comment)
                    
                    # Write the updated FITS file back
                    fits.writeto(fits_path, data, header, overwrite=True)
                    print(f"FITS header updated and saved successfully.")
                    
                    # Clean up the .wcs file after successful merge
                    os.remove(wcs_file)
                except Exception as e:
                    print(f"Manual header update failed: {e}")

            # 4. Load the new WCS to verify
            try:
                with fits.open(fits_path) as hdul:
                    new_wcs = WCS(hdul[0].header)
                # v4.15: a WCS object is always "truthy" - require a real celestial solution
                if not new_wcs.has_celestial:
                    print(f"No celestial WCS found in {fits_path} after solving - treating as NOT solved.")
                    return None
                return new_wcs
            except Exception as e:
                print(f"Verification failed: Could not read WCS from {fits_path}. Error: {e}")
                return None
        else:
            print(f"ASTAP finished but no solution was found for {fits_path}.")
            return None

    except subprocess.TimeoutExpired:
        print(f"ASTAP did not finish within 10 minutes for {fits_path} - aborted.")
        return None
    except FileNotFoundError:
        print(f"ASTAP executable not found: {astap_exe} (check the path in Settings).")
        return None
    except subprocess.CalledProcessError as e:
        print(f"ASTAP failed with error code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return None

def plate_solvem(input_pattern, suffix="wcs", astap_exe="astap", search_radius=5.0, annotate=False):
    """
    Solves multiple FITS files matching a pattern.
    Does not overwrite original files; creates a copy with the given suffix.
    """
    files = glob.glob(input_pattern)
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return []

    solved_files = []
    for f in files:
        if suffix in f:
            print(f"Skipping already solved file: {f}")
            continue
            
        base, ext = os.path.splitext(f)
        new_filename = f"{base}_{suffix}{ext}"
        
        print(f"Copying {f} to {new_filename}...")
        shutil.copy2(f, new_filename)
        
        # Solve the new file
        res = solve_with_astap(new_filename, astap_exe=astap_exe, search_radius=search_radius, annotate=annotate)
        if res:
            solved_files.append(new_filename)
        else:
            # If solve failed, maybe we should remove the copy?
            # Or keep it for inspection. Let's keep it for now.
            pass
            
    return solved_files

def plate_solve_files(files, suffix="wcs", astap_exe="astap", search_radius=5.0, annotate=False,
                      ra_hint_deg=None, dec_hint_deg=None, chain=True, fov=0.0, allow_blind=True):
    """
    Solves a provided list of FITS file paths.
    Does not overwrite original files; creates a copy with the given suffix.
    v4.22: optional position guess; with chain=True the centre of each solved image
    is the guess for the next one (see solve_series).
    """
    if not files:
        print("No files provided for plate solving.")
        return []
    todo = []
    for f in files:
        if suffix and f"_{suffix}" in f:
            print(f"Skipping potentially already solved file: {f}")
            continue
        base, ext = os.path.splitext(f)
        new_filename = f"{base}_{suffix}{ext}" if suffix else f
        if new_filename != f:
            print(f"Copying {f} to {new_filename}...")
            shutil.copy2(f, new_filename)
        todo.append(new_filename)
    results = solve_series(todo, astap_exe=astap_exe, search_radius=search_radius, annotate=annotate,
                           ra_hint_deg=ra_hint_deg, dec_hint_deg=dec_hint_deg, chain=chain, fov=fov,
                           allow_blind=allow_blind)
    return [f for f, w in zip(todo, results) if w is not None]


def solve_series(paths, astap_exe="astap", search_radius=5.0, annotate=False, ra_hint_deg=None,
                 dec_hint_deg=None, chain=True, fov=0.0, allow_blind=True, solver=None,
                 on_result=None, cancel_event=None):
    """
    v4.22: solve a series of images (in place).
    Guess for each image, in this order of attempts:
      1. the centre of the last successfully solved image (if chain),
      2. the user's guess (ra_hint_deg/dec_hint_deg: object name or manual entry),
      3. the FITS header position, or a blind solve if the header has none (if allow_blind).
    An attempt is only repeated with the next source if it differs from the one tried.
    solver: function with the signature of solve_with_astap (for tests).
    on_result(i, path, wcs_or_None, guess_label) is called after every image.
    Returns a list of WCS objects (or None) in the order of paths.
    """
    solver = solver or solve_with_astap
    last = None
    out = []
    for i, p in enumerate(paths):
        if cancel_event is not None and cancel_event.is_set():
            out.append(None)
            continue
        attempts = []
        if chain and last is not None:
            attempts.append(('previous image', last))
        if ra_hint_deg is not None and dec_hint_deg is not None:
            g = (float(ra_hint_deg), float(dec_hint_deg))
            if not attempts or _sep_deg(attempts[0][1], g) > 0.05:
                attempts.append(('your guess', g))
        attempts.append(('header/blind', None))
        res, used = None, None
        for label, g in attempts:
            print(f"[{i + 1}/{len(paths)}] {os.path.basename(p)}: guess = {label}"
                  + (f" ({g[0]:.4f}, {g[1]:+.4f})" if g else ""))
            if g is None:
                res = solver(p, astap_exe=astap_exe, search_radius=search_radius, annotate=annotate,
                             fov=fov, allow_blind=allow_blind)
            else:
                res = solver(p, astap_exe=astap_exe, search_radius=search_radius, annotate=annotate,
                             ra_hint_deg=g[0], dec_hint_deg=g[1], fov=fov, allow_blind=allow_blind)
            if res is not None:
                used = label
                break
        if res is not None:
            try:
                with fits.open(p) as h:
                    shape = (int(h[0].header['NAXIS2']), int(h[0].header['NAXIS1']))
                last = wcs_center(res.celestial if hasattr(res, 'celestial') else res, shape)
            except Exception as e:
                print(f"  (could not determine the image centre for the next guess: {e})")
        out.append(res)
        if on_result is not None:
            on_result(i, p, res, used)
    return out


def _sep_deg(a, b):
    import numpy as np
    ra1, d1, ra2, d2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(ra1 - ra2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# --- Example Usage ---
# Ensure 'astap' is in your system PATH, or provide full path to the .exe
# my_wcs = solve_with_astap("m42_test.fits", astap_exe="C:/Program Files/astap/astap.exe")