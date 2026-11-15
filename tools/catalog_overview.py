#!/usr/bin/env python3
"""
Overview of local catalogue copies (ATLAS-RefCat2, Gaia, APASS, ...) - which files
can be deleted (or reduced) without losing anything Calibra needs.

    python tools/catalog_overview.py /path/to/catalogs [--dec-min -30] [--mag-max 18]
                                     [--reduce /path/to/reduced] [--report overview.csv]

READ-ONLY: nothing is deleted or changed. The script
  * lists every data file with size, number of stars, Dec range and magnitude range,
  * marks files that are entirely outside the limits ("DELETE: all Dec < -30",
    "DELETE: all fainter than 18") and files that are only partly useful,
  * for ATLAS-RefCat2 tiles (RRR+DD files in magnitude folders 00_m_16, 16_m_17 ...)
    decides from the folder and file names alone - no file needs to be read,
  * prints a summary (how much disk space deleting would free) and writes the full
    list to a CSV file,
  * with --reduce writes REDUCED COPIES of the table files (only rows inside the limits,
    same columns and header) into another folder, with the same subfolders - useful for
    large Gaia/APASS files that contain all magnitudes and therefore cannot simply be
    deleted. (ATLAS tiles are not copied: for them deleting tiles/folders is enough.)

Limits: --dec-min (default -30): keep stars with Dec >= this value.
        --mag-max (default 18):  keep stars brighter than this (Gaia G, APASS V,
        RefCat2 magnitude folders; a small margin of +0.5 mag is kept for tables).
"""
import argparse
import csv
import gzip
import io
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry import local_catalogs as lc          # noqa: E402

_emit = print
MAG_COLS = ('phot_g_mean_mag', 'gmag', 'johnson_v', 'vmag', 'v', 'v_mag', 'g', 'r', 'rmag')


def human(n):
    for u in ('B', 'kB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB':
            return f"{n:.1f} {u}" if u != 'B' else f"{n} B"
        n /= 1024.0


def tile_verdict(path, root, dec_min, mag_max):
    """ATLAS-RefCat2 tile: verdict from names only."""
    m = lc._TILE_RE.match(os.path.basename(path))
    dec0 = int(m.group(2))                       # tile covers dec0 .. dec0+1
    md = lc._MAGDIR_RE.search(os.path.relpath(path, root))
    reasons = []
    if dec0 + 1 <= dec_min:
        reasons.append(f"all Dec < {dec_min:g}")
    if md and int(md.group(1)) >= mag_max:
        reasons.append(f"all fainter than {mag_max:g} (folder {md.group(0)})")
    rng = f"{dec0:+d}..{dec0 + 1:+d}"
    mags = f"{md.group(1)}..{md.group(2)}" if md else ""
    return rng, mags, reasons


def scan_table(path, dec_min, mag_max, reduce_to=None):
    """Read a table file once: n, Dec range, mag range, n inside limits; optionally write a
    reduced copy. Returns dict."""
    layout, header, delim = lc._file_layout(path)
    out = {'n': 0, 'dec0': None, 'dec1': None, 'm0': None, 'm1': None, 'n_keep': 0, 'magcol': '', 'note': ''}
    cols = header or (lc.REFCAT2_COLS if layout == 'tiles' else None)
    if not cols:
        out['note'] = 'no header - not understood'
        return out
    i_ra, i_dec = lc._find(cols, lc._RA), lc._find(cols, lc._DEC)
    i_mag = lc._find(cols, MAG_COLS)
    if i_dec is None:
        out['note'] = 'no Dec column found'
        return out
    out['magcol'] = cols[i_mag] if i_mag is not None else ''
    w = fout = None
    if reduce_to:
        os.makedirs(os.path.dirname(reduce_to), exist_ok=True)
        fout = gzip.open(reduce_to, 'wt', encoding='utf-8') if reduce_to.endswith('.gz') else open(reduce_to, 'w', encoding='utf-8')
    scale = None
    first_data = header is not None
    with lc._open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith('#'):
                if fout:
                    fout.write(line)
                continue
            if first_data:                       # header line
                first_data = False
                if fout:
                    fout.write(line)
                continue
            t = lc._split(line.rstrip('\n\r'), delim)
            try:
                d = float(t[i_dec])
            except (ValueError, IndexError):
                continue
            if scale is None:
                scale = 1e-8 if abs(d) > 90 else 1.0
            d *= scale
            mag = None
            if i_mag is not None:
                try:
                    mag = float(t[i_mag])
                    if abs(mag) >= 100:
                        mag /= 1000.0
                    if not (0 < mag < 30):
                        mag = None
                except (ValueError, IndexError):
                    mag = None
            out['n'] += 1
            out['dec0'] = d if out['dec0'] is None else min(out['dec0'], d)
            out['dec1'] = d if out['dec1'] is None else max(out['dec1'], d)
            if mag is not None:
                out['m0'] = mag if out['m0'] is None else min(out['m0'], mag)
                out['m1'] = mag if out['m1'] is None else max(out['m1'], mag)
            keep = d >= dec_min and (mag is None or mag <= mag_max + 0.5)
            if keep:
                out['n_keep'] += 1
                if fout:
                    fout.write(line)
    if fout:
        fout.close()
    return out


def main(argv=None, emit=None):
    global _emit
    _emit = emit or print
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='folder with the catalogue subfolders')
    ap.add_argument('--dec-min', type=float, default=-30.0)
    ap.add_argument('--mag-max', type=float, default=18.0)
    ap.add_argument('--reduce', default=None, help='write reduced copies of table files into this folder')
    ap.add_argument('--report', default='catalog_overview.csv')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    found = lc.detect_catalogs(root)
    if not found:
        _emit(f"No catalogues recognised in {root}.")
        return 1
    rows = []
    for cat, info in found.items():
        if info['layout'] == 'list':
            continue
        files = lc._data_files(info['path'], max_depth=4)
        _emit(f"\n=== {cat}: {info['path']} ({len(files)} files, layout: {info['layout']}) ===")
        tot = dele = 0
        for k, p in enumerate(files):
            size = os.path.getsize(p)
            tot += size
            if info['layout'] == 'tiles' and lc._TILE_RE.match(os.path.basename(p)):
                rng, mags, reasons = tile_verdict(p, info['path'], a.dec_min, a.mag_max)
                verdict = "DELETE: " + "; ".join(reasons) if reasons else "keep"
                rows.append([cat, os.path.relpath(p, root), size, '', rng, mags, '', verdict])
            else:
                red = None
                if a.reduce:
                    red = os.path.join(os.path.abspath(a.reduce), os.path.relpath(p, root))
                r = scan_table(p, a.dec_min, a.mag_max, red)
                if red and r['n_keep'] == 0 and os.path.exists(red):
                    os.remove(red)                  # nothing inside the limits - no copy needed
                if r['n'] == 0:
                    verdict = "? " + (r['note'] or 'no rows read')
                elif r['n_keep'] == 0:
                    why = []
                    if r['dec1'] is not None and r['dec1'] < a.dec_min:
                        why.append(f"all Dec < {a.dec_min:g}")
                    if r['m0'] is not None and r['m0'] > a.mag_max + 0.5:
                        why.append(f"all fainter than {a.mag_max:g}")
                    verdict = "DELETE: " + ("; ".join(why) or "no star inside the limits")
                elif r['n_keep'] < r['n']:
                    verdict = f"partly useful: {100.0 * r['n_keep'] / r['n']:.0f} % of the stars inside the limits" + \
                              (" (reduced copy written)" if red else " (use --reduce)")
                else:
                    verdict = "keep"
                f = lambda v, nd: '' if v is None else f"{v:.{nd}f}"
                rows.append([cat, os.path.relpath(p, root), size, r['n'], f"{f(r['dec0'], 2)}..{f(r['dec1'], 2)}",
                             f"{f(r['m0'], 2)}..{f(r['m1'], 2)}", r['magcol'], verdict])
                _emit(f"  [{k + 1}/{len(files)}] {os.path.basename(p)}: {verdict}")
            if rows[-1][-1].startswith('DELETE'):
                dele += size
        _emit(f"  {cat}: total {human(tot)}; files that can be deleted: {human(dele)}")
    with open(a.report, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['catalogue', 'file', 'bytes', 'n_stars', 'dec_range', 'mag_range', 'mag_column', 'verdict'])
        w.writerows(rows)
    n_del = sum(1 for r in rows if r[-1].startswith('DELETE'))
    b_del = sum(r[2] for r in rows if r[-1].startswith('DELETE'))
    _emit(f"\n{n_del} of {len(rows)} files ({human(b_del)}) lie entirely outside Dec >= {a.dec_min:g} / "
          f"mag <= {a.mag_max:g}. Full list: {os.path.abspath(a.report)}")
    _emit("Nothing was deleted. Check the list, then delete those files yourself (keep a copy until "
          "Calibra has worked with the reduced set). After deleting, press 'Rescan' in Calibra's settings.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
