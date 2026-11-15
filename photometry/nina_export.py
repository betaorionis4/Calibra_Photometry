"""
Night plan and NINA export (v4.38).

From the targets chosen in the Observation Planner a night plan is made and written as
a NINA *simple sequencer* target set (ArrayOfCaptureSequenceList XML, the format of
'Load Target Set' in NINA) - one entry per target with filter, exposure time, number of
exposures, gain/offset from the instrument setup.

Per target:
  * exposure time: from the standard exposure times of the setup - the longest one that does
    not saturate the star at its brightest, not longer than period/50 (time resolution), with
    the signal-to-noise ratio at its faintest reported; shorter times if even the shortest
    standard time saturates. Zero point: measured value of the setup (planning 'zp_v_e' in
    e-/s for V = 0), else estimated from the aperture.
  * how long: eclipsing binaries with epoch and period - the predicted minimum +-
    (6 % of the period, at least 45 min, at most 3 h) if it falls in the night; short periods
    (<= 0.5 d) - one period (plus 30 min, at most 5 h); 0.5-1 d - 2 h; longer periods / Miras /
    semiregulars / unknown - a few exposures ('snapshot').
  * when: blocks are placed in the time in which the target is observable (planner),
    eclipses at their time, time series near the best time, snapshots in the gaps.

The simple sequencer has no start times: the targets run one after the other in the order
of the plan. Flexible blocks are moved together; a remaining gap (before an eclipse, or until
a target is high enough) is written as 'Delay' of the next target, so the plan holds if the
sequence is started at the planned start time.
"""
import datetime as _dt
import math
import os
import re
from xml.sax.saxutils import escape

import numpy as np

# zero-magnitude photon flux (photons/s/cm2) and typical system throughput per band
_BAND_PHOTONS = {'V': 8.7e5, 'B': 1.3e6, 'R': 6.0e5, 'G': 1.0e6, 'I': 3.5e5, 'L': 2.6e6}
_DEFAULT_THROUGHPUT = 0.45
SHORT_FALLBACK = (1, 2, 5, 10, 15, 20)


def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ photometric model
def zp_e(setup, band='V'):
    """Zero point in electrons/s (magnitude giving 1 e-/s) for a band. Measured value of the
    setup ('planning'->'zp_v_e', V band; other bands scaled) or estimated from the aperture."""
    pl = setup.get('planning') or {}
    meas = _f(pl.get('zp_v_e'))
    ratio = 2.5 * math.log10(_BAND_PHOTONS.get(band, _BAND_PHOTONS['V']) / _BAND_PHOTONS['V'])
    if meas is not None:
        return meas + ratio, 'measured'
    d_mm = _f((setup.get('instrument') or {}).get('aperture_mm')) or 100.0
    area = math.pi * (d_mm / 20.0) ** 2 * 0.85                          # cm2, ~15 % obstruction/losses
    ph = _BAND_PHOTONS.get(band, _BAND_PHOTONS['V']) * area * _DEFAULT_THROUGHPUT
    return 2.5 * math.log10(ph), 'estimated from the aperture'


def star_model(setup, band='V', sky_mag=None):
    """Parameters for signal/noise estimates: zp (e-/s), gain (e-/ADU), saturation (ADU),
    read noise (e-), dark (e-/s/px), FWHM (px), pixel scale, sky (e-/s/px)."""
    from photometry import instruments as ins
    sett = setup.get('settings') or {}
    pl = setup.get('planning') or {}
    z, src = zp_e(setup, band)
    scale = ins.derived(setup).get('pixel_scale') or 1.5
    fwhm_px = _f(sett.get('detect_fwhm'))
    if not fwhm_px:
        fwhm_px = (_f(pl.get('seeing_arcsec')) or 3.0) / scale
    sky_mag = sky_mag if sky_mag is not None else (_f(pl.get('sky_mag')) or 20.0)
    return {'zp': z, 'zp_source': src, 'gain': _f(sett.get('ccd_gain')) or 1.0,
            'sat': _f(sett.get('saturation_limit')) or 60000.0, 'rn': _f(sett.get('ccd_read_noise')) or 3.0,
            'dark': _f(sett.get('ccd_dark_current')) or 0.0, 'fwhm': max(1.0, fwhm_px), 'scale': scale,
            'sky': 10 ** (-0.4 * (sky_mag - z)) * scale ** 2}


def peak_adu(mag, t, m):
    """Peak pixel (ADU, above the background) of a star of magnitude mag in t seconds."""
    e = 10 ** (-0.4 * (mag - m['zp'])) * t
    return e * 4 * math.log(2) / (math.pi * m['fwhm'] ** 2) / m['gain']


def snr(mag, t, m):
    e = 10 ** (-0.4 * (mag - m['zp'])) * t * 0.9
    npix = math.pi * (1.5 * m['fwhm']) ** 2
    return e / math.sqrt(e + npix * (m['sky'] * t + m['dark'] * t + m['rn'] ** 2))


def choose_exposure(target, setup, band='V', std=None, sky_mag=None, sat_fraction=0.7, max_t=None):
    """(exposure s, info dict) for a target. info: snr_bright, snr_faint, peak_frac, note."""
    m = star_model(setup, band, sky_mag)
    std = sorted({float(x) for x in (std or (setup.get('planning') or {}).get('exposures_s') or [30, 60, 120])})
    bright = _f(target.get('max'))
    faint = _f(target.get('min'))
    if bright is None and faint is None:
        bright = faint = None
    else:
        bright = bright if bright is not None else faint
        faint = faint if faint is not None else bright
    p = _f(target.get('period'))
    t_res = p * 86400.0 / 50.0 if p else None
    cap = min(x for x in (max_t, t_res, max(std)) if x) if any((max_t, t_res)) else max(std)
    notes = []
    if bright is None:
        t = min(std, key=lambda x: abs(x - min(cap, 60.0)))
        return t, {'note': "no magnitudes - standard exposure", 'snr_bright': None, 'snr_faint': None,
                   'peak_frac': None, 'zp_source': m['zp_source']}
    ok = [x for x in std if x <= cap + 1e-9 and peak_adu(bright, x, m) <= sat_fraction * m['sat']]
    if ok:
        t = max(ok)
    else:
        short = [x for x in SHORT_FALLBACK if x < min(std) and peak_adu(bright, x, m) <= sat_fraction * m['sat']]
        if short:
            t = max(short)
            notes.append(f"bright: {t:g} s (standard times saturate)")
        else:
            t = min(SHORT_FALLBACK)
            notes.append("saturates even at 1 s - use a narrower filter / defocus / skip")
        if t_res and t > t_res:
            notes.append("time resolution limited")
    if t_res and t >= t_res * 0.99 and t < max(std) and ok:
        notes.append(f"limited to P/50 = {t_res:.0f} s")
    sb, sf = snr(bright, t, m), snr(faint, t, m)
    if sf < 10:
        notes.append(f"faint at minimum (S/N {sf:.0f})")
    return t, {'note': "; ".join(notes), 'snr_bright': sb, 'snr_faint': sf,
               'peak_frac': peak_adu(bright, t, m) / m['sat'], 'zp_source': m['zp_source']}


# ------------------------------------------------------------------ what and how long
_LONG_TYPES = ('M', 'SR', 'SRA', 'SRB', 'SRC', 'SRD', 'L', 'LB', 'LC', 'RV', 'RVA', 'RVB', 'ZAND', 'RCB', 'GCAS', 'BE')


def minima_in(target, jd0, jd1):
    """Predicted times (JD) of primary minima of an eclipsing binary between jd0 and jd1
    (epoch and period in the target list; GCVS epochs are HJD - the difference is <= 8 min)."""
    p, e = _f(target.get('period')), _f(target.get('epoch'))
    if not p or not e:
        return []
    if e < 2400000:
        e += 2400000.0                    # reduced JD
    n0 = math.ceil((jd0 - e) / p)
    out = []
    n = n0
    while e + n * p <= jd1:
        out.append(e + n * p)
        n += 1
    return out


def block_request(target, jd0, jd1, snapshot_frames=5):
    """What to observe: {'kind': 'eclipse'|'series'|'snapshot', 'hours' or 'frames', 'center_jd'}."""
    t = str(target.get('type') or '').upper()
    parts = re.split(r'[/|+:\s]+', t)
    p = _f(target.get('period'))
    if t == 'CLUSTER':
        return {'kind': 'series', 'hours': 2.0, 'why': "cluster field - 2 h series for all its variables"}
    if t.startswith('E') and not t.startswith('EW') and p:
        mins = minima_in(target, jd0, jd1)
        half = min(3.0, max(0.75, 0.06 * p * 24.0))
        if mins:
            return {'kind': 'eclipse', 'hours': 2 * half, 'center_jd': mins[0],
                    'why': f"primary minimum predicted (+-{half:.1f} h)"}
        if target.get('epoch') not in (None, ''):
            return {'kind': 'snapshot', 'frames': snapshot_frames, 'why': "no primary minimum in this night"}
        return {'kind': 'series', 'hours': min(4.0, max(2.0, p * 24 * 0.3)),
                'why': "eclipsing binary without epoch - time series"}
    if p and p <= 0.5:
        return {'kind': 'series', 'hours': min(5.0, p * 24.0 + 0.5), 'why': "short period - one full cycle"}
    if p and p <= 1.0:
        return {'kind': 'series', 'hours': 2.0, 'why': "period 0.5-1 d - 2 h series"}
    if any(x in _LONG_TYPES for x in parts) or (p and p > 1.0):
        return {'kind': 'snapshot', 'frames': snapshot_frames if not (p and p > 20) else 3,
                'why': "long period - a few exposures"}
    return {'kind': 'snapshot', 'frames': snapshot_frames, 'why': "no period - a few exposures"}


# ------------------------------------------------------------------ schedule
def _intervals(ok, jd):
    """Contiguous observable intervals [(jd_start, jd_end)] from a mask on the time grid."""
    out, i, n = [], 0, len(ok)
    step = (jd[1] - jd[0]) if n > 1 else 0
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            out.append((jd[i], jd[j] + step))
            i = j + 1
        else:
            i += 1
    return out


def make_plan(items, jd, overhead_min=3.0, per_frame_s=4.0):
    """
    items: [{'target', 'ok' (mask on jd), 'best_jd', 'exposure', 'filters' [names], 'request'}]
    Returns the scheduled blocks sorted by start ({'start','end','frames' per filter, ...}) and
    the items that did not fit (with 'why').
    """
    busy = []                                    # (start, end)
    blocks, rejected = [], []
    ov = overhead_min / 1440.0

    def free(a, b):
        return all(b <= s or a >= e for s, e in busy)

    def place(it, length, around):
        ivs = _intervals(it['ok'], jd)
        cands = []
        for s, e in ivs:
            if e - s < length:
                continue
            # candidate starts: as close to 'around' as possible, then along the interval
            st0 = min(max(around - length / 2, s), e - length)
            grid = np.unique(np.concatenate([[st0], np.arange(s, e - length + 1e-9, 5 / 1440.0)]))
            for st in sorted(grid, key=lambda x: abs(x + length / 2 - around)):
                if free(st, st + length):
                    cands.append(st)
                    break
        if not cands:
            return None
        return min(cands, key=lambda x: abs(x + length / 2 - around))

    order = sorted(items, key=lambda it: {'eclipse': 0, 'series': 1, 'snapshot': 2}[it['request']['kind']])
    for it in order:
        req = it['request']
        nf = max(1, len(it['filters']))
        cycle = (it['exposure'] + per_frame_s) / 86400.0
        if req['kind'] == 'snapshot':
            frames = int(req.get('frames', 5))
            length = ov + frames * nf * cycle
            around = it['best_jd']
        else:
            length = ov + req['hours'] / 24.0
            around = req.get('center_jd', it['best_jd'])
            frames = max(1, int((req['hours'] / 24.0) / (nf * cycle)))
        st = place(it, length, around)
        if st is None and req['kind'] != 'snapshot':
            # shorten a time series to the longest free piece (at least half)
            for frac in (0.75, 0.5):
                st = place(it, ov + req['hours'] / 24.0 * frac, around)
                if st is not None:
                    length = ov + req['hours'] / 24.0 * frac
                    frames = max(1, int((length - ov) / (nf * cycle)))
                    it = dict(it, shortened=frac)
                    break
        if st is None:
            rejected.append(dict(it, why="no free observable time"))
            continue
        busy.append((st, st + length))
        blocks.append(dict(it, start_jd=st, end_jd=st + length, frames=frames))
    blocks.sort(key=lambda b: b['start_jd'])
    # The simple sequencer runs the targets one after the other: close gaps by moving flexible
    # blocks (not eclipses) to the end of the previous block if the target is observable then.
    for i in range(1, len(blocks)):
        b, prev_end = blocks[i], blocks[i - 1]['end_jd']
        if b['request']['kind'] == 'eclipse' or b['start_jd'] <= prev_end + 1e-9:
            continue
        length = b['end_jd'] - b['start_jd']
        if _observable(b['ok'], jd, prev_end, prev_end + length):
            b['start_jd'], b['end_jd'] = prev_end, prev_end + length
    blocks.sort(key=lambda b: b['start_jd'])
    return blocks, rejected


def _observable(ok, jd, a, b):
    """Target observable during the whole interval [a, b] (time grid with its step)."""
    step = (jd[1] - jd[0]) if len(jd) > 1 else 0
    idx = np.where((jd + step > a) & (jd < b))[0]
    return len(idx) > 0 and bool(np.all(ok[idx])) and a >= jd[0] - 1e-9 and b <= jd[-1] + step + 1e-9


# ------------------------------------------------------------------ NINA XML
_ATTR_DEFAULTS = dict(PositionAngle="0", AutoFocusOnStart="false", RotateTarget="false",
                      AutoFocusOnFilterChange="false", AutoFocusAfterSetTime="false", AutoFocusSetTime="30",
                      AutoFocusAfterSetExposures="false", AutoFocusSetExposures="10",
                      AutoFocusAfterTemperatureChange="false", AutoFocusAfterTemperatureChangeAmount="5",
                      AutoFocusAfterHFRChange="false", AutoFocusAfterHFRChangeAmount="10")


def _sexa(ra_deg, dec_deg):
    h = (ra_deg % 360.0) / 15.0
    rh = int(h); rm = int((h - rh) * 60); rs = int(round(((h - rh) * 60 - rm) * 60))
    if rs == 60:
        rs = 0; rm += 1
    if rm == 60:
        rm = 0; rh = (rh + 1) % 24
    d = abs(dec_deg)
    dd = int(d); dm = int((d - dd) * 60); ds = int(round(((d - dd) * 60 - dm) * 60))
    if ds == 60:
        ds = 0; dm += 1
    if dm == 60:
        dm = 0; dd += 1
    return rh, rm, rs, dd, dm, ds, dec_deg < 0


def _bool(x):
    return "true" if x else "false"


def capture_xml(filter_name, position, exposure, count, gain, offset, binning=1, dither=False, indent="    "):
    i, j, k = indent + "  ", indent + "    ", indent + "      "
    return (f"{indent}<CaptureSequence>\n"
            f"{i}<Enabled>true</Enabled>\n"
            f"{i}<ExposureTime>{exposure:g}</ExposureTime>\n"
            f"{i}<ImageType>LIGHT</ImageType>\n"
            f"{i}<FilterType>\n"
            f"{j}<Name>{escape(str(filter_name))}</Name>\n"
            f"{j}<FocusOffset>0</FocusOffset>\n"
            f"{j}<Position>{int(position)}</Position>\n"
            f"{j}<AutoFocusExposureTime>-1</AutoFocusExposureTime>\n"
            f"{j}<AutoFocusFilter>false</AutoFocusFilter>\n"
            f"{j}<FlatWizardFilterSettings>\n"
            f"{k}<FlatWizardMode>DYNAMICEXPOSURE</FlatWizardMode>\n"
            f"{k}<HistogramMeanTarget>0.5</HistogramMeanTarget>\n"
            f"{k}<HistogramTolerance>0.1</HistogramTolerance>\n"
            f"{k}<MaxFlatExposureTime>10</MaxFlatExposureTime>\n"
            f"{k}<MinFlatExposureTime>0.1</MinFlatExposureTime>\n"
            f"{k}<MaxAbsoluteFlatDeviceBrightness>30</MaxAbsoluteFlatDeviceBrightness>\n"
            f"{k}<MinAbsoluteFlatDeviceBrightness>0</MinAbsoluteFlatDeviceBrightness>\n"
            f"{k}<Gain>-1</Gain>\n"
            f"{k}<Offset>-1</Offset>\n"
            f"{k}<Binning>\n{k}  <X>1</X>\n{k}  <Y>1</Y>\n{k}</Binning>\n"
            f"{j}</FlatWizardFilterSettings>\n"
            f"{j}<AutoFocusBinning>\n{j}  <X>1</X>\n{j}  <Y>1</Y>\n{j}</AutoFocusBinning>\n"
            f"{j}<AutoFocusGain>-1</AutoFocusGain>\n"
            f"{j}<AutoFocusOffset>-1</AutoFocusOffset>\n"
            f"{i}</FilterType>\n"
            f"{i}<Binning>\n{j}<X>{int(binning)}</X>\n{j}<Y>{int(binning)}</Y>\n{i}</Binning>\n"
            f"{i}<Gain>{int(gain) if gain is not None else -1}</Gain>\n"
            f"{i}<Offset>{int(offset) if offset is not None else -1}</Offset>\n"
            f"{i}<TotalExposureCount>{int(count)}</TotalExposureCount>\n"
            f"{i}<ProgressExposureCount>0</ProgressExposureCount>\n"
            f"{i}<Dither>{_bool(dither)}</Dither>\n"
            f"{i}<DitherAmount>1</DitherAmount>\n"
            f"{indent}</CaptureSequence>\n")


def target_xml(name, ra_deg, dec_deg, captures, mode="STANDARD", delay_s=0, slew=True, center=True, guide=True):
    rh, rm, rs, dd, dm, ds, neg = _sexa(ra_deg, dec_deg)
    attrs = dict(TargetName=str(name), Mode=mode, RAHours=str(rh), RAMinutes=str(rm), RASeconds=str(rs),
                 DecDegrees=str(-dd if neg and dd else dd), DecMinutes=str(dm), DecSeconds=str(ds),
                 PositionAngle="0", Delay=str(int(max(0, delay_s))), SlewToTarget=_bool(slew),
                 AutoFocusOnStart="false", CenterTarget=_bool(center), RotateTarget="false",
                 StartGuiding=_bool(guide))
    for k, v in _ATTR_DEFAULTS.items():
        attrs.setdefault(k, v)
    a = " ".join(f'{k}="{escape(v, {chr(34): "&quot;"})}"' for k, v in attrs.items())
    return (f"  <CaptureSequenceList {a}>\n" + "".join(captures) +
            f"    <Coordinates>\n      <RA>{(ra_deg % 360.0) / 15.0!r}</RA>\n      <Dec>{float(dec_deg)!r}</Dec>\n"
            f"      <Epoch>J2000</Epoch>\n    </Coordinates>\n"
            f"    <NegativeDec>{_bool(neg)}</NegativeDec>\n  </CaptureSequenceList>\n")


def write_target_set(path, entries):
    """entries: list of target_xml strings. Writes an ArrayOfCaptureSequenceList file."""
    txt = ('<?xml version="1.0" encoding="utf-8"?>\n<ArrayOfCaptureSequenceList '
           'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n'
           + "".join(entries) + "</ArrayOfCaptureSequenceList>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)
    return path


def plan_to_entries(blocks, setup, filters, mode="ROTATE", start_jd=None, slew=True, center=True, guide=True,
                    dither=False):
    """NINA entries for scheduled blocks. filters: [(name, position)]. Delay = gap before a block
    if the block has a fixed time (eclipse) and the sequence starts at start_jd."""
    ins = setup.get('instrument') or {}
    gain, offset = _f(ins.get('gain')), _f(ins.get('offset'))
    binning = int(_f(ins.get('binning')) or 1)
    out = []
    t_cur = start_jd if start_jd is not None else (blocks[0]['start_jd'] if blocks else 0)
    for b in blocks:
        delay = 0.0
        if b['start_jd'] > t_cur + 2 / 1440.0:            # wait (eclipse, target not yet up ...)
            delay = (b['start_jd'] - t_cur) * 86400.0
        caps = [capture_xml(fn, pos, b['exposure'], b['frames'], gain, offset, binning, dither) for fn, pos in filters]
        t = b['target']
        out.append(target_xml(t.get('name', 'target'), float(t['ra']), float(t['dec']), caps,
                              mode=mode if len(filters) > 1 else "STANDARD", delay_s=delay, slew=slew,
                              center=center, guide=guide))
        t_cur = max(t_cur, b['start_jd']) + (b['end_jd'] - b['start_jd'])
    return out


def filters_from_nina_file(path):
    """{filter name: position} from an existing NINA sequence / target set."""
    import xml.etree.ElementTree as ET
    out = {}
    root = ET.parse(path).getroot()
    for ft in root.iter('FilterType'):
        n, p = ft.findtext('Name'), ft.findtext('Position')
        if n is not None and p is not None:
            try:
                out[n.strip()] = int(p)
            except ValueError:
                pass
    return out


def default_folder(calibra_dir):
    """NINA folder next to the Calibra folder (e.g. Variables/NINA), else 'nina_sequences'."""
    cand = os.path.join(os.path.dirname(os.path.abspath(calibra_dir)), "NINA")
    return cand if os.path.isdir(cand) else os.path.join(os.path.abspath(calibra_dir), "nina_sequences")
