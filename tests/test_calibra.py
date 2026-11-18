"""
Calibra regression tests (v4.16).  Run from the Calibra folder with:

    python -m pytest tests -q

Only synthetic data are used (no internet, no GUI).
"""
import os
import sys
import csv
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# v5.0: never touch the real user folder (~/Calibra) during tests
import tempfile
os.environ["CALIBRA_HOME"] = tempfile.mkdtemp(prefix="calibra_test_home_")
from python import paths


# ---------------------------------------------------------------- filters
from python.filters import classify_filter


@pytest.fixture(autouse=True)
def _fresh_user_dir(tmp_path):
    """v5.0: every test gets its own empty user folder (settings, cache, output)."""
    old = paths.USER_DIR
    paths.set_user_dir(tmp_path / "calibra_home", "test")
    yield
    paths.set_user_dir(old, "test")


@pytest.fixture(autouse=True)
def _reset_online_state():
    """v4.34: failed online services are remembered - start every test with a clean state."""
    from python import online
    online._DOWN.clear()
    yield
    online._DOWN.clear()

@pytest.mark.parametrize("filt,expected", [
    ("Vmag", "V"), ("Bmag", "B"), ("V", "V"), ("B", "B"), ("Bessell V", "V"), ("Baader V", "V"),
    ("SBIG V", "V"), ("Johnson-B", "B"), ("JohnsonV", "V"), ("Vmag_2", "V"),
    ("L", None), ("R", None), ("Master_L", None), ("", None),
])
def test_classify_filter(filt, expected):
    assert classify_filter(filt, "Vmag", "Bmag") == expected


# ---------------------------------------------------------------- noise model
from python.noise import effective_gain, flux_variance, stack_count

def test_effective_gain_average_stack():
    g, n = effective_gain(1.27, {'STACKCNT': 49}, 'average')
    assert n == 49 and g == pytest.approx(1.27 * 49)
    g2, _ = effective_gain(1.27, {'STACKCNT': 49}, 'sum')
    assert g2 == pytest.approx(1.27)
    assert stack_count({}) == 1

def test_flux_variance_uses_net_flux():
    # pure Poisson star, no sky noise
    assert flux_variance(10000, 50, 0.0, 500, 2.0) == pytest.approx(5000)
    # sky term: A s^2 (1 + A/n)
    assert flux_variance(0, 50, 3.0, 500, 2.0) == pytest.approx(50 * 9 * 1.1)

def test_aperture_errors_match_scatter():
    """Simulated star + sky: the quoted flux error must match the real scatter."""
    from python.aperture_phot import perform_aperture_photometry
    rng = np.random.default_rng(3)
    gain, sky, F = 2.0, 400.0, 20000.0
    yy, xx = np.mgrid[:61, :61]
    psf = np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / (2 * 2.0 ** 2)); psf /= psf.sum()
    fluxes, errs = [], []
    for _ in range(200):
        img = rng.poisson((sky + F * psf) * gain) / gain
        res = [{'id': 'S', 'x': 31.0, 'y': 31.0}]
        perform_aperture_photometry(img, res, 6.0, 10.0, 16.0, print_table=False, gain=gain)
        fluxes.append(res[0]['net_flux']); errs.append(res[0]['flux_err'])
    ratio = np.std(fluxes) / np.median(errs)
    assert 0.85 < ratio < 1.15


# ---------------------------------------------------------------- differential photometry
from python.diff_photometry import compute_zero_points, compute_target_BV

def test_B_minus_V_consistent():
    Z = compute_zero_points(0.95, 0.10, -0.03, 0.35, 0.2, 11.0, 10.4, -12.0, -13.2, 1.1, 1.1)
    B, V, BV = compute_target_BV(-11.5, -12.3, 1.2, 1.2, 0.95, 0.10, -0.03, *Z, 0.35, 0.2)
    assert B - V == pytest.approx(BV, abs=1e-12)

def test_comparison_star_recovers_itself():
    Z = compute_zero_points(0.95, 0.10, -0.03, 0.35, 0.2, 11.0, 10.4, -12.0, -13.2, 1.1, 1.1)
    B, V, BV = compute_target_BV(-12.0, -13.2, 1.1, 1.1, 0.95, 0.10, -0.03, *Z, 0.35, 0.2)
    assert V == pytest.approx(10.4) and BV == pytest.approx(0.6)


# ---------------------------------------------------------------- colour fit
from python.color_calibration import perform_robust_fit

def test_robust_weighted_fit_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1.5, 300)
    y = 0.05 * x + 0.3 + rng.normal(0, 0.01, 300)
    y[:15] += 0.5   # outliers
    fit, keep, *_ = perform_robust_fit(x, y, sig=np.full(300, 0.01))
    assert fit.slope == pytest.approx(0.05, abs=0.005)
    assert keep[:15].sum() == 0


# ---------------------------------------------------------------- image calibration
def test_calibrate_image_float_and_naming(tmp_path):
    from astropy.io import fits
    from python.image_calibration import calibrate_image
    bias = np.full((20, 30), 500.0)
    flat = 20000.0 * np.linspace(0.8, 1.2, 30)[None, :] * np.ones((20, 1))
    fits.PrimaryHDU(bias.astype(np.uint16)).writeto(tmp_path / "b.fits")
    fits.PrimaryHDU(flat.astype(np.uint16)).writeto(tmp_path / "f.fits")
    light = bias + 5 * flat / 20000.0          # very low sky -> negative noise values must survive
    h = fits.Header(); h['FILENAME'] = 'wrong_name.fits'
    _, _, out = calibrate_image(light, h, str(tmp_path / "b.fits"), str(tmp_path / "f.fits"),
                                out_dir=str(tmp_path / "cal"), input_path="my_stack.fits", return_path=True, verbose=False)
    assert os.path.basename(out) == "cal_my_stack.fits"
    d = fits.getdata(out)
    assert d.dtype.kind == 'f'
    assert np.allclose(d, 5.0, atol=0.05)


# ---------------------------------------------------------------- data mining alignment
def test_data_mining_partial_frame_alignment(tmp_path):
    from python import data_mining as dm
    dm.cross_match_vsx_suspects = lambda *a, **k: None
    dm.cross_match_deep_catalog = lambda *a, **k: None
    rng = np.random.default_rng(0)
    cat = [{'id': f'S{i}', 'ra_deg': 10 + i * 0.01, 'dec_deg': 20.0} for i in range(20)]
    tr = []
    for f in range(20):
        meas = {}
        for i, s in enumerate(cat):
            if f == 5 and i >= 10:
                continue
            m = -10 + 0.1 * i + rng.normal(0, 0.003) + (0.4 * np.sin(f / 3) if i == 15 else 0)
            meas[s['id']] = {'mag_inst': m, 'mag_err': 0.003, 'snr': 300, 'bkg_std': 5}
        tr.append({'measurements': meas, 'img_std': 5})
    stats, _ = dm.calibrate_and_analyze(tr, cat, str(tmp_path / "dm.csv"), ensemble_size=10)
    top = max(stats, key=lambda s: s['rms'])
    assert top['id'] == 'S15'


# ---------------------------------------------------------------- v4.17 speed code
from python.fastphot import fast_sky_stats, windowed_centroid, track_centroid

def test_fast_sky_stats_equals_astropy():
    from astropy.stats import sigma_clip
    rng = np.random.default_rng(5)
    for _ in range(100):
        x = rng.normal(500, 20, 400); x[:20] += rng.uniform(100, 3000, 20)
        g = sigma_clip(x, sigma=3, maxiters=5, masked=True).compressed()
        med, std, n = fast_sky_stats(x)
        assert med == pytest.approx(np.median(g)) and std == pytest.approx(g.std()) and n == g.size

def test_windowed_centroid_accuracy():
    rng = np.random.default_rng(2)
    errs = []
    for _ in range(100):
        tx, ty = 30 + rng.uniform(-1.5, 1.5), 30 + rng.uniform(-1.5, 1.5)
        yy, xx = np.mgrid[:61, :61]
        img = 500 + 20000 * np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2 * 1.5 ** 2)) / (2 * np.pi * 1.5 ** 2)
        img = rng.poisson(img).astype(float)
        x, y, ok = track_centroid(img, tx + rng.uniform(-3, 3), ty + rng.uniform(-3, 3), max_shift=6, sigma_w=1.5)
        assert ok
        errs.append(np.hypot(x - tx, y - ty))
    assert np.sqrt(np.mean(np.square(errs))) < 0.05

def test_track_centroid_rejects_far_jump():
    yy, xx = np.mgrid[:61, :61]
    img = 500 + 50000 * np.exp(-((xx - 36) ** 2 + (yy - 30) ** 2) / 4.5)   # only a star 6 px away
    x, y, ok = track_centroid(img.astype(float), 30.0, 30.0, max_shift=4.0, sigma_w=1.5)
    assert not ok and (x, y) == (30.0, 30.0)

def test_data_mining_cache_roundtrip(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    from python import data_mining as dm
    w = WCS(naxis=2); w.wcs.crpix = [50, 50]; w.wcs.cdelt = [-1.4/3600, 1.4/3600]; w.wcs.crval = [150, 30]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    yy, xx = np.mgrid[:100, :100]
    img = 300 + 20000 * np.exp(-((xx - 40) ** 2 + (yy - 55) ** 2) / 4.5) / (2 * np.pi * 2.25)
    fn = str(tmp_path / "f.fits"); fits.PrimaryHDU(img.astype(np.float32), header=w.to_header()).writeto(fn)
    c = w.pixel_to_world(40, 55)
    cat = [{'id': 'S0', 'ra_deg': float(c.ra.deg), 'dec_deg': float(c.dec.deg)}]
    kw = dict(saturation_limit=1e9, edge_buffer=15, use_cache=True, cache_dir=str(tmp_path / "cache"))
    r1, _ = dm.track_and_measure([fn], cat, 5, 8, 13, **kw)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    r2, _ = dm.track_and_measure([fn], cat, 5, 8, 13, **kw)
    assert r1[0]['measurements']['S0']['mag_inst'] == pytest.approx(r2[0]['measurements']['S0']['mag_inst'])
    r3, _ = dm.track_and_measure([fn], cat, 6, 8, 13, **kw)   # other aperture -> new measurement
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2


# ---------------------------------------------------------------- field variables (v4.19)
import python.field_variables as fvm

def test_parse_target_list_formats(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("# comment\nAE UMa 09 36 53.2 +44 04 00\nOff field, 10.0, 10.0\n"
                 "RR Lyr, 19 25 27.9, +42 47 03.7\nV1 Her;19:25:27.9;+42:47:03.7\nOnlyName\n")
    t, probs = fvm.parse_target_list(str(p))
    assert [x['name'] for x in t] == ['AE UMa', 'Off field', 'RR Lyr', 'V1 Her', 'OnlyName']
    assert t[0]['ra'] == pytest.approx(144.22167, abs=1e-4) and t[0]['dec'] == pytest.approx(44.06667, abs=1e-4)
    assert t[1]['ra'] == 10.0 and t[2]['ra'] == pytest.approx(291.36625, abs=1e-4)
    assert t[4]['ra'] is None and not probs
    h = tmp_path / "h.csv"
    h.write_text("Name,RA,Dec,Type,Period\nAE UMa,09:36:53.2,+44:04:00,HADS,0.086\nbad,abc,def,,\n")
    t, probs = fvm.parse_target_list(str(h))
    assert len(t) == 1 and t[0]['period'] == pytest.approx(0.086) and t[0]['type'] == 'HADS' and len(probs) == 1
    fvm.save_target_list(str(tmp_path / "s.csv"), t)
    t2, _ = fvm.parse_target_list(str(tmp_path / "s.csv"))
    assert t2[0]['name'] == 'AE UMa' and t2[0]['ra'] == pytest.approx(t[0]['ra'], abs=1e-6)


def test_query_gcvs_simulated_and_filter():
    from astropy.table import Table
    class FakeViz:
        def __init__(self, **kw): pass
        def query_region(self, c, radius):
            t = Table({'GCVS': ['AE     UMa', 'BB  UMa', 'CC UMa'], 'VarType': ['DSCTC', 'EA', 'M'],
                       'magMax': [10.86, 12.0, 15.5], 'Min1': [11.51, 0.0, 17.0], 'Period': [0.086, 2.5, 300.0],
                       '_RAJ2000': [144.22, 144.5, 144.3], '_DEJ2000': [44.07, 44.2, 44.0]}, masked=True)
            t['Min1'].mask = [False, True, False]
            return [t]
    g = fvm.query_gcvs(144.3, 44.07, 30, vizier_cls=FakeViz)
    assert [s['name'] for s in g] == ['AE UMa', 'BB UMa', 'CC UMa'] and g[1]['min'] is None
    assert [s['name'] for s in fvm.filter_variables(g, faintest_max=14)] == ['AE UMa', 'BB UMa']
    assert [s['name'] for s in fvm.filter_variables(g, min_amplitude=0.3)] == ['AE UMa', 'CC UMa']
    assert [s['name'] for s in fvm.filter_variables(g, min_period=1, max_period=100)] == ['BB UMa']
    assert [s['name'] for s in fvm.filter_variables(g, type_filter='M SR')] == ['CC UMa']


def test_query_vsx_refreshes_incomplete_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(paths.CATALOG_CACHE_DIR)
    with open(os.path.join(paths.CATALOG_CACHE_DIR, "VSX_10.0000_20.0000_30.csv"), "w") as f:
        f.write("id,ra_deg,dec_deg\nX Cnc,10.01,20.01\n")      # old cache: no type/range/period
    fresh = [{'id': 'X Cnc', 'ra_deg': 10.01, 'dec_deg': 20.01, 'Type': 'SRB', 'max': 5.6, 'min': 7.5, 'Period': 195.0}]
    out = fvm.query_vsx(10.0, 20.0, 30, fetch=lambda *a, **k: fresh)
    assert out[0]['type'] == 'SRB' and out[0]['period'] == 195.0
    out2 = fvm.query_vsx(10.0, 20.0, 30, fetch=lambda *a, **k: [])   # now read from the new cache
    assert out2[0]['type'] == 'SRB'


def _star_field(tmp_path, n_frames=3, var_amp=0.3, shifts=None):
    """Synthetic frames with WCS: 30 constant stars + one variable (index 0).
    shifts: optional list of (dx, dy) pointing offsets [px] per frame (other nights)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    rng = np.random.default_rng(5)
    ny = nx = 300
    w = WCS(naxis=2); w.wcs.crpix = [150, 150]; w.wcs.cdelt = [-2.0/3600, 2.0/3600]; w.wcs.crval = [150, 30]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    xs = rng.uniform(30, 270, 31); ys = rng.uniform(30, 270, 31)
    mags = rng.uniform(11, 13.5, 31)
    yy, xx = np.mgrid[0:ny, 0:nx]
    files = []
    for k in range(n_frames):
        dx, dy = shifts[k] if shifts else (0.0, 0.0)
        img = np.full((ny, nx), 500.0)
        for i in range(31):
            m = mags[i] + (var_amp * np.sin(2 * np.pi * k / n_frames) if i == 0 else 0.0)
            flux = 10 ** (-0.4 * (m - 25.0))
            img += flux / (2 * np.pi * 2.0 ** 2) * np.exp(-((xx - xs[i] - dx) ** 2 + (yy - ys[i] - dy) ** 2) / (2 * 2.0 ** 2))
        img = rng.poisson(img).astype(np.float32)
        wk = w.deepcopy(); wk.wcs.crpix = [150 + dx, 150 + dy]
        h = wk.to_header(); h['DATE-OBS'] = f"2026-04-13T2{k}:00:00"; h['EXPTIME'] = 60.0; h['FILTER'] = 'V'
        fn = str(tmp_path / f"f{k}.fts"); fits.PrimaryHDU(img, header=h).writeto(fn); files.append(fn)
    sky = w.pixel_to_world(xs, ys)
    cat = [{'id': f"S{i}", 'ra_deg': float(sky[i].ra.deg), 'dec_deg': float(sky[i].dec.deg),
            'V_mag': float(mags[i]), 'B_mag': float(mags[i]) + 0.6} for i in range(31)]
    return files, cat, w, (ny, nx)


def test_multi_target_light_curves(tmp_path):
    from python.time_series import run_multi_target_photometry, run_time_series_photometry
    files, cat, w, shape = _star_field(tmp_path)
    var = {'name': 'VAR', 'ra': cat[0]['ra_deg'], 'dec': cat[0]['dec_deg']}
    other = {'name': 'C5', 'ra': cat[5]['ra_deg'], 'dec': cat[5]['dec_deg']}
    targets = []
    for t in (var, other):
        comps, chk, tmag, tbv = fvm.choose_catalog_comparisons(t, cat, [var], w, shape, n_comp=4,
                                                                 radius_arcmin=10, blend_arcsec=10, edge_px=20)
        assert len(comps) == 4 and chk is not None and tbv == pytest.approx(0.6)
        assert all(c['name'] not in ('S0', t['name']) for c in comps)
        targets.append({'name': t['name'], 'ra': t['ra'], 'dec': t['dec'], 'ensemble': comps, 'check': chk,
                        'target_bv': tbv, 'lc_mode': 'calibrated'})
    targets.append(dict(targets[0], name='VAR raw', lc_mode='raw', ensemble=[], check=None))
    out = run_multi_target_photometry(files, targets, aperture_radius=6, annulus_inner=9, annulus_outer=14,
                                      gain=1.0, k_coeff=0.0, verbose=False)
    st_var = fvm.summarize_light_curve(out[0][0]); st_c = fvm.summarize_light_curve(out[1][0])
    assert st_var['n'] == 3 and st_c['n'] == 3
    assert st_var['p2p'] > 0.3 and st_c['std'] < 0.02
    assert out[1][0][0]['mag'] == pytest.approx(cat[5]['V_mag'], abs=0.03)
    assert out[2][0][0]['lc_mode'] == 'raw'
    # identical to the single-target function
    r1, _ = run_time_series_photometry(files, targets[0]['ra'], targets[0]['dec'], targets[0]['ensemble'],
                                       targets[0]['check'], targets[0]['target_bv'], 0.0, 0.0, 6, 9, 14,
                                       gain=1.0, k_coeff=0.0)
    assert [r['mag'] for r in r1] == [r['mag'] for r in out[0][0]]
    png = fvm.plot_overview([(t['name'], r) for t, (r, _) in zip(targets, out)], str(tmp_path / "ov.png"))
    assert os.path.exists(png)


def test_footprints_and_shifted_nights(tmp_path):
    """v4.20: pointing differs between nights - target off the image on one night,
    a comparison star missing on another: frames are skipped / the relative-flux
    ensemble is rescaled, so the light curve of a constant star stays flat."""
    from python.time_series import run_multi_target_photometry
    shifts = [(0, 0), (0, 0), (120, 0), (-100, 0), (-100, 0)]
    files, cat, w, shape = _star_field(tmp_path, n_frames=5, var_amp=0.0, shifts=shifts)
    fps = fvm.image_footprints(files)
    assert all(f[1] is not None for f in fps)
    ra = np.array([c['ra_deg'] for c in cat]); dec = np.array([c['dec_deg'] for c in cat])
    inside = fvm.in_footprints(ra, dec, fps, margin_px=20)
    assert inside.shape == (5, 31)
    # a star near the right edge: on the image in frames 0,1,3,4 but not in frame 2
    xs, ys = w.world_to_pixel_values(ra, dec)
    tgt = int(np.argmax(np.where((xs > 200) & (xs < 250), xs, -1)))
    assert not inside[2, tgt] and inside[0, tgt]
    # comparison stars: one of them off the image in frames 3/4 (left part of the field)
    comps = [c for k, c in enumerate(cat) if k != tgt and k != 0][:4]
    left = int(np.argmin(np.where(np.arange(31) != tgt, xs, 1e9)))
    if all(c['id'] != cat[left]['id'] for c in comps):
        comps = comps[:3] + [cat[left]]
    ens = [{'name': c['id'], 'ra': c['ra_deg'], 'dec': c['dec_deg'], 'mag_std': c['V_mag'], 'bv_std': 0.6} for c in comps]
    out = run_multi_target_photometry(files, [{'name': 'T', 'ra': ra[tgt], 'dec': dec[tgt], 'ensemble': ens,
                                               'check': None, 'target_bv': 0.6, 'lc_mode': 'relative'}],
                                      aperture_radius=6, annulus_inner=9, annulus_outer=14, verbose=False)
    res, msg = out[0]
    assert len(res) == 4                       # frame 2 skipped (target outside)
    f = np.array([r['flux'] for r in res])
    assert np.std(f) / np.mean(f) < 0.01       # flat although one comparison star is missing in 2 frames
    assert any(r.get('ens_complete') is False for r in res)
    # target off every image
    far = {'name': 'far', 'ra': 151.0, 'dec': 30.0, 'ensemble': [], 'check': None, 'lc_mode': 'raw'}
    r2, m2 = run_multi_target_photometry(files, [far], verbose=False)[0]
    assert r2 is None and 'outside' in m2
    # automatic comparison stars respect all footprints that contain the target
    comps2, chk, _, _ = fvm.choose_catalog_comparisons({'name': 'T', 'ra': ra[tgt], 'dec': dec[tgt]}, cat, (),
                                                        w, shape, n_comp=3, radius_arcmin=15, blend_arcsec=5,
                                                        edge_px=20, footprints=fps)
    t_in = fvm.in_footprints(ra[tgt], dec[tgt], fps, margin_px=10)
    cin = fvm.in_footprints(np.array([c['ra'] for c in comps2]), np.array([c['dec'] for c in comps2]),
                            [f for f, k in zip(fps, t_in) if k], margin_px=20)
    assert cin.all()


# ---------------------------------------------------------------- period analysis (v4.21)
import python.period_analysis as pam

def _multi_periodic(seed=1):
    rng = np.random.default_rng(seed)
    t = np.concatenate([2461144.35 + n + np.arange(0, 0.25, 5 / 1440) for n in (0, 1, 3, 7, 8)])
    y = (11.3 + 0.12 * np.sin(2 * np.pi * 11.6257 * (t - 2461144)) + 0.04 * np.sin(2 * np.pi * 2 * 11.6257 * (t - 2461144) + 1)
         + 0.03 * np.sin(2 * np.pi * 14.9 * (t - 2461144)) + rng.normal(0, 0.01, t.size))
    return t, y, np.full(t.size, 0.01)


def test_prewhitening_finds_main_and_secondary_frequency():
    t, y, e = _multi_periodic()
    fr = pam.frequency_grid(t)
    s = pam.data_summary(t)
    assert s['nights'] == 5 and s['nyquist'] == pytest.approx(144, rel=0.01)
    amp = pam.amplitude_spectrum(t, y, fr)
    f1 = pam.find_peak(fr, amp)
    assert f1 == pytest.approx(11.6257, abs=0.02)
    fit = pam.fit_frequencies(t, y, [f1], [2], e)
    f2 = pam.find_peak(fr, pam.amplitude_spectrum(t, fit['residuals'], fr), exclude=fit['freqs'], resolution=s['resolution'])
    fit = pam.fit_frequencies(t, y, fit['freqs'] + [f2], [2, 1], e)
    assert fit['freqs'][0] == pytest.approx(11.6257, abs=0.002) and fit['freqs'][1] == pytest.approx(14.9, abs=0.005)
    assert fit['components'][0][0][1] == pytest.approx(0.12, abs=0.006)
    assert fit['components'][0][1][1] == pytest.approx(0.04, abs=0.005)
    assert fit['components'][1][0][1] == pytest.approx(0.03, abs=0.005)
    assert fit['rms'] == pytest.approx(0.01, abs=0.002)
    snr, _ = pam.snr_all(t, fit, fr)
    assert min(snr) > 10
    assert np.allclose(pam.model_curve(fit, t), y - fit['residuals'])


def test_identify_alias_harmonic_and_window():
    assert pam.identify(23.2514, [11.6257], 0.1) == "2f1"
    assert "alias" in pam.identify(12.6257, [11.6257], 0.1)
    assert pam.identify(11.6257 + 14.9, [11.6257, 14.9], 0.1) == "f1+f2"
    t = np.concatenate([n + np.linspace(0, 0.25, 50) for n in range(6)])
    f = np.linspace(-3, 3, 6001)
    w = pam.spectral_window(t, f)
    assert w[3000] == pytest.approx(1.0)
    assert w[np.argmin(np.abs(f - 1.0))] > 0.7          # daily alias of single-site data


def test_pdm_and_bls_and_csv(tmp_path):
    t, y, e = _multi_periodic()
    fr = pam.frequency_grid(t, fmin=5, fmax=20, oversample=10)
    th = pam.pdm(t, y, fr)
    assert fr[np.argmin(th)] == pytest.approx(11.6257, abs=0.05)
    # eclipse (box) signal
    tt = np.linspace(0, 20, 3000); yy = np.where(((tt * 0.5) % 1.0) < 0.04, 0.5, 0.0) + 10
    p, _ = pam.bls(tt, yy, np.linspace(0.2, 1.5, 3000), magnitudes=True)     # eclipse = fainter = larger mag
    assert np.linspace(0.2, 1.5, 3000)[np.argmax(p)] == pytest.approx(0.5, abs=0.01)
    # reading Calibra and generic CSV files, combining with mean subtraction
    p1 = tmp_path / "a.csv"
    p1.write_text("file,hjd,mag,mag_err,flag,lc_mode\n" + "".join(
        f"x,{a},{b},0.01,{'OK' if i % 7 else 'SUSPECT'},calibrated\n" for i, (a, b) in enumerate(zip(t[:100], y[:100]))))
    p2 = tmp_path / "b.csv"
    p2.write_text("JD;V;err\n" + "".join(f"{a};{b + 0.3};0.01\n" for a, b in zip(t[100:], y[100:])))
    lc1, lc2 = pam.read_light_curve(str(p1)), pam.read_light_curve(str(p2))
    assert lc1['value_col'] == 'mag' and lc2['time_col'] == 'JD' and lc2['value_col'] == 'V'
    s1 = pam.extract_series(lc1); s2 = pam.extract_series(lc2, 'V')
    assert len(s1[0]) == 100 - 15 and pam.error_column(lc2['columns'], 'V') is None
    tc, yc, ec, idx = pam.combine_series([s1, s2], subtract_mean=True)
    assert abs(yc[idx == 0].mean()) < 1e-9 and abs(yc[idx == 1].mean()) < 1e-9 and ec is not None
    fit = pam.fit_frequencies(tc, yc, [11.6257], [1], ec)
    fit['snr'] = [20.0]; fit['notes'] = ['']
    pam.save_results(str(tmp_path / "r.csv"), fit, {'test': 1})
    assert "f1,1,11.62" in (tmp_path / "r.csv").read_text()


# ---------------------------------------------------------------- plate solving guess (v4.22)
import python.plate_solve as pls

def test_header_position_variants():
    from astropy.io import fits
    h = fits.Header(); h['OBJCTRA'] = '09 36 53.2'; h['OBJCTDEC'] = '+44 04 00'
    ra, dec = pls.header_position(h)
    assert ra == pytest.approx(144.2217, abs=1e-3) and dec == pytest.approx(44.0667, abs=1e-3)
    h = fits.Header(); h['RA'] = 144.2217; h['DEC'] = 44.0667
    assert pls.header_position(h) == pytest.approx((144.2217, 44.0667))
    h = fits.Header(); h['RA'] = '09:36:53.2'; h['DEC'] = '+44:04:00'
    assert pls.header_position(h)[0] == pytest.approx(144.2217, abs=1e-3)
    assert pls.header_position(fits.Header()) is None


def test_astap_command_with_guess(tmp_path, monkeypatch):
    from astropy.io import fits
    fn = tmp_path / "a.fits"; fits.PrimaryHDU(np.zeros((10, 10), np.uint16)).writeto(fn)
    calls = []
    class R:  # no solution produced
        stdout = ""; stderr = ""
    monkeypatch.setattr(pls.subprocess, "run", lambda cmd, **k: (calls.append(cmd), R())[1])
    monkeypatch.setattr(pls.time, "sleep", lambda s: None)
    assert pls.solve_with_astap(str(fn), "astap", 3.0, ra_hint_deg=144.2217, dec_hint_deg=44.0667, fov=1.2) is None
    c = calls[-1]
    assert c[c.index("-ra") + 1] == f"{144.2217 / 15:.6f}" and c[c.index("-spd") + 1] == f"{134.0667:.6f}"
    assert c[c.index("-r") + 1] == "3" and c[c.index("-fov") + 1] == "1.2000"
    pls.solve_with_astap(str(fn), "astap", 3.0)                       # no header position -> blind
    assert calls[-1][calls[-1].index("-r") + 1] == "180" and "-ra" not in calls[-1]
    n = len(calls)
    assert pls.solve_with_astap(str(fn), "astap", 3.0, allow_blind=False) is None
    assert len(calls) == n                                            # skipped, ASTAP not called


def test_solve_series_chains_solutions(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    paths = []
    for k in range(4):
        fn = tmp_path / f"f{k}.fits"; fits.PrimaryHDU(np.zeros((100, 200), np.uint16)).writeto(fn); paths.append(str(fn))
    log = []
    def fake(p, ra_hint_deg=None, dec_hint_deg=None, **kw):
        log.append((os.path.basename(p), None if ra_hint_deg is None else round(ra_hint_deg, 3)))
        if os.path.basename(p) == "f2.fits" and ra_hint_deg is not None and ra_hint_deg > 100.5:
            return None          # the chained guess fails for f2 -> falls back to the user's guess
        if ra_hint_deg is None:
            return None          # no header, no blind
        w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crpix = [100.5, 50.5]
        w.wcs.cdelt = [-0.001, 0.001]; w.wcs.crval = [ra_hint_deg + 1.0, dec_hint_deg]   # pointing drifts by 1 deg
        return w
    res = pls.solve_series(paths, ra_hint_deg=100.0, dec_hint_deg=20.0, chain=True, solver=fake)
    assert [r is not None for r in res] == [True, True, True, True]
    assert log[0] == ("f0.fits", 100.0) and log[1] == ("f1.fits", 101.0)       # 2nd image guided by the 1st
    assert ("f2.fits", 102.0) in log and ("f2.fits", 100.0) in log            # fallback to the user's guess
    res2 = pls.solve_series(paths[:2], chain=False, solver=fake)
    assert res2 == [None, None]


# ---------------------------------------------------------------- v4.23
def test_parse_saaf_gcvs_export(tmp_path):
    p = tmp_path / "GCVS.csv"
    p.write_bytes(('"Var ID","Namn","Typ","RA","Dekl","Max","Min","Band","Epok","Period","Stig%","Sp Type","Nova"\r\n'
                   '"6301090","V1090 Per","DSCT","03 22 42.50","+39 06 34.6","11.600","12.300","CV","2455515.73900","0.1101054100","28","",""\r\n'
                   '"6301015","V1015 Per","RS","03 20 43.87","+39 23 48.2","11.800","(0.20)","*","","12.2800000000","","",""\r\n'
                   '"6301016","V1016 Per","SR","","","11.8","12.0","*","","","","",""\r\n').encode('latin-1'))
    t, probs = fvm.parse_target_list(str(p))
    assert [x['name'] for x in t] == ['V1090 Per', 'V1015 Per']
    assert t[0]['ra'] == pytest.approx(50.67708, abs=1e-4) and t[0]['dec'] == pytest.approx(39.10961, abs=1e-4)
    assert t[0]['type'] == 'DSCT' and t[0]['period'] == pytest.approx(0.11010541) and t[0]['epoch'] == pytest.approx(2455515.739)
    assert t[1]['min'] == pytest.approx(12.0) and t[1]['epoch'] is None        # amplitude in parentheses
    assert len(probs) == 1 and 'V1016 Per' in probs[0] and 'missing' in probs[0]


def test_parse_list_requires_ra_dec_and_reminds_name(tmp_path):
    p = tmp_path / "noradec.csv"
    p.write_text("Namn,Typ,Period\nX Per,M,300\n")
    with pytest.raises(ValueError) as e:
        fvm.parse_target_list(str(p))
    assert "RA and Dec" in str(e.value) and "name, RA and Dec" in str(e.value)
    p2 = tmp_path / "noname.csv"
    p2.write_text("RA,Dec,Type\n03 22 42.5,+39 06 34.6,DSCT\n")
    t, probs = fvm.parse_target_list(str(p2))
    assert t[0]['name'] == 'Star_1' and any('no name column' in x for x in probs)


def test_classify_band_luminance():
    from python.filters import classify_band
    for f in ('L', 'Lum', 'Luminance', 'Baader L', 'Lmag', 'UV/IR-cut'):
        assert classify_band(f, 'Vmag', 'Bmag', 'L') == 'L', f
    assert classify_band('Vmag') == 'V' and classify_band('Bmag') == 'B' and classify_band('R') is None
    assert classify_band('Clear', 'Vmag', 'Bmag', 'Clear') == 'L'          # user keyword
    assert classify_filter('L', 'Vmag', 'Bmag') is None                    # B/V-only function unchanged


# ---------------------------------------------------------------- v4.24: fit quality, O-C
def _hads(t, dphase=None, f0=11.6257, f1=15.0314, a1=0.03):
    ph = f0 * (t - 2461144) + (0 if dphase is None else dphase)
    x = 2 * np.pi * ph
    y = 11.3 - (0.20 * np.sin(x) + 0.08 * np.sin(2 * x + 0.6) + 0.04 * np.sin(3 * x + 1.2) + 0.02 * np.sin(4 * x + 1.8))
    return y + a1 * np.sin(2 * np.pi * f1 * (t - 2461144))


def test_fit_quality_and_period_ratio():
    rng = np.random.default_rng(3)
    t = np.concatenate([2461144.35 + n + np.arange(0, 0.17, 2 / 1440) for n in (0, 1, 2, 5, 6, 9)])
    e = np.full(t.size, 0.008)
    y = _hads(t) + rng.normal(0, 0.008, t.size)
    q1 = pam.fit_quality(pam.fit_frequencies(t, y, [11.6257], [1], e), t, e)
    q4 = pam.fit_quality(pam.fit_frequencies(t, y, [11.6257], [4], e), t, e)
    fit = pam.fit_frequencies(t, y, [11.6257, 15.0314], [4, 1], e)
    q = pam.fit_quality(fit, t, e)
    assert q1['ratio'] > 5 and 1.5 < q4['ratio'] < q1['ratio'] and q['ratio'] == pytest.approx(1.0, abs=0.1)
    assert len(q['nights']) == 6 and q['night_scatter'] < 0.003
    assert "HADS" in pam.period_ratio_note(15.0314, 11.6257)
    assert "HADS" in pam.identify(15.0314, [11.6257], 0.1)
    assert pam.period_ratio_note(1.0, 3.0) == ""


def test_times_of_maximum_and_oc():
    rng = np.random.default_rng(4)
    e0 = 0.008
    # 20 nights over 190 d, slowly increasing period: phase lag = c*E^2 cycles
    t = np.concatenate([2461144.35 + n + np.arange(0, 0.17, 2 / 1440) for n in range(0, 200, 10)])
    E = (t - 2461144) * 11.6257
    lag = 2e-8 * E ** 2
    y = _hads(t - lag / 11.6257) + rng.normal(0, e0, t.size)
    e = np.full(t.size, e0)
    fit = pam.fit_frequencies(t, y, [11.6257, 15.0314], [4, 1], e)
    T0, P, sP = pam.ephemeris(fit, 0)
    assert P == pytest.approx(1 / 11.6257, rel=1e-4)
    mx = pam.times_of_maximum(t, y, fit, 0, e, magnitudes=True, method='template')
    assert len(mx) >= 30 and np.median([m['err'] for m in mx]) < 0.0005
    oc = pam.oc_analysis(mx, T0, P, quadratic=True)
    c_true = 2e-8 / 11.6257                       # O-C = c*E^2 (days)
    assert oc['coef'][2] == pytest.approx(c_true, rel=0.15)
    assert oc['dP_dE'] > 0 and oc['dP_dE'] / oc['dP_dE_err'] > 5
    # constant period: straight line, period recovered
    y2 = _hads(t) + rng.normal(0, e0, t.size)
    fit2 = pam.fit_frequencies(t, y2, [11.6257, 15.0314], [4, 1], e)
    T0, P, _ = pam.ephemeris(fit2, 0)
    oc2 = pam.oc_analysis(pam.times_of_maximum(t, y2, fit2, 0, e), T0, P)
    assert oc2['P_new'] == pytest.approx(1 / 11.6257, abs=3 * oc2['P_new_err'] + 1e-8)
    assert oc2['chi2_red'] < 2.5
    mxp = pam.times_of_maximum(t, y2, fit2, 0, e, method='poly')
    assert len(mxp) > 10


def test_harmonic_family():
    fit = {'freqs': [11.6257, 23.2514, 15.0314, 34.8771, 26.6571]}
    assert pam.harmonic_family(fit, 0) == [0, 1, 3]
    assert pam.harmonic_family(fit, 2) == [2]


# ---------------------------------------------------------------- v4.26: local catalogues
def _make_local_root(tmp_path):
    root = tmp_path / "catalogs"
    rng = np.random.default_rng(1)
    ra = 144.0 + rng.uniform(0.2, 0.8, 60); dec = 44.0 + rng.uniform(0.2, 0.8, 60)
    g = rng.uniform(11, 17.5, 60); r = g - rng.uniform(0.2, 0.9, 60)
    for sub, sel in (("00_m_16", r < 16), ("16_m_17", (r >= 16) & (r < 17)), ("17_m_18", r >= 17)):
        d = root / "ATLAS_refcat2" / sub; d.mkdir(parents=True)
        with open(d / "144+44.rc2", "w") as f:
            for k in np.where(sel)[0]:
                v = [0] * 44; v[0] = int(round(ra[k] * 1e8)); v[1] = int(round(dec[k] * 1e8))
                v[21] = int(round(g[k] * 1000)); v[25] = int(round(r[k] * 1000))
                f.write(",".join(map(str, v)) + "\n")
    (root / "ATLAS_refcat2" / "18_m_19").mkdir()
    (root / "ATLAS_refcat2" / "18_m_19" / "144+44.rc2").write_text("14450000000,4450000000" + ",0" * 19 + ",18500,0,0,0,18200\n")
    d = root / "gaia"; d.mkdir()
    with open(d / "extract.csv", "w") as f:
        f.write("# comment\nsource_id,ra,dec,b,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag\n")
        for k in range(60):
            f.write(f"{5000 + k},{ra[k]},{dec[k]},48.0,{g[k] - 0.3},{g[k] - 0.1},{g[k] - 0.8}\n")
    (root / "GCVS_selected.csv").write_text(
        '"Var ID","Namn","Typ","RA","Dekl","Max","Min","Band","Epok","Period"\n'
        '"1","AA Tst","M","09 37 12.0","+44 30 00","11.0","14.0","V","","300.0"\n'
        '"2","BB Tst","EA","03 22 42.5","+39 06 34","12.0","13.0","V","","2.0"\n')
    return root, ra, dec, g, r


def test_local_catalogs_detection_and_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc
    from python import calibration as cal
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True, 'loaded': True})
    root, ra, dec, g, r = _make_local_root(tmp_path)
    found = lc.set_root(str(root))
    assert set(found) >= {'ATLAS', 'GAIA_DR3', 'GCVS'} and found['ATLAS']['layout'] == 'tiles'
    assert os.path.exists(paths.LOCAL_CATALOGS_FILE)
    class NoViz:                                  # any online access fails the test
        def __init__(self, *a, **k): raise AssertionError("went online")
    monkeypatch.setattr(cal, "Vizier", NoViz)
    st = cal.fetch_online_catalog(144.5, 44.5, 30, "ATLAS refcat2", verbose=False)
    k = int(np.argmin(np.abs(np.array(ra) - st[0]['ra_deg'])))
    v_exp, _ = cal.atlas_to_vb(round(g[k], 3), round(r[k], 3))
    assert st[0]['V_mag'] == pytest.approx(v_exp, abs=1e-6)
    assert len(st) == sum(1 for kk in range(60) if cal.atlas_to_vb(g[kk], r[kk])[0] < 18)    # 18_m_19 ignored
    assert all(s['gaia_id'].startswith('Gaia 50') for s in st)                               # local Gaia cross-match
    gs = cal.fetch_online_catalog(144.5, 44.5, 30, "GAIA_DR3", verbose=False)
    assert len(gs) > 0 and gs[0]['cat_id'].startswith('Gaia ')
    assert lc.query("ATLAS", 10.0, 10.0, 10) is None            # not covered -> caller goes online
    gc = lc.gcvs_in_field(144.3, 44.3, 60)
    assert [t['name'] for t in gc] == ['AA Tst'] and gc[0]['period'] == 300.0
    from python.field_variables import query_field_variables
    assert [t['name'] for t in query_field_variables('GCVS', 144.3, 44.3, 60)] == ['AA Tst']
    lc.set_options(allow_online=False)
    assert cal.fetch_online_catalog(10, 10, 15, "ATLAS", verbose=False) == []


# ---------------------------------------------------------------- v4.27: master frames, calibration
def _cal_set(tmp_path, n_bias=5, n_flat=5, darks=False):
    from astropy.io import fits
    rng = np.random.default_rng(7)
    ny, nx = 64, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    bias = 500 + 20 * np.sin(xx / 7.0)                                  # fixed pattern
    flat_true = {'L': 1 - 0.3 * ((xx - 40) ** 2 + (yy - 32) ** 2) / 2600.0, 'Vmag': 1 - 0.2 * (xx / 80.0)}
    hot = np.zeros((ny, nx)); hot[10, 10] = 800.0                        # dark current hot pixel (per 60 s)
    def hdr(typ, filt=None, exp=0.001):
        h = fits.Header(); h['IMAGETYP'] = typ; h['XBINNING'] = 1; h['YBINNING'] = 1; h['GAIN'] = 100; h['OFFSET'] = 50
        h['EXPTIME'] = exp; h['CCD-TEMP'] = -10.0
        if filt: h['FILTER'] = filt
        return h
    def write(p, data, h):
        p.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(np.clip(data, 0, 65535).astype(np.uint16), header=h).writeto(p)
    for k in range(n_bias):
        write(tmp_path / "calib" / "Bias" / f"b{k}.fits", bias + rng.normal(0, 5, (ny, nx)), hdr('Bias Frame'))
    for filt in ('L', 'Vmag'):
        for k in range(n_flat):
            write(tmp_path / "calib" / f"Flats_{filt}" / f"f{k}.fits",
                  bias + 20000 * (1 + 0.1 * k) * flat_true[filt] + rng.normal(0, 50, (ny, nx)), hdr('Flat Field', filt, 2.0))
    if darks:
        for k in range(5):
            write(tmp_path / "calib" / "Darks" / f"d{k}.fits", bias + hot + rng.normal(0, 5, (ny, nx)), hdr('Dark Frame', None, 60.0))
    star = 3000 * np.exp(-((xx - 50) ** 2 + (yy - 20) ** 2) / 8.0) + 300
    light = bias + hot + star * flat_true['L'] + rng.normal(0, 3, (ny, nx))
    lp = tmp_path / "lights" / "light1.fits"
    write(lp, light, hdr('Light Frame', 'Lum', 60.0))
    return str(tmp_path / "calib"), str(lp), bias, flat_true, star, hot


def test_build_masters_and_reuse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from python import master_frames as mf
    folder, lp, bias, flat_true, star, hot = _cal_set(tmp_path)
    res = mf.build_masters(folder, log=lambda *a: None)
    names = sorted(os.path.basename(r['path']) for r in res)
    assert names == ['master_bias_80x64_bin1_g100_o50.fits', 'master_flat_L_80x64_bin1_g100_o50.fits',
                     'master_flat_V_80x64_bin1_g100_o50.fits']
    assert all(r['status'] == 'built' for r in res)
    from astropy.io import fits
    # memmap=False: a memory-mapped master cannot be overwritten on Windows (WinError 32)
    mb = fits.getdata(os.path.join(folder, 'Bias', names[0]), memmap=False)
    assert np.abs(mb - bias).max() < 12                                   # 5 frames, noise 5
    mfl = fits.getdata(os.path.join(folder, 'Flats_L', names[1]), memmap=False)
    ref = flat_true['L'] / np.median(flat_true['L'])
    assert np.abs(mfl - ref).max() < 0.01 and fits.getheader(os.path.join(folder, 'Flats_L', names[1]))['NCOMBINE'] == 5
    res2 = mf.build_masters(folder, log=lambda *a: None)
    assert all(r['status'] == 'reused' for r in res2)
    import shutil
    shutil.copy(os.path.join(folder, 'Bias', 'b0.fits'), os.path.join(folder, 'Bias', 'b5.fits'))
    res3 = {os.path.basename(r['path']): r['status'] for r in mf.build_masters(folder, log=lambda *a: None)}
    assert res3['master_bias_80x64_bin1_g100_o50.fits'] == 'built'


def test_calibration_on_the_fly_and_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from python import master_frames as mf
    from astropy.io import fits
    folder, lp, bias, flat_true, star, hot = _cal_set(tmp_path, darks=True)
    mf.build_masters(folder, log=lambda *a: None)
    monkeypatch.setattr(mf, "_STATE", dict(mf._STATE, loaded=True))
    mf.configure(save=False, on_the_fly=False, folder=folder, mode='auto', use_darks=False, sat_limit=63000)
    raw = fits.getdata(lp).astype(float); h = fits.getheader(lp)
    d, done = mf.apply_on_the_fly(raw, h, lp)
    assert done is False and d is raw and mf.state_tag() == ''
    mf.configure(save=False, on_the_fly=True)
    d, done = mf.apply_on_the_fly(raw, h, lp)
    fm = float(np.median(flat_true['L']))                               # flats are normalised to median 1
    assert done and abs(np.median(d) - 300 * fm) < 5                    # flat sky after bias + flat
    assert np.std(d[30:40, 0:20]) < 10                                  # vignetting removed
    assert abs(d[10, 10] - 300 * fm) > 400                              # hot pixel left without darks
    assert mf.state_tag().startswith('cal')
    mf.configure(save=False, use_darks=True)
    d2, _ = mf.apply_on_the_fly(raw, h, lp)
    assert abs(d2[10, 10] - 300 * fm) < 30                              # dark removes the hot pixel
    peak = d2[20, 50] - 300 * fm
    assert peak == pytest.approx(3000 * fm, rel=0.03)                   # flat-field corrected star
    out, msg = mf.calibrate_to_file(lp)
    assert out and os.path.basename(out) == 'cal_light1.fits' and 'calibrated' in out
    assert fits.getheader(out)['FLATFILE'].startswith('master_flat_L')
    out2, msg2 = mf.calibrate_to_file(lp)
    assert 'reused' in msg2
    d3, done3 = mf.apply_on_the_fly(fits.getdata(out), fits.getheader(out), out)
    assert done3 is False                                               # already calibrated
    h2 = h.copy(); h2['FILTER'] = 'R'
    d4, done4 = mf.apply_on_the_fly(raw, h2, lp)
    assert done4 is False                                               # no flat for R -> raw, warning
    assert mf.frame_type({'IMAGETYP': 'Flat Field'}) == 'FLAT' and mf.frame_type({'IMAGETYP': 'Dark Frame'}) == 'DARK'
    assert mf.frame_type({}, '/x/FlatDarks/fd1.fits') == 'DARKFLAT'
    assert mf.skip_reason({'PEDESTAL': 0}) == ''                        # raw MaxIm frames have PEDESTAL


def test_catalog_overview_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
    import catalog_overview as co
    root, *_ = _make_local_root(tmp_path)
    (root / "ATLAS_refcat2" / "00_m_16" / "010-45.rc2").write_text("1000000000,-4450000000" + ",0" * 19 + ",12000,0,0,0,11500\n")
    (root / "gaia" / "south.csv").write_text("source_id,ra,dec,phot_g_mean_mag\n1,10,-50,12\n")
    assert co.main([str(root), '--report', 'ov.csv', '--reduce', str(tmp_path / 'red')]) == 0
    # Windows: the report has native paths ('\'); one backslash, not two (v5.0.1)
    rows = {r['file'].replace('\\', '/'): r['verdict'] for r in csv.DictReader(open('ov.csv'))}
    assert rows['ATLAS_refcat2/00_m_16/010-45.rc2'].startswith('DELETE: all Dec')
    assert rows['ATLAS_refcat2/18_m_19/144+44.rc2'].startswith('DELETE: all fainter')
    assert rows['ATLAS_refcat2/00_m_16/144+44.rc2'] == 'keep'
    assert rows['gaia/south.csv'].startswith('DELETE') and rows['gaia/extract.csv'] == 'keep'
    assert (tmp_path / 'red' / 'gaia' / 'extract.csv').exists() and not (tmp_path / 'red' / 'gaia' / 'south.csv').exists()
    assert (root / "gaia" / "south.csv").exists()                         # nothing deleted


def test_master_frames_from_type_folders(tmp_path, monkeypatch):
    """v4.29: one folder per type - names and IMAGETYP do not matter, masters written there,
    flats bias-corrected once, a rebuilt bias forces new flats, a 'Light' header in the bias folder
    is used as bias with a warning (v4.33)."""
    monkeypatch.chdir(tmp_path)
    from python import master_frames as mf
    from astropy.io import fits
    rng = np.random.default_rng(3)
    ny, nx = 48, 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    bias = 500 + 0.5 * xx
    flat_true = 1 - 0.25 * ((xx - 30) ** 2 + (yy - 24) ** 2) / 1500.0
    def write(p, data, **kw):
        p.parent.mkdir(parents=True, exist_ok=True)
        h = fits.Header(); h['XBINNING'] = 1; h['YBINNING'] = 1
        for k, v in kw.items():
            h[k] = v
        fits.PrimaryHDU(np.clip(data, 0, 65535).astype(np.uint16), header=h).writeto(p)
    for k in range(5):                                       # no IMAGETYP, meaningless names
        write(tmp_path / "B1" / f"IMG_{k:04d}.fits", bias + rng.normal(0, 4, (ny, nx)), EXPTIME=0.001)
    write(tmp_path / "B1" / "IMG_9999.fits", bias + rng.normal(0, 4, (ny, nx)), IMAGETYP='Light Frame',
          EXPTIME=0.001)                                     # bias with a wrong header: used, with a warning
    for k in range(4):                                       # filter in header
        write(tmp_path / "F1" / f"x{k}.fit", bias + 25000 * flat_true + rng.normal(0, 40, (ny, nx)), EXPTIME=2.0, FILTER='V')
    for k in range(4):                                       # filter only in the name
        write(tmp_path / "F1" / f"flat_B_{k}.fit", bias + 15000 * flat_true + rng.normal(0, 40, (ny, nx)), EXPTIME=3.0)
    logs = []
    res = mf.build_masters(bias_folder=str(tmp_path / "B1"), flat_folder=str(tmp_path / "F1"), log=logs.append)
    names = sorted(os.path.basename(r['path']) for r in res)
    assert names == ['master_bias_60x48_bin1_gx_ox.fits', 'master_flat_B_60x48_bin1_gx_ox.fits',
                     'master_flat_V_60x48_bin1_gx_ox.fits']
    assert any('IMG_9999' in m and 'WARNING' in m and 'used as bias' in m for m in logs)
    assert (tmp_path / "B1" / names[0]).exists() and (tmp_path / "F1" / names[2]).exists()
    # memmap=False: a memory-mapped master cannot be overwritten on Windows (WinError 32)
    mb = fits.getdata(tmp_path / "B1" / names[0], memmap=False)
    assert np.abs(mb - bias).max() < 10 and fits.getheader(tmp_path / "B1" / names[0])['NCOMBINE'] == 6
    fv = fits.getdata(tmp_path / "F1" / names[2], memmap=False)
    assert np.abs(fv - flat_true / np.median(flat_true)).max() < 0.01  # bias removed before normalising
    assert fits.getheader(tmp_path / "F1" / names[2])['FLATCORR'].startswith('master_bias')
    # light: bias subtracted once, divided by the normalised flat -> flat sky
    write(tmp_path / "L" / "l1.fits", bias + 400 * flat_true + rng.normal(0, 2, (ny, nx)), EXPTIME=30.0, FILTER='V',
          IMAGETYP='Light Frame')
    monkeypatch.setattr(mf, "_STATE", dict(mf._STATE, loaded=True))
    mf.configure(save=False, on_the_fly=True, folder='', bias_folder=str(tmp_path / "B1"),
                 flat_folder=str(tmp_path / "F1"), dark_folder='', mode='auto', use_darks=False)
    lp = str(tmp_path / "L" / "l1.fits")
    d, done = mf.apply_on_the_fly(fits.getdata(lp).astype(float), fits.getheader(lp), lp)
    assert done
    level = 400 * np.median(flat_true)
    assert abs(np.median(d) - level) < 3 and np.std(d) < 6            # no residual vignetting / bias pattern
    # second run: everything reused; a changed bias set rebuilds the bias AND the flats
    assert all(r['status'] == 'reused' for r in mf.build_masters(bias_folder=str(tmp_path / "B1"),
                                                                  flat_folder=str(tmp_path / "F1"), log=lambda *a: None))
    write(tmp_path / "B1" / "IMG_0005.fits", bias + rng.normal(0, 4, (ny, nx)), EXPTIME=0.001)
    st = {os.path.basename(r['path']): r['status'] for r in mf.build_masters(
        bias_folder=str(tmp_path / "B1"), flat_folder=str(tmp_path / "F1"), log=lambda *a: None)}
    assert set(st.values()) == {'built'}


def test_name_lookup_local_gcvs(tmp_path, monkeypatch):
    """v4.30: names are looked up in the local GCVS list first (no internet needed)."""
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc, names
    from astropy.coordinates import SkyCoord
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True, 'loaded': True})
    root, *_ = _make_local_root(tmp_path)
    with open(root / "GCVS_selected.csv", "a") as f:
        f.write('"3","HU Aqr","AM","21 07 58.2","-05 17 40","14.6","18.5","V","",""\n')
    lc.set_root(str(root))
    def no_net(*a, **k):
        raise AssertionError("went online")
    monkeypatch.setattr(SkyCoord, "from_name", staticmethod(no_net))
    for nm in ("HU Aqr", "hu aqr", "HU  AQR", "HU_Aqr"):
        c = names.from_name(nm)
        assert c.ra.deg == pytest.approx(316.99250, abs=1e-4) and c.dec.deg == pytest.approx(-5.29444, abs=1e-4)
    assert names.LAST_SOURCE['source'] == 'local GCVS list'
    t = lc.gcvs_lookup("bb tst")
    assert t['type'] == 'EA' and t['source'] == 'GCVS (local)'
    assert lc.name_key("V 1500 Cyg") == lc.name_key("V1500 Cyg")
    with pytest.raises(AssertionError):                 # unknown name -> online (SIMBAD)
        names.from_name("SS Cyg")
    with pytest.raises(ValueError):                     # v4.34: SIMBAD failed -> skipped for a while
        names.from_name("SS Cyg")
    from python import online
    online.ok('simbad')
    lc._STATE['allow_online'] = False
    with pytest.raises(ValueError):                     # online switched off -> clear message
        names.from_name("SS Cyg")
    lc._STATE['allow_online'] = True
    lc._STATE['use_local'] = False                      # local copies off -> straight to SIMBAD
    with pytest.raises(AssertionError):
        names.from_name("HU Aqr")
    from python import field_variables as fv
    lc._STATE['use_local'] = True
    tg = [{'name': 'HU Aqr', 'ra': None, 'dec': None}]
    assert fv.resolve_missing_coordinates(tg) == [] and tg[0]['ra'] == pytest.approx(316.9925, abs=1e-4)


APASS_DR10_SAMPLE = (
    'radeg,raerr("),decdeg,decerr("),Johnson_V (V),Verr,Vnobs,Johnson_B (B),Berr,Bnobs,Sloan_u (SU),SUerr,SUnobs,'
    'Sloan_g (SG),SGerr,SGnobs,Sloan_r (SR),SRerr,SRnobs,Sloan_i (SI),SIerr,SInobs,Sloan_z (SZ),SZerr,SZnobs,'
    'PanSTARRS_Y (Y),Yerr,Ynobs\n'
    '316.966311,0.367,-5.29276,0.13,NA,NA,0,NA,NA,0,NA,NA,0,17.906,0.153,1,16.621,0.137,1,NA,NA,0,NA,NA,0,NA,NA,0\n'
    '316.967633,0.258,-5.3209,0.171,13.69,0.06,42,14.368,0.068,42,NA,NA,0,14,0.04,42,13.487,0.108,42,13.258,0.163,42,13.301,0.066,3,NA,NA,0\n'
    '316.982072,0.49,-5.271034,0.408,15.133,0.113,41,15.719,0.144,41,NA,NA,0,15.387,0.07,42,14.924,0.156,42,14.666,0.259,21,NA,NA,0,NA,NA,0\n')


def test_apass_dr10_download_files(tmp_path, monkeypatch):
    """v4.31: CSV files from archive.aavso.org/download-apass-data (APASS DR10): column names
    'Johnson_V (V)', 'NA' for missing values, overlapping downloads without duplicates,
    recognised by content also in a folder without 'apass' in its name."""
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc, calibration as cal
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True, 'loaded': True})
    d = tmp_path / "cat" / "dr10_downloads"; d.mkdir(parents=True)
    (d / "apass_316.9925_-5.29444444444444_0.05_20_none.csv").write_text(APASS_DR10_SAMPLE)
    (d / "apass_316.99_-5.30_0.10_20_none.csv").write_text(APASS_DR10_SAMPLE)          # overlapping box
    found = lc.set_root(str(tmp_path / "cat"))
    assert found['APASS']['layout'] == 'table'
    rows = lc.query("APASS", 316.975, -5.29, 3.0)
    assert len(rows) == 3                                                           # no duplicates
    st = cal.stars_from_local_rows(rows, "APASS")
    assert len(st) == 2 and {round(s['V_mag'], 3) for s in st} == {13.69, 15.133}   # 'NA' row skipped
    assert next(s for s in st if abs(s['V_mag'] - 13.69) < 1e-6)['B_mag'] == pytest.approx(14.368)
    assert lc._norm_col('Johnson_V (V)') == 'johnson_v' and lc._norm_col('ra(deg)') == 'ra'


def test_apass_dr10_download_and_auto(tmp_path, monkeypatch):
    """v4.32: APASS DR10 download (server answer simulated), reuse, automatic download for a
    missing field in fetch_online_catalog, pause after a failure, fields from images."""
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc, calibration as cal, apass_dr10 as ad
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True,
                                       'apass_auto': False, 'loaded': True})
    (tmp_path / "cat").mkdir()
    lc.set_root(str(tmp_path / "cat"))
    calls = []
    def fake_post(params, timeout):
        calls.append(params)
        return APASS_DR10_SAMPLE.encode(), 'attachment;filename=apass_316.9925_-5.29_0.2_20_none.csv'
    monkeypatch.setattr(ad, "_http_post", fake_post)
    assert ad.sexagesimal(316.9925, -5.294444) == ('21:07:58.20', '-05:17:40.0')
    path, n, msg = ad.download_field(316.975, -5.29, 0.2, log=lambda *a: None)
    assert n == 3 and os.path.basename(path) == 'apass_316.9925_-5.29_0.2_20_none.csv'
    assert os.path.dirname(path) == os.path.join(str(tmp_path / "cat"), "APASS")
    assert calls[0]['outtype'] == '1' and calls[0]['radius'] == '0.2' and calls[0]['ra'].startswith('21:07:54')
    assert lc.get_config()['catalogs']['APASS']['layout'] == 'table'
    assert ad.download_field(316.975, -5.29, 0.1, log=lambda *a: None)[1] is None and len(calls) == 1   # reused
    class NoViz:
        def __init__(self, *a, **k): raise AssertionError("went to VizieR")
    monkeypatch.setattr(cal, "Vizier", NoViz)
    st = cal.fetch_online_catalog(316.975, -5.29, 3, "APASS DR9", verbose=False)
    assert len(st) == 2                                                     # from the downloaded file
    # automatic: a field that is not local is downloaded once
    def post2(params, timeout):
        calls.append(params)
        return APASS_DR10_SAMPLE.replace('316.9', '100.9').replace(',-5.', ',+20.').encode(), ''
    monkeypatch.setattr(ad, "_http_post", post2)
    monkeypatch.setattr(ad, "MIN_INTERVAL", 0.0)
    lc.set_options(apass_auto=True)
    st2 = cal.fetch_online_catalog(100.975, 20.29, 3, "APASS DR9", verbose=False)
    assert len(calls) == 2 and len(st2) == 2 and st2[0]['dec_deg'] > 20
    assert ad.radius_for_field(3) == ad.MIN_RADIUS and ad.radius_for_field(60) == 1.2
    # failure (server answers HTML) -> no further automatic tries for a while
    monkeypatch.setattr(ad, "_http_post", lambda p, t: (calls.append(p) or (b"<html>Error</html>", '')))
    assert ad.auto_download(200.0, 10.0, 10, log=lambda *a: None) is False
    assert ad.auto_download(201.0, 10.0, 10, log=lambda *a: None) is False and len(calls) == 3
    with pytest.raises(RuntimeError):
        ad.download_field(202.0, 10.0, 0.2, log=lambda *a: None)
    # fields from plate-solved images: two pointings -> two fields
    from astropy.io import fits
    from astropy.wcs import WCS
    def img(name, ra, dec):
        w = WCS(naxis=2); w.wcs.crpix = [50.5, 40.5]; w.wcs.crval = [ra, dec]
        w.wcs.cdelt = [-0.005, 0.005]; w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        p = tmp_path / name
        fits.PrimaryHDU(np.zeros((80, 100), np.float32), header=w.to_header()).writeto(p)
        return str(p)
    paths = [img('a1.fits', 150.0, 30.0), img('a2.fits', 150.02, 30.01), img('b.fits', 155.0, 30.0)]
    fits.PrimaryHDU(np.zeros((10, 10), np.float32)).writeto(tmp_path / 'nowcs.fits')
    fields, no_wcs = ad.fields_from_images(paths + [str(tmp_path / 'nowcs.fits')])
    assert len(fields) == 2 and no_wcs == [str(tmp_path / 'nowcs.fits')]
    f1 = next(f for f in fields if f[3] == 2)
    assert f1[0] == pytest.approx(150.01, abs=0.01) and ad.MIN_RADIUS <= f1[2] < 0.5


def test_online_service_pause(tmp_path, monkeypatch):
    """v4.34: after a failed VizieR request further requests are skipped (no long timeouts)."""
    monkeypatch.chdir(tmp_path)
    from python import online, calibration as cal, local_catalogs as lc
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True,
                                       'apass_auto': False, 'loaded': True})
    n = {'calls': 0}
    class DeadViz:
        def __init__(self, *a, **k): pass
        def query_region(self, *a, **k):
            n['calls'] += 1
            raise ConnectionError("timed out")
    monkeypatch.setattr(cal, "Vizier", DeadViz)
    assert cal.fetch_online_catalog(10.0, 20.0, 5, "ATLAS refcat2", verbose=False) == []
    assert not online.available('vizier') and n['calls'] == 1
    assert cal.fetch_online_catalog(11.0, 20.0, 5, "ATLAS refcat2", verbose=False) == []
    assert cal.fetch_vsx_catalog(11.0, 20.0, 5, verbose=False) == []
    assert n['calls'] == 1                                   # skipped, no second timeout
    monkeypatch.setattr(online, "PAUSE", 0.0)
    assert online.available('vizier')                        # pause over -> tried again


def test_offline_first_catalogues(tmp_path, monkeypatch):
    """v4.46: offline first - chosen catalogue's local copy, then cache, then the local
    ATLAS-RefCat2 copy as substitute, VizieR only after that; old online caches no longer
    hide the local copy; Gaia IDs not fetched online; VSX falls back to the local GCVS list."""
    import shutil
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc, calibration as cal, online
    import astroquery.vizier as aqv
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True,
                                       'apass_auto': False, 'loaded': True})
    monkeypatch.setattr(online, "_DOWN", {}); monkeypatch.setattr(online, "_FAILS", {})
    root, ra, dec, g, r = _make_local_root(tmp_path)
    shutil.rmtree(root / "gaia")                                   # RefCat2 + GCVS list only (Stephan's case)
    found = lc.set_root(str(root))
    assert 'GAIA_DR3' not in found and 'ATLAS' in found and lc.prefer_offline()
    n = {'viz': 0}
    class CountViz:
        def __init__(self, *a, **k): pass
        def query_region(self, *a, **k):
            n['viz'] += 1
            raise ConnectionError("VizieR down")
    monkeypatch.setattr(cal, "Vizier", CountViz); monkeypatch.setattr(aqv, "Vizier", CountViz)
    n_ref = sum(1 for kk in range(60) if cal.atlas_to_vb(g[kk], r[kk])[0] < 18)
    for name in ("GAIA_DR3", "APASS DR9", "Landolt Standard Star Catalogue"):
        st = cal.fetch_online_catalog(144.5, 44.5, 30, name, verbose=False)
        assert len(st) == n_ref and st[0]['cat_source'].startswith("local ATLAS copy instead of")
        assert st[0]['cat_id'].startswith("ATLAS") and st[0]['gaia_id'] == ''
    assert n['viz'] == 0                                           # no online query at all (also no Gaia IDs)
    rows, used = lc.query_best("GAIA_DR3", 144.5, 44.5, 10)
    assert used == 'ATLAS' and rows
    assert lc.query_best("GAIA_DR3", 10.0, 10.0, 10) == (None, None)
    # an old online cache file of the same catalogue no longer wins over the local copy
    os.makedirs(paths.CATALOG_CACHE_DIR, exist_ok=True)
    cal.save_to_cache([{'id': 'x', 'cat_id': 'OLD', 'ra_deg': 144.5, 'dec_deg': 44.5, 'V_mag': 9.0, 'B_mag': 9.5}],
                      144.5, 44.5, 30, "ATLAS REFCAT2", verbose=False)
    st = cal.get_ref_stars("ATLAS refcat2", 144.5, 44.5, radius_arcmin=30, verbose=False)
    assert len(st) == n_ref and all(s['cat_id'] != 'OLD' for s in st)
    # ...but a cached (online) result of the chosen catalogue comes before the substitute
    cal.save_to_cache([{'id': 'y', 'cat_id': 'APASS 1', 'ra_deg': 144.5, 'dec_deg': 44.5, 'V_mag': 12.0, 'B_mag': 12.5}],
                      144.5, 44.5, 30, "APASS DR9", verbose=False)
    st = cal.fetch_online_catalog(144.5, 44.5, 30, "APASS DR9", verbose=False)
    assert [s['cat_id'] for s in st] == ['APASS 1'] and st[0]['cat_source'].startswith("cache")
    # option off: VizieR first, the local substitute only as last resort
    lc.set_options(prefer_offline=False)
    assert not lc.prefer_offline()
    st = cal.fetch_online_catalog(144.6, 44.6, 20, "GAIA_DR3", verbose=False)
    assert n['viz'] >= 1 and st and st[0]['cat_source'].startswith("local ATLAS")
    lc.set_options(prefer_offline=True)
    # not covered locally, online off -> nothing (and no query)
    lc.set_options(allow_online=False)
    k0 = n['viz']
    assert cal.fetch_online_catalog(10.0, 10.0, 10, "GAIA_DR3", verbose=False) == [] and n['viz'] == k0
    # VSX: no local copy and VizieR off -> variables of the local GCVS list
    vs = cal.get_vsx_stars(144.3, 44.3, 60, verbose=False)
    assert [v['id'] for v in vs] == ['AA Tst'] and vs[0]['Type'] == 'M' and vs[0]['Period'] == 300.0
    assert not any(f.startswith("VSX_") for f in os.listdir(paths.CATALOG_CACHE_DIR))   # not cached as VSX
    marked = cal.mark_variable_stars([{'ra_deg': 144.3, 'dec_deg': 44.5}], 144.3, 44.3, 60, verbose=False)
    assert marked[0]['is_variable'] and marked[0]['vsx_name'] == 'AA Tst'
    # online back-off: the pause grows with repeated failures
    online.ok('vizier')
    online.failed('vizier', 'x'); p1 = online.pause('vizier')
    online._DOWN.pop('vizier'); online.failed('vizier', 'x'); p2 = online.pause('vizier')
    for _ in range(5):
        online.failed('vizier', 'x')
    assert p1 == online.PAUSE and p2 == 2 * online.PAUSE and online.pause('vizier') == online.PAUSE_MAX
    online.ok('vizier')
    assert online.available('vizier') and online.pause('vizier') == online.PAUSE


def test_unreadable_local_catalogue_and_vsx_gcvs_first(tmp_path, monkeypatch):
    """v5.0.1: a local table catalogue that cannot be read (Cartes du Ciel binary Gaia,
    'CDCSTAR1' .dat files) is marked once and then skipped (before: every look-up re-checked
    all files, ~30 s); with 'prefer local catalogues' VSX comes from the local GCVS list
    without asking VizieR, even when online queries are allowed."""
    import shutil, json
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc, calibration as cal, online
    import astroquery.vizier as aqv
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True,
                                       'apass_auto': False, 'loaded': True})
    monkeypatch.setattr(online, "_DOWN", {}); monkeypatch.setattr(online, "_FAILS", {})
    root, ra, dec, g, r = _make_local_root(tmp_path)
    shutil.rmtree(root / "gaia")
    gd = root / "Gaia DR3" / "gaia" / "gaia1"; (gd / "n0000").mkdir(parents=True)
    (gd / "gaia.hdr").write_bytes(bytes(2) + b"CDCSTAR1GaiaGaia DR3" + bytes(200))
    for k in range(3):
        (gd / "n0000" / f"gaia00{k}.dat").write_bytes(bytes(range(256)) * 4)
    found = lc.set_root(str(root))
    assert 'GAIA_DR3' in found and lc.has_local("GAIA_DR3")
    n = {'viz': 0}
    class CountViz:
        def __init__(self, *a, **k): pass
        def query_region(self, *a, **k):
            n['viz'] += 1
            raise ConnectionError("VizieR down")
    monkeypatch.setattr(cal, "Vizier", CountViz); monkeypatch.setattr(aqv, "Vizier", CountViz)
    assert lc.query("GAIA_DR3", 144.5, 44.5, 20) is None              # not readable -> not local
    assert lc.get_config()['catalogs']['GAIA_DR3'].get('unreadable') and not lc.has_local("GAIA_DR3")
    saved = json.load(open(paths.LOCAL_CATALOGS_FILE, encoding='utf-8'))
    assert saved['catalogs']['GAIA_DR3']['unreadable'] is True        # remembered for the next start
    assert any("NOT READABLE" in line for line in lc.summary_lines())
    called = []
    monkeypatch.setattr(lc, "_table_index", lambda *a, **k: called.append(1) or {})
    assert lc.query("GAIA_DR3", 144.5, 44.5, 20) is None and not called   # skipped without checking files
    st = cal.fetch_online_catalog(144.5, 44.5, 30, "GAIA_DR3", verbose=False)
    assert st and st[0]['cat_source'].startswith("local ATLAS copy instead of") and n['viz'] == 0
    lc.set_root(str(root))                                            # rescan clears the mark
    assert not lc.get_config()['catalogs']['GAIA_DR3'].get('unreadable')
    # VSX: online allowed and VizieR "available", but the local GCVS list comes first
    assert lc.online_allowed() and online.available('vizier')
    fetched = []
    monkeypatch.setattr(cal, "fetch_vsx_catalog", lambda *a, **k: fetched.append(1) or [])
    vs = cal.get_vsx_stars(144.3, 44.3, 60, verbose=False)
    assert [v['id'] for v in vs] == ['AA Tst'] and not fetched
    lc.set_options(prefer_offline=False)                              # option off: VizieR (VSX) first
    cal.get_vsx_stars(144.3, 44.3, 60, verbose=False)
    assert fetched


def test_aavso_sequences_store(tmp_path, monkeypatch):
    """v4.35: AAVSO sequences stored per variable, names matched loosely, cross-match kept."""
    monkeypatch.chdir(tmp_path)
    from python import aavso_sequences as aseq
    stars = [{'auid': '000-AAA-001', 'name': '120', 'ra': 316.9, 'dec': -5.3, 'V_mag': 12.0, 'B_mag': 12.6,
              'cat_v': 12.01, 'cat_b': float('nan'), 'cat_match': True}]
    p = aseq.save("HU Aqr", stars, 30.0, "ATLAS refcat2")
    assert os.path.basename(p) == "huaqr.json"
    d = aseq.load("hu  aqr")
    assert d and d['stars'][0]['auid'] == '000-AAA-001' and d['fov_arcmin'] == 30.0 and d['catalog'] == "ATLAS refcat2"
    assert np.isnan(d['stars'][0]['cat_b']) and aseq.age_days(d) < 0.01
    assert aseq.load("SS Cyg") is None
    d['stars'][0]['cat_v'] = 12.5
    aseq.update_crossmatch("HU_Aqr", d['stars'], "APASS DR9")
    d2 = aseq.load("HU Aqr")
    assert d2['catalog'] == "APASS DR9" and d2['stars'][0]['cat_v'] == 12.5 and d2['fetched'] == d['fetched']
    assert [t for t, *_ in aseq.list_stored()] == ["HU Aqr"]


def test_instrument_setups(tmp_path, monkeypatch):
    """v4.36: instrument setups - store/load, settings snapshot, site/filter sync, derived
    pixel scale and field, values from a FITS header, recognising frames of a setup."""
    monkeypatch.chdir(tmp_path)
    from python import instruments as ins
    from astropy.io import fits
    s = ins.template("Refractor 102")
    s['instrument'].update(focal_length_mm=714.0, aperture_mm=102.0, pixel_um=3.76, sensor_x=6248, sensor_y=4176,
                           camera='ZWO ASI6200MM Pro', telescope='TS 102 APO')
    s['site'].update(name='Uppsala', lat=59.86, lon=17.64)
    s['filters']['V']['fits_name'] = 'Vmag'; s['filters']['V']['nina_position'] = 6
    ins.snapshot(s, {'ccd_gain': 0.25, 'saturation_limit': 60000.0, 'ts_lat': 59.8, 'not_a_setup_key': 1,
                     'filter_b_keyword': 'Bmag'}, sync_descriptive=False)
    ins.save(s)
    d = ins.derived(s)
    assert d['pixel_scale'] == pytest.approx(1.086, abs=0.002) and d['fov_x_deg'] == pytest.approx(1.885, abs=0.005)
    assert d['focal_ratio'] == pytest.approx(7.0, abs=0.01)
    s2 = ins.load("Refractor 102")
    assert s2['settings'] == {'ccd_gain': 0.25, 'saturation_limit': 60000.0, 'ts_lat': 59.8, 'filter_b_keyword': 'Bmag'}
    v = ins.values_to_apply(s2)
    assert v['ts_lat'] == 59.86 and v['ts_lon'] == 17.64 and v['filter_v_keyword'] == 'Vmag' and v['ccd_gain'] == 0.25
    ins.snapshot(s2, {'ts_lat': 59.85, 'filter_v_keyword': 'V'})          # edits in the Settings tab
    assert s2['site']['lat'] == 59.85 and s2['filters']['V']['fits_name'] == 'V'
    assert "1.09\"/px" in ins.summary(s) and "Uppsala" in ins.summary(s)
    # a second setup and frames from it
    m = ins.template("MWT"); m['instrument'].update(focal_length_mm=2900.0, sensor_x=4096, sensor_y=4096, pixel_um=9.0)
    ins.save(m)
    assert ins.list_names() == ["MWT", "Refractor 102"]
    h = fits.Header()
    h['NAXIS'] = 2; h['NAXIS1'] = 3124; h['NAXIS2'] = 2088; h['XBINNING'] = 2; h['XPIXSZ'] = 7.52
    h['FOCALLEN'] = 714; h['INSTRUME'] = 'ZWO ASI6200MM Pro'; h['SITELAT'] = '59 51 36'; h['FILTER'] = 'Vmag'
    fh = ins.from_header(h)
    assert fh['instrument']['sensor_x'] == 6248 and fh['instrument']['pixel_um'] == pytest.approx(3.76)
    assert fh['site']['lat'] == pytest.approx(59.86, abs=0.001) and fh['filter'] == 'Vmag'
    assert ins.best_match(h)[0] == "Refractor 102"
    assert ins.match_score(ins.load("MWT"), h) < 0
    new = ins.template("Vespera 2")
    filled = ins.fill_from_header(new, h)
    assert new['instrument']['focal_length_mm'] == 714 and 'instrument.camera' in filled
    ins.rename("MWT", "MWT Raspay")
    assert ins.list_names() == ["MWT Raspay", "Refractor 102"] and ins.load("MWT Raspay")['instrument']['focal_length_mm'] == 2900
    with pytest.raises(ValueError):
        ins.rename("MWT Raspay", "Refractor 102")
    assert ins.delete("MWT Raspay") and ins.list_names() == ["Refractor 102"]


def test_observation_planner(tmp_path, monkeypatch):
    """v4.37: planner - time window over midnight, sectors, filters, observability, Sun/Moon
    (checked against astropy), priority list, cluster file."""
    monkeypatch.chdir(tmp_path)
    import datetime as dt
    from python import planner as P
    tz = dt.timezone(dt.timedelta(hours=1))
    times, jd = P.time_grid("2026-11-08", "18:00", "02:00", tz, step_min=30)
    assert times[0].hour == 18 and times[-1].day == 9 and times[-1].hour == 2 and len(times) == 17
    assert P.parse_sector("E") == (45, 135) and P.parse_sector("60-200") == (60.0, 200.0)
    assert P.in_sector(np.array([350.0, 10.0, 100.0]), P.SECTORS['N']).tolist() == [True, True, False]
    assert P.type_matches("EA/SD", ["EA"]) and P.type_matches("DSCTC", ["DSCT*"]) and not P.type_matches("M", ["EA"])
    assert P.mag_ok({'max': 2.0, 'min': 3.3}, bright=6, faint=14) is False          # always too bright
    assert P.mag_ok({'max': 11.5, 'min': 12.2}, bright=6, faint=14) and P.mag_ok({}, 6, 14)
    tg = [{'name': 'AE UMa', 'ra': 144.22, 'dec': 44.07, 'type': 'DSCT', 'period': 0.086, 'max': 10.9, 'min': 11.5},
          {'name': 'Sgr star', 'ra': 270.0, 'dec': -40.0, 'type': 'EA', 'period': 2.0, 'max': 10, 'min': 11},
          {'name': 'bet Per', 'ra': 47.04, 'dec': 40.96, 'type': 'EA/SD', 'period': 2.867, 'max': 2.1, 'min': 3.3}]
    f = P.filter_targets(tg, bright=6, faint=14, types=None, pmin=0.05, pmax=10)
    assert [t['name'] for t in f] == ['AE UMa', 'Sgr star']
    rows, night = P.plan(f, times, jd, 59.86, 17.64, alt_min=30, sector=(0, 360), dark_sun_alt=90)
    assert [r['target']['name'] for r in rows] == ['AE UMa']               # never above 30 deg: not listed
    r = rows[0]
    assert r['first'].hour == 23 and 40 < r['max_alt'] < 55                # rising until 02:00
    # same altitude with astropy (independent check)
    from astropy.utils import iers
    iers.conf.auto_download = False
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    import astropy.units as u
    aa = SkyCoord(144.22 * u.deg, 44.07 * u.deg).transform_to(
        AltAz(obstime=Time(jd, format='jd'), location=EarthLocation(lat=59.86 * u.deg, lon=17.64 * u.deg)))
    assert np.max(np.abs(aa.alt.deg - r['alt'])) < 0.05
    ev, mo = P.twilight_times("2026-11-08", tz, 59.86, 17.64, -12)
    assert ev.hour == 17 and mo.hour == 5
    assert P.twilight_times("2026-06-21", tz, 59.86, 17.64, -12) == (None, None)    # bright summer night
    # priority list and clusters
    assert P.add_priority(tg[:2], notes="campaign") == 2 and P.add_priority(tg[:1]) == 0
    pl = P.load_priority()
    assert [t['name'] for t in pl] == ['AE UMa', 'Sgr star'] and pl[0]['notes'] == 'campaign'
    assert [t['name'] for t in P.remove_priority(['ae uma'])] == ['Sgr star']
    cl = P.load_clusters()
    assert os.path.exists(paths.CLUSTER_FILE) and len(cl) == 10
    h = next(c for c in cl if c['name'] == 'NGC 869')
    assert h['ra'] == pytest.approx(34.75, abs=0.01) and h['dec'] == pytest.approx(57.133, abs=0.01)
    vs = P.cluster_variables(h, [{'name': 'X', 'ra': 34.80, 'dec': 57.15}, {'name': 'Y', 'ra': 40.0, 'dec': 57.0}])
    assert [v['name'] for v in vs] == ['X'] and vs[0]['dist_arcmin'] < 3


def test_nina_night_plan(tmp_path, monkeypatch):
    """v4.38: exposure choice, eclipse prediction, what/how long, schedule without overlaps,
    NINA simple-sequencer XML (structure like NINA's own target sets), filters from NINA files."""
    monkeypatch.chdir(tmp_path)
    import xml.etree.ElementTree as ET
    from python import nina_export as NE, instruments as I
    s = I.template("Refractor")
    s['instrument'].update(aperture_mm=102, focal_length_mm=714, pixel_um=3.76, sensor_x=6248, sensor_y=4176, gain=0, offset=50)
    s['settings'].update(ccd_gain=0.8, saturation_limit=60000, ccd_read_noise=3.5, detect_fwhm=3.0)
    s['filters']['V'].update(fits_name='Vmag', nina_position=6)
    t_ae = {'name': 'AE UMa', 'ra': 144.22, 'dec': 44.07, 'type': 'DSCT', 'max': 10.86, 'min': 11.52, 'period': 0.086}
    e, inf = NE.choose_exposure(t_ae, s, 'V')
    assert e == 120 and 0 < inf['peak_frac'] < 0.7 and inf['snr_faint'] > 50
    e2, inf2 = NE.choose_exposure({'max': 2.1, 'min': 3.3}, s, 'V')
    assert e2 < 30 and 'saturat' in inf2['note']
    assert NE.choose_exposure(dict(t_ae, period=0.02), s, 'V')[0] <= 0.02 * 86400 / 50   # time resolution
    s['planning']['zp_v_e'] = 16.0                                     # measured, 2.6 mag less sensitive
    assert NE.choose_exposure({'max': 15.5, 'min': 17}, s, 'V')[1]['snr_faint'] < 5
    s['planning']['zp_v_e'] = None
    # eclipses and requests
    jd0, jd1 = 2461353.2, 2461353.7
    ea = {'name': 'EA star', 'ra': 47.0, 'dec': 41.0, 'type': 'EA', 'period': 2.0, 'epoch': 2461353.45 - 20 * 2.0}
    assert NE.minima_in(ea, jd0, jd1) == pytest.approx([2461353.45])
    r = NE.block_request(ea, jd0, jd1)
    assert r['kind'] == 'eclipse' and r['center_jd'] == pytest.approx(2461353.45) and 1.5 <= r['hours'] <= 6
    assert NE.block_request(dict(ea, epoch=2461353.45 - 20 * 2.0 + 1.0), jd0, jd1)['kind'] == 'snapshot'
    assert NE.block_request(t_ae, jd0, jd1)['kind'] == 'series'
    assert NE.block_request({'type': 'M', 'period': 332}, jd0, jd1) == {'kind': 'snapshot', 'frames': 3,
                                                                       'why': "long period - a few exposures"}
    # schedule
    jd = np.arange(jd0, jd1 + 1e-9, 10 / 1440.0)
    ok_all = np.ones(len(jd), bool)
    items = [{'target': ea, 'ok': ok_all, 'best_jd': jd0 + 0.1, 'exposure': 60, 'filters': ['Vmag'],
              'request': NE.block_request(ea, jd0, jd1)},
             {'target': t_ae, 'ok': ok_all, 'best_jd': 2461353.45, 'exposure': 120, 'filters': ['Vmag'],
              'request': NE.block_request(t_ae, jd0, jd1)},
             {'target': {'name': 'Mira', 'ra': 34.8, 'dec': -3.0, 'type': 'M', 'period': 332}, 'ok': ok_all,
              'best_jd': 2461353.45, 'exposure': 30, 'filters': ['Vmag'], 'request': {'kind': 'snapshot', 'frames': 3}}]
    blocks, rej = NE.make_plan(items, jd)
    assert not rej and len(blocks) == 3
    for a, b in zip(blocks, blocks[1:]):
        assert a['end_jd'] <= b['start_jd'] + 1e-9                     # no overlaps, in time order
    ecl = next(b for b in blocks if b['request']['kind'] == 'eclipse')
    assert abs((ecl['start_jd'] + ecl['end_jd']) / 2 - 2461353.45) < 0.01          # centred on the minimum
    assert next(b for b in blocks if b['target']['name'] == 'Mira')['frames'] == 3
    # NINA XML
    entries = NE.plan_to_entries(blocks, s, [('Vmag', 6)], start_jd=jd0)
    p = NE.write_target_set(str(tmp_path / "plan.ninaTargetSet.xml"), entries)
    root = ET.parse(p).getroot()
    assert root.tag == 'ArrayOfCaptureSequenceList' and len(root) == 3
    tl = root.find("CaptureSequenceList[@TargetName='AE UMa']")
    assert tl.find('CaptureSequence/FilterType/Position').text == '6' and tl.find('CaptureSequence/Gain').text == '0'
    assert tl.find('CaptureSequence/ExposureTime').text == '120' and float(tl.find('Coordinates/RA').text) == pytest.approx(9.6147, abs=1e-3)
    assert [c.tag for c in tl.find('CaptureSequence')][:4] == ['Enabled', 'ExposureTime', 'ImageType', 'FilterType']
    mira = root.find("CaptureSequenceList[@TargetName='Mira']")
    assert mira.get('DecDegrees') == '-3' and mira.find('NegativeDec').text == 'true'
    assert NE.filters_from_nina_file(p) == {'Vmag': 6}


def test_planner_refinements():
    """v4.39: sorting by constellation (GCVS order of designations), meridian passage."""
    import datetime as dt
    from python import planner as P
    names = ['V1090 Per', 'bet Per', 'AE UMa', 'RZ Cas', 'AA Per', 'ZZ Per', 'RR Per', 'S Per', 'NGC 869',
             'V335 Per', 'del Cep', 'R Sct']
    assert sorted(names, key=P.name_sort_key) == ['RZ Cas', 'del Cep', 'bet Per', 'S Per', 'RR Per', 'ZZ Per',
                                                  'AA Per', 'V335 Per', 'V1090 Per', 'R Sct', 'AE UMa', 'NGC 869']
    tz = dt.timezone(dt.timedelta(hours=1))
    times, jd = P.time_grid("2026-11-08", "18:00", "04:00", tz, step_min=5)
    rows, _ = P.plan([{'name': 'V1090 Per', 'ra': 35.25, 'dec': 57.17}], times, jd, 59.86, 17.64, alt_min=10,
                     dark_sun_alt=90)
    r = rows[0]
    k = int(np.argmax(r['alt']))
    assert r['transit_in_window'] and abs((r['transit'] - times[k]).total_seconds()) < 400   # = highest point
    assert r['transit'].strftime('%H:%M') == '22:59'


def test_hjd_sexagesimal_site():
    """v4.41: HJD with sexagesimal SITELAT/SITELONG (MaxIm/ACP at the MWT) - before, the
    heliocentric correction was dropped (JD used). Reference: ACP's HJD-OBS (start) + half
    the exposure, header of HU Aqr 2026-11-13 21:19:30.69 UTC, 60 s."""
    from astropy.utils import iers
    iers.conf.auto_download = False
    from astropy.io import fits
    from python.time_series import get_hjd, _site_deg
    h = fits.Header()
    h['DATE-OBS'] = '2026-11-13T21:19:30.690'; h['EXPTIME'] = 60.0; h['JD'] = 2461358.3885496529
    h['SITELAT'] = '38 27 33'; h['SITELONG'] = '-01 04 57'; h['ALT-OBS'] = 673.0
    assert _site_deg(h, ('SITELAT',)) == pytest.approx(38.4592, abs=1e-4)
    assert _site_deg(h, ('SITELONG',)) == pytest.approx(-1.0825, abs=1e-4)
    hjd = get_hjd(None, 316.991, -5.29444, h)
    assert hjd is not None
    assert (hjd - (2461358.3882240 + 30 / 86400)) * 86400 == pytest.approx(0, abs=0.1)     # ACP HJD-OBS + 30 s
    assert (hjd - (2461358.3885496529 + 30 / 86400)) * 86400 == pytest.approx(-28.1, abs=0.2)  # correction -28 s
    h2 = h.copy(); h2['SITELAT'] = 'garbage'
    assert abs(get_hjd(None, 316.991, -5.29444, h2) - hjd) * 86400 < 0.05                 # site barely matters


def test_light_curve_ut_column():
    """v4.42: UT of mid-exposure (HH:MM:SS) instead of JD in light-curve tables / CSV."""
    from python.time_series import ut_hms, csv_fields
    assert ut_hms(2461358.3888969) == "21:20:01"            # 21:19:30.69 UTC start + 30 s
    assert ut_hms(2461358.5) == "00:00:00" and ut_hms(None) == "" and ut_hms(float('nan')) == ""
    assert csv_fields([{'file': 'a', 'ut_mid': '21:20:01', 'jd': 1.0, 'hjd': 2.0}, {'mag': 1}]) == \
        ['file', 'ut_mid', 'hjd', 'mag']


def test_light_curve_ut_date_column():
    """v4.43: UT date (YYYY-MM-DD) of mid-exposure next to ut_mid."""
    from python.time_series import ut_date, ut_hms, csv_fields
    assert ut_date(2461358.3888969) == "2026-11-13"
    assert ut_date(2461358.5) == "2026-11-14" and ut_hms(2461358.5) == "00:00:00"
    assert ut_date(2461358.4999999) == "2026-11-14"          # 23:59:59.99 rounds to 00:00:00 of the next day
    assert ut_date(None) == "" and ut_date(float('nan')) == ""
    assert csv_fields([{'file': 'a', 'date_ut': '2026-11-13', 'ut_mid': '21:20:01', 'jd': 1.0, 'hjd': 2.0}]) == \
        ['file', 'date_ut', 'ut_mid', 'hjd']


def _faint_frames(tmp_path, n=6, t_flux=300.0, shift=(0.6, -0.4), sky=100.0, seed=1):
    """Synthetic frames: 2 bright refs + faint target, WCS off by `shift` px, Gaussian noise."""
    from astropy.io import fits
    from astropy.wcs import WCS
    rng = np.random.default_rng(seed)
    w = WCS(naxis=2)
    w.wcs.crpix = [100, 100]; w.wcs.crval = [317.0, -5.3]; w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    stars = {'T': (317.0, -5.3, t_flux), 'R1': (317.0 + 40 / 3600, -5.3 + 30 / 3600, 20000.0),
             'R2': (317.0 - 35 / 3600, -5.3 - 45 / 3600, 40000.0), 'C': (317.0 + 50 / 3600, -5.3 - 40 / 3600, 10000.0)}
    yy, xx = np.mgrid[0:200, 0:200]
    sig = 3.0 / 2.355
    files = []
    for k in range(n):
        img = np.full((200, 200), sky)
        for ra, dec, F in stars.values():
            x, y = w.world_to_pixel_values(ra, dec)
            x += shift[0]; y += shift[1]
            img += F / (2 * np.pi * sig ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
        img += rng.normal(0, 10.0, img.shape)
        h = w.to_header()
        h['EXPTIME'] = 60.0; h['JD'] = 2461358.4 + k * 70 / 86400; h['AIRMASS'] = 1.3; h['FWHM'] = 3.0
        p = tmp_path / f"f{k}.fits"
        fits.writeto(p, img.astype(np.float32), h)
        files.append(str(p))
    ens = [{'name': 'R1', 'ra': stars['R1'][0], 'dec': stars['R1'][1], 'mag_std': 14.0, 'bv_std': 0.5},
           {'name': 'R2', 'ra': stars['R2'][0], 'dec': stars['R2'][1], 'mag_std': 14.0 - 2.5 * np.log10(2), 'bv_std': 0.5}]
    chk = {'name': 'C', 'ra': stars['C'][0], 'dec': stars['C'][1], 'mag_std': 14.0 + 2.5 * np.log10(2), 'bv_std': 0.5}
    return files, stars, ens, chk


def test_faint_target_forced_photometry(tmp_path):
    """v4.44: forced photometry at the ref-corrected WCS position; combining = summing fluxes."""
    from python.faint_target import measure_frames, combine, groups_consecutive, groups_single
    files, stars, ens, chk = _faint_frames(tmp_path)
    frames, info = measure_frames(files, {'ra': stars['T'][0], 'dec': stars['T'][1]}, ens, chk,
                                  fwhm_factor=1.5, gain=1e9, log=lambda *a: None)
    assert len(frames) == 6
    assert info['offset_mean'][0] == pytest.approx(0.6, abs=0.1)
    assert info['offset_mean'][1] == pytest.approx(-0.4, abs=0.1)
    # same aperture for all stars -> ratio target / R1 independent of the aperture loss
    one = combine(frames, groups_single(frames), 'calibrated', ens, chk, snr_limit=3.0)
    all6 = combine(frames, groups_consecutive(frames, 6), 'calibrated', ens, chk, snr_limit=3.0)
    assert len(one) == 6 and len(all6) == 1 and all6[0]['n_frames'] == 6
    true_mag = 14.0 - 2.5 * np.log10(300.0 / 20000.0)            # 18.56
    assert all6[0]['mag'] == pytest.approx(true_mag, abs=3 * all6[0]['mag_err'] + 0.02)
    assert all6[0]['check_mag'] == pytest.approx(chk['mag_std'], abs=0.02)
    s1 = np.array([r['snr'] for r in one])
    assert all6[0]['snr'] == pytest.approx(np.sqrt(6) * np.mean(s1), rel=0.35)  # sum of 6 ~ sqrt(6) x S/N
    assert all6[0]['exptime'] == 360.0 and all6[0]['t_span_min'] == pytest.approx(5 * 70 / 60 + 1, abs=0.01)
    # relative mode = sum target / sum refs
    rel = combine(frames, groups_consecutive(frames, 3), 'relative', ens)
    assert len(rel) == 2 and rel[0]['flux'] == pytest.approx(300.0 / 60000.0, rel=0.25)


def test_faint_target_limits_and_groups(tmp_path):
    """v4.44: upper limits below the S/N threshold, grouping helpers, AAVSO '<' entries."""
    from python.faint_target import (measure_frames, combine, groups_consecutive, merge_groups,
                                         split_groups, groups_single)
    from python.time_series import save_aavso_report
    files, stars, ens, chk = _faint_frames(tmp_path, n=4, t_flux=0.0, seed=3)
    frames, _ = measure_frames(files, {'ra': stars['T'][0], 'dec': stars['T'][1]}, ens, None,
                               fwhm_factor=1.5, gain=1e9, log=lambda *a: None)
    res = combine(frames, groups_single(frames), 'calibrated', ens, None, snr_limit=3.0)
    assert all(r['upper_limit'] and r['flag'] == 'LIMIT' and np.isnan(r['mag']) for r in res)
    assert all(18.0 < r['mag_limit'] < 21.0 for r in res)
    out = tmp_path / "a.txt"
    save_aavso_report(res, str(out), "X", "V", "OBS")
    lines = [l for l in out.read_text().splitlines() if l.startswith("X,")]
    assert len(lines) == 4 and all(l.split(",")[2].startswith("<") and l.split(",")[3] == "na" for l in lines)
    # grouping helpers
    fr = [{'jd': t} for t in (0.0, 0.001, 0.002, 0.003, 0.5, 0.501, 0.502)]
    assert groups_consecutive(fr, 2) == [[0, 1], [2, 3], [4, 5], [6]]
    assert groups_consecutive(fr, 3) == [[0, 1, 2], [3], [4, 5, 6]]          # never across the gap
    g = merge_groups(groups_single(fr), [1, 3, 4])
    assert g == [[0], [1, 3, 4], [2], [5], [6]]
    assert split_groups(g, [1]) == groups_single(fr)


# ---------------------------------------------------------------- v4.45: real RefCat2 tile format
_RC2_LINES = [   # first lines of the real tile 000+00.rc2 (MAST RefCat2, 00_m_16): hex flags '2d', '0d'
    "192775,52528876,44,9,-71,21,-307,9,17091,1,17480,7,16528,4,0,0,2,78,256,999,999,17502,10,248,2d,17046,10,1041,2d,16880,10,415,0d,16838,10,21,0d,0,15891,65,15641,140,15267,148",
    "253790,10177984,307,16,3256,25,-2070,14,17654,1,19171,23,16412,4,0,0,2,87,999,999,999,19913,106,2,01,18086,101,1,01,16968,100,0,01,16364,100,0,01,0,14693,31,14000,55,13829,54",
    "1209324,91636405,295,13,-734,23,2782,10,17112,1,18220,12,16050,4,0,0,2,75,999,999,999,18611,10,97,0d,17356,10,55,0d,16460,10,158,0d,16036,9,72,0d,0,14719,34,14142,30,13907,46",
]


def test_refcat2_real_tile_format(tmp_path, monkeypatch):
    """v4.45: a tile whose first line contains hexadecimal flag fields ('2d') was taken as
    a header line and the whole file was lost (79 instead of ~1000 stars in a MWT frame)."""
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc
    from python import calibration as cal
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True, 'loaded': True})
    d = tmp_path / "cats" / "ATLAS_refcat2" / "00_m_16"
    d.mkdir(parents=True)
    (d / "000+00.rc2").write_text("\n".join(_RC2_LINES) + "\n")
    found = lc.set_root(str(tmp_path / "cats"))
    assert found['ATLAS']['layout'] == 'tiles' and found['ATLAS']['tile_ext'] == ['.rc2']
    assert lc._file_layout(str(d / "000+00.rc2"))[1] is None          # data line, not a header
    rows = lc.query("ATLAS refcat2", 0.5, 0.5, 40)
    assert len(rows) == 3
    assert rows[0]['_ra'] == pytest.approx(0.00192775) and rows[0]['_dec'] == pytest.approx(0.52528876)
    st = cal.stars_from_local_rows(rows, "ATLAS refcat2")
    v, b = cal.atlas_to_vb(17.502, 17.046)
    assert st[0]['V_mag'] == pytest.approx(v) and st[0]['B_mag'] == pytest.approx(b)
    assert st[0]['cat_id'].startswith('ATLAS J0000')
    # a header line of a table catalogue is still recognised as a header
    t = tmp_path / "t.csv"
    t.write_text("RA,Dec,Vmag,Bmag\n10.0,20.0,12.0,12.5\n")
    assert lc._file_layout(str(t))[1] == ['ra', 'dec', 'vmag', 'bmag']


# ---------------------------------------------------------------- v5.0 user folder
def test_paths_layout_and_legacy_translation(tmp_path):
    paths.set_user_dir(tmp_path / "home", "test")
    assert paths.SESSION_FILE == str(tmp_path / "home" / "config" / "calibra_session.json")
    assert paths.CATALOG_CACHE_DIR == str(tmp_path / "home" / "cache" / "catalogs")
    tl = paths.translate_legacy
    assert tl("photometry_output\\targets_auto_x.csv") == os.path.join(paths.OUTPUT_DIR, "targets_auto_x.csv")
    assert tl("photometry_refstars/reference_stars.csv") == paths.REFERENCE_STARS_FILE
    assert tl("photometry_output/dm_cache/a.json") == os.path.join(paths.DM_CACHE_DIR, "a.json")
    assert tl("photometry_refstars/cache") == paths.CATALOG_CACHE_DIR
    old = str(tmp_path / "old")
    assert tl(old + "/photometry_output/period_analysis", old) == os.path.join(paths.OUTPUT_DIR, "period_analysis")
    # not part of the old layout: unchanged
    for v in ("ATLAS refcat2", "C:\\Astro\\bias.fits", "/data/frames/x.fits", "Standard", "", None, 3.5):
        assert tl(v) == v
    paths.ensure_user_dirs()
    assert os.path.isdir(paths.INSTRUMENTS_DIR) and os.path.isdir(paths.PLOTS_DIR)
    assert os.path.exists(paths.REFERENCE_STARS_FILE)          # example list copied


def test_migrate_from_old_calibra_folder(tmp_path):
    import json
    old = tmp_path / "Calibra_Photometry"
    for d in ("photometry_output/dm_cache", "photometry_output/logs", "photometry_plots",
              "photometry_refstars/cache", "photometry_refstars/aavso_sequences", "instruments", "setups",
              "python", "__pycache__"):
        (old / d).mkdir(parents=True)
    (old / "calibra_session.json").write_text(json.dumps({
        "color_b_csv": "photometry_output\\targets_auto_B.csv",
        "reference_catalog": "photometry_refstars/reference_stars.csv",
        "fv_target_list": [{"name": "X", "out": str(old / "photometry_output" / "lc.csv")}],
        "astap_path": "C:\\Program Files\\astap\\astap.exe"}))
    (old / "calibration_config.json").write_text("{}")
    (old / "photometry_output" / "light_curve_X.csv").write_text("a\n")
    (old / "photometry_output" / "dm_cache" / "d.json").write_text("{}")
    (old / "photometry_output" / "logs" / "log.txt").write_text("x")
    (old / "photometry_plots" / "p.png").write_bytes(b"png")
    (old / "photometry_refstars" / "cache" / "VSX_1_2_3.csv").write_text("id\n")
    (old / "photometry_refstars" / "aavso_sequences" / "x_cnc.json").write_text("{}")
    (old / "photometry_refstars" / "local_catalogs.json").write_text(json.dumps({"root": "/cat"}))
    (old / "instruments" / "mwt.json").write_text("{}")
    (old / "setups" / "s.json").write_text("{}")
    (old / "gui.py").write_text("# program file, not copied")
    (old / "python" / "x.py").write_text("")

    n, skipped, msgs = paths.migrate_from(old)
    assert n == 11 and skipped == 0
    with open(paths.SESSION_FILE) as f:
        sess = json.load(f)
    assert sess["color_b_csv"] == os.path.join(paths.OUTPUT_DIR, "targets_auto_B.csv")
    assert sess["reference_catalog"] == paths.REFERENCE_STARS_FILE
    assert sess["fv_target_list"][0]["out"] == os.path.join(paths.OUTPUT_DIR, "lc.csv")
    assert sess["astap_path"] == "C:\\Program Files\\astap\\astap.exe"
    for p in (paths.CALIBRATION_CONFIG_FILE, paths.LOCAL_CATALOGS_FILE,
              os.path.join(paths.OUTPUT_DIR, "light_curve_X.csv"), os.path.join(paths.DM_CACHE_DIR, "d.json"),
              os.path.join(paths.LOGS_DIR, "log.txt"), os.path.join(paths.PLOTS_DIR, "p.png"),
              os.path.join(paths.CATALOG_CACHE_DIR, "VSX_1_2_3.csv"),
              os.path.join(paths.AAVSO_SEQUENCES_DIR, "x_cnc.json"),
              os.path.join(paths.INSTRUMENTS_DIR, "mwt.json"), os.path.join(paths.SETUPS_DIR, "s.json")):
        assert os.path.exists(p), p
    assert not os.path.exists(os.path.join(paths.USER_DIR, "gui.py"))
    assert os.path.exists(old / "calibra_session.json")          # nothing moved or deleted
    # second run: everything already there -> kept
    n2, skipped2, _ = paths.migrate_from(old)
    assert n2 == 0 and skipped2 == 11


def test_user_dir_from_env_and_pointer_file(tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("CALIBRA_HOME", str(tmp_path / "envhome"))
    assert paths._resolve_user_dir() == (str(tmp_path / "envhome"), "environment variable CALIBRA_HOME")
    monkeypatch.delenv("CALIBRA_HOME")
    ptr = tmp_path / "calibra_home.txt"
    monkeypatch.setattr(paths, "POINTER_FILE", str(ptr))
    ptr.write_text("# comment\n" + str(tmp_path / "ptrhome") + "\n")
    assert paths._resolve_user_dir()[0] == str(tmp_path / "ptrhome")
    monkeypatch.setattr(paths, "PROGRAM_DIR", str(tmp_path / "prog"))
    ptr.write_text(".\n")                                          # old behaviour: program folder
    assert paths._resolve_user_dir()[0] == str(tmp_path / "prog")
    ptr.unlink()
    assert paths._resolve_user_dir() == (paths.DEFAULT_USER_DIR, "default")


def test_migrate_replaces_shipped_reference_list(tmp_path):
    old = tmp_path / "old"
    (old / "photometry_refstars").mkdir(parents=True)
    (old / "calibra_session.json").write_text("{}")
    (old / "photometry_refstars" / "reference_stars.csv").write_text("AUID,RA,Dec,V,B-V\nmine,1,2,3,0.5\n")
    paths.ensure_user_dirs()                       # example list copied at first start
    paths.migrate_from(old)
    with open(paths.REFERENCE_STARS_FILE) as f:
        assert "mine" in f.read()                  # the user's own list wins over the example


# ---------------------------------------------------------------- v5.0 version and dependencies
def test_version_in_one_place():
    import re
    import python
    from python.version import __version__, VERSION_DATE
    assert python.__version__ == __version__
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", __version__)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", VERSION_DATE)
    root = os.path.join(os.path.dirname(__file__), "..")
    gui_src = open(os.path.join(root, "gui.py"), encoding="utf-8").read()
    assert "APP_VERSION = __version__" in gui_src
    # no other hard-coded version numbers in the code
    for dirpath, _, files in os.walk(root):
        if any(p in dirpath for p in ("tests", "__pycache__", ".git")):
            continue
        for fn in files:
            if fn.endswith(".py") and fn != "version.py":
                src = open(os.path.join(dirpath, fn), encoding="utf-8").read()
                assert not re.search(r"""(VERSION|software_version)\s*=\s*['"]\d+\.\d+""", src), fn


def test_aavso_report_software_version(tmp_path):
    from python.time_series import save_aavso_report
    from python.version import __version__
    out = tmp_path / "r.txt"
    save_aavso_report([], str(out), "X Cnc", "V", "XXX")
    assert f"#SOFTWARE=Calibra {__version__}\n" in out.read_text()


def test_requirements_match_pyproject():
    import re
    root = os.path.join(os.path.dirname(__file__), "..")
    req = [l.split("#")[0].strip() for l in open(os.path.join(root, "requirements.txt"))]
    req = sorted(r.lower() for r in req if r)
    toml = open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
    block = re.search(r"dependencies = \[(.*?)\]", toml, re.S).group(1)
    deps = sorted(d.lower() for d in re.findall(r'"([^"]+)"', block))
    assert req == deps


def test_fits_fixed_warning_silenced():
    """v5.0.1: old-style headers (RADECSYS, DATE-OBS without MJD-OBS) give no
    FITSFixedWarning anywhere in Calibra (filter set in python/__init__.py)."""
    import subprocess
    code = (
        "import warnings\n"
        "import python\n"
        "from astropy.io import fits\n"
        "from astropy.wcs import WCS\n"
        "h = fits.Header()\n"
        "h['NAXIS'] = 2; h['NAXIS1'] = 100; h['NAXIS2'] = 100\n"
        "h['CTYPE1'] = 'RA---TAN'; h['CTYPE2'] = 'DEC--TAN'\n"
        "h['CRVAL1'] = 10.0; h['CRVAL2'] = 20.0; h['CRPIX1'] = 50; h['CRPIX2'] = 50\n"
        "h['CDELT1'] = -0.0003; h['CDELT2'] = 0.0003\n"
        "h['RADECSYS'] = 'FK5'; h['EQUINOX'] = 2000.0\n"
        "h['DATE-OBS'] = '2026-11-13T21:58:12'\n"
        "with warnings.catch_warnings(record=True) as rec:\n"
        "    w = WCS(h)\n"
        "print(sum('FITSFixed' in type(r.message).__name__ for r in rec))\n"
    )
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out = subprocess.run([sys.executable, "-c", code], cwd=root,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0", out.stdout + out.stderr
    assert "FITSFixedWarning" not in out.stderr


def test_parse_coordinates_fast_path():
    """v5.0.1: sexagesimal text converted without astropy (GCVS lists ~25x faster); same
    values as SkyCoord, sign of '-00 ..' declinations kept, other forms still accepted."""
    from python.field_variables import parse_coordinates
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    for ra_t, dec_t in (("21 07 58.19", "-05 17 40.5"), ("23 59 46.49", "-00 16 48.4"), ("00 00 00.0", "+89 59 59"),
                        ("09:36:53.2", "+44:04:00"), ("5 3", "-0 30"), ("12 30 00", "+00 00 01.5")):
        c = SkyCoord(ra_t, dec_t, unit=(u.hourangle, u.deg))
        ra, dec = parse_coordinates(ra_t, dec_t)
        assert abs(ra - c.ra.deg) < 1e-9 and abs(dec - c.dec.deg) < 1e-9, (ra_t, dec_t)
    assert parse_coordinates("23 59 46.49", "-00 16 48.4")[1] < 0
    assert parse_coordinates("9h36m53.2s", "+44d04m00s") == pytest.approx((144.2216667, 44.0666667))
    assert parse_coordinates("144.2", "44.1") == (144.2, 44.1)
    with pytest.raises(Exception):
        parse_coordinates("25 61 00", "+95 00 00")


def test_landolt_cds_tables(tmp_path, monkeypatch):
    """v5.0.1: original Landolt CDS tables (raw fixed-width + ReadMe, '|' downloads) read;
    the same star in 1992 and 2009 kept once (newest), renamed stars (92 311 -> 92 312)
    recognised, tables without V/B-V ignored; a field without Landolt stars -> None
    (local RefCat2 substitute); an old 'unreadable' mark is cleared."""
    monkeypatch.chdir(tmp_path)
    from python import local_catalogs as lc, calibration as cal
    root = tmp_path / "cat"
    d92 = root / "Landolt" / "1992"; d92.mkdir(parents=True)
    readme = "\n".join([
        "II/183A          UBVRI Photometric Standards              (Landolt  1992)",
        "================================================================================",
        "Byte-by-byte Description of file: table2.dat",
        "--------------------------------------------------------------------------------",
        "   Bytes Format Units   Label   Explanations",
        "--------------------------------------------------------------------------------",
        "   1- 11  A11   ---     Star    Star Designation",
        "  13- 14  I2    h       RAh     Right Ascension J2000 (hours)",
        "  16- 17  I2    min     RAm     Right Ascension J2000 (minutes)",
        "  19- 20  I2    s       RAs     Right Ascension J2000 (seconds)",
        "      22  A1    ---     DE-     Declination J2000 (sign)",
        "  23- 24  I2    deg     DEd     Declination J2000 (degrees)",
        "  26- 27  I2    arcmin  DEm     Declination J2000 (minutes)",
        "  29- 30  I2    arcsec  DEs     Declination J2000 (seconds)",
        "  33- 38  F6.3  mag     Vmag    V magnitude",
        "  40- 45  F6.3  mag     B-V     B-V color",
        "--------------------------------------------------------------------------------", ""])
    (d92 / "ReadMe.txt").write_text(readme)
    (d92 / "table2.dat").write_text(
        "TPHE A      00 30 09 -46 31 22  14.651  0.793\n"
        "92 311      00 53 16 +00 48 29  10.595  1.638\n"
        "WOLF 918    21 09 21 -13 18 09  10.868  1.491\n")
    (d92 / "filters.dat").write_text(" 2750   0.00\n 2760   1.12\n")
    d09 = root / "Landolt" / "2009"; d09.mkdir()
    (d09 / "ReadMe.txt").write_text("J/AJ/137/4186    UBVRI standards around celestial equator  (Landolt, 2009)\n")
    head = ("#Table: J/AJ/137/4186/table2.dat (http://cdsarc.cds.unistra.fr)\n"
            "-------------------------------------------------\n"
            "           | R  R          D  D      |      |      \n"
            "Name       |Ah Am    RAs  Ed Em   DEs|  Vmag|   B-V\n"
            "-----------|-------------------------|------|------\n")
    (d09 / "J_AJ_137_4186_table2.dat.txt").write_text(head +
        "TPhe A     |00 30 09.594 -46 31 28.91|14.651|+0.793\n"
        "92 312     |00 53 16.533 +00 48 28.90|10.598|+1.636\n"
        "92 309     |00 53 14.047 +00 46 02.14|13.842|+0.513\n")
    (d09 / "J_AJ_137_4186_table5.dat.txt").write_text(
        "#Table: J/AJ/137/4186/table5.dat (http://cdsarc.cds.unistra.fr)\n----------\n"
        "Name       |Ah Am    RAs  Ed Em   DEs|UCAC2   \n-----------|-------------------------|--------\n"
        "TPhe A     |00 30 09.594 -46 31 28.91|11679110\n")
    stars = lc.landolt_stars(str(root / "Landolt"))
    names = sorted(s['name'] for s in stars)
    assert names == ['92 309', '92 312', 'TPhe A', 'WOLF 918'], names
    tp = next(s for s in stars if s['name'] == 'TPhe A')
    assert tp['year'] == 2009 and abs(tp['dec'] - (-46 - 31 / 60 - 28.91 / 3600)) < 1e-9
    wolf = next(s for s in stars if s['name'] == 'WOLF 918')
    assert wolf['year'] == 1992 and wolf['dec'] < 0 and abs(wolf['ra'] - 15 * (21 + 9 / 60 + 21 / 3600)) < 1e-9
    monkeypatch.setattr(lc, "_STATE", {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': False,
                                       'apass_auto': False, 'loaded': True})
    lc.set_root(str(root))
    lc._STATE['catalogs']['LANDOLT']['unreadable'] = True            # mark of the first v5.0.1 version
    rows = lc.query("LANDOLT", 13.33, 0.80, 10)
    assert sorted(r['name'] for r in rows) == ['92 309', '92 312']
    assert not lc._STATE['catalogs']['LANDOLT'].get('unreadable')
    st = cal.fetch_online_catalog(13.33, 0.80, 10, "Landolt Standard Star Catalogue", verbose=False)
    s312 = next(s for s in st if s['cat_id'] == '92 312')
    assert st[0]['cat_source'] == 'local copy' and abs(s312['V_mag'] - 10.598) < 1e-6
    assert abs(s312['B_mag'] - (10.598 + 1.636)) < 1e-6
    assert lc.query("LANDOLT", 200.0, 30.0, 10) is None                # no Landolt star -> not covered
    assert any('4 standard stars' in l for l in lc.summary_lines())
