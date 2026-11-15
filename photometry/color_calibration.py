import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import linregress
import json

class FitResult:
    """Minimal linregress-like result (slope, intercept, stderr, intercept_stderr, rvalue, chi2_red, n)."""
    def __init__(self, slope, intercept, stderr, intercept_stderr, rvalue, chi2_red, n):
        self.slope, self.intercept = slope, intercept
        self.stderr, self.intercept_stderr = stderr, intercept_stderr
        self.rvalue, self.chi2_red, self.n = rvalue, chi2_red, n


def _wls(x, y, sig):
    """Weighted least squares straight line y = a + b x. Returns FitResult."""
    w = 1.0 / np.maximum(sig, 1e-4) ** 2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx ** 2
    b = (S * Sxy - Sx * Sy) / D
    a = (Sxx * Sy - Sx * Sxy) / D
    res = y - (a + b * x)
    dof = max(len(x) - 2, 1)
    chi2_red = float((w * res ** 2).sum() / dof)
    scale = max(chi2_red, 1.0)          # inflate errors if the scatter exceeds the error bars
    sb = np.sqrt(S / D * scale)
    sa = np.sqrt(Sxx / D * scale)
    r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
    return FitResult(float(b), float(a), float(sb), float(sa), float(r), chi2_red, len(x))


def perform_robust_fit(x, y, sigma_clip=3.0, sig=None, maxiters=10):
    """
    v4.16: weighted linear fit with iterative, robust outlier rejection.
      * weights 1/sigma^2 (sigma = per-star errors; equal weights if None)
      * rejection on normalised residuals res/sigma using a MAD-based scale
        (was: one pass with the non-robust std, 2 sigma)
    Returns (final_fit, keep_mask, std_final, first_fit, std_first) - same
    structure as before so the plotting code is unchanged.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    sig = np.ones_like(x) * 0.02 if sig is None else np.asarray(sig, float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(sig)
    first = _wls(x[keep], y[keep], sig[keep])
    std_first = float(np.std(y[keep] - (first.intercept + first.slope * x[keep])))
    fit = first
    for _ in range(maxiters):
        nres = (y - (fit.intercept + fit.slope * x)) / sig
        base = nres[keep]
        scale = 1.4826 * np.median(np.abs(base - np.median(base)))
        scale = max(scale, 1e-6)
        new_keep = np.isfinite(nres) & (np.abs(nres - np.median(base)) <= sigma_clip * scale)
        if new_keep.sum() < 5:
            break
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
        fit = _wls(x[keep], y[keep], sig[keep])
    std_final = float(np.std(y[keep] - (fit.intercept + fit.slope * x[keep])))
    return fit, keep, std_final, first, std_first


def derive_color_terms(results_b, results_v, catalog_stars, output_dir, airmass_b=1.0, airmass_v=1.0, k_b=0.35, k_v=0.20, axes=None,
                       min_snr=50.0, peak_limit=30000.0, max_v_mag=None, catalog_error=0.02, exclude_blended=True):
    """
    Derives instrumental colour terms by cross-matching B and V photometry results.
    results_b/v: list of dicts from the pipeline CSVs (targets_auto_*.csv)
    catalog_stars: list of dicts from fetch_online_catalog
    k_b, k_v: Extinction coefficients

    v4.16 changes (code review S2): only well-measured, unsaturated, unblended
    stars are used (min_snr, peak_limit, optional max_v_mag), the fits are
    weighted (photometric + catalogue errors) with robust iterative clipping,
    Tbv is obtained by fitting (b-v) against (B-V) and inverting (the catalogue
    colour is the less noisy variable, as in the AAVSO Transform Generator),
    coefficient uncertainties are reported, and the consistency relation
    Tbv = 1/(1 - Tb_bv + Tv_bv) is checked. Previously all matched stars were
    used unweighted, which on the AE UMa field gave Tb_bv = 0.148 (all stars)
    versus -0.007 for V < 12.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "color_transformation_report.md")

    def _f(d, k, default=np.nan):
        try:
            v = d.get(k, default)
            return float(v) if v not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def _usable(d):
        if str(d.get('saturated', 'No')).lower() in ('yes', 'true', '1'):
            return False
        if exclude_blended and str(d.get('blended', 'No')).lower() in ('yes', 'true', '1'):
            return False
        if _f(d, 'snr', 0.0) < min_snr:
            return False
        peak = _f(d, 'peak_raw')
        if not np.isfinite(peak):
            peak = _f(d, 'peak_adu')
        if peak_limit and np.isfinite(peak) and peak > peak_limit:
            return False
        return np.isfinite(_f(d, 'mag_inst'))

    n_b_all, n_v_all = len(results_b), len(results_v)
    results_b = [r for r in results_b if _usable(r)]
    results_v = [r for r in results_v if _usable(r)]
    if len(results_b) < 5 or len(results_v) < 5:
        return (f"Error: too few usable stars after selection (B: {len(results_b)}/{n_b_all}, V: {len(results_v)}/{n_v_all}; "
                f"SNR >= {min_snr}, peak <= {peak_limit}). Lower the limits or check the input CSVs.")
    
    # 1. Match B and V detections
    coords_b = SkyCoord(ra=[float(r['ra_deg']) for r in results_b]*u.deg, dec=[float(r['dec_deg']) for r in results_b]*u.deg)
    coords_v = SkyCoord(ra=[float(r['ra_deg']) for r in results_v]*u.deg, dec=[float(r['dec_deg']) for r in results_v]*u.deg)
    
    idx, d2d, d3d = coords_b.match_to_catalog_sky(coords_v)
    max_sep = 2.0 * u.arcsec
    mask = d2d < max_sep
    
    matched_pairs = []
    for i, is_match in enumerate(mask):
        if is_match:
            b_star = results_b[i]
            v_star = results_v[idx[i]]
            matched_pairs.append({'b': b_star, 'v': v_star})
            
    if not matched_pairs:
        return "Error: No matching stars found between B and V images."

    # 2. Match pairs to Catalog
    coords_pairs = SkyCoord(ra=[float(p['v']['ra_deg']) for p in matched_pairs]*u.deg, 
                            dec=[float(p['v']['dec_deg']) for p in matched_pairs]*u.deg)
    coords_cat = SkyCoord(ra=[s['ra_deg'] for s in catalog_stars]*u.deg, 
                          dec=[s['dec_deg'] for s in catalog_stars]*u.deg)
    
    idx_cat, d2d_cat, _ = coords_pairs.match_to_catalog_sky(coords_cat)
    mask_cat = d2d_cat < max_sep
    
    final_data = []
    for i, is_match in enumerate(mask_cat):
        if is_match:
            pair = matched_pairs[i]
            cat = catalog_stars[idx_cat[i]]
            
            # Exclude variable stars from color term derivation
            if cat.get('is_variable', False):
                continue
            if pair['b'].get('is_variable') == 'Yes' or pair['v'].get('is_variable') == 'Yes':
                continue
            
            b_raw = _f(pair['b'], 'mag_inst')
            v_raw = _f(pair['v'], 'mag_inst')
            eb = _f(pair['b'], 'mag_inst_err', 0.01)
            ev = _f(pair['v'], 'mag_inst_err', 0.01)
            eb = eb if np.isfinite(eb) else 0.01
            ev = ev if np.isfinite(ev) else 0.01
            
            # Apply Extinction Correction: m_corr = m_inst - k*X
            b_inst = b_raw - (k_b * airmass_b)
            v_inst = v_raw - (k_v * airmass_v)
            
            b_cat = cat.get('B_mag', np.nan)
            v_cat = cat.get('V_mag', np.nan)
            if max_v_mag is not None and np.isfinite(v_cat) and v_cat > max_v_mag:
                continue
            
            if not np.isnan([b_inst, v_inst, b_cat, v_cat]).any():
                final_data.append({
                    'ra_deg': float(pair['v']['ra_deg']),
                    'dec_deg': float(pair['v']['dec_deg']),
                    'b_inst': b_inst,
                    'v_inst': v_inst,
                    'b_inst_err': eb,
                    'v_inst_err': ev,
                    'b_cat': b_cat,
                    'v_cat': v_cat,
                    'color_inst': b_inst - v_inst,
                    'color_cat': b_cat - v_cat,
                    'diff_b': b_cat - b_inst,
                    'diff_v': v_cat - v_inst
                })

    if len(final_data) < 5:
        return f"Error: Too few stars ({len(final_data)}) matched to catalog (after selection and excluding variables) for reliable fitting."

    # Save matched data to CSV for transparency
    matched_csv = os.path.join(output_dir, "color_matched_stars.csv")
    import csv
    with open(matched_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_data[0].keys())
        writer.writeheader()
        writer.writerows(final_data)

    # 3. Weighted robust fits
    c_cat = np.array([d['color_cat'] for d in final_data])
    c_inst = np.array([d['color_inst'] for d in final_data])
    db = np.array([d['diff_b'] for d in final_data])
    dv = np.array([d['diff_v'] for d in final_data])
    eb = np.array([d['b_inst_err'] for d in final_data])
    ev = np.array([d['v_inst_err'] for d in final_data])
    v_cat_arr = np.array([d['v_cat'] for d in final_data])
    s_b = np.sqrt(eb ** 2 + catalog_error ** 2)
    s_v = np.sqrt(ev ** 2 + catalog_error ** 2)
    s_c = np.sqrt(eb ** 2 + ev ** 2 + 2 * catalog_error ** 2)

    # Tbv: fit (b-v) = a + s (B-V) and invert -> (B-V) = Tbv (b-v) + zp
    res_inv, mask_mu, _, res1_inv, _ = perform_robust_fit(c_cat, c_inst, sig=s_c)
    Tbv = 1.0 / res_inv.slope
    Tbv_err = res_inv.stderr / res_inv.slope ** 2
    zp_bv = -res_inv.intercept / res_inv.slope
    res_mu = FitResult(Tbv, zp_bv, Tbv_err, res_inv.intercept_stderr / abs(res_inv.slope), res_inv.rvalue, res_inv.chi2_red, res_inv.n)
    res1_mu = FitResult(1.0 / res1_inv.slope, -res1_inv.intercept / res1_inv.slope, 0, 0, res1_inv.rvalue, res1_inv.chi2_red, res1_inv.n)
    res_resid = c_cat - (res_mu.intercept + res_mu.slope * c_inst)
    std_mu = float(np.std(res_resid[mask_mu])); std1_mu = float(np.std(res_resid))

    res_psi, mask_psi, std_psi, res1_psi, std1_psi = perform_robust_fit(c_cat, db, sig=s_b)
    res_eps, mask_eps, std_eps, res1_eps, std1_eps = perform_robust_fit(c_cat, dv, sig=s_v)

    # Consistency: Tbv should equal 1 / (1 - Tb_bv + Tv_bv)
    denom = 1.0 - res_psi.slope + res_eps.slope
    Tbv_implied = 1.0 / denom if denom != 0 else np.nan
    Tbv_implied_err = abs(Tbv_implied) ** 2 * np.hypot(res_psi.stderr, res_eps.stderr) if np.isfinite(Tbv_implied) else np.nan
    diff_sig = (abs(Tbv - Tbv_implied) / np.hypot(Tbv_err, Tbv_implied_err)) if np.isfinite(Tbv_implied) else np.nan
    consistent = bool(np.isfinite(diff_sig) and diff_sig < 3.0)

    # Residuals of the V term versus magnitude (reveals faint-end / non-linearity bias)
    resid_v = dv - (res_eps.intercept + res_eps.slope * c_cat)
    mag_bins = []
    if len(v_cat_arr) >= 10:
        edges = np.unique(np.percentile(v_cat_arr, np.linspace(0, 100, 6)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (v_cat_arr >= lo) & (v_cat_arr <= hi) & mask_eps
            if sel.sum() >= 3:
                mag_bins.append((lo, hi, int(sel.sum()), float(np.median(resid_v[sel]))))

    # 4. Generate Plots
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        is_standalone = True
    else:
        fig = axes[0].figure
        for ax in axes: ax.clear()
        is_standalone = False
    
    def plot_fit(ax, x, y, mask, res, std, res1, std1, xlabel, ylabel, title, label_pref):
        # Plot Outliers
        ax.scatter(x[~mask], y[~mask], color='red', marker='x', alpha=0.4, label='Outliers (Clipped)')
        # Plot Kept Points
        ax.scatter(x[mask], y[mask], color='blue', alpha=0.6, label='Data Points')
        
        # Plot 1st iteration window (just outer lines)
        if len(x) > 0:
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit1 = res1.slope * x_fit + res1.intercept
            ax.plot(x_fit, y_fit1 + 2*std1, color='r', linestyle=':', alpha=0.4, linewidth=1, label='Initial 2σ')
            ax.plot(x_fit, y_fit1 - 2*std1, color='r', linestyle=':', alpha=0.4, linewidth=1)

            # Plot Final Fit
            y_fit = res.slope * x_fit + res.intercept
            ax.plot(x_fit, y_fit, 'k-', linewidth=2, label=f'Final {label_pref}={res.slope:.3f}')
            
            # Plot 2-sigma window (use the std from the final fit)
            ax.plot(x_fit, y_fit + 2*std, color='#555', linestyle='--', alpha=0.8, linewidth=1, label=r'Final ±2σ')
            ax.plot(x_fit, y_fit - 2*std, color='#555', linestyle='--', alpha=0.8, linewidth=1)
            
            # Add fill for the window to make it obvious
            ax.fill_between(x_fit, y_fit - 2*std, y_fit + 2*std, color='gray', alpha=0.1)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plot_fit(axes[0], c_inst, c_cat, mask_mu, res_mu, std_mu, res1_mu, std1_mu,
             "(b-v) Extinction Corrected", "(B-V) Catalog", "Color Transformation", "mu")
             
    plot_fit(axes[1], c_cat, db, mask_psi, res_psi, std_psi, res1_psi, std1_psi,
             "(B-V) Catalog", "B_cat - b_corr", "B-Filter Color Term", "psi")

    plot_fit(axes[2], c_cat, dv, mask_eps, res_eps, std_eps, res1_eps, std1_eps,
             "(B-V) Catalog", "V_cat - v_corr", "V-Filter Color Term", "eps")
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "color_plots.png")
    if is_standalone:
        plt.savefig(plot_path)
        plt.close(fig)
    else:
        fig.savefig(plot_path)

    # 5. Write Report
    with open(report_path, "w") as f:
        f.write("# Color Transformation Calibration Report\n\n")
        f.write(f"Analyzed {len(final_data)} common stars (selection: SNR >= {min_snr:g}, peak <= {peak_limit:g} ADU"
                f"{', V <= %.1f' % max_v_mag if max_v_mag is not None else ''}, not saturated"
                f"{', no neighbour in the annulus' if exclude_blended else ''}, known variables excluded).\n")
        f.write(f"Weighted fits (photometric errors + {catalog_error:.3f} mag catalogue error) with iterative robust 3-sigma rejection.\n")
        f.write(f"Applied Extinction Correction:\n")
        f.write(f"- B-Filter: k={k_b:.2f}, X={airmass_b:.3f}\n")
        f.write(f"- V-Filter: k={k_v:.2f}, X={airmass_v:.3f}\n\n")
        
        f.write("## Derived Coefficients\n")
        f.write(f"- **$T_{{bv}}$ ($\\mu$, colour scale):** {Tbv:.4f} ± {Tbv_err:.4f}  (fit of (b-v) vs (B-V), inverted; N={res_inv.n}, $\\chi^2_\\nu$={res_inv.chi2_red:.2f})\n")
        f.write(f"- **$T_{{b,bv}}$ ($\\psi$, B term):** {res_psi.slope:.4f} ± {res_psi.stderr:.4f}  (N={res_psi.n}, $\\chi^2_\\nu$={res_psi.chi2_red:.2f})\n")
        f.write(f"- **$T_{{v,bv}}$ ($\\epsilon$, V term):** {res_eps.slope:.4f} ± {res_eps.stderr:.4f}  (N={res_eps.n}, $\\chi^2_\\nu$={res_eps.chi2_red:.2f})\n\n")
        f.write("## Consistency Check\n")
        f.write(f"$T_{{bv}}$ implied by the B and V terms, $1/(1 - T_{{b,bv}} + T_{{v,bv}})$ = {Tbv_implied:.4f} ± {Tbv_implied_err:.4f}; "
                f"difference to the fitted $T_{{bv}}$: {diff_sig:.1f} sigma -> **{'consistent' if consistent else 'INCONSISTENT - check star selection / catalogue'}**.\n\n")
        if mag_bins:
            f.write("## V-term residuals versus catalogue magnitude\n")
            f.write("A trend here indicates magnitude-dependent errors (faint-end bias, non-linearity).\n\n")
            f.write("| V range | N | median residual (mag) |\n|---|---|---|\n")
            for lo, hi, n, med in mag_bins:
                f.write(f"| {lo:.2f} - {hi:.2f} | {n} | {med:+.4f} |\n")
            f.write("\n")
        
        f.write("## Transformation Equations\n")
        f.write("Using these coefficients, your calibrated magnitudes are:\n")
        f.write(f"1. $(B-V)_{{std}} = {res_mu.slope:.3f} \\cdot (b-v)_{{corr}} + {res_mu.intercept:.3f}$\n")
        f.write(f"2. $B_{{std}} = b_{{corr}} + {res_psi.slope:.3f} \\cdot (B-V)_{{std}} + {res_psi.intercept:.3f}$\n")
        f.write(f"3. $V_{{std}} = v_{{corr}} + {res_eps.slope:.3f} \\cdot (B-V)_{{std}} + {res_eps.intercept:.3f}$\n")
        f.write("*(Note: v_corr and b_corr are the instrumental magnitudes corrected for extinction)*\n\n")
        
        f.write("![Color Diagnostic Plots](color_plots.png)\n")
        
    # Save coefficients to JSON for later use in Differential Photometry
    json_path = os.path.join(output_dir, "color_coefficients.json")
    coeffs = {
        "Tbv": float(Tbv),
        "Tb_bv": float(res_psi.slope),
        "Tv_bv": float(res_eps.slope),
        "Tbv_err": float(Tbv_err),
        "Tb_bv_err": float(res_psi.stderr),
        "Tv_bv_err": float(res_eps.stderr),
        "Tbv_implied": float(Tbv_implied) if np.isfinite(Tbv_implied) else None,
        "consistent": consistent,
        "n_stars": int(len(final_data)),
        "selection": {"min_snr": float(min_snr), "peak_limit": float(peak_limit) if peak_limit else None,
                      "max_v_mag": max_v_mag, "exclude_blended": bool(exclude_blended)}
    }
    try:
        with open(json_path, "w") as jf:
            json.dump(coeffs, jf, indent=4)
    except Exception as e:
        print(f"Warning: Could not save coefficients JSON: {e}")

    status = (f"Success: {len(final_data)} stars. Tbv={Tbv:.3f}±{Tbv_err:.3f}, Tb_bv={res_psi.slope:.3f}±{res_psi.stderr:.3f}, "
              f"Tv_bv={res_eps.slope:.3f}±{res_eps.stderr:.3f} ({'consistent' if consistent else 'INCONSISTENT'}). Report: {report_path}")
    return status
