import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import linregress
import json

def perform_robust_fit(x, y, sigma_clip=2.0):
    """
    Performs a linear fit with iterative outlier rejection.
    1. Initial fit
    2. Identify points > sigma_clip * std_dev of residuals
    3. Final fit on cleaned data
    """
    x = np.array(x)
    y = np.array(y)
    
    # 1. Initial Fit
    res1 = linregress(x, y)
    y_pred = res1.slope * x + res1.intercept
    residuals = y - y_pred
    std_dev = np.std(residuals)
    
    # 2. Filter outliers
    mask = np.abs(residuals) <= (sigma_clip * std_dev)
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 3. Final Fit
    if len(x_clean) < 3: # Fallback if too many clipped
        return res1, mask, std_dev, res1, std_dev
        
    res2 = linregress(x_clean, y_clean)
    # Re-calculate std_dev for the final window plotting
    y_pred_final = res2.slope * x_clean + res2.intercept
    std_dev_final = np.std(y_clean - y_pred_final)
    
    return res2, mask, std_dev_final, res1, std_dev

def derive_color_terms(results_b, results_v, catalog_stars, output_dir, airmass_b=1.0, airmass_v=1.0, k_b=0.35, k_v=0.20):
    """
    Derives instrumental color terms by cross-matching B and V photometry results.
    results_b/v: list of dicts from process_file
    catalog_stars: list of dicts from fetch_online_catalog
    k_b, k_v: Extinction coefficients
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "color_transformation_report.md")
    
    # 1. Match B and V detections
    coords_b = SkyCoord(ra=[r['ra_deg'] for r in results_b]*u.deg, dec=[r['dec_deg'] for r in results_b]*u.deg)
    coords_v = SkyCoord(ra=[r['ra_deg'] for r in results_v]*u.deg, dec=[r['dec_deg'] for r in results_v]*u.deg)
    
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
    coords_pairs = SkyCoord(ra=[p['v']['ra_deg'] for p in matched_pairs]*u.deg, 
                            dec=[p['v']['dec_deg'] for p in matched_pairs]*u.deg)
    coords_cat = SkyCoord(ra=[s['ra_deg'] for s in catalog_stars]*u.deg, 
                          dec=[s['dec_deg'] for s in catalog_stars]*u.deg)
    
    idx_cat, d2d_cat, _ = coords_pairs.match_to_catalog_sky(coords_cat)
    mask_cat = d2d_cat < max_sep
    
    final_data = []
    for i, is_match in enumerate(mask_cat):
        if is_match:
            pair = matched_pairs[i]
            cat = catalog_stars[idx_cat[i]]
            
            # Instrumental magnitudes (raw)
            b_raw = pair['b'].get('mag_inst', np.nan)
            v_raw = pair['v'].get('mag_inst', np.nan)
            
            # Apply Extinction Correction: m_corr = m_inst - k*X
            b_inst = b_raw - (k_b * airmass_b)
            v_inst = v_raw - (k_v * airmass_v)
            
            # Catalog magnitudes
            b_cat = cat.get('B_mag', np.nan)
            v_cat = cat.get('V_mag', np.nan)
            
            if not np.isnan([b_inst, v_inst, b_cat, v_cat]).any():
                final_data.append({
                    'b_inst': b_inst,
                    'v_inst': v_inst,
                    'b_cat': b_cat,
                    'v_cat': v_cat,
                    'color_inst': b_inst - v_inst,
                    'color_cat': b_cat - v_cat,
                    'diff_b': b_cat - b_inst,
                    'diff_v': v_cat - v_inst
                })

    if len(final_data) < 5:
        return f"Error: Too few stars ({len(final_data)}) matched to catalog for reliable fitting."

    # 3. Perform Robust Fits (with 2-sigma clipping)
    c_cat = np.array([d['color_cat'] for d in final_data])
    c_inst = np.array([d['color_inst'] for d in final_data])
    db = np.array([d['diff_b'] for d in final_data])
    dv = np.array([d['diff_v'] for d in final_data])
    
    res_mu, mask_mu, std_mu, res1_mu, std1_mu = perform_robust_fit(c_inst, c_cat)
    res_psi, mask_psi, std_psi, res1_psi, std1_psi = perform_robust_fit(c_cat, db)
    res_eps, mask_eps, std_eps, res1_eps, std1_eps = perform_robust_fit(c_cat, dv)

    # 4. Generate Plots
    plt.style.use('bmh')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    def plot_fit(ax, x, y, mask, res, std, res1, std1, xlabel, ylabel, title, label_pref):
        # Plot Outliers
        ax.scatter(x[~mask], y[~mask], color='red', marker='x', alpha=0.4, label='Outliers (Clipped)')
        # Plot Kept Points
        ax.scatter(x[mask], y[mask], color='blue', alpha=0.6, label='Data Points')
        
        # Plot 1st iteration window (just outer lines)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit1 = res1.slope * x_fit + res1.intercept
        ax.plot(x_fit, y_fit1 + 2*std1, color='r', linestyle=':', alpha=0.4, linewidth=1, label='Initial 2$\sigma$')
        ax.plot(x_fit, y_fit1 - 2*std1, color='r', linestyle=':', alpha=0.4, linewidth=1)

        # Plot Final Fit
        y_fit = res.slope * x_fit + res.intercept
        ax.plot(x_fit, y_fit, 'k-', linewidth=2, label=f'Final {label_pref}={res.slope:.3f}')
        
        # Plot 2-sigma window (use the std from the final fit)
        ax.plot(x_fit, y_fit + 2*std, color='#555', linestyle='--', alpha=0.8, linewidth=1, label=r'Final $\pm$2$\sigma$')
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
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "color_plots.png")
    plt.savefig(plot_path)
    plt.close()

    # 5. Write Report
    with open(report_path, "w") as f:
        f.write("# Color Transformation Calibration Report\n\n")
        f.write(f"Analyzed {len(final_data)} common stars.\n")
        f.write(f"Applied 2-sigma iterative outlier rejection.\n")
        f.write(f"Applied Extinction Correction:\n")
        f.write(f"- B-Filter: k={k_b:.2f}, X={airmass_b:.3f}\n")
        f.write(f"- V-Filter: k={k_v:.2f}, X={airmass_v:.3f}\n\n")
        
        f.write("## Derived Coefficients (Cleaned)\n")
        f.write(f"- **$\\mu$ (Color Scale):** {res_mu.slope:.4f}  (R={res_mu.rvalue:.3f})\n")
        f.write(f"- **$\\psi$ (B-Term):** {res_psi.slope:.4f}  (R={res_psi.rvalue:.3f})\n")
        f.write(f"- **$\\epsilon$ (V-Term):** {res_eps.slope:.4f}  (R={res_eps.rvalue:.3f})\n\n")
        
        f.write("## Transformation Equations\n")
        f.write("Using these coefficients, your calibrated magnitudes are:\n")
        f.write(f"1. $(B-V)_{{std}} = {res_mu.slope:.3f} \cdot (b-v)_{{corr}} + {res_mu.intercept:.3f}$\n")
        f.write(f"2. $B_{{std}} = b_{{corr}} + {res_psi.slope:.3f} \cdot (B-V)_{{std}} + {res_psi.intercept:.3f}$\n")
        f.write(f"3. $V_{{std}} = v_{{corr}} + {res_eps.slope:.3f} \cdot (B-V)_{{std}} + {res_eps.intercept:.3f}$\n")
        f.write("*(Note: v_corr and b_corr are the instrumental magnitudes corrected for extinction)*\n\n")
        
        f.write("![Color Diagnostic Plots](color_plots.png)\n")
        
    # Save coefficients to JSON for later use in Differential Photometry
    json_path = os.path.join(output_dir, "color_coefficients.json")
    coeffs = {
        "Tbv": float(res_mu.slope),
        "Tb_bv": float(res_psi.slope),
        "Tv_bv": float(res_eps.slope)
    }
    try:
        with open(json_path, "w") as jf:
            json.dump(coeffs, jf, indent=4)
    except Exception as e:
        print(f"Warning: Could not save coefficients JSON: {e}")

    return f"Success: Derived coefficients from {len(final_data)} stars. Report: {report_path}"
