import tkinter as tk
from tkinter import ttk, messagebox
import os

def run_config_gui():
    """
    Launches a Tkinter GUI for pipeline configuration.
    Returns a dictionary of settings if "Run" is clicked, or None if closed/cancelled.
    """
    root = tk.Tk()
    root.title("StarID Photometry Pipeline Configuration")
    root.geometry("650x750")
    root.resizable(False, False)

    # Output dictionary
    config = None

    # Variable storage
    vars_dict = {}

    def add_entry(parent, label_text, var_name, default_val, row, col_offset=0, vtype=float):
        ttk.Label(parent, text=label_text).grid(row=row, column=col_offset*2, sticky=tk.W, padx=10, pady=5)
        if vtype == str:
            var = tk.StringVar(value=str(default_val))
        elif vtype == int:
            var = tk.IntVar(value=int(default_val))
        else:
            var = tk.DoubleVar(value=float(default_val))
        vars_dict[var_name] = (var, vtype)
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=col_offset*2+1, sticky=tk.W, padx=10, pady=5)
        return var

    def add_check(parent, label_text, var_name, default_val, row, col_offset=0):
        var = tk.BooleanVar(value=bool(default_val))
        vars_dict[var_name] = (var, bool)
        chk = ttk.Checkbutton(parent, text=label_text, variable=var)
        chk.grid(row=row, column=col_offset*2, columnspan=2, sticky=tk.W, padx=10, pady=5)
        return var

    def add_dropdown(parent, label_text, var_name, options, default_val, row, col_offset=0):
        ttk.Label(parent, text=label_text).grid(row=row, column=col_offset*2, sticky=tk.W, padx=10, pady=5)
        var = tk.StringVar(value=str(default_val))
        vars_dict[var_name] = (var, str)
        cb = ttk.Combobox(parent, textvariable=var, values=options, state="readonly", width=13)
        cb.grid(row=row, column=col_offset*2+1, sticky=tk.W, padx=10, pady=5)
        return var

    # Create Notebook for Tabs
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, fill='both')

    # --- TAB 1: I/O & Filtering ---
    tab_io = ttk.Frame(notebook)
    notebook.add(tab_io, text="I/O & Filtering")

    # Files
    lf_files = ttk.LabelFrame(tab_io, text="Files")
    lf_files.pack(fill="x", padx=10, pady=10)
    add_entry(lf_files, "Input Pattern:", "input_pattern", os.path.join('fitsfiles', '*.fits'), 0, vtype=str)
    add_entry(lf_files, "Ref Catalog:", "reference_catalog", os.path.join('photometry_refstars', 'reference_stars.csv'), 1, vtype=str)

    # Filtering
    lf_filt = ttk.LabelFrame(tab_io, text="Region Filtering")
    lf_filt.pack(fill="x", padx=10, pady=10)
    add_dropdown(lf_filt, "Filter Mode:", "filter_mode", ["all", "xy", "radec"], "all", 0)
    
    ttk.Label(lf_filt, text="XY Bounds (Pixels)").grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(10,0))
    add_entry(lf_filt, "X Min:", "xy_x_min", 100, 2, col_offset=0, vtype=int)
    add_entry(lf_filt, "X Max:", "xy_x_max", 500, 2, col_offset=1, vtype=int)
    add_entry(lf_filt, "Y Min:", "xy_y_min", 100, 3, col_offset=0, vtype=int)
    add_entry(lf_filt, "Y Max:", "xy_y_max", 500, 3, col_offset=1, vtype=int)

    ttk.Label(lf_filt, text="RADEC Bounds").grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(10,0))
    add_entry(lf_filt, "RA Min:", "ra_min", "10h34m00s", 5, col_offset=0, vtype=str)
    add_entry(lf_filt, "RA Max:", "ra_max", "10h35m00s", 5, col_offset=1, vtype=str)
    add_entry(lf_filt, "DEC Min:", "dec_min", "+43d00m00s", 6, col_offset=0, vtype=str)
    add_entry(lf_filt, "DEC Max:", "dec_max", "+43d30m00s", 6, col_offset=1, vtype=str)


    # --- TAB 2: Camera & Detection ---
    tab_cam = ttk.Frame(notebook)
    notebook.add(tab_cam, text="Camera & Detection")

    lf_ccd = ttk.LabelFrame(tab_cam, text="CCD Settings (Error Analysis)")
    lf_ccd.pack(fill="x", padx=10, pady=10)
    add_entry(lf_ccd, "Gain (e-/ADU):", "ccd_gain", 1.27, 0)
    add_entry(lf_ccd, "Read Noise (e-):", "ccd_read_noise", 3.3, 1)
    add_entry(lf_ccd, "Dark Current (e-/s/px):", "ccd_dark_current", 0.0007, 2)
    add_entry(lf_ccd, "Saturation Limit (ADU):", "saturation_limit", 63000, 3, vtype=int)

    lf_det = ttk.LabelFrame(tab_cam, text="Detection (DAOStarFinder)")
    lf_det.pack(fill="x", padx=10, pady=10)
    add_entry(lf_det, "Detection Sigma:", "detect_sigma", 5.0, 0)
    add_entry(lf_det, "Sharpness Low:", "dao_sharplo", 0.2, 1, col_offset=0)
    add_entry(lf_det, "Sharpness High:", "dao_sharphi", 1.0, 1, col_offset=1)
    add_entry(lf_det, "Roundness Low:", "dao_roundlo", -1.2, 2, col_offset=0)
    add_entry(lf_det, "Roundness High:", "dao_roundhi", 1.2, 2, col_offset=1)

    # --- TAB 3: Photometry & Calibration ---
    tab_phot = ttk.Frame(notebook)
    notebook.add(tab_phot, text="Photometry & Calibration")

    lf_ap = ttk.LabelFrame(tab_phot, text="Aperture Photometry")
    lf_ap.pack(fill="x", padx=10, pady=10)
    add_entry(lf_ap, "PSF Box Size (px):", "box_size", 15, 0, vtype=int)
    add_entry(lf_ap, "Aperture Radius (px):", "aperture_radius", 5.0, 1)
    add_entry(lf_ap, "Annulus Inner (px):", "annulus_inner", 7.0, 2)
    add_entry(lf_ap, "Annulus Outer (px):", "annulus_outer", 13.0, 3)

    lf_cal = ttk.LabelFrame(tab_phot, text="Zero Point Calibration")
    lf_cal.pack(fill="x", padx=10, pady=10)
    add_entry(lf_cal, "Match Tolerance (arcsec):", "match_tolerance_arcsec", 8.0, 0)
    add_entry(lf_cal, "Default Zero Point:", "default_zero_point", 23.399, 1)
    add_check(lf_cal, "Run New ZP Calibration (Overwrite Default)", "run_new_calibration", False, 2)
    add_check(lf_cal, "Run Positional Shift Analysis", "run_shift_analysis", True, 3)

    # --- TAB 4: Output & Displays ---
    tab_out = ttk.Frame(notebook)
    notebook.add(tab_out, text="Output Toggles")

    lf_out = ttk.LabelFrame(tab_out, text="Console & Plot Toggles")
    lf_out.pack(fill="x", padx=10, pady=10)
    add_check(lf_out, "Print Massive Aperture Photometry Table", "print_star_detection_table", False, 0)
    add_check(lf_out, "Print Individual PSF Fitting Results", "print_psf_fitting", False, 1)
    add_check(lf_out, "Display Matplotlib Plots (Blocking)", "display_plots", False, 2)
    add_entry(lf_out, "Max Plots to Show/Save per file:", "max_plots_to_show_per_file", 3, 3, vtype=int)

    def on_run():
        nonlocal config
        config = {}
        try:
            for k, (var, vtype) in vars_dict.items():
                config[k] = vtype(var.get())
            
            # Reconstruct dictionary bounds
            config['xy_bounds'] = {
                'x_min': config.pop('xy_x_min'),
                'x_max': config.pop('xy_x_max'),
                'y_min': config.pop('xy_y_min'),
                'y_max': config.pop('xy_y_max')
            }
            config['radec_bounds'] = {
                'ra_min': config.pop('ra_min'),
                'ra_max': config.pop('ra_max'),
                'dec_min': config.pop('dec_min'),
                'dec_max': config.pop('dec_max')
            }
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Input Error", "Please ensure all numerical fields contain valid numbers.")

    # Action Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=15)
    tk.Button(btn_frame, text="Cancel", command=root.destroy, width=15).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Run Pipeline", command=on_run, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT, padx=10)

    # Run the UI loop
    root.mainloop()

    return config

if __name__ == "__main__":
    # Test the GUI
    cfg = run_config_gui()
    print(cfg)
