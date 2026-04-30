import tkinter as tk
from tkinter import ttk, messagebox
import os

def run_config_gui():
    """
    Launches a Tkinter GUI for pipeline configuration.
    Returns a dictionary of settings if "Run" is clicked, or None if closed/cancelled.
    """
    root = tk.Tk()
    root.title("Calibra: Automated Photometric Analysis & Calibration Toolkit")
    root.geometry("720x820")
    root.resizable(False, False)
    root.configure(bg="#f0f2f5") # Light professional gray background

    # --- MODERN STYLING ---
    style = ttk.Style()
    style.theme_use('clam') # Clam is more customizable than default
    
    # Configure Colors
    primary_blue = "#1a3a5f" # Deep space blue
    accent_green = "#2e7d32" # Forest green for "Run"
    text_dark = "#333333"
    
    style.configure("TNotebook", background="#f0f2f5", padding=5)
    style.configure("TNotebook.Tab", background="#e1e4e8", padding=[12, 4], font=("Arial", 10))
    style.map("TNotebook.Tab", background=[("selected", "white")], font=[("selected", ("Arial", 10, "bold"))])
    
    style.configure("TLabelframe", background="white", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background="white", font=("Arial", 10, "bold"), foreground=primary_blue)
    
    style.configure("TLabel", background="white", font=("Arial", 9))
    style.configure("TEntry", fieldbackground="#f8f9fa", borderwidth=1)
    style.configure("TCheckbutton", background="white")
    style.configure("TCombobox", fieldbackground="#f8f9fa")

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

    # --- TAB 0: About ---
    tab_about = ttk.Frame(notebook)
    notebook.add(tab_about, text="About")
    
    about_container = tk.Frame(tab_about, padx=30, pady=30, bg="white")
    about_container.pack(fill="both", expand=True)
    
    # Try to load Logo
    logo_path = os.path.join(os.path.dirname(__file__), "calibra_logo.png")

    try:
        from PIL import Image, ImageTk
        img = Image.open(logo_path)
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        logo_img = ImageTk.PhotoImage(img)
        lbl_logo = tk.Label(about_container, image=logo_img, bg="white")
        lbl_logo.image = logo_img # Keep reference
        lbl_logo.pack(pady=(0, 20))
    except Exception as e:
        # Fallback if PIL is missing or file not found
        tk.Label(about_container, text="[ CALIBRA ]", font=("Arial", 24, "bold"), bg="white", fg=primary_blue).pack(pady=(0, 20))
    
    tk.Label(about_container, text="Calibra: An automated photometric analysis & calibration toolkit", font=("Arial", 16, "bold"), anchor="w", bg="white", fg=primary_blue).pack(fill="x")
#    tk.Label(about_container, text="An automated photometric analysis & calibration toolkit", font=("Arial", 11, "italic"), anchor="w", bg="white", fg="#666").pack(fill="x", pady=(0, 20))
    
    info_frame = tk.Frame(about_container, bg="white")
    info_frame.pack(fill="x", pady=10)
    tk.Label(info_frame, text="Version: 1.1", font=("Arial", 10), anchor="w", bg="white").pack(fill="x")
    tk.Label(info_frame, text="Latest Update: 2026-04-30", font=("Arial", 10), anchor="w", bg="white").pack(fill="x")
    
    tk.Label(about_container, text="Description:", font=("Arial", 11, "bold"), anchor="w", bg="white", fg=primary_blue).pack(fill="x", pady=(10, 5))
    desc_text = (
        "Calibra is a toolkit for the automated analysis of astronomical \n"
        "FITS images. It uses star detection, sub-pixel PSF fitting, \n"
        "aperture photometry, for the zero-point calibration \n"
        "using online catalogs as ATLAS-RefCat2, APASS and others.\n\n"
        "Designed for astronomers and enthusiasts to explore the \n"
        "principles of CCD/CMOS photometry."
    )
    tk.Label(about_container, text=desc_text, justify=tk.LEFT, font=("Arial", 10), anchor="w", bg="white").pack(fill="x")

    # --- TAB 1: I/O & Filtering ---
    tab_io = ttk.Frame(notebook)
    notebook.add(tab_io, text="I/O & Filtering")

    # Files
    lf_files = ttk.LabelFrame(tab_io, text="Files")
    lf_files.pack(fill="x", padx=10, pady=10)
    add_entry(lf_files, "Input Pattern:", "input_pattern", os.path.join('fitsfiles', '*.fits'), 0, vtype=str)
    
    ttk.Label(lf_files, text="Ref Catalog:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
    cat_var = tk.StringVar(value="ATLAS refcat2")
    vars_dict["reference_catalog"] = (cat_var, str)
    cat_cb = ttk.Combobox(lf_files, textvariable=cat_var, values=["ATLAS refcat2", "APASS DR9", "GAIA_DR3", os.path.join('photometry_refstars', 'reference_stars.csv')], width=35)
    cat_cb.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
    
    def browse_catalog():
        from tkinter import filedialog
        filename = filedialog.askopenfilename(initialdir="photometry_refstars", title="Select Reference Catalog", filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
        if filename:
            cat_var.set(filename)
            
    ttk.Button(lf_files, text="Browse...", command=browse_catalog).grid(row=1, column=2, padx=5)

    # Filtering
    lf_filt = ttk.LabelFrame(tab_io, text="Region Filtering")
    lf_filt.pack(fill="x", padx=10, pady=10)
    add_dropdown(lf_filt, "Filter Mode:", "filter_mode", ["all", "xy", "radec"], "all", 0)
    
    ttk.Label(lf_filt, text="XY Bounds (Pixels)").grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(10,0))
    add_entry(lf_filt, "X Min:", "xy_x_min", 200, 2, col_offset=0, vtype=int)
    add_entry(lf_filt, "X Max:", "xy_x_max", 6000, 2, col_offset=1, vtype=int)
    add_entry(lf_filt, "Y Min:", "xy_y_min", 200, 3, col_offset=0, vtype=int)
    add_entry(lf_filt, "Y Max:", "xy_y_max", 4000, 3, col_offset=1, vtype=int)

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
    add_entry(lf_cal, "Min SNR for Calib:", "calib_snr_threshold", 10.0, 2)
    add_check(lf_cal, "Run New ZP Calibration (Overwrite Default)", "run_new_calibration", False, 3)
    add_check(lf_cal, "Run Positional Shift Analysis", "run_shift_analysis", True, 4)

    # --- TAB 4: Output & Displays ---
    tab_out = ttk.Frame(notebook)
    notebook.add(tab_out, text="Output Toggles")

    lf_out = ttk.LabelFrame(tab_out, text="Console & Plot Toggles")
    lf_out.pack(fill="x", padx=10, pady=10)
    add_check(lf_out, "Print Detailed Calibration to Console", "print_detailed_calibration", False, 0)
    add_check(lf_out, "Print Massive Aperture Photometry Table", "print_star_detection_table", False, 1)
    add_check(lf_out, "Print Individual PSF Fitting Results", "print_psf_fitting", False, 2)
    add_check(lf_out, "Display Matplotlib Plots (Blocking)", "display_plots", False, 3)
    add_entry(lf_out, "Max Plots to Show/Save per file:", "max_plots_to_show_per_file", 3, 4, vtype=int)

    # --- TAB 5: Help ---
    tab_help = ttk.Frame(notebook)
    notebook.add(tab_help, text="Help")
    
    help_frame = tk.Frame(tab_help, padx=20, pady=20)
    help_frame.pack(fill="both", expand=True)
    
    tk.Label(help_frame, text="Documentation & Support", font=("Arial", 12, "bold"), bg="white").pack(pady=(0,10), ipady=2)
    tk.Label(help_frame, text="For detailed instructions on how to use Calibra, please refer to the documentation files in the program directory:", justify=tk.LEFT, wraplength=550).pack(pady=5)
    
    import webbrowser
    def open_readme():
        webbrowser.open("README.md")
    def open_manual():
        webbrowser.open("photometry_user_manual.md")
        
    tk.Button(help_frame, text="Open README.md", command=open_readme, width=30).pack(pady=5)
    tk.Button(help_frame, text="Open User Manual (PDF/MD)", command=open_manual, width=30).pack(pady=5)
    
    help_info = (
        "\nQuick Tips:\n"
        "- Ensure your FITS headers have valid WCS (RA/Dec) for online calibration.\n"
        "- Set the Aperture Radius to approximately 2x the FWHM of your stars.\n"
        "- Use the 'Region Filtering' tab to focus on specific targets or avoid edges."
    )
    tk.Label(help_frame, text=help_info, justify=tk.LEFT, wraplength=550, font=("Arial", 9, "italic")).pack(pady=10)

    # Developer Info
    dev_info = (
        "\nDevelopers & Contact:\n"
        "Developed by Stephan Pomp & Google DeepMind Antigravity\n"
        "Contact: stephan.pomp@gmail.com"
    )
    tk.Label(help_frame, text=dev_info, justify=tk.LEFT, font=("Arial", 9), fg="#555").pack(side=tk.BOTTOM, anchor="w")

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
    btn_frame = tk.Frame(root, bg="#f0f2f5")
    btn_frame.pack(pady=20)
    
    cancel_btn = tk.Button(btn_frame, text="Cancel", command=root.destroy, width=15, 
                           font=("Arial", 10), relief="flat", bg="#ccc")
    cancel_btn.pack(side=tk.LEFT, padx=10)
    
    run_btn = tk.Button(btn_frame, text="Run Pipeline", command=on_run, 
                        bg=accent_green, fg="white", font=("Arial", 10, "bold"), 
                        width=25, relief="flat", pady=8)
    run_btn.pack(side=tk.LEFT, padx=10)

    # Run the UI loop
    root.mainloop()

    return config

if __name__ == "__main__":
    # Test the GUI
    cfg = run_config_gui()
    print(cfg)
