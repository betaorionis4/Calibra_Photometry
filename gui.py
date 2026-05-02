import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import sys
import threading

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.log_file = None

    def set_log_file(self, file_path):
        self.log_file = file_path

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(string)

    def flush(self):
        pass

def run_config_gui(pipeline_callback=None):
    """
    Launches a persistent Tkinter GUI for pipeline configuration.
    pipeline_callback: A function that takes (config) and runs the analysis.
    """
    root = tk.Tk()
    root.title("Calibra: Automated Photometric Analysis & Calibration Toolkit")
    root.geometry("900x750")
    root.resizable(True, True)
    root.configure(bg="#f0f2f5") 

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

    def add_entry(parent, label_text, var_name, default_val, row, col_offset=0, vtype=float, width=15):
        ttk.Label(parent, text=label_text).grid(row=row, column=col_offset*2, sticky=tk.W, padx=10, pady=5)
        if vtype == str:
            var = tk.StringVar(value=str(default_val))
        elif vtype == int:
            var = tk.IntVar(value=int(default_val))
        else:
            var = tk.DoubleVar(value=float(default_val))
        vars_dict[var_name] = (var, vtype)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=col_offset*2+1, sticky=tk.W, padx=10, pady=5)
        return var

    def add_check(parent, label_text, var_name, default_val, row, col_offset=0):
        var = tk.BooleanVar(value=bool(default_val))
        vars_dict[var_name] = (var, bool)
        chk = ttk.Checkbutton(parent, text=label_text, variable=var)
        chk.grid(row=row, column=col_offset*2, columnspan=2, sticky=tk.W, padx=10, pady=5)
        return var

    def add_dropdown(parent, label_text, var_name, options, default_val, row, col_offset=0, width=13):
        ttk.Label(parent, text=label_text).grid(row=row, column=col_offset*2, sticky=tk.W, padx=10, pady=5)
        var = tk.StringVar(value=str(default_val))
        vars_dict[var_name] = (var, str)
        cb = ttk.Combobox(parent, textvariable=var, values=options, state="readonly", width=width)
        cb.grid(row=row, column=col_offset*2+1, sticky=tk.W, padx=10, pady=5)
        return var

    # Create Notebook for Tabs
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, fill='both')

    # --- TAB 0: About ---
    tab_about = ttk.Frame(notebook)
    notebook.add(tab_about, text="About")
    
    about_container = tk.Frame(tab_about, padx=30, pady=10, bg="white")
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
        lbl_logo.pack(pady=(0, 5))
    except Exception as e:
        # Fallback if PIL is missing or file not found
        tk.Label(about_container, text="[ CALIBRA ]", font=("Arial", 24, "bold"), bg="white", fg=primary_blue).pack(pady=(0, 20))
    
    tk.Label(about_container, text="Calibra: An automated photometric analysis & calibration toolkit", font=("Arial", 16, "bold"), anchor="w", bg="white", fg=primary_blue).pack(fill="x")
#    tk.Label(about_container, text="An automated photometric analysis & calibration toolkit", font=("Arial", 11, "italic"), anchor="w", bg="white", fg="#666").pack(fill="x", pady=(0, 20))
    
    info_frame = tk.Frame(about_container, bg="white")
    info_frame.pack(fill="x", pady=10)
    tk.Label(info_frame, text="Version: 1.3 \tLatest Update: 2026-05-02", font=("Arial", 10), anchor="w", bg="white").pack(fill="x")
    #tk.Label(info_frame, text="Latest Update: 2026-04-30", font=("Arial", 10), anchor="w", bg="white").pack(fill="x")
    
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
    
    ttk.Label(lf_files, text="Input Pattern:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
    input_var = tk.StringVar(value=r"C:\Astro\Photometry_Calibra\fitsfiles\*.fits")
    vars_dict["input_pattern"] = (input_var, str)
    ttk.Entry(lf_files, textvariable=input_var, width=65).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
    
    def browse_input_dir():
        from tkinter import filedialog
        dirname = filedialog.askdirectory(initialdir=r"C:\Astro\Photometry_Calibra", title="Select FITS Directory")
        if dirname:
            input_var.set(os.path.join(dirname, "*.fits"))
            
    ttk.Button(lf_files, text="Browse...", command=browse_input_dir).grid(row=0, column=2, padx=5)
    
    ttk.Label(lf_files, text="Ref Catalog:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
    cat_var = tk.StringVar(value="GAIA_DR3")
    vars_dict["reference_catalog"] = (cat_var, str)
    cat_cb = ttk.Combobox(lf_files, textvariable=cat_var, values=["GAIA_DR3", "ATLAS refcat2", "APASS DR9", os.path.join('photometry_refstars', 'reference_stars.csv')], width=62)
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

    # --- TAB 1.5: Pre-processing ---
    tab_pre = ttk.Frame(notebook)
    notebook.add(tab_pre, text="Pre-processing")
    
    lf_calib = ttk.LabelFrame(tab_pre, text="FITS Calibration (Bias & Flats)")
    lf_calib.pack(fill="x", padx=10, pady=10)
    
    add_check(lf_calib, "Enable Pre-processing (Apply Bias/Flats)", "enable_calibration", False, 0)
    
    # Plate solving reminder - positioned to avoid overlap
    ttk.Label(lf_calib, text="* Note: Input FITS files must still be plate solved (contain WCS headers).", 
              foreground="#a00", font=("Arial", 8, "italic")).grid(row=0, column=1, sticky=tk.W, padx=(200, 10))
    
    def add_file_selector(parent, label, var_name, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        var = tk.StringVar(value=default)
        vars_dict[var_name] = (var, str)
        ttk.Entry(parent, textvariable=var, width=65).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
        
        def browse():
            from tkinter import filedialog
            fname = filedialog.askopenfilename(initialdir="bias_and_flats", title=f"Select {label}")
            if fname: var.set(fname)
            
        ttk.Button(parent, text="Browse...", command=browse).grid(row=row, column=2, padx=5)
        return var

    add_file_selector(lf_calib, "Master Bias:", "bias_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\Master_Bias_1x1_gain_0.fits", 1)
    add_file_selector(lf_calib, "Master Flat (V):", "flat_v_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\FLAT_Vmag_1x1_gain_0.fits", 2)
    add_file_selector(lf_calib, "Master Flat (B):", "flat_b_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\FLAT_Bmag_1x1_gain_0.fits", 3)


    # --- TAB 2: Camera & Detection ---
    tab_cam = ttk.Frame(notebook)
    notebook.add(tab_cam, text="Camera & Detection")
    
    # --- TAB 2.5: Color Calibration ---
    tab_color = ttk.Frame(notebook)
    notebook.add(tab_color, text="Color Calibration")
    
    lf_color = ttk.LabelFrame(tab_color, text="Derive Transformation Coefficients (B-V Pairs)")
    lf_color.pack(fill="x", padx=10, pady=10)
    
    add_file_selector(lf_color, "B-Filter Results (CSV):", "color_b_csv", "", 0)
    add_file_selector(lf_color, "V-Filter Results (CSV):", "color_v_csv", "", 1)
    
    ttk.Label(lf_color, text="Airmass B:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
    air_b_var = tk.DoubleVar(value=1.0)
    vars_dict["air_b"] = (air_b_var, float)
    ttk.Entry(lf_color, textvariable=air_b_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
    
    ttk.Label(lf_color, text="Airmass V:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
    air_v_var = tk.DoubleVar(value=1.0)
    vars_dict["air_v"] = (air_v_var, float)
    ttk.Entry(lf_color, textvariable=air_v_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)

    ttk.Label(lf_color, text="Extinction k_B (Sea Level ~0.35):").grid(row=2, column=2, sticky=tk.W, padx=10, pady=5)
    k_b_var = tk.DoubleVar(value=0.35)
    vars_dict["k_b"] = (k_b_var, float)
    ttk.Entry(lf_color, textvariable=k_b_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=10, pady=5)
    
    ttk.Label(lf_color, text="Extinction k_V (Sea Level ~0.20):").grid(row=3, column=2, sticky=tk.W, padx=10, pady=5)
    k_v_var = tk.DoubleVar(value=0.20)
    vars_dict["k_v"] = (k_v_var, float)
    ttk.Entry(lf_color, textvariable=k_v_var, width=10).grid(row=3, column=3, sticky=tk.W, padx=10, pady=5)

    color_status_var = tk.StringVar(value="Select B and V result files to begin.")
    tk.Label(tab_color, textvariable=color_status_var, fg="#333", font=("Arial", 9, "italic")).pack(pady=5)

    def on_run_color():
        try:
            b_csv = vars_dict["color_b_csv"][0].get()
            v_csv = vars_dict["color_v_csv"][0].get()
            
            if not os.path.exists(b_csv) or not os.path.exists(v_csv):
                messagebox.showerror("File Error", "Please select valid CSV result files for both filters.")
                return
            
            import csv
            
            color_status_var.set("Reading results and fetching catalog data...")
            root.update_idletasks()
            
            # Load results using standard csv module
            def read_csv_to_dicts(path):
                with open(path, mode='r', encoding='utf-8') as f:
                    return [row for row in csv.DictReader(f)]

            data_b = read_csv_to_dicts(b_csv)
            data_v = read_csv_to_dicts(v_csv)
            
            # Auto-extract Airmass from CSV if present
            if data_b and 'airmass' in data_b[0]:
                try: 
                    am_b = float(data_b[0]['airmass'])
                    air_b_var.set(am_b)
                except: pass
            if data_v and 'airmass' in data_v[0]:
                try: 
                    am_v = float(data_v[0]['airmass'])
                    air_v_var.set(am_v)
                except: pass

            # Convert numeric fields
            for d in data_b:
                for k in ['ra_deg', 'dec_deg', 'mag_inst', 'snr']:
                    if k in d and d[k]: d[k] = float(d[k])
            for d in data_v:
                for k in ['ra_deg', 'dec_deg', 'mag_inst', 'snr']:
                    if k in d and d[k]: d[k] = float(d[k])

            # Get catalog (use center of V image)
            ra_c = sum(d['ra_deg'] for d in data_v) / len(data_v) if data_v else 0
            dec_c = sum(d['dec_deg'] for d in data_v) / len(data_v) if data_v else 0
            
            cat_type = vars_dict["reference_catalog"][0].get()
            from photometry.color_calibration import derive_color_terms
            from photometry.calibration import fetch_online_catalog
            
            catalog = fetch_online_catalog(ra_c, dec_c, catalog_name=cat_type)
            
            if not catalog:
                color_status_var.set("Error: Could not fetch online catalog.")
                return
                
            res = derive_color_terms(data_b, data_v, 
                                     catalog, "photometry_output", 
                                     airmass_b=air_b_var.get(), airmass_v=air_v_var.get(),
                                     k_b=k_b_var.get(), k_v=k_v_var.get())
            color_status_var.set(res)
            
        except Exception as e:
            color_status_var.set(f"Error: {e}")
            messagebox.showerror("Analysis Error", str(e))

    tk.Button(tab_color, text="Run Color Transformation Analysis", command=on_run_color,
              bg="#673ab7", fg="white", font=("Arial", 10, "bold"), pady=8).pack(pady=10)

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
    add_check(lf_cal, "Run New ZP Calibration (Overwrite Default)", "run_new_calibration", True, 3)
    add_check(lf_cal, "Run Positional Shift Analysis", "run_shift_analysis", False, 4)

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

    # --- OUTPUT CONSOLE (Separate Window) ---
    console_win = tk.Toplevel(root)
    console_win.title("Calibra: Process Console")
    console_win.geometry("800x500")
    console_win.configure(bg="#f0f2f5")
    
    console_frame = tk.LabelFrame(console_win, text="Log Output", bg="#f0f2f5", font=("Arial", 10, "bold"))
    console_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    console = scrolledtext.ScrolledText(console_frame, font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
    console.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Redirect stdout and stderr
    sys.stdout = StdoutRedirector(console)
    sys.stderr = StdoutRedirector(console)

    def on_run():
        vars_vals = {}
        try:
            for k, (var, vtype) in vars_dict.items():
                vars_vals[k] = vtype(var.get())
            
            # Reconstruct dictionary bounds
            config_run = {
                'input_pattern': vars_vals.pop('input_pattern'),
                'reference_catalog': vars_vals.pop('reference_catalog'),
                'detect_sigma': vars_vals.pop('detect_sigma'),
                'saturation_limit': vars_vals.pop('saturation_limit'),
                'box_size': vars_vals.pop('box_size'),
                'aperture_radius': vars_vals.pop('aperture_radius'),
                'annulus_inner': vars_vals.pop('annulus_inner'),
                'annulus_outer': vars_vals.pop('annulus_outer'),
                'match_tolerance_arcsec': vars_vals.pop('match_tolerance_arcsec'),
                'default_zero_point': vars_vals.pop('default_zero_point'),
                'calib_snr_threshold': vars_vals.pop('calib_snr_threshold'),
                'run_new_calibration': vars_vals.pop('run_new_calibration'),
                'run_shift_analysis': vars_vals.pop('run_shift_analysis'),
                'ccd_gain': vars_vals.pop('ccd_gain'),
                'ccd_read_noise': vars_vals.pop('ccd_read_noise'),
                'ccd_dark_current': vars_vals.pop('ccd_dark_current'),
                'print_detailed_calibration': vars_vals.pop('print_detailed_calibration'),
                'print_star_detection_table': vars_vals.pop('print_star_detection_table'),
                'print_psf_fitting': vars_vals.pop('print_psf_fitting'),
                'display_plots': vars_vals.pop('display_plots'),
                'max_plots_to_show_per_file': vars_vals.pop('max_plots_to_show_per_file'),
                'dao_sharplo': vars_vals.pop('dao_sharplo'),
                'dao_sharphi': vars_vals.pop('dao_sharphi'),
                'dao_roundlo': vars_vals.pop('dao_roundlo'),
                'dao_roundhi': vars_vals.pop('dao_roundhi'),
                'filter_mode': vars_vals.pop('filter_mode'),
            }
            
            config_run['xy_bounds'] = {
                'x_min': vars_vals.pop('xy_x_min'),
                'x_max': vars_vals.pop('xy_x_max'),
                'y_min': vars_vals.pop('xy_y_min'),
                'y_max': vars_vals.pop('xy_y_max')
            }
            config_run['radec_bounds'] = {
                'ra_min': vars_vals.pop('ra_min'),
                'ra_max': vars_vals.pop('ra_max'),
                'dec_min': vars_vals.pop('dec_min'),
                'dec_max': vars_vals.pop('dec_max')
            }
            config_run['calibration_settings'] = {
                'enable': vars_vals.pop('enable_calibration'),
                'bias_path': vars_vals.pop('bias_path'),
                'flat_v_path': vars_vals.pop('flat_v_path'),
                'flat_b_path': vars_vals.pop('flat_b_path')
            }
            
            if pipeline_callback:
                # Run in a separate thread to keep UI alive
                run_btn.config(state=tk.DISABLED, text="Processing...")
                
                def thread_target():
                    try:
                        results = pipeline_callback(config_run)
                        # Auto-populate Color Calibration tab if B/V pairs found
                        if results:
                            for csv_path, filt in results:
                                f_upper = filt.upper()
                                if 'B' in f_upper:
                                    vars_dict['color_b_csv'][0].set(csv_path)
                                elif 'V' in f_upper:
                                    vars_dict['color_v_csv'][0].set(csv_path)
                    finally:
                        run_btn.config(state=tk.NORMAL, text="Run Pipeline")
                
                thread = threading.Thread(target=thread_target)
                thread.daemon = True
                thread.start()
            else:
                print("No pipeline callback provided.")
                
        except ValueError as e:
            messagebox.showerror("Input Error", "Please ensure all numerical fields contain valid numbers.")

    # Action Buttons
    btn_frame = tk.Frame(root, bg="#f0f2f5")
    btn_frame.pack(pady=10)
    
    exit_btn = tk.Button(btn_frame, text="Exit Calibra", command=root.destroy, width=15, 
                           font=("Arial", 10), relief="flat", bg="#f44336", fg="white")
    exit_btn.pack(side=tk.LEFT, padx=10)
    
    run_btn = tk.Button(btn_frame, text="Run Pipeline", command=on_run, 
                        bg=accent_green, fg="white", font=("Arial", 10, "bold"), 
                        width=25, relief="flat", pady=8)
    run_btn.pack(side=tk.LEFT, padx=10)

    # Ensure closing main window closes everything
    def on_closing():
        root.destroy()
        sys.exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    console_win.protocol("WM_DELETE_WINDOW", lambda: None) # Prevent closing console individually if desired, or just let it close

    # Run the UI loop
    root.mainloop()

if __name__ == "__main__":
    # Test the GUI with a dummy callback
    def dummy_pipeline(cfg):
        import time
        print("Starting dummy pipeline...")
        for i in range(5):
            print(f"Step {i+1}/5 complete...")
            time.sleep(1)
        print("Done!")

    run_config_gui(dummy_pipeline)
