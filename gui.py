import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import numpy as np
import sys
import threading
import csv
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from astropy.coordinates import SkyCoord
import astropy.units as u
from photometry.plate_solve import plate_solve_files, solve_with_astap
from photometry.image_calibration import calibrate_image
from photometry.gui_utils import add_copy_context_menu, SelectableLabel, add_treeview_copy_menu

# Platform-agnostic safe icons to prevent broken emoji boxes on Linux/Ubuntu
IS_LINUX = sys.platform.startswith("linux")

def get_icon(emoji_win, fallback_linux):
    return fallback_linux if IS_LINUX else emoji_win

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widgets = [text_widget]
        self.log_file = None

    def add_widget(self, text_widget):
        if text_widget not in self.text_widgets:
            self.text_widgets.append(text_widget)

    def remove_widget(self, text_widget):
        if text_widget in self.text_widgets:
            self.text_widgets.remove(text_widget)

    def set_log_file(self, file_path):
        self.log_file = file_path

    def write(self, string):
        for w in self.text_widgets:
            try:
                w.insert(tk.END, string)
                w.see(tk.END)
            except:
                pass
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(string)

    def flush(self):
        pass

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="#f0f2f5")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        def update_scrollregion(e):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            req_height = self.scrollable_frame.winfo_reqheight()
            canvas_height = self.canvas.winfo_height()
            if req_height < canvas_height:
                self.canvas.itemconfig(self.canvas_window, height=canvas_height)
            else:
                self.canvas.itemconfig(self.canvas_window, height=0)
            self.reset_view_if_fits()

        self.scrollable_frame.bind("<Configure>", update_scrollregion)

        def _on_mousewheel(event):
            if self.scrollable_frame.winfo_height() > self.canvas.winfo_height():
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                self.canvas.yview_moveto(0)
            
        self.canvas.bind("<Enter>", lambda _: self.canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.canvas.bind("<Leave>", lambda _: self.canvas.unbind_all("<MouseWheel>"))

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        # Stretch inner frame to fill canvas height if it is smaller than the canvas viewport
        req_height = self.scrollable_frame.winfo_reqheight()
        if req_height < event.height:
            self.canvas.itemconfig(self.canvas_window, height=event.height)
        else:
            self.canvas.itemconfig(self.canvas_window, height=0)
        self.reset_view_if_fits()

    def reset_view_if_fits(self):
        if self.scrollable_frame.winfo_height() <= self.canvas.winfo_height():
            self.canvas.yview_moveto(0)

def run_config_gui(pipeline_callback=None):
    """
    Launches a persistent Tkinter GUI for pipeline configuration.
    pipeline_callback: A function that takes (config) and runs the analysis.
    """
    APP_VERSION = "4.0"
    root = tk.Tk()
    root.title(f"Calibra v{APP_VERSION} - Astro Analysis & Photometry Suite")
    root.geometry("1100x750")
    root.minsize(950, 650)
    root.resizable(True, True)
    root.configure(bg="#f0f2f5") 

    # Fix for Windows Taskbar Icon
    if sys.platform == 'win32':
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("google.calibra.v3")
        except:
            pass

    # Set Window Icon
    logo_path = os.path.join(os.path.dirname(__file__), "calibra_logo.png")
    if os.path.exists(logo_path):
        try:
            from PIL import Image, ImageTk
            icon_img = Image.open(logo_path)
            icon_img = icon_img.resize((32, 32), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_img)
            root.iconphoto(True, icon_photo)
        except Exception as e:
            print(f"Error loading icon: {e}")

    # --- MODERN STYLING ---
    style = ttk.Style()
    style.theme_use('clam') # Clam is more customizable than default
    
    # Configure Colors
    primary_blue = "#1a3a5f" # Deep space blue
    accent_green = "#2e7d32" # Forest green for "Run"
    text_dark = "#333333"
    
    style.configure("TNotebook", background="#f0f2f5", padding=5)
    # Unselected tabs: small padding, greyish
    style.configure("TNotebook.Tab", background="#ccd0d5", padding=[10, 2], font=("Segoe UI", 9))
    # Selected tab: larger padding, white, bold
    style.map("TNotebook.Tab", 
              background=[("selected", "white")], 
              padding=[("selected", [20, 8])],
              font=[("selected", ("Segoe UI", 10, "bold"))])
    
    style.configure("TLabelframe", background="white", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background="white", font=("Arial", 10, "bold"), foreground=primary_blue)
    
    style.configure("TLabel", background="white", font=("Arial", 9))
    style.configure("TEntry", fieldbackground="#f8f9fa", borderwidth=1)
    style.configure("TCheckbutton", background="white")
    style.configure("TCombobox", fieldbackground="#f8f9fa")

    # Output dictionary
    config = None

    # Shared File State
    loaded_files = [] # List of dictionaries containing file path and metadata
    vars_dict = {}    # Global-like storage for variable access
    ts_widgets = {}   # Shared widgets for dynamic updates (e.g., filter dropdown)
    
    # Initialize global observer variables early to avoid KeyErrors in tabs
    aavso_obs_var = tk.StringVar(value="XXXX")
    vars_dict["aavso_obs_code"] = (aavso_obs_var, str)
    obs_name_var = tk.StringVar(value="Calibra User")
    vars_dict["observer_name"] = (obs_name_var, str)
    
    # --- MAIN LAYOUT STRUCTURE ---
    # 1. Bottom Bar (Locked to bottom)
    btn_frame = tk.Frame(root, bg="#f0f2f5")
    btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
    
    # Main container frame (fills all remaining space)
    main_container = tk.Frame(root, bg="#f0f2f5")
    main_container.pack(side=tk.TOP, fill="both", expand=True)
    
    # Left Sidebar Frame
    sidebar_frame = tk.Frame(main_container, bg="#132b47", width=180)
    sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
    sidebar_frame.pack_propagate(False)
    

    
    # Content Frame (Right side stacked container)
    content_container = tk.Frame(main_container, bg="white")
    content_container.pack(side=tk.RIGHT, fill="both", expand=True)
    content_container.grid_rowconfigure(0, weight=1)
    content_container.grid_columnconfigure(0, weight=1)

    # Sidebar button dictionary and tab switching logic
    sidebar_buttons = {}
    
    def switch_tab(tab_name):
        # Raise target frame
        if tab_name == "files":
            tab_files_outer.tkraise()
        elif tab_name == "pre":
            tab_pre_outer.tkraise()
        elif tab_name == "analysis":
            tab_analysis_outer.tkraise()
        elif tab_name == "ts":
            tab_ts_outer.tkraise()
        elif tab_name == "datamining":
            tab_datamining_outer.tkraise()
        elif tab_name == "settings":
            tab_settings_outer.tkraise()
        elif tab_name == "about":
            tab_about_outer.tkraise()
        elif tab_name == "help":
            tab_help_outer.tkraise()
            
        # Update button highlighting
        for name, btn in sidebar_buttons.items():
            if name == tab_name:
                btn.config(bg="#2e5b8a", activebackground="#2e5b8a")
            else:
                btn.config(bg="#132b47", activebackground="#132b47")

    def create_sidebar_button(text, tab_name):
        btn = tk.Button(sidebar_frame, text=text, font=("Segoe UI", 9, "bold"), fg="white", bg="#132b47",
                        activeforeground="white", activebackground="#132b47", relief="flat", bd=0,
                        padx=15, pady=12, anchor="w", command=lambda: switch_tab(tab_name))
        btn.pack(fill=tk.X, side=tk.TOP, pady=1)
        
        # Add hover effects
        def on_enter(e):
            if sidebar_buttons[tab_name].cget("bg") != "#2e5b8a":
                btn.config(bg="#1f446e")
        def on_leave(e):
            if sidebar_buttons[tab_name].cget("bg") != "#2e5b8a":
                btn.config(bg="#132b47")
                
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        sidebar_buttons[tab_name] = btn
        return btn
    
    def scan_fits_header(filepath):
        from astropy.io import fits
        metadata = {
            'path': filepath,
            'filename': os.path.basename(filepath),
            'filter': '',
            'binning': '',
            'airmass': '',
            'date_obs': '',
            'exposure': '',
            'wcs': False,
            'object': '',
            'size': f"{os.path.getsize(filepath) / 1024:.1f} KB"
        }
        try:
            def get_hdr_val(hdul, key, default=None):
                for hdu in hdul:
                    if key in hdu.header:
                        return hdu.header[key]
                return default

            with fits.open(filepath) as hdul:
                filt_val = get_hdr_val(hdul, 'FILTER', '')
                metadata['filter'] = str(filt_val)
                xbin = get_hdr_val(hdul, 'XBINNING', '')
                ybin = get_hdr_val(hdul, 'YBINNING', '')
                if xbin and ybin:
                    metadata['binning'] = f"{xbin}x{ybin}"
                # Rounded airmass
                am = get_hdr_val(hdul, 'AIRMASS')
                if am is not None:
                    try:
                        metadata['airmass'] = f"{float(am):.3f}"
                    except:
                        metadata['airmass'] = str(am)
                
                date_obs = get_hdr_val(hdul, 'DATE-OBS', '')
                metadata['date_obs'] = str(date_obs)
                exposure = get_hdr_val(hdul, 'EXPTIME', '')
                metadata['exposure'] = str(exposure)
                
                # Check WCS
                wcs_val = '✗'
                for hdu in hdul:
                    if 'CRVAL1' in hdu.header:
                        wcs_val = '✓'
                        break
                metadata['wcs'] = wcs_val
                
                obj_val = get_hdr_val(hdul, 'OBJECT', '')
                metadata['object'] = str(obj_val)
        except Exception as e:
            print(f"Error reading header for {filepath}: {e}")
        return metadata

    def show_pipeline_info():
        pop = tk.Toplevel(root)
        pop.title("About the Photometric Analysis Pipeline")
        pop.geometry("620x550")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        # Center popup over root window
        pop.update_idletasks()
        rx = root.winfo_x()
        ry = root.winfo_y()
        rw = root.winfo_width()
        rh = root.winfo_height()
        px = rx + (rw - 620) // 2
        py = ry + (rh - 550) // 2
        pop.geometry(f"+{px}+{py}")
        
        # Header Frame
        header = tk.Frame(pop, bg="#1a3a5f", pady=15)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text=f"{get_icon('🌌', '')}  Photometric Analysis Pipeline".strip(), font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        # Main Content Area
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        # Text box (using a premium styled readonly scrolledtext)
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT THE PIPELINE DOES:\n"
            "------------------------\n"
            "When you click 'Run Analysis Pipeline on Selected', Calibra processes each checked FITS file through a series of rigid, automated, consecutive steps:\n\n"
            "1. Star Detection (DAOStarFinder):\n"
            "   Scans the image to find stellar sources matching your detection parameters (sigma threshold, sharpness, roundness bounds).\n\n"
            "2. PSF Coordinate Refinement:\n"
            "   Fits a 2D Gaussian PSF (Point Spread Function) model to each star to obtain sub-pixel centroid coordinates and measures the stellar FWHM.\n\n"
            "3. Aperture Photometry:\n"
            "   Measures the background-subtracted flux of every star. It uses a surrounding annulus to subtract sky background and calculates instrumental magnitudes. Can use fixed or flexible apertures (set to 2x FWHM).\n\n"
            "4. Zero-Point Calibration:\n"
            "   Matches coordinates against your selected online catalog (ATLAS refcat2, APASS DR9, Gaia DR3, or Landolt) using WCS headers. It derives the calculated Zero Point (ZP) of the image to standardize magnitudes.\n\n"
            "5. Positional Shift Analysis (optional):\n"
            "   Analyzes pixel shift offsets between your image and the reference catalog.\n\n"
            "6. Smart CSV Export:\n"
            "   Saves coordinates, flux, SNR, and calibrated magnitudes into 'targets_auto_<filename>.csv'. These files are automatically fed into the downstream Color Transformation & Differential Photometry tools for you!\n\n"
            "------------------------\n"
            "REQUIRED INPUTS:\n"
            "- FITS files loaded in the File Manager with active [X] checkboxes.\n"
            "- FITS files MUST have valid WCS headers (plate-solved RA/Dec coordinates). If they don't, run the 'WCS Plate Solving' tool in the Pre-processing tab first.\n"
            "- Files should ideally be bias-subtracted and flat-field corrected first using 'FITS Calibration' under Pre-processing."
        )
        
        info_box.insert(tk.END, details)
        info_box.config(state=tk.DISABLED) # Read only
        
        btn_close = tk.Button(content_frame, text="Got it!", command=pop.destroy, 
                              bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"), 
                              relief="flat", width=15, pady=8)
        btn_close.pack(pady=(15, 0))

    def show_color_info():
        pop = tk.Toplevel(root)
        pop.title("About Color Coefficient Calibration (B-V Pairs)")
        pop.geometry("620x550")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 620) // 2
        py = ry + (rh - 550) // 2
        pop.geometry(f"+{px}+{py}")
        
        header = tk.Frame(pop, bg="#1a3a5f", pady=15)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text=f"{get_icon('🌈', '')}  Color Coefficient Calibration (B-V Pairs)".strip(), font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT COLOR CALIBRATION DOES:\n"
            "-----------------------------\n"
            "Stellar photometry requires converting instrumental magnitudes (derived from pixel counts) to standard system magnitudes (like the standard Johnson B and V system).\n\n"
            "Because CCD camera sensitivities and filter transmission curves differ slightly from the standard definitions, standard transformation coefficients are needed:\n\n"
            "- Tbv (Color Index Slope):\n"
            "  Transforms the instrumental (b_inst - v_inst) color index into standard (B - V).\n\n"
            "- Tb_bv & Tv_bv (Filter Coefficients):\n"
            "  Apply a second-order standard color correction slope to standardise individual filters.\n\n"
            "How it works:\n"
            "1. It reads the CSV results from standard star fields taken in both B and V filters.\n"
            "2. It matches stellar centroids between the B and V frames.\n"
            "3. It extracts standard (B-V) colors for these stars from online catalogs.\n"
            "4. It runs a linear regression plotting (Standard - Instrumental) magnitudes against the standard (B-V) colors, calculating the slope coefficients automatically!\n\n"
            "-----------------------------\n"
            "REQUIRED INPUTS:\n"
            "- A B-Filter results CSV and a V-Filter results CSV (generated by the star detection pipeline).\n"
            "- Airmass values (k_B, k_V) from your observations. If FITS headers contain airmass, they are loaded automatically unless the override checkbox is checked.\n\n"
            "-----------------------------\n"
            "OUTPUTS:\n"
            "- Regression fit lines and residual plots shown in the preview window.\n"
            "- Extracted slope coefficients (Tbv, Tb_bv, Tv_bv) saved in 'photometry_output/color_coefficients.json' and automatically updated inside the next tab!"
        )
        
        info_box.insert(tk.END, details)
        info_box.config(state=tk.DISABLED)
        
        btn_close = tk.Button(content_frame, text="Got it!", command=pop.destroy, 
                              bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"), 
                              relief="flat", width=15, pady=8)
        btn_close.pack(pady=(15, 0))

    def show_diff_info():
        pop = tk.Toplevel(root)
        pop.title("About Differential Photometry & Light Curves")
        pop.geometry("620x550")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 620) // 2
        py = ry + (rh - 550) // 2
        pop.geometry(f"+{px}+{py}")
        
        header = tk.Frame(pop, bg="#1a3a5f", pady=15)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text=f"{get_icon('📊', '')}  Differential Photometry & Light Curves".strip(), font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT DIFFERENTIAL PHOTOMETRY DOES:\n"
            "-----------------------------------\n"
            "Differential Photometry is the standard method for obtaining extremely high precision light curves of variable stars.\n\n"
            "Atmospheric transparency, clouds, and instrumental tracking errors vary over time. By comparing the brightness of your variable target star against stable comparison (reference) stars in the exact same image frame, these atmospheric variations cancel out completely.\n\n"
            "How it works:\n"
            "1. Select your target star (either by name using the Simbad resolver or by entering manual RA/Dec coordinates).\n"
            "2. Define up to 5 comparison stars (reference stars) in the field of view.\n"
            "3. For each loaded FITS file, it calculates the magnitude difference between the target star and the ensemble average of the comparison stars.\n"
            "4. Applies the color transformation coefficients (Tbv, Tb_bv, Tv_bv) derived in the second tab to yield absolute, standard calibrated magnitudes.\n"
            "5. Automatically extracts the time of observation (Julian Date or JD) from each FITS header to build a light curve.\n\n"
            "-----------------------------------\n"
            "REQUIRED INPUTS:\n"
            "- A time series of B or V FITS files loaded in the File Manager with valid WCS solution.\n"
            "- Valid coordinates for the Target star and stable Comparison stars.\n"
            "- Color coefficients (automatically loaded from color_coefficients.json).\n\n"
            "-----------------------------------\n"
            "OUTPUTS:\n"
            "- An interactive Matplotlib light curve plot.\n"
            "- A detailed results table displaying observation dates, target instrumental magnitudes, and calibrated magnitudes.\n"
            "- Exported photometry results saved to standard CSV and ready-to-submit AAVSO formatted files!"
        )
        
        info_box.insert(tk.END, details)
        info_box.config(state=tk.DISABLED)
        
        btn_close = tk.Button(content_frame, text="Got it!", command=pop.destroy, 
                              bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"), 
                              relief="flat", width=15, pady=8)
        btn_close.pack(pady=(15, 0))

    def show_calibration_info():
        pop = tk.Toplevel(root)
        pop.title("About FITS Calibration (Bias & Flats)")
        pop.geometry("620x550")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 620) // 2
        py = ry + (rh - 550) // 2
        pop.geometry(f"+{px}+{py}")
        
        header = tk.Frame(pop, bg="#1a3a5f", pady=15)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text=f"{get_icon('⚙', '')}  FITS Instrumental Calibration".strip(), font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT FITS CALIBRATION DOES:\n"
            "----------------------------\n"
            "Raw astronomical images suffer from noise, pixel imperfections, and dust. Calibration cleans raw observations, leaving only genuine starlight:\n\n"
            "1. Master Bias Subtraction:\n"
            "   Removes read-out noise, fixed-pattern electronic noise, and internal offsets generated by the camera CCD sensor.\n\n"
            "2. Master Flat-Field Division:\n"
            "   Corrects for pixel-to-pixel sensitivity variances, dust artifacts ('dust donuts'), and optical vignetting (dark corners). Flats standardize light throughput across the frame.\n\n"
            "How it works:\n"
            "- You check raw FITS files in the File Manager.\n"
            "- The engine automatically maps and applies the correct Master Flat corresponding to the filter specified inside the FITS header (e.g. standard B or V flats).\n\n"
            "----------------------------\n"
            "REQUIRED INPUTS:\n"
            "- FITS files loaded and checked in the File Manager.\n"
            "- A Master Bias FITS file.\n"
            "- Master B-Filter and V-Filter Flats.\n\n"
            "----------------------------\n"
            "OUTPUTS:\n"
            "- Cleaned, calibrated FITS copies created under 'fitsfiles/calibrated/' prefixed with 'cal_'.\n"
            "- The File Manager table is automatically updated with these calibrated files in-place!"
        )
        
        info_box.insert(tk.END, details)
        info_box.config(state=tk.DISABLED)
        
        btn_close = tk.Button(content_frame, text="Got it!", command=pop.destroy, 
                              bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"), 
                              relief="flat", width=15, pady=8)
        btn_close.pack(pady=(15, 0))

    def show_plate_solve_info():
        pop = tk.Toplevel(root)
        pop.title("About ASTAP Plate Solving")
        pop.geometry("620x550")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 620) // 2
        py = ry + (rh - 550) // 2
        pop.geometry(f"+{px}+{py}")
        
        header = tk.Frame(pop, bg="#1a3a5f", pady=15)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text="🌌  ASTAP Plate Solving & WCS Alignment", font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT PLATE SOLVING DOES:\n"
            "-------------------------\n"
            "Plate solving compares the star patterns in your FITS image against standard celestial star catalogs to calculate the precise RA/Dec coordinates, scale, and rotation of your image.\n\n"
            "This embeds standard WCS (World Coordinate System) metadata into the FITS header, which is absolutely required to match stars automatically against online catalogs for zero-point and magnitude calibration.\n\n"
            "How it works:\n"
            "1. Invokes the external ASTAP astronomical command-line solver.\n"
            "2. Detects constellations in the image, maps catalog matches, and solves the center RA/Dec coordinates.\n"
            "3. Writes WCS coordinates directly into the new FITS header copy.\n\n"
            "-------------------------\n"
            "REQUIRED INPUTS:\n"
            "- Checked FITS files in the File Manager.\n"
            "- External ASTAP executable path installed on your computer.\n"
            "- A rough search radius around the target field.\n\n"
            "-------------------------\n"
            "OUTPUTS:\n"
            "- Solved FITS copy generated with standard suffix (e.g. '_wcs').\n"
            "- File Manager table automatically displays WCS success checkmarks (✓) for solved files!"
        )
        
        info_box.insert(tk.END, details)
        info_box.config(state=tk.DISABLED)
        
        btn_close = tk.Button(content_frame, text="Got it!", command=pop.destroy, 
                              bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"), 
                              relief="flat", width=15, pady=8)
        btn_close.pack(pady=(15, 0))

    def show_lightcurve_info():
        pop = tk.Toplevel(root)
        pop.title("About Time-Series Photometry & Light Curves")
        pop.geometry("620x550")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 620) // 2
        py = ry + (rh - 550) // 2
        pop.geometry(f"+{px}+{py}")
        
        header = tk.Frame(pop, bg="#1a3a5f", pady=15)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text="📈  Time-Series Photometry & Light Curves", font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT TIME-SERIES PHOTOMETRY DOES:\n"
            "-----------------------------------\n"
            "Tracks the changing brightness of a variable target star over a sequence of observations. It performs differential ensemble photometry to yield high precision light curves.\n\n"
            "How it works:\n"
            "1. Target Setup: Enter or resolve coordinates of the target star.\n"
            "2. Ensemble Setup: Add stable reference/comparison stars. Click 'Get AAVSO Ref Stars' to automatically query AAVSO coordinate chart magnitudes for your field.\n"
            "3. Processing Sequence: Iterates through checked FITS files matching the chosen filter. Subtracts background sky, scales magnitude variations against comparison stars, applies standard color index adjustments, and computes precise observation times (Julian Date).\n\n"
            "-----------------------------------\n"
            "REQUIRED INPUTS:\n"
            "- A sequence of B or V FITS files loaded and checked in the File Manager (bias-subtracted, flat-fielded, and plate solved).\n"
            "- Target and Ensemble star RA/Dec coordinates.\n"
            "- Color transformation coefficients (loaded from previous runs).\n\n"
            "-----------------------------\n"
            "OUTPUTS:\n"
            "- Interactive plotted light curve preview.\n"
            "- Scrollable results table containing Julian dates, calibrated magnitudes, SNR, and errors.\n"
            "- Photometry results saved to standard CSV and AAVSO-format text reports ready for upload!"
        )
        
        info_box.insert(tk.END, details)
        info_box.config(state=tk.DISABLED)
        
        btn_close = tk.Button(content_frame, text="Got it!", command=pop.destroy, 
                              bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"), 
                              relief="flat", width=15, pady=8)
        btn_close.pack(pady=(15, 0))

    def open_fits_viewer_for_item(item):
        if not item: return
        try:
            from photometry.fits_viewer import FITSViewer
            ref_cat = vars_dict["reference_catalog"][0].get() if "reference_catalog" in vars_dict else "ATLAS"
            idx = int(item)
            file_path = loaded_files[idx]['path']
            filt = loaded_files[idx]['filter'].upper()
            b_key = vars_dict["filter_b_keyword"][0].get().upper() if "filter_b_keyword" in vars_dict else "BMAG"
            
            if b_key in filt:
                def_zp = float(vars_dict["default_zp_b"][0].get()) if "default_zp_b" in vars_dict else 23.399
            else:
                def_zp = float(vars_dict["default_zp_v"][0].get()) if "default_zp_v" in vars_dict else 23.399
            
            if os.path.exists(file_path):
                # 1. Gather aperture/annulus configuration
                viewer_config = {
                    'aperture_radius': float(vars_dict["aperture_radius"][0].get()),
                    'annulus_inner': float(vars_dict["annulus_inner"][0].get()),
                    'annulus_outer': float(vars_dict["annulus_outer"][0].get()),
                    'use_flexible_aperture': vars_dict["use_flexible_aperture"][0].get(),
                    'aperture_fwhm_factor': float(vars_dict["aperture_fwhm_factor"][0].get()),
                    'annulus_inner_gap': float(vars_dict["annulus_inner_gap"][0].get()),
                    'annulus_width': float(vars_dict["annulus_width"][0].get()),
                }
                
                # 2. Gather INITIAL STARS from Light Curves tab
                initial_stars = {'variable': None, 'check': None, 'refs': []}
                
                # Target
                t_mode = vars_dict["ts_target_mode"][0].get()
                t_name = vars_dict["ts_target_name"][0].get().strip()
                if t_mode == "name" and t_name:
                    try:
                        c = SkyCoord.from_name(t_name)
                        initial_stars['variable'] = {'ra': c.ra.deg, 'dec': c.dec.deg, 'name': t_name}
                    except: pass
                elif t_mode == "manual":
                    try:
                        c = SkyCoord(vars_dict["ts_target_ra"][0].get(), vars_dict["ts_target_dec"][0].get(), unit=(u.hourangle, u.deg))
                        initial_stars['variable'] = {'ra': c.ra.deg, 'dec': c.dec.deg, 'name': t_name or "Target"}
                    except: pass
                
                # Refs & Check
                cs_idx = ts_check_star_idx_var.get()
                for i in range(5):
                    name = vars_dict[f"ts_ref_{i}_name"][0].get().strip()
                    if vars_dict[f"ts_ref_{i}_has_manual"][0].get():
                        ra = vars_dict[f"ts_ref_{i}_ra"][0].get()
                        dec = vars_dict[f"ts_ref_{i}_dec"][0].get()
                        s_data = {'ra': ra, 'dec': dec, 'name': name or f"Star_{i+1}"}
                        if i == cs_idx: initial_stars['check'] = s_data
                        elif vars_dict[f"ts_ref_{i}_use"][0].get(): initial_stars['refs'].append(s_data)
                    elif name:
                        try:
                            c = SkyCoord.from_name(name)
                            s_data = {'ra': c.ra.deg, 'dec': c.dec.deg, 'name': name}
                            if i == cs_idx: initial_stars['check'] = s_data
                            elif vars_dict[f"ts_ref_{i}_use"][0].get(): initial_stars['refs'].append(s_data)
                        except: pass

                # 3. Define EXPORT Callbacks
                def update_light_curve_stars(data):
                    if data['variable']:
                        v = data['variable']
                        vars_dict["ts_target_mode"][0].set("manual")
                        vars_dict["ts_target_name"][0].set(v['name'])
                        c = SkyCoord(v['ra'], v['dec'], unit=u.deg)
                        vars_dict["ts_target_ra"][0].set(c.ra.to_string(unit='hour', sep=':', precision=2))
                        vars_dict["ts_target_dec"][0].set(c.dec.to_string(unit='degree', sep=':', precision=2, alwayssign=True))
                        
                        # Use exported mag/bv if available to skip re-fetch
                        if v.get('mag') is not None: vars_dict["ts_target_mag"][0].set(v['mag'])
                        if v.get('bv') is not None: vars_dict["ts_target_bv"][0].set(v['bv'])
                    
                    # Check & Refs
                    slots_filled = 0
                    if data['check']:
                        c = data['check']
                        vars_dict["ts_ref_0_name"][0].set(c['name'])
                        vars_dict["ts_ref_0_ra"][0].set(c['ra'])
                        vars_dict["ts_ref_0_dec"][0].set(c['dec'])
                        vars_dict["ts_ref_0_has_manual"][0].set(True)
                        vars_dict["ts_ref_0_use"][0].set(False)
                        ts_check_star_idx_var.set(0)
                        
                        if c.get('mag') is not None: vars_dict["ts_ref_0_mag"][0].set(c['mag'])
                        if c.get('bv') is not None: vars_dict["ts_ref_0_bv"][0].set(c['bv'])
                        slots_filled = 1
                    else:
                        ts_check_star_idx_var.set(-1)
                    
                    current_idx = slots_filled
                    for r in data['refs']:
                        if current_idx >= 5: break
                        vars_dict[f"ts_ref_{current_idx}_name"][0].set(r['name'])
                        vars_dict[f"ts_ref_{current_idx}_ra"][0].set(r['ra'])
                        vars_dict[f"ts_ref_{current_idx}_dec"][0].set(r['dec'])
                        vars_dict[f"ts_ref_{current_idx}_has_manual"][0].set(True)
                        vars_dict[f"ts_ref_{current_idx}_use"][0].set(True)
                        
                        if r.get('mag') is not None: vars_dict[f"ts_ref_{current_idx}_mag"][0].set(r['mag'])
                        if r.get('bv') is not None: vars_dict[f"ts_ref_{current_idx}_bv"][0].set(r['bv'])
                        current_idx += 1
                    
                    for idx in range(current_idx, 5):
                        vars_dict[f"ts_ref_{idx}_name"][0].set("")
                        vars_dict[f"ts_ref_{idx}_has_manual"][0].set(False)
                        vars_dict[f"ts_ref_{idx}_use"][0].set(False)
                        if f"ts_ref_{idx}_coord_label" in vars_dict:
                            vars_dict[f"ts_ref_{idx}_coord_label"][0].set("")
                    
                    # TRIGGER AUTO-FETCH (Only if data is missing)
                    def trigger_fetch_if_needed():
                        # Target
                        if data['variable'] and data['variable'].get('mag') is None:
                            if 'ts_target_fetch_func' in vars_dict:
                                root.after(100, vars_dict['ts_target_fetch_func'])
                        
                        # Refs
                        if 'ts_ref_fetch_funcs' in vars_dict:
                            # Check
                            if data['check'] and data['check'].get('mag') is None:
                                if 0 in vars_dict['ts_ref_fetch_funcs']:
                                    root.after(200, vars_dict['ts_ref_fetch_funcs'][0])
                            
                            # Ensemble Refs
                            for i in range(slots_filled, current_idx):
                                if i in vars_dict['ts_ref_fetch_funcs'] and data['refs'][i-slots_filled].get('mag') is None:
                                    root.after(i*300, vars_dict['ts_ref_fetch_funcs'][i])
                                    
                    root.after(500, trigger_fetch_if_needed)
                    
                    # Autosave session to disk (Fix)
                    save_session()

                def update_apertures(ap_data):
                    vars_dict["aperture_radius"][0].set(ap_data['aperture'])
                    vars_dict["annulus_inner"][0].set(ap_data['annulus_in'])
                    vars_dict["annulus_outer"][0].set(ap_data['annulus_out'])
                    ts_status_var.set("Aperture settings updated from FITS Viewer.")
                    
                    # Autosave session to disk (Fix)
                    save_session()

                # Pass the AAVSO cache if available
                aavso_cache = getattr(on_get_aavso_refs, 'cache', None)
                
                viewer_win = tk.Toplevel(root)
                FITSViewer(viewer_win, file_path, ref_catalog=ref_cat, default_zp=def_zp, 
                           config=viewer_config, initial_stars=initial_stars, 
                           aavso_stars=aavso_cache,
                           export_callback=update_light_curve_stars,
                           aperture_export_callback=update_apertures)
            else:
                messagebox.showerror("Error", f"File not found: {file_path}")
        except Exception as e:
            print(f"Error opening viewer: {e}")

    def on_double_click(event):
        item = tree.identify_row(event.y)
        if not item: return
        open_fits_viewer_for_item(item)

    def on_open_viewer_button():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a FITS file in the table first.")
            return
        open_fits_viewer_for_item(selected[0])

    def add_entry(parent, label_text, var_name, default_val, row, col_offset=0, vtype=float, width=15):
        ttk.Label(parent, text=label_text).grid(row=row, column=col_offset*2, sticky=tk.W, padx=10, pady=5)
        if var_name in vars_dict:
            var = vars_dict[var_name][0]
        else:
            if vtype == str:
                var = tk.StringVar(value=str(default_val))
            elif vtype == int:
                var = tk.IntVar(value=int(default_val))
            else:
                var = tk.DoubleVar(value=float(default_val))
            vars_dict[var_name] = (var, vtype)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col_offset*2+1, sticky=tk.W, padx=10, pady=5)
        return var

    def add_check(parent, label_text, var_name, default_val, row, col_offset=0):
        var = tk.BooleanVar(value=bool(default_val))
        vars_dict[var_name] = (var, bool)
        ttk.Checkbutton(parent, text=label_text, variable=var).grid(row=row, column=col_offset*2, columnspan=2, sticky=tk.W, padx=10, pady=5)
        return var

    def add_dropdown(parent, label_text, var_name, options, default_val, row, col_offset=0, width=13):
        tk.Label(parent, text=label_text).grid(row=row, column=col_offset*2, sticky=tk.W, padx=10, pady=5)
        var = tk.StringVar(value=str(default_val))
        vars_dict[var_name] = (var, str)
        cb = ttk.Combobox(parent, textvariable=var, values=options, state="readonly", width=width)
        cb.grid(row=row, column=col_offset*2+1, sticky=tk.W, padx=10, pady=5)
        return var

    def add_file_selector(parent, label, var_name, default, row, initial_dir="."):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        var = tk.StringVar(value=default)
        vars_dict[var_name] = (var, str)
        entry = ttk.Entry(parent, textvariable=var, width=65)
        entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
        
        def browse():
            from tkinter import filedialog
            fname = filedialog.askopenfilename(initialdir=initial_dir, title=f"Select {label}")
            if fname: var.set(fname)
            
        ttk.Button(parent, text="Browse...", command=browse).grid(row=row, column=2, padx=5)
        return var

    def on_run():
        vars_vals = {}
        try:
            for k, val in vars_dict.items():
                if not isinstance(val, (tuple, list)) or len(val) < 2:
                    continue
                var, vtype = val[0], val[1]
                try:
                    vars_vals[k] = vtype(var.get())
                except ValueError as ve:
                    raise ValueError(f"Field '{k}' (current value: '{var.get()}') must be a valid {vtype.__name__}.") from ve
            
            selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
            if not selected_iids:
                messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager.")
                return
            
            file_list = [loaded_files[int(iid)]['path'] for iid in selected_iids]

            # Reconstruct dictionary bounds
            config_run = {
                'input_pattern': file_list,
                'reference_catalog': vars_vals.pop('reference_catalog'),
                'detect_sigma': vars_vals.pop('detect_sigma'),
                'saturation_limit': vars_vals.pop('saturation_limit'),
                'box_size': vars_vals.pop('box_size'),
                'aperture_radius': vars_vals.pop('aperture_radius'),
                'annulus_inner': vars_vals.pop('annulus_inner'),
                'annulus_outer': vars_vals.pop('annulus_outer'),
                'match_tolerance_arcsec': vars_vals.pop('match_tolerance_arcsec'),
                'default_zp_v': vars_vals.pop('default_zp_v'),
                'default_zp_b': vars_vals.pop('default_zp_b'),
                'filter_v_keyword': vars_vals.pop('filter_v_keyword'),
                'filter_b_keyword': vars_vals.pop('filter_b_keyword'),
                'calib_snr_threshold': vars_vals.pop('calib_snr_threshold'),
                'catalog_search_radius': vars_vals.pop('catalog_search_radius'),
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
                'run_star_detection': vars_vals.pop('run_star_detection'),
                'dao_roundhi': vars_vals.pop('dao_roundhi'),
                'filter_mode': vars_vals.pop('filter_mode'),
                'use_flexible_aperture': vars_vals.pop('use_flexible_aperture'),
                'aperture_fwhm_factor': vars_vals.pop('aperture_fwhm_factor'),
                'annulus_inner_gap': vars_vals.pop('annulus_inner_gap'),
                'annulus_width': vars_vals.pop('annulus_width'),
                'dao_sharplo': vars_vals.pop('dao_sharplo'),
                'dao_sharphi': vars_vals.pop('dao_sharphi'),
                'dao_roundlo': vars_vals.pop('dao_roundlo'),
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
                'enable': False,
                'bias_path': vars_vals.pop('bias_path'),
                'flat_v_path': vars_vals.pop('flat_v_path'),
                'flat_b_path': vars_vals.pop('flat_b_path')
            }
            
            if pipeline_callback:
                # Run in a separate thread to keep UI alive
                run_btn.config(state=tk.DISABLED, text="Processing...")
                
                def thread_target():
                    try:
                        pipeline_terminal.delete("1.0", tk.END)
                        if hasattr(sys.stdout, "add_widget"):
                            sys.stdout.add_widget(pipeline_terminal)
                        if hasattr(sys.stderr, "add_widget"):
                            sys.stderr.add_widget(pipeline_terminal)

                        results = pipeline_callback(config_run)
                        if results:
                            last_zp_v = None
                            last_zp_b = None
                            summary_lines = []
                            for idx, res in enumerate(results):
                                csv_path = res[0]
                                filt = res[1]
                                zp_val = res[2] if len(res) >= 3 else None
                                shift_val = res[3] if len(res) >= 4 else None
                                det_val = res[4] if len(res) >= 5 else None
                                
                                f_upper = str(filt).upper()
                                b_key = config_run.get('filter_b_keyword', 'BMAG').upper()
                                if b_key in f_upper:
                                    vars_dict['color_b_csv'][0].set(csv_path)
                                    last_zp_b = zp_val
                                else:
                                    vars_dict['color_v_csv'][0].set(csv_path)
                                    last_zp_v = zp_val
                                
                                # Build summary info for this file
                                f_name = os.path.basename(csv_path).replace("targets_auto_", "")
                                line = f"File {idx+1}: {f_name}\n"
                                line += f"  • Output CSV: {os.path.abspath(csv_path)}\n"
                                if zp_val is not None:
                                    line += f"  • Filter: {filt}  |  Calculated Zero Point: {zp_val:.4f}\n"
                                if det_val:
                                    det_parts = []
                                    if det_val.get('mag_calibrated') is not None and not np.isnan(det_val['mag_calibrated']):
                                        det_parts.append(f"Calibrated: {det_val['mag_calibrated']:.2f}")
                                    if det_val.get('mag_inst') is not None and not np.isnan(det_val['mag_inst']):
                                        det_parts.append(f"Instrumental: {det_val['mag_inst']:.2f}")
                                    if det_parts:
                                        line += f"  • Detection Limit (@SNR={det_val.get('snr', 3.0):.1f}): " + " | ".join(det_parts) + "\n"
                                if shift_val:
                                    line += f"  • Shift Analysis: Matched {shift_val['count']} stars\n"
                                    line += f"    Median Shift: dX = {shift_val['med_dx']:+.2f} px, dY = {shift_val['med_dy']:+.2f} px\n"
                                    line += f"    Arcsec Shift: dRA = {shift_val['med_dra']:+.2f}\", dDec = {shift_val['med_ddec']:+.2f}\"\n"
                                else:
                                    if not config_run.get('run_shift_analysis'):
                                        line += "  • Shift Analysis: Disabled in Pipeline Configuration.\n"
                                    else:
                                        line += "  • Shift Analysis: No matching catalog stars found (verify FITS has valid WCS).\n"
                                summary_lines.append(line)
                            
                            full_summary = f"Pipeline execution completed successfully on {len(results)} file(s).\n\n"
                            full_summary += "\n".join(summary_lines)
                            root.after(0, lambda: update_pipeline_results_display(full_summary))
                            
                            # Update GUI with latest calculated ZPs
                            if last_zp_v is not None:
                                root.after(0, lambda v=last_zp_v: vars_dict["default_zp_v"][0].set(round(v, 3)))
                            if last_zp_b is not None:
                                root.after(0, lambda v=last_zp_b: vars_dict["default_zp_b"][0].set(round(v, 3)))
                    finally:
                        if hasattr(sys.stdout, "remove_widget"):
                            sys.stdout.remove_widget(pipeline_terminal)
                        if hasattr(sys.stderr, "remove_widget"):
                            sys.stderr.remove_widget(pipeline_terminal)
                        run_btn.config(state=tk.NORMAL, text="Run Analysis Pipeline on Selected")
                
                thread = threading.Thread(target=thread_target)
                thread.daemon = True
                thread.start()
            else:
                print("No pipeline callback provided.")
                
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please ensure all numerical fields contain valid numbers.\n\nDetail: {e}")

    def on_run_color():
        bc, vc = get_checked_b_v_counts()
        if bc != 1 or vc != 1:
            if not messagebox.askyesno("Input Warning", 
                f"Color Transformation Analysis requires exactly one B and one V file.\n\n"
                f"Currently checked in File Manager: {bc} B files, {vc} V files.\n\n"
                f"Do you want to ignore this and proceed with the manually selected CSV files?"):
                return
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
            
            # Auto-extract Airmass from CSV if present (unless overridden)
            if not override_airmass_var.get():
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
            valid_coords = [d for d in data_v if isinstance(d.get('ra_deg'), float) and isinstance(d.get('dec_deg'), float)]
            ra_c = sum(d['ra_deg'] for d in valid_coords) / len(valid_coords) if valid_coords else 0
            dec_c = sum(d['dec_deg'] for d in valid_coords) / len(valid_coords) if valid_coords else 0
            
            cat_type = vars_dict["reference_catalog"][0].get()
            search_radius = float(vars_dict["catalog_search_radius"][0].get())
            from photometry.color_calibration import derive_color_terms
            from photometry.calibration import get_ref_stars
            
            catalog = get_ref_stars(cat_type, ra_c, dec_c, radius_arcmin=search_radius, verbose=True)
            
            if not catalog:
                color_status_var.set("Error: Could not fetch online catalog.")
                return
                
            res = derive_color_terms(data_b, data_v, 
                                     catalog, "photometry_output", 
                                     airmass_b=air_b_var.get(), airmass_v=air_v_var.get(),
                                     k_b=vars_dict["extinction_kb"][0].get(), k_v=vars_dict["extinction_kv"][0].get(),
                                     axes=color_coeff_axes)
            color_coeff_canvas.draw()
            color_status_var.set(res)
            
        except Exception as e:
            color_status_var.set(f"Error: {e}")
            messagebox.showerror("Analysis Error", str(e))

    def on_run_diff():
        bc, vc = get_checked_b_v_counts()
        if bc != 1 or vc != 1:
            if not messagebox.askyesno("Input Warning", 
                f"Differential Photometry requires exactly one B and one V file.\n\n"
                f"Currently checked in File Manager: {bc} B files, {vc} V files.\n\n"
                f"Do you want to ignore this and proceed with the manually selected CSV files?"):
                return
        try:
            b_csv = vars_dict["diff_b_csv"][0].get()
            v_csv = vars_dict["diff_v_csv"][0].get()
            cat_type = vars_dict["reference_catalog"][0].get()
            
            if not os.path.exists(b_csv) or not os.path.exists(v_csv):
                messagebox.showerror("File Error", "Please select valid CSV result files for both filters.")
                return
                
            from photometry.diff_photometry import run_differential_photometry
            diff_status_var.set("Running differential photometry...")
            root.update_idletasks()
            
            manual_coord = None
            if ref_mode_var.get() == "manual":
                ra_str = f"{ra_h_var.get()}h{ra_m_var.get()}m{ra_s_var.get()}s"
                dec_str = f"{dec_d_var.get()}d{dec_m_var.get()}m{dec_s_var.get()}s"
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                try:
                    c = SkyCoord(f"{ra_str} {dec_str}")
                    manual_coord = (c.ra.deg, c.dec.deg)
                except Exception as e:
                    messagebox.showerror("Coordinate Error", f"Invalid manual coordinates format.\n{e}")
                    diff_status_var.set("Error: Invalid manual coordinates.")
                    return
            elif ref_mode_var.get() == "name":
                star_name = star_name_var.get().strip()
                if not star_name:
                    messagebox.showerror("Name Error", "Please enter a star name.")
                    diff_status_var.set("Error: Empty star name.")
                    return
                from astropy.coordinates import SkyCoord
                from astropy.coordinates.name_resolve import NameResolveError
                diff_status_var.set(f"Resolving '{star_name}' via Simbad...")
                root.update_idletasks()
                try:
                    c = SkyCoord.from_name(star_name)
                    manual_coord = (c.ra.deg, c.dec.deg)
                    print(f"Resolved '{star_name}' to RA: {c.ra.deg:.5f}, Dec: {c.dec.deg:.5f}")
                except NameResolveError as e:
                    # Fallback to local catalog
                    try:
                        from photometry.calibration import read_reference_catalog
                        cat_file = vars_dict["reference_catalog"][0].get()
                        if os.path.exists(cat_file):
                            ref_stars = read_reference_catalog(cat_file)
                            for s in ref_stars:
                                if str(s.get('id', '')).upper() == star_name.upper():
                                    manual_coord = (s['ra_deg'], s['dec_deg'])
                                    print(f"Resolved '{star_name}' to RA: {manual_coord[0]:.5f}, Dec: {manual_coord[1]:.5f} from local catalog.")
                                    break
                    except Exception:
                        pass
                    
                    if manual_coord is None:
                        messagebox.showerror("Resolution Error", f"Could not resolve name '{star_name}' via Simbad or Local Catalog.\n\nNote: Make sure your AAVSO CSV is loaded in the Analysis tab if you are using AUIDs like 000-BJS-555.")
                        diff_status_var.set(f"Error: Could not resolve '{star_name}'.")
                        return
                except Exception as e:
                    messagebox.showerror("Resolution Error", f"Error looking up '{star_name}':\n{e}")
                    diff_status_var.set("Error during name resolution.")
                    return
            
            manual_target_coord = None
            target_mode = target_mode_var.get()
            if target_mode == "manual":
                ra_str = f"{target_ra_h_var.get()}h{target_ra_m_var.get()}m{target_ra_s_var.get()}s"
                dec_str = f"{target_dec_d_var.get()}d{target_dec_m_var.get()}m{target_dec_s_var.get()}s"
                try:
                    c = SkyCoord(f"{ra_str} {dec_str}")
                    manual_target_coord = (c.ra.deg, c.dec.deg)
                except Exception as e:
                    messagebox.showerror("Coordinate Error", f"Invalid manual target coordinates format.\n{e}")
                    diff_status_var.set("Error: Invalid target coordinates.")
                    return
            elif target_mode == "name":
                star_name = target_name_var.get().strip()
                if not star_name:
                    messagebox.showerror("Name Error", "Please enter a target star name.")
                    diff_status_var.set("Error: Empty target star name.")
                    return
                from astropy.coordinates import SkyCoord
                from astropy.coordinates.name_resolve import NameResolveError
                diff_status_var.set(f"Resolving '{star_name}' via Simbad...")
                root.update_idletasks()
                try:
                    c = SkyCoord.from_name(star_name)
                    manual_target_coord = (c.ra.deg, c.dec.deg)
                    print(f"Resolved target '{star_name}' to RA: {c.ra.deg:.5f}, Dec: {c.dec.deg:.5f}")
                except NameResolveError as e:
                    # Fallback to local catalog
                    try:
                        from photometry.calibration import read_reference_catalog
                        cat_file = vars_dict["reference_catalog"][0].get()
                        if os.path.exists(cat_file):
                            ref_stars = read_reference_catalog(cat_file)
                            for s in ref_stars:
                                if str(s.get('id', '')).upper() == star_name.upper():
                                    manual_target_coord = (s['ra_deg'], s['dec_deg'])
                                    print(f"Resolved target '{star_name}' to RA: {manual_target_coord[0]:.5f}, Dec: {manual_target_coord[1]:.5f} from local catalog.")
                                    break
                    except Exception:
                        pass
                        
                    if manual_target_coord is None:
                        messagebox.showerror("Resolution Error", f"Could not resolve target name '{star_name}' via Simbad or Local Catalog.\n\nNote: Make sure your AAVSO CSV is loaded in the Analysis tab if you are using AUIDs like 000-BJS-555.")
                        diff_status_var.set(f"Error: Could not resolve target '{star_name}'.")
                        return
                except Exception as e:
                    messagebox.showerror("Resolution Error", f"Error looking up target '{star_name}':\n{e}")
                    diff_status_var.set("Error during target name resolution.")
                    return

            res = run_differential_photometry(
                csv_b=b_csv, csv_v=v_csv, ref_catalog=cat_type,
                k_b=vars_dict["extinction_kb"][0].get(), k_v=vars_dict["extinction_kv"][0].get(),
                Tbv=diff_tbv_var.get(), Tb_bv=diff_tbbv_var.get(), Tv_bv=diff_tvbv_var.get(),
                radius_arcmin=float(vars_dict["catalog_search_radius"][0].get()),
                manual_ref_coord=manual_coord,
                target_mode=target_mode,
                manual_target_coord=manual_target_coord,
                axes=accuracy_axes
            )
            accuracy_canvas.draw()
            diff_status_var.set(res)
        except Exception as e:
            diff_status_var.set(f"Error: {e}")
            messagebox.showerror("Analysis Error", str(e))

    def save_session():
        import json
        data = {}
        for key, val in vars_dict.items():
            # Robust check: only process (variable, type) pairs
            if not isinstance(val, (tuple, list)) or len(val) < 2:
                continue
                
            var, vtype = val[0], val[1]
            try:
                if hasattr(var, 'get'):
                    data[key] = var.get()
            except:
                pass # Skip if variable is destroyed or invalid
        
        # Add loaded file paths
        data['loaded_file_paths'] = [f['path'] for f in loaded_files]
        
        try:
            with open("calibra_session.json", "w") as f:
                json.dump(data, f, indent=4)
            print("Session saved to calibra_session.json")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save session: {e}")

    def load_session():
        import json
        if not os.path.exists("calibra_session.json"):
            return
        
        try:
            with open("calibra_session.json", "r") as f:
                data = json.load(f)
            
            for key, value in data.items():
                if key == 'loaded_file_paths':
                    for path in value:
                        if os.path.exists(path):
                            loaded_files.append(scan_fits_header(path))
                    continue
                if key in vars_dict:
                    val = vars_dict[key]
                    if not isinstance(val, (tuple, list)) or len(val) < 2:
                        continue
                    var, vtype = val[0], val[1]
                    try:
                        if hasattr(var, 'set'):
                            var.set(value)
                    except:
                        # Silently skip errors (e.g. type mismatch if session file is old)
                        pass
            print("Session loaded from calibra_session.json")
        except Exception as e:
            print(f"Error loading session: {e}")

    # --- TAB 1: File Manager ---
    tab_files_outer = ttk.Frame(content_container)
    tab_files_outer.grid(row=0, column=0, sticky="nsew")
    file_manager_frame = tab_files_outer
    
    # Button Toolbar
    toolbar_frame = ttk.Frame(file_manager_frame)
    toolbar_frame.pack(fill="x", padx=5, pady=5)
    
    def update_file_table():
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Populate with loaded_files
        for idx, file_data in enumerate(loaded_files):
            tag = 'even' if idx % 2 == 0 else 'odd'
            # Default to checked '[X]'
            tree.insert('', tk.END, iid=str(idx), values=(
                '[X]',
                file_data['filename'],
                file_data['filter'],
                file_data['binning'],
                file_data['airmass'],
                file_data['date_obs'],
                file_data['exposure'],
                file_data['wcs'],
                file_data['object'],
                file_data['size']
            ), tags=(tag,))
           # Update status bar
        if "filter_v_keyword" in vars_dict:
            v_key = vars_dict["filter_v_keyword"][0].get().upper().strip()
        else:
            v_key = "VMAG"
            
        if "filter_b_keyword" in vars_dict:
            b_key = vars_dict["filter_b_keyword"][0].get().upper().strip()
        else:
            b_key = "BMAG"

        v_count = 0
        b_count = 0
        for f in loaded_files:
            filt_str = str(f.get('filter', '')).upper().strip()
            if filt_str:
                if v_key and (v_key in filt_str or filt_str in v_key):
                    v_count += 1
                elif b_key and (b_key in filt_str or filt_str in b_key):
                    b_count += 1

        status_text = f"Loaded: {len(loaded_files)} files ({v_count}× V, {b_count}× B)"
        file_manager_status.set(status_text)
        
        # Update light curve filter dropdown with unique translated filters
        unique_translated = set()
        for f in loaded_files:
            filt_str = str(f.get('filter', '')).upper().strip()
            if filt_str:
                if v_key and (v_key in filt_str or filt_str in v_key):
                    unique_translated.add("V")
                elif b_key and (b_key in filt_str or filt_str in b_key):
                    unique_translated.add("B")
                else:
                    unique_translated.add(filt_str)
        
        unique_filters = sorted(list(unique_translated))
        if 'filter_cb' in ts_widgets:
            ts_widgets['filter_cb']['values'] = unique_filters
            # If current selection is not in new list and list is not empty, select first
            if unique_filters and vars_dict.get("ts_filter", [None])[0]:
                curr = vars_dict["ts_filter"][0].get()
                if curr not in unique_filters:
                    vars_dict["ts_filter"][0].set(unique_filters[0])
        
        # Select first item if nothing selected
        if tree.get_children() and not tree.selection():
            first_iid = tree.get_children()[0]
            tree.selection_set(first_iid)
            tree.see(first_iid)
            # Manually trigger header update
            on_tree_select(None)

    def on_load_files():
        from tkinter import filedialog
        files = filedialog.askopenfilenames(title="Select FITS Files", filetypes=(("FITS files", "*.fits *.fit"), ("all files", "*.*")))
        if files:
            for f in files:
                # Avoid duplicates
                if not any(existing['path'] == f for existing in loaded_files):
                    loaded_files.append(scan_fits_header(f))
            update_file_table()

    def on_load_dir():
        from tkinter import filedialog
        import glob
        dirname = filedialog.askdirectory(title="Select FITS Directory")
        if dirname:
            pattern = os.path.join(dirname, "*.fits")
            files = glob.glob(pattern)
            for f in files:
                if not any(existing['path'] == f for existing in loaded_files):
                    loaded_files.append(scan_fits_header(f))
            update_file_table()

    def on_remove_selected():
        selected = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected:
            # Fallback to highlighted if none are checked
            selected = tree.selection()
            if not selected: return
        
        # We need to remove from the back to keep indices valid if we were removing from list directly,
        # but here we can just rebuild the list from the remaining IIDs.
        indices_to_remove = sorted([int(iid) for iid in selected], reverse=True)
        for idx in indices_to_remove:
            loaded_files.pop(idx)
        update_file_table()

    def on_clear_all():
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to remove all files from the list?"):
            loaded_files.clear()
            update_file_table()

    def on_refresh_headers():
        for i in range(len(loaded_files)):
            loaded_files[i] = scan_fits_header(loaded_files[i]['path'])
        update_file_table()

    def on_check_all():
        for iid in tree.get_children():
            vals = list(tree.item(iid, 'values'))
            vals[0] = '[X]'
            tree.item(iid, values=vals)

    def on_uncheck_all():
        for iid in tree.get_children():
            vals = list(tree.item(iid, 'values'))
            vals[0] = '[ ]'
            tree.item(iid, values=vals)

    ttk.Button(toolbar_frame, text="Load Files...", command=on_load_files).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Load Directory...", command=on_load_dir).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Check All", command=on_check_all).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Uncheck All", command=on_uncheck_all).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Remove Checked/Selected", command=on_remove_selected).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Clear All", command=on_clear_all).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Refresh Headers", command=on_refresh_headers).pack(side=tk.LEFT, padx=5)

    # Informative tip text below buttons
    tip_label = tk.Label(file_manager_frame, text=f"{get_icon('💡', '')} Tip: Double-click any file in the table below to open it in the interactive FITS Viewer.".strip(),
                         font=("Arial", 9, "italic"), fg="#00796b", bg="#f0f2f5", anchor="w")
    tip_label.pack(fill="x", padx=10, pady=(0, 5))

    # Treeview Table & Header Panel (Side by Side)
    tree_frame = ttk.Frame(file_manager_frame)
    tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    paned = tk.PanedWindow(tree_frame, orient=tk.HORIZONTAL, bg="#f0f2f5", sashwidth=4)
    paned.pack(fill="both", expand=True)
    
    # Left side: Tree
    left_side = ttk.Frame(paned)
    paned.add(left_side, stretch="always")
    
    columns = ("use", "filename", "filter", "binning", "airmass", "date_obs", "exposure", "wcs", "object", "size")
    tree = ttk.Treeview(left_side, columns=columns, show='headings', height=6, selectmode='extended')
    
    # Configure Columns
    column_configs = {
        "use": ("Use", 40),
        "filename": ("Filename", 150),
        "filter": ("Filter", 60),
        "binning": ("Binning", 70),
        "airmass": ("Airmass", 70),
        "date_obs": ("Date-Obs", 130),
        "exposure": ("Exposure", 70),
        "wcs": ("WCS?", 50),
        "object": ("Object", 80),
        "size": ("Size", 70)
    }
    
    # Symmetrical grid layout in left_side to support both scrollbars
    left_side.grid_rowconfigure(0, weight=1)
    left_side.grid_columnconfigure(0, weight=1)
    
    for col, (text, width) in column_configs.items():
        tree.heading(col, text=text)
        tree.column(col, width=width, anchor=tk.CENTER, stretch=False)
    
    tree.grid(row=0, column=0, sticky="nsew")
    
    # Vertical scrollbar
    ts_vsb = ttk.Scrollbar(left_side, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=ts_vsb.set)
    ts_vsb.grid(row=0, column=1, sticky="ns")
    
    # Horizontal scrollbar (allows scrolling rather than squishing column widths)
    ts_hsb = ttk.Scrollbar(left_side, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=ts_hsb.set)
    ts_hsb.grid(row=1, column=0, sticky="ew")
    
    add_treeview_copy_menu(tree)
    
    # Right side: Header Viewer
    right_side = tk.LabelFrame(paned, text=f"{get_icon('📄', '')} FITS Header Preview".strip())
    paned.add(right_side, width=350)
    
    header_text = tk.Text(right_side, font=("Courier", 9), wrap=tk.NONE, bg="#fdfdfe")
    h_hsb = ttk.Scrollbar(right_side, orient="horizontal", command=header_text.xview)
    h_vsb = ttk.Scrollbar(right_side, orient="vertical", command=header_text.yview)
    header_text.configure(xscrollcommand=h_hsb.set, yscrollcommand=h_vsb.set)
    
    h_vsb.pack(side=tk.RIGHT, fill="y")
    h_hsb.pack(side=tk.BOTTOM, fill="x")
    header_text.pack(fill="both", expand=True)
    add_copy_context_menu(header_text)
    
    def on_tree_select(event):
        selected = tree.selection()
        if not selected:
            return
        iid = selected[0]
        try:
            file_path = loaded_files[int(iid)]['path']
            from astropy.io import fits
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                header_text.delete(1.0, tk.END)
                # Formatted header display
                header_text.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
                header_text.insert(tk.END, "="*40 + "\n")
                for key, val in header.items():
                    try:
                        comment = header.comments[key]
                        line = f"{key:<8}= {str(val):<20} / {comment}\n"
                    except:
                        line = f"{key:<8}= {str(val):<20}\n"
                    header_text.insert(tk.END, line)
        except Exception as e:
            header_text.delete(1.0, tk.END)
            header_text.insert(tk.END, f"Error reading header: {e}")

    tree.bind("<<TreeviewSelect>>", on_tree_select)
    
    # Tags for alternating colors
    tree.tag_configure('even', background='#ffffff')
    tree.tag_configure('odd', background='#f7f9fc')
    
    def on_tree_click(event):
        region = tree.identify_region(event.x, event.y)
        if region == 'cell':
            column = tree.identify_column(event.x)
            if column == '#1': # The 'use' column
                iid = tree.identify_row(event.y)
                if iid:
                    vals = list(tree.item(iid, 'values'))
                    if vals[0] == '[X]':
                        vals[0] = '[ ]'
                    else:
                        vals[0] = '[X]'
                    tree.item(iid, values=vals)

    tree.bind("<ButtonRelease-1>", on_tree_click)
    tree.bind("<Double-1>", on_double_click)
    
    file_manager_status = tk.StringVar(value="No files loaded.")
    file_manager_status_label = SelectableLabel(file_manager_frame, textvariable=file_manager_status, font=("Arial", 8, "italic"))
    file_manager_status_label.pack(anchor=tk.W, padx=10, pady=2, fill="x")

    # Notebook is already created and packed early

    # Update the table after notebook is created (for initial load_session)
    root.after(100, update_file_table)

    # --- TAB 2: Pre-processing (NEW) ---
    tab_pre_outer = ttk.Frame(content_container)
    tab_pre_outer.grid(row=0, column=0, sticky="nsew")
    
    # Elegant sub-notebook for organized tab-within-tab pre-processing layout
    pre_notebook = ttk.Notebook(tab_pre_outer)
    pre_notebook.pack(fill="both", expand=True, padx=5, pady=5)
    
    sub_calib_scroll = ScrollableFrame(pre_notebook)
    tab_pre_left = sub_calib_scroll.scrollable_frame
    
    sub_solve_scroll = ScrollableFrame(pre_notebook)
    tab_pre_right = sub_solve_scroll.scrollable_frame
    
    pre_notebook.add(sub_calib_scroll, text=f"{get_icon('🧪', '')}  FITS Calibration".strip())
    pre_notebook.add(sub_solve_scroll, text=f"{get_icon('🧭', '')}  WCS Plate Solving".strip())
    
    tab_pre = tab_pre_left

    # --- TAB 3: Analysis & Calibration (Unified) ---
    tab_analysis_outer = ttk.Frame(content_container)
    tab_analysis_outer.grid(row=0, column=0, sticky="nsew")
    
    # Elegant sub-notebook for organized tab-within-tab analysis layout
    analysis_notebook = ttk.Notebook(tab_analysis_outer)
    analysis_notebook.pack(fill="both", expand=True, padx=5, pady=5)
    
    sub_detect_scroll = ScrollableFrame(analysis_notebook)
    sub_detect = sub_detect_scroll.scrollable_frame
    
    sub_color_scroll = ScrollableFrame(analysis_notebook)
    sub_color = sub_color_scroll.scrollable_frame
    
    sub_diff_scroll = ScrollableFrame(analysis_notebook)
    sub_diff = sub_diff_scroll.scrollable_frame
    
    analysis_notebook.add(sub_detect_scroll, text=f"{get_icon('🌌', '')}  1. Detection & Zero-Point".strip())
    analysis_notebook.add(sub_color_scroll, text=f"{get_icon('🌈', '')}  2. Color Coefficients".strip())
    analysis_notebook.add(sub_diff_scroll, text=f"{get_icon('📊', '')}  3. Differential Photometry".strip())
    
    tab_analysis = sub_detect
    

    # --- TAB 4: Light Curves ---
    tab_ts_outer = ttk.Frame(content_container)
    tab_ts_outer.grid(row=0, column=0, sticky="nsew")
    
    ts_scroll = ScrollableFrame(tab_ts_outer)
    ts_scroll.pack(fill="both", expand=True)
    tab_ts = ts_scroll.scrollable_frame

    # --- TAB 5: Data Mining ---
    tab_datamining_outer = ttk.Frame(content_container)
    tab_datamining_outer.grid(row=0, column=0, sticky="nsew")
    
    dm_scroll = ScrollableFrame(tab_datamining_outer)
    dm_scroll.pack(fill="both", expand=True)
    tab_datamining = dm_scroll.scrollable_frame

    # --- TAB 6: Settings ---
    tab_settings_outer = ttk.Frame(content_container)
    tab_settings_outer.grid(row=0, column=0, sticky="nsew")
    
    # Elegant sub-notebook for organized tab-within-tab settings layout
    settings_notebook = ttk.Notebook(tab_settings_outer)
    settings_notebook.pack(fill="both", expand=True, padx=5, pady=5)
    
    sub_astro_scroll = ScrollableFrame(settings_notebook)
    sub_astro = sub_astro_scroll.scrollable_frame
    sub_phot_scroll = ScrollableFrame(settings_notebook)
    sub_phot = sub_phot_scroll.scrollable_frame
    sub_ops_scroll = ScrollableFrame(settings_notebook)
    sub_ops = sub_ops_scroll.scrollable_frame
    
    settings_notebook.add(sub_astro_scroll, text=f"{get_icon('🌌', '')} Astro & Catalog".strip())
    settings_notebook.add(sub_phot_scroll, text=f"{get_icon('📷', '')} Camera & Photometry".strip())
    settings_notebook.add(sub_ops_scroll, text=f"{get_icon('👤', '')} Operator & Session".strip())
    
    # --- TAB 5: Settings CONTENT ---
    # Session Management - MOVED TO TOP
    lf_session = tk.LabelFrame(sub_ops, text="Session Management")
    lf_session.pack(fill="x", padx=10, pady=10)
    
    ttk.Button(lf_session, text="Save Session", command=save_session).grid(row=0, column=0, padx=10, pady=5)
    ttk.Button(lf_session, text="Load Session", command=load_session).grid(row=0, column=1, padx=10, pady=5)
    tk.Label(lf_session, text="* Settings are saved to calibra_session.json and auto-load on startup.", 
              fg="#555", font=("Arial", 8, "italic")).grid(row=0, column=2, padx=10)

    # Files & Catalog (from old TAB 1) - MOVED UP
    lf_files = tk.LabelFrame(sub_astro, text="Reference Catalog Selection")
    lf_files.pack(fill="x", padx=10, pady=10)
    
    tk.Label(lf_files, text="Ref Catalog:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
    cat_var = tk.StringVar(value="ATLAS refcat2")
    vars_dict["reference_catalog"] = (cat_var, str)
    cat_cb = ttk.Combobox(lf_files, textvariable=cat_var, values=["ATLAS refcat2", "APASS DR9", "Landolt Standard Star Catalogue", "GAIA_DR3", os.path.join('photometry_refstars', 'reference_stars.csv')], width=62)
    cat_cb.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
    
    def browse_catalog():
        from tkinter import filedialog
        filename = filedialog.askopenfilename(initialdir="photometry_refstars", title="Select Reference Catalog", filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
        if filename:
            cat_var.set(filename)
            
    ttk.Button(lf_files, text="Browse...", command=browse_catalog).grid(row=0, column=2, padx=5)

    # Filter Keyword Mapping
    lf_map = tk.LabelFrame(sub_astro, text="Filter Keyword Mapping")
    lf_map.pack(fill="x", padx=10, pady=10)
    tk.Label(lf_map, text="Define keywords in your FITS headers that identify the B and V filters.", 
              fg="#555", font=("Arial", 8, "italic")).grid(row=0, column=0, columnspan=4, padx=10, pady=(0, 10))
    add_entry(lf_map, "B-Filter Keyword:", "filter_b_keyword", "Bmag", 1, col_offset=0, vtype=str)
    add_entry(lf_map, "V-Filter Keyword:", "filter_v_keyword", "Vmag", 1, col_offset=1, vtype=str)

    # Region Selection (from old TAB 1)
    lf_filt = tk.LabelFrame(sub_astro, text="Region Selection")
    lf_filt.pack(fill="x", padx=10, pady=10)
    add_dropdown(lf_filt, "Region:", "filter_mode", ["all", "xy", "radec"], "all", 0)
    
    tk.Label(lf_filt, text="XY Bounds (Pixels)").grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(10,0))
    add_entry(lf_filt, "X Min:", "xy_x_min", 200, 2, col_offset=0, vtype=int)
    add_entry(lf_filt, "X Max:", "xy_x_max", 4200, 2, col_offset=1, vtype=int)
    add_entry(lf_filt, "Y Min:", "xy_y_min", 200, 3, col_offset=0, vtype=int)
    add_entry(lf_filt, "Y Max:", "xy_y_max", 2800, 3, col_offset=1, vtype=int)
    
    tk.Label(lf_filt, text="RA/Dec Bounds (Degrees)").grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(10,0))
    add_entry(lf_filt, "RA Min:", "ra_min", 0.0, 5, col_offset=0)
    add_entry(lf_filt, "RA Max:", "ra_max", 360.0, 5, col_offset=1)
    add_entry(lf_filt, "Dec Min:", "dec_min", -90.0, 6, col_offset=0)
    add_entry(lf_filt, "Dec Max:", "dec_max", 90.0, 6, col_offset=1)



    # --- TAB 1: Pre-processing CONTENT ---
    
    # 1. Calibration (Bias & Flats)
    lf_calib = tk.LabelFrame(tab_pre, text="FITS Calibration (Bias & Flats)")
    lf_calib.pack(fill="x", padx=10, pady=10)
    
    add_file_selector(lf_calib, "Master Bias:", "bias_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\Master_Bias_1x1_gain_0.fits", 0, initial_dir="bias_and_flats")
    add_file_selector(lf_calib, "Master Flat (V):", "flat_v_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\FLAT_Vmag_1x1_gain_0.fits", 1, initial_dir="bias_and_flats")
    add_file_selector(lf_calib, "Master Flat (B):", "flat_b_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\FLAT_Bmag_1x1_gain_0.fits", 2, initial_dir="bias_and_flats")

    def on_run_calibration():
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager.")
            return
        
        bias_path = vars_dict["bias_path"][0].get()
        flat_v = vars_dict["flat_v_path"][0].get()
        flat_b = vars_dict["flat_b_path"][0].get()
        
        if not os.path.exists(bias_path):
            messagebox.showerror("Error", f"Master Bias not found at:\n{bias_path}")
            return

        def cal_thread():
            try:
                from astropy.io import fits
                print(f"\n--- Starting Batch Calibration ---")
                total = len(selected_iids)
                success_count = 0
                
                for idx, iid in enumerate(selected_iids):
                    iid_int = int(iid)
                    orig_path = loaded_files[iid_int]['path']
                    
                    filt = loaded_files[iid_int]['filter'].upper()
                    b_key = vars_dict["filter_b_keyword"][0].get().upper()
                    v_key = vars_dict["filter_v_keyword"][0].get().upper()
                    
                    flat_path = None
                    if b_key and b_key in filt:
                        flat_path = flat_b
                    elif v_key and v_key in filt:
                        flat_path = flat_v
                    
                    if not flat_path:
                        print(f"[{idx+1}/{total}] Error: Filter '{filt}' does not match B ({b_key}) or V ({v_key}) mapping. Skipping {os.path.basename(orig_path)}.")
                        continue
                    
                    if not os.path.exists(flat_path):
                        print(f"[{idx+1}/{total}] Warning: Master Flat for '{filt}' not found at {flat_path}. Skipping.")
                        continue

                    print(f"[{idx+1}/{total}] Calibrating {os.path.basename(orig_path)}...")
                    with fits.open(orig_path) as hdul:
                        data = hdul[0].data
                        header = hdul[0].header
                        if 'FILENAME' not in header:
                            header['FILENAME'] = os.path.basename(orig_path)
                        
                        # Apply calibration
                        calibrate_image(data, header, bias_path, flat_path, verbose=True)
                        
                        # Expected output path
                        out_dir = "fitsfiles/calibrated"
                        new_path = os.path.join(out_dir, f"cal_{os.path.basename(orig_path)}")
                        
                        if os.path.exists(new_path):
                            success_count += 1
                            loaded_files[iid_int] = scan_fits_header(new_path)
                            root.after(0, update_file_table)
                
                msg = f"Calibration complete.\n{success_count} of {total} files were successfully calibrated."
                print(f"\n{msg}")
                root.after(0, lambda: messagebox.showinfo("Batch Complete", msg))
            except Exception as e:
                print(f"Calibration error: {e}")
                root.after(0, lambda: messagebox.showerror("Error", f"Calibration failed: {e}"))

        threading.Thread(target=cal_thread, daemon=True).start()

    cal_btn_frame = ttk.Frame(tab_pre)
    cal_btn_frame.pack(pady=(10, 20))
    
    run_cal_btn = tk.Button(cal_btn_frame, text="Run Calibration (Bias/Flat) on Selected", command=on_run_calibration,
                            bg="#388e3c", fg="white", font=("Arial", 11, "bold"), pady=10, width=35)
    run_cal_btn.pack(side=tk.LEFT, padx=5)
    
    cal_info_btn = tk.Button(cal_btn_frame, text=f"{get_icon('❓', '?')} What does this do?", command=show_calibration_info,
                             bg="#f0f2f5", fg="#388e3c", font=("Arial", 11, "bold"), pady=10, padx=15)
    cal_info_btn.pack(side=tk.LEFT, padx=5)

    # 2. Plate Solving (ASTAP)
    lf_plate_info = tk.LabelFrame(tab_pre_right, text="Plate Solving (ASTAP Integration)")
    lf_plate_info.pack(fill="x", padx=10, pady=10)
    
    tk.Label(lf_plate_info, text="Automatically solve FITS coordinates using ASTAP command-line solver.\nSolved files will be updated in the File Manager with a '_wcs' suffix.", justify=tk.LEFT).pack(padx=10, pady=10)

    lf_plate_settings = tk.LabelFrame(tab_pre_right, text="Solver Settings")
    lf_plate_settings.pack(fill="x", padx=10, pady=10)
    
    add_entry(lf_plate_settings, "Output Suffix:", "plate_suffix", "wcs", 0, col_offset=0, vtype=str)
    add_entry(lf_plate_settings, "Search Radius (deg):", "plate_radius", 5.0, 0, col_offset=1)

    # ASTAP Path
    tk.Label(lf_plate_settings, text="ASTAP Executable:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
    astap_path_var = tk.StringVar(value=r"C:\Program Files\astap\astap.exe")
    vars_dict["astap_path"] = (astap_path_var, str)
    ttk.Entry(lf_plate_settings, textvariable=astap_path_var, width=50).grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
    
    def browse_astap():
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Select ASTAP Executable", filetypes=(("Executable", "*.exe"), ("all files", "*.*")))
        if path: astap_path_var.set(path)
    ttk.Button(lf_plate_settings, text="Browse...", command=browse_astap).grid(row=1, column=2, padx=5)

    add_check(lf_plate_settings, "Annotate Image", "plate_annotate", False, 2)

    plate_status_var = tk.StringVar(value="Ready")
    plate_status_label = SelectableLabel(tab_pre_right, textvariable=plate_status_var, font=("Arial", 9, "italic"), justify=tk.CENTER)
    plate_status_label.pack(pady=5, fill="x")

    def on_run_plate_solve():
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager.")
            return
        
        # 1. Pre-check for existing WCS
        files_to_solve = []
        already_solved = []
        for iid in selected_iids:
            file_data = loaded_files[int(iid)]
            if file_data['wcs'] == '✓':
                already_solved.append(iid)
            else:
                files_to_solve.append(iid)
        
        if already_solved:
            msg = f"{len(already_solved)} files already have WCS (plate solved).\n\n" \
                  "Do you want to re-solve them anyway?\n" \
                  "[Yes] - Re-solve all selected files.\n" \
                  "[No]  - Skip already solved files.\n" \
                  "[Cancel] - Abort operation."
            ans = messagebox.askyesnocancel("Existing WCS Found", msg)
            if ans is None: # Cancel
                return
            if ans: # Yes
                files_to_solve = list(selected_iids)
            # else: No (Skip already solved), files_to_solve is already filtered
        
        if not files_to_solve:
            messagebox.showinfo("Nothing to do", "No files to solve (all were skipped).")
            return

        suffix = vars_dict["plate_suffix"][0].get()
        radius = vars_dict["plate_radius"][0].get()
        exe = astap_path_var.get()
        annotate = vars_dict["plate_annotate"][0].get()
        
        plate_status_var.set(f"Solving {len(files_to_solve)} files...")
        run_plate_btn.config(state=tk.DISABLED, text="Solving...")
        
        def plate_thread():
            try:
                print(f"\n--- Starting Batch Plate Solve ---")
                print(f"Target files: {len(files_to_solve)}")
                print(f"Suffix: {suffix}")
                
                solved_count = 0
                for idx, iid in enumerate(files_to_solve):
                    iid_int = int(iid)
                    orig_path = loaded_files[iid_int]['path']
                    
                    base, ext = os.path.splitext(orig_path)
                    new_filename = f"{base}_{suffix}{ext}" if suffix else orig_path
                    
                    if new_filename != orig_path:
                        print(f"[{idx+1}/{len(files_to_solve)}] Copying {orig_path} to {new_filename}...")
                        shutil.copy2(orig_path, new_filename)
                    
                    print(f"[{idx+1}/{len(files_to_solve)}] Solving {new_filename}...")
                    res = solve_with_astap(new_filename, astap_exe=exe, search_radius=radius, annotate=annotate)
                    
                    if res:
                        solved_count += 1
                        # Update the loaded_files entry in-place
                        loaded_files[iid_int] = scan_fits_header(new_filename)
                        root.after(0, update_file_table)
                    else:
                        print(f"[{idx+1}/{len(files_to_solve)}] Solve failed for {new_filename}")
                
                msg = f"Plate solve complete.\n{solved_count} of {len(files_to_solve)} files were successfully solved."
                print(f"\n{msg}")
                root.after(0, lambda: messagebox.showinfo("Batch Complete", msg))
                root.after(0, lambda: plate_status_var.set("Complete."))
            except Exception as e:
                print(f"Plate solve error: {e}")
                root.after(0, lambda: messagebox.showerror("Error", f"Plate solve failed: {e}"))
                root.after(0, lambda: plate_status_var.set("Error occurred."))
            finally:
                root.after(0, lambda: run_btn_plate_solve_relabel())

        def run_btn_plate_solve_relabel():
            run_plate_btn.config(state=tk.NORMAL, text="Run Plate Solver on Selected")

        threading.Thread(target=plate_thread, daemon=True).start()

    plate_btn_frame = ttk.Frame(tab_pre_right)
    plate_btn_frame.pack(pady=20)
    
    run_plate_btn = tk.Button(plate_btn_frame, text="Run Plate Solver on Selected", command=on_run_plate_solve,
                               bg="#0288d1", fg="white", font=("Arial", 11, "bold"), pady=10, width=35)
    run_plate_btn.pack(side=tk.LEFT, padx=5)
    
    plate_info_btn = tk.Button(plate_btn_frame, text=f"{get_icon('❓', '?')} What does this do?", command=show_plate_solve_info,
                               bg="#f0f2f5", fg="#0288d1", font=("Arial", 11, "bold"), pady=10, padx=15)
    plate_info_btn.pack(side=tk.LEFT, padx=5)



    # --- TAB 2: Detect & Measure CONTENT ---

    # 1. Pipeline Configuration
    lf_pipe_cfg = tk.LabelFrame(tab_analysis, text="Pipeline Configuration")
    lf_pipe_cfg.pack(fill="x", padx=10, pady=10)
    
    add_check(lf_pipe_cfg, "Run Star Detection (DAOStarFinder)", "run_star_detection", True, 0, col_offset=0)
    add_check(lf_pipe_cfg, "Perform Zero-Point Calibration", "run_new_calibration", True, 0, col_offset=1)
    add_check(lf_pipe_cfg, "Run Positional Shift Analysis", "run_shift_analysis", False, 1, col_offset=0)
    
    tk.Label(lf_pipe_cfg, text="* Tip: Disable 'ZP Calibration' if using pre-calibrated magnitudes.", fg="#555", font=("Arial", 8, "italic")).grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(2, 5))

    # Analysis & Calibration Run Button with Premium Info Pop-up
    run_btn_frame = ttk.Frame(tab_analysis)
    run_btn_frame.pack(pady=15)

    run_btn = tk.Button(run_btn_frame, text="Run Analysis Pipeline on Selected", command=on_run, 
                        bg="#1a3a5f", fg="white", font=("Arial", 10, "bold"), 
                        width=35, relief="flat", pady=10)
    run_btn.pack(side=tk.LEFT, padx=5)

    info_btn = tk.Button(run_btn_frame, text=f"{get_icon('❓', '?')} What does this do?", command=show_pipeline_info,
                         bg="#f0f2f5", fg="#1a3a5f", font=("Arial", 10, "bold"),
                         relief="flat", pady=10, padx=15)
    info_btn.pack(side=tk.LEFT, padx=5)

    # 2. Pipeline Run Results Summary Panel (Multiline & Monospaced)
    lf_pipeline_results = tk.LabelFrame(tab_analysis, text="Last Pipeline Run Results Summary")
    lf_pipeline_results.pack(fill="x", padx=10, pady=10)
    
    pipeline_results_text = scrolledtext.ScrolledText(lf_pipeline_results, font=("Consolas", 9), bg="#f8f9fa", fg="#1a3a5f",
                                                      height=9, wrap=tk.WORD, relief="flat", borderwidth=0, highlightthickness=0)
    pipeline_results_text.pack(fill="both", expand=True, padx=15, pady=10)
    
    def update_pipeline_results_display(text_content):
        pipeline_results_text.config(state=tk.NORMAL)
        pipeline_results_text.delete("1.0", tk.END)
        pipeline_results_text.insert(tk.END, text_content)
        pipeline_results_text.config(state=tk.DISABLED)
        
    update_pipeline_results_display("No analysis run yet in this session. Click 'Run Analysis Pipeline on Selected' to begin.")

    # 3. Live Pipeline Console Output Panel (Premium Black Terminal Style)
    lf_pipeline_terminal = tk.LabelFrame(tab_analysis, text="Live Pipeline Console Output")
    lf_pipeline_terminal.pack(fill="both", expand=True, padx=10, pady=10)
    
    pipeline_terminal = scrolledtext.ScrolledText(lf_pipeline_terminal, font=("Consolas", 8), bg="#1e1e1e", fg="#d4d4d4", height=16)
    pipeline_terminal.pack(fill="both", expand=True, padx=5, pady=5)
    add_copy_context_menu(pipeline_terminal)

    # 2. Color Transformation Section
    lf_color = tk.LabelFrame(sub_color, text="Color Transformation Analysis (B-V Pairs)")
    lf_color.pack(fill="x", padx=10, pady=10)
    
    import glob
    def get_latest_csv(pattern):
        files = glob.glob(os.path.join("photometry_output", pattern))
        return max(files, key=os.path.getmtime) if files else ""
        
    recent_b_csv = get_latest_csv("*Bmag*.csv") or get_latest_csv("*_B_*.csv")
    recent_v_csv = get_latest_csv("*Vmag*.csv") or get_latest_csv("*_V_*.csv")
    
    add_file_selector(lf_color, "B-Filter Results (CSV):", "color_b_csv", recent_b_csv, 0, initial_dir="photometry_output")
    add_file_selector(lf_color, "V-Filter Results (CSV):", "color_v_csv", recent_v_csv, 1, initial_dir="photometry_output")
    
    tk.Label(lf_color, text="Airmass B*:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
    air_b_var = tk.DoubleVar(value=1.0)
    vars_dict["air_b"] = (air_b_var, float)
    ttk.Entry(lf_color, textvariable=air_b_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)
    
    tk.Label(lf_color, text="Airmass V*:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
    air_v_var = tk.DoubleVar(value=1.0)
    vars_dict["air_v"] = (air_v_var, float)
    ttk.Entry(lf_color, textvariable=air_v_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)

    tk.Label(lf_color, text="* Global extinction (k_B, k_V) settings from the Settings tab will be used.", fg="#555", font=("Arial", 8, "italic")).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=10, pady=5)

    override_airmass_var = tk.BooleanVar(value=False)
    vars_dict["override_airmass"] = (override_airmass_var, bool)
    tk.Label(lf_color, text="* 1.0 used if FITS values not found. To override FITS airmass values, check the box below and enter new values.", fg="#555", font=("Arial", 8, "italic")).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
    ttk.Checkbutton(lf_color, text="Override FITS Airmass", variable=override_airmass_var).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)

    color_btn_frame = ttk.Frame(sub_color)
    color_btn_frame.pack(pady=5)
    
    tk.Button(color_btn_frame, text="Run Color Transformation Analysis", command=on_run_color,
              bg="#1a3a5f", fg="white", font=("Arial", 10, "bold"), width=35, relief="flat", pady=10).pack(side=tk.LEFT, padx=5)
              
    color_info_btn = tk.Button(color_btn_frame, text=f"{get_icon('❓', '?')} What does this do?", command=show_color_info,
                               bg="#f0f2f5", fg="#1a3a5f", font=("Arial", 10, "bold"),
                               relief="flat", pady=10, padx=15)
    color_info_btn.pack(side=tk.LEFT, padx=5)

    color_status_var = tk.StringVar(value="Ready")
    color_status_label = SelectableLabel(sub_color, textvariable=color_status_var, font=("Arial", 9, "italic"), justify=tk.CENTER)
    color_status_label.pack(pady=5, fill="x")

    # Preview for Color Transformation
    lf_color_preview = tk.LabelFrame(sub_color, text="Color Transformation Preview")
    lf_color_preview.pack(fill="x", padx=10, pady=5)
    
    color_coeff_fig, color_coeff_axes = plt.subplots(1, 3, figsize=(10, 3.5))
    color_coeff_canvas = FigureCanvasTkAgg(color_coeff_fig, master=lf_color_preview)
    color_coeff_canvas.get_tk_widget().pack(fill="x", expand=True)
    color_coeff_toolbar = NavigationToolbar2Tk(color_coeff_canvas, lf_color_preview)
    color_coeff_toolbar.update()

    def get_checked_b_v_counts():
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        b_count = 0
        v_count = 0
        b_key = vars_dict["filter_b_keyword"][0].get().upper() if "filter_b_keyword" in vars_dict else "BMAG"
        v_key = vars_dict["filter_v_keyword"][0].get().upper() if "filter_v_keyword" in vars_dict else "VMAG"
        
        for iid in selected_iids:
            finfo = loaded_files[int(iid)]
            filt = str(finfo.get('filter', '')).upper()
            if b_key in filt or (len(filt)==1 and filt=='B'): b_count += 1
            elif v_key in filt or (len(filt)==1 and filt=='V'): v_count += 1
        return b_count, v_count

    # --- Differential Photometry Section ---
    lf_diff = tk.LabelFrame(sub_diff, text="2. Compute B/V relative to a reference star")
    lf_diff.pack(fill="x", padx=10, pady=10)
    
    add_file_selector(lf_diff, "B-Filter Results (CSV):", "diff_b_csv", recent_b_csv, 0, initial_dir="photometry_output")
    add_file_selector(lf_diff, "V-Filter Results (CSV):", "diff_v_csv", recent_v_csv, 1, initial_dir="photometry_output")
    
    tk.Label(lf_diff, text="* Global extinction (k_B, k_V) settings from the Settings tab will be used.", fg="#555", font=("Arial", 8, "italic")).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
    
    tk.Label(lf_diff, text="Color Term Tbv:").grid(row=2, column=2, sticky=tk.W, padx=10, pady=5)
    diff_tbv_var = tk.DoubleVar(value=1.0)
    vars_dict["diff_tbv"] = (diff_tbv_var, float)
    ttk.Entry(lf_diff, textvariable=diff_tbv_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=10, pady=5)
    
    tk.Label(lf_diff, text="B Correction Tb_bv:").grid(row=3, column=2, sticky=tk.W, padx=10, pady=5)
    diff_tbbv_var = tk.DoubleVar(value=0.0)
    vars_dict["diff_tbbv"] = (diff_tbbv_var, float)
    ttk.Entry(lf_diff, textvariable=diff_tbbv_var, width=10).grid(row=3, column=3, sticky=tk.W, padx=10, pady=5)
    
    tk.Label(lf_diff, text="V Correction Tv_bv:").grid(row=4, column=2, sticky=tk.W, padx=10, pady=5)
    diff_tvbv_var = tk.DoubleVar(value=0.0)
    vars_dict["diff_tvbv"] = (diff_tvbv_var, float)
    ttk.Entry(lf_diff, textvariable=diff_tvbv_var, width=10).grid(row=4, column=3, sticky=tk.W, padx=10, pady=5)
    
    diff_status_var = tk.StringVar(value="Load coefficients and select CSV files to begin.")
    
    lf_ref = tk.LabelFrame(sub_diff, text="Reference Star Selection")
    lf_ref.pack(fill="x", padx=10, pady=10)
    
    ref_mode_var = tk.StringVar(value="auto")
    vars_dict["ref_mode"] = (ref_mode_var, str)
    ttk.Radiobutton(lf_ref, text="Automatic (Brightest star with 0.4 <= B-V <= 0.8)", variable=ref_mode_var, value="auto").grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    ttk.Radiobutton(lf_ref, text="Resolve Star by Name (via Simbad)", variable=ref_mode_var, value="name").grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    
    tk.Label(lf_ref, text="Star Name:").grid(row=2, column=0, sticky=tk.E, padx=(10, 2), pady=5)
    star_name_var = tk.StringVar(value="AE UMa")
    vars_dict["ref_star_name"] = (star_name_var, str)
    star_name_entry = ttk.Entry(lf_ref, textvariable=star_name_var, width=20)
    star_name_entry.grid(row=2, column=1, sticky=tk.W, padx=2)
    
    name_resolve_status_var = tk.StringVar(value="")
    name_resolve_status_label = SelectableLabel(lf_ref, textvariable=name_resolve_status_var)
    name_resolve_status_label.grid(row=2, column=3, columnspan=4, sticky=tk.W, padx=5)

    def check_star_name(*args):
        if ref_mode_var.get() != "name": return
        star_name = star_name_var.get().strip()
        if not star_name:
            name_resolve_status_var.set("Please enter a name.")
            return
        name_resolve_status_var.set("Resolving...")
        root.update_idletasks()
        
        def resolve_thread():
            from astropy.coordinates import SkyCoord
            from astropy.coordinates.name_resolve import NameResolveError
            try:
                c = SkyCoord.from_name(star_name)
                ra_hms = c.ra.to_string(unit='hour', sep='hms', precision=1)
                dec_dms = c.dec.to_string(unit='degree', sep='dms', precision=1)
                root.after(0, lambda: name_resolve_status_var.set(f"Found: {ra_hms}, {dec_dms}"))
            except NameResolveError:
                root.after(0, lambda: name_resolve_status_var.set("Not found in Simbad."))
            except Exception:
                root.after(0, lambda: name_resolve_status_var.set("Error connecting."))
        
        import threading
        threading.Thread(target=resolve_thread, daemon=True).start()

    check_name_btn = ttk.Button(lf_ref, text="Check", command=check_star_name, width=8)
    check_name_btn.grid(row=2, column=2, sticky=tk.W, padx=2)
    star_name_entry.bind('<Return>', check_star_name)

    ttk.Radiobutton(lf_ref, text="Manual Coordinates", variable=ref_mode_var, value="manual").grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    
    # RA boxes
    tk.Label(lf_ref, text="RA:").grid(row=4, column=0, sticky=tk.E, padx=(10, 2), pady=5)
    ra_h_var = tk.StringVar(value="14")
    vars_dict["ref_ra_h"] = (ra_h_var, str)
    ttk.Entry(lf_ref, textvariable=ra_h_var, width=4).grid(row=4, column=1, sticky=tk.W, padx=2)
    tk.Label(lf_ref, text="h").grid(row=4, column=2, sticky=tk.W, padx=0)
    ra_m_var = tk.StringVar(value="34")
    vars_dict["ref_ra_m"] = (ra_m_var, str)
    ttk.Entry(lf_ref, textvariable=ra_m_var, width=4).grid(row=4, column=3, sticky=tk.W, padx=2)
    tk.Label(lf_ref, text="m").grid(row=4, column=4, sticky=tk.W, padx=0)
    ra_s_var = tk.StringVar(value="00.00")
    vars_dict["ref_ra_s"] = (ra_s_var, str)
    ttk.Entry(lf_ref, textvariable=ra_s_var, width=6).grid(row=4, column=5, sticky=tk.W, padx=2)
    tk.Label(lf_ref, text="s").grid(row=4, column=6, sticky=tk.W, padx=0)
    
    # Dec boxes
    tk.Label(lf_ref, text="Dec:").grid(row=5, column=0, sticky=tk.E, padx=(10, 2), pady=5)
    dec_d_var = tk.StringVar(value="+43")
    vars_dict["ref_dec_d"] = (dec_d_var, str)
    ttk.Entry(lf_ref, textvariable=dec_d_var, width=4).grid(row=5, column=1, sticky=tk.W, padx=2)
    tk.Label(lf_ref, text="d").grid(row=5, column=2, sticky=tk.W, padx=0)
    dec_m_var = tk.StringVar(value="30")
    vars_dict["ref_dec_m"] = (dec_m_var, str)
    ttk.Entry(lf_ref, textvariable=dec_m_var, width=4).grid(row=5, column=3, sticky=tk.W, padx=2)
    tk.Label(lf_ref, text="m").grid(row=5, column=4, sticky=tk.W, padx=0)
    dec_s_var = tk.StringVar(value="00.0")
    vars_dict["ref_dec_s"] = (dec_s_var, str)
    ttk.Entry(lf_ref, textvariable=dec_s_var, width=6).grid(row=5, column=5, sticky=tk.W, padx=2)
    tk.Label(lf_ref, text="s").grid(row=5, column=6, sticky=tk.W, padx=0)
    
    def toggle_ref_entries(*args):
        mode = ref_mode_var.get()
        name_state = tk.NORMAL if mode == "name" else tk.DISABLED
        manual_state = tk.NORMAL if mode == "manual" else tk.DISABLED
        
        star_name_entry.config(state=name_state)
        check_name_btn.config(state=name_state)
        for child in lf_ref.winfo_children():
            if isinstance(child, ttk.Entry) and child != star_name_entry:
                child.config(state=manual_state)
    
    ref_mode_var.trace("w", toggle_ref_entries)
    toggle_ref_entries()

    lf_target = tk.LabelFrame(sub_diff, text="Target Star Selection")
    lf_target.pack(fill="x", padx=10, pady=10)
    
    target_mode_var = tk.StringVar(value="all")
    vars_dict["target_mode"] = (target_mode_var, str)
    ttk.Radiobutton(lf_target, text="Analyze all stars", variable=target_mode_var, value="all").grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    ttk.Radiobutton(lf_target, text="Resolve Target by Name (via Simbad)", variable=target_mode_var, value="name").grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    
    tk.Label(lf_target, text="Star Name:").grid(row=2, column=0, sticky=tk.E, padx=(10, 2), pady=5)
    target_name_var = tk.StringVar(value="")
    vars_dict["target_star_name"] = (target_name_var, str)
    target_name_entry = ttk.Entry(lf_target, textvariable=target_name_var, width=20)
    target_name_entry.grid(row=2, column=1, sticky=tk.W, padx=2)
    
    target_resolve_status_var = tk.StringVar(value="")
    target_resolve_status_label = SelectableLabel(lf_target, textvariable=target_resolve_status_var)
    target_resolve_status_label.grid(row=2, column=3, columnspan=4, sticky=tk.W, padx=5)

    def check_target_name(*args):
        if target_mode_var.get() != "name": return
        star_name = target_name_var.get().strip()
        if not star_name:
            target_resolve_status_var.set("Please enter a name.")
            return
        target_resolve_status_var.set("Resolving...")
        root.update_idletasks()
        
        def resolve_thread():
            from astropy.coordinates import SkyCoord
            from astropy.coordinates.name_resolve import NameResolveError
            try:
                c = SkyCoord.from_name(star_name)
                ra_hms = c.ra.to_string(unit='hour', sep='hms', precision=1)
                dec_dms = c.dec.to_string(unit='degree', sep='dms', precision=1)
                root.after(0, lambda: target_resolve_status_var.set(f"Found: {ra_hms}, {dec_dms}"))
            except NameResolveError:
                root.after(0, lambda: target_resolve_status_var.set("Not found in Simbad."))
            except Exception:
                root.after(0, lambda: target_resolve_status_var.set("Error connecting."))
        
        import threading
        threading.Thread(target=resolve_thread, daemon=True).start()

    check_target_btn = ttk.Button(lf_target, text="Check", command=check_target_name, width=8)
    check_target_btn.grid(row=2, column=2, sticky=tk.W, padx=2)
    target_name_entry.bind('<Return>', check_target_name)

    ttk.Radiobutton(lf_target, text="Manual Coordinates", variable=target_mode_var, value="manual").grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    
    # RA boxes
    tk.Label(lf_target, text="RA:").grid(row=4, column=0, sticky=tk.E, padx=(10, 2), pady=5)
    target_ra_h_var = tk.StringVar(value="14")
    ttk.Entry(lf_target, textvariable=target_ra_h_var, width=4).grid(row=4, column=1, sticky=tk.W, padx=2)
    tk.Label(lf_target, text="h").grid(row=4, column=2, sticky=tk.W, padx=0)
    target_ra_m_var = tk.StringVar(value="34")
    ttk.Entry(lf_target, textvariable=target_ra_m_var, width=4).grid(row=4, column=3, sticky=tk.W, padx=2)
    tk.Label(lf_target, text="m").grid(row=4, column=4, sticky=tk.W, padx=0)
    target_ra_s_var = tk.StringVar(value="00.00")
    ttk.Entry(lf_target, textvariable=target_ra_s_var, width=6).grid(row=4, column=5, sticky=tk.W, padx=2)
    tk.Label(lf_target, text="s").grid(row=4, column=6, sticky=tk.W, padx=0)
    
    # Dec boxes
    tk.Label(lf_target, text="Dec:").grid(row=5, column=0, sticky=tk.E, padx=(10, 2), pady=5)
    target_dec_d_var = tk.StringVar(value="+43")
    ttk.Entry(lf_target, textvariable=target_dec_d_var, width=4).grid(row=5, column=1, sticky=tk.W, padx=2)
    tk.Label(lf_target, text="d").grid(row=5, column=2, sticky=tk.W, padx=0)
    target_dec_m_var = tk.StringVar(value="30")
    ttk.Entry(lf_target, textvariable=target_dec_m_var, width=4).grid(row=5, column=3, sticky=tk.W, padx=2)
    tk.Label(lf_target, text="m").grid(row=5, column=4, sticky=tk.W, padx=0)
    target_dec_s_var = tk.StringVar(value="00.0")
    ttk.Entry(lf_target, textvariable=target_dec_s_var, width=6).grid(row=5, column=5, sticky=tk.W, padx=2)
    tk.Label(lf_target, text="s").grid(row=5, column=6, sticky=tk.W, padx=0)
    
    def toggle_target_entries(*args):
        mode = target_mode_var.get()
        name_state = tk.NORMAL if mode == "name" else tk.DISABLED
        manual_state = tk.NORMAL if mode == "manual" else tk.DISABLED
        
        target_name_entry.config(state=name_state)
        check_target_btn.config(state=name_state)
        for child in lf_target.winfo_children():
            if isinstance(child, ttk.Entry) and child != target_name_entry:
                child.config(state=manual_state)
    
    target_mode_var.trace("w", toggle_target_entries)
    toggle_target_entries()

    def load_color_coefficients():
        import json
        json_path = os.path.join("photometry_output", "color_coefficients.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    coeffs = json.load(f)
                diff_tbv_var.set(coeffs.get('Tbv', 1.0))
                diff_tbbv_var.set(coeffs.get('Tb_bv', 0.0))
                diff_tvbv_var.set(coeffs.get('Tv_bv', 0.0))
                diff_status_var.set("Loaded coefficients from previous run.")
            except Exception as e:
                diff_status_var.set(f"Error loading JSON: {e}")
        else:
            diff_status_var.set("No previous coefficients found. Enter manually.")
            
    tk.Button(lf_diff, text="Load Last Coefficients", command=load_color_coefficients, bg="#f0f2f5", relief="flat").grid(row=5, column=0, columnspan=2, pady=5)
            
    diff_btn_frame = ttk.Frame(sub_diff)
    diff_btn_frame.pack(pady=5)
    
    tk.Button(diff_btn_frame, text="Execute Differential Photometry", command=on_run_diff,
              bg="#1a3a5f", fg="white", font=("Arial", 10, "bold"), width=35, relief="flat", pady=10).pack(side=tk.LEFT, padx=5)
              
    diff_info_btn = tk.Button(diff_btn_frame, text=f"{get_icon('❓', '?')} What does this do?", command=show_diff_info,
                              bg="#f0f2f5", fg="#1a3a5f", font=("Arial", 10, "bold"),
                              relief="flat", pady=10, padx=15)
    diff_info_btn.pack(side=tk.LEFT, padx=5)

    diff_status_label = SelectableLabel(sub_diff, textvariable=diff_status_var, font=("Arial", 9, "italic"), justify=tk.CENTER)
    diff_status_label.pack(pady=5, fill="x")

    # Preview for Accuracy
    lf_accuracy_preview = tk.LabelFrame(sub_diff, text="Accuracy Evaluation Preview")
    lf_accuracy_preview.pack(fill="x", padx=10, pady=5)
    
    accuracy_fig, accuracy_axes = plt.subplots(1, 3, figsize=(10, 3.5))
    accuracy_canvas = FigureCanvasTkAgg(accuracy_fig, master=lf_accuracy_preview)
    accuracy_canvas.get_tk_widget().pack(fill="x", expand=True)
    accuracy_toolbar = NavigationToolbar2Tk(accuracy_canvas, lf_accuracy_preview)
    accuracy_toolbar.update()

    ts_container = tab_ts
    
    # Filter selection for Light Curves
    filter_frame = ttk.Frame(ts_container)
    filter_frame.pack(fill="x", padx=10, pady=5)
    tk.Label(filter_frame, text="Light Curve Filter:").pack(side=tk.LEFT, padx=5)
    ts_filter_var = tk.StringVar(value="V")
    vars_dict["ts_filter"] = (ts_filter_var, str)
    ts_filter_cb = ttk.Combobox(filter_frame, textvariable=ts_filter_var, values=["V", "B"], state="readonly", width=5)
    ts_filter_cb.pack(side=tk.LEFT, padx=5)
    ts_widgets['filter_cb'] = ts_filter_cb
    def get_coords_ts(mode, name, ra_s, dec_s):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        if mode == "name":
            return SkyCoord.from_name(name)
        else:
            return SkyCoord(f"{ra_s} {dec_s}", unit=(u.hourangle, u.deg))

    # --- Ensemble Reference Stars ---
    lf_ts_ensemble = tk.LabelFrame(ts_container, text="Ensemble Reference Stars (Comparison)")

    # Coefficients & Metadata
    lf_ts_coeff = tk.LabelFrame(ts_container, text="Coefficients & Metadata")
    
    ts_check_star_idx_var = tk.IntVar(value=-1)
    vars_dict["ts_check_star_idx"] = (ts_check_star_idx_var, int)
    
    def create_ensemble_row(idx, container):
        row_f = ttk.Frame(container)
        row_f.pack(fill="x", pady=2)
        
        tk.Label(row_f, text=f"Star {idx+1}:", width=6).pack(side=tk.LEFT, padx=5)
        
        name_v = tk.StringVar(value="")
        vars_dict[f"ts_ref_{idx}_name"] = (name_v, str)
        ttk.Entry(row_f, textvariable=name_v, width=15).pack(side=tk.LEFT, padx=2)
        
        tk.Label(row_f, text="Mag:").pack(side=tk.LEFT, padx=2)
        mag_v = tk.DoubleVar(value=10.0)
        vars_dict[f"ts_ref_{idx}_mag"] = (mag_v, float)
        ttk.Entry(row_f, textvariable=mag_v, width=6).pack(side=tk.LEFT, padx=2)
        
        tk.Label(row_f, text="B-V:").pack(side=tk.LEFT, padx=2)
        bv_v = tk.DoubleVar(value=0.5)
        vars_dict[f"ts_ref_{idx}_bv"] = (bv_v, float)
        ttk.Entry(row_f, textvariable=bv_v, width=6).pack(side=tk.LEFT, padx=2)
        
        # Manual Coords
        ra_val = tk.DoubleVar(value=0.0)
        dec_val = tk.DoubleVar(value=0.0)
        has_manual = tk.BooleanVar(value=False)
        vars_dict[f"ts_ref_{idx}_ra"] = (ra_val, float)
        vars_dict[f"ts_ref_{idx}_dec"] = (dec_val, float)
        vars_dict[f"ts_ref_{idx}_has_manual"] = (has_manual, bool)
        
        use_v = tk.BooleanVar(value=(idx == 0))
        vars_dict[f"ts_ref_{idx}_use"] = (use_v, bool)
        use_chk = ttk.Checkbutton(row_f, text="Use", variable=use_v)
        use_chk.pack(side=tk.LEFT, padx=2)
        
        def on_use_change(*args):
            # If Use is checked, make sure it's not the check star
            if use_v.get() and ts_check_star_idx_var.get() == idx:
                ts_check_star_idx_var.set(-1)
                
        use_v.trace_add("write", on_use_change)
        
        ttk.Radiobutton(row_f, text="Check", variable=ts_check_star_idx_var, value=idx).pack(side=tk.LEFT, padx=2)
        
        coord_v = tk.StringVar(value="")
        vars_dict[f"ts_ref_{idx}_coord_label"] = (coord_v, str)
        coord_label = SelectableLabel(row_f, textvariable=coord_v, font=("Arial", 8, "italic"))
        coord_label.pack(side=tk.LEFT, padx=5)
        
        def on_fetch():
            name = name_v.get().strip()
            from photometry.calibration import resolve_auid, fetch_online_catalog
            import re
            
            c = None
            # 1. Detect and Resolve AUID
            if re.match(r'^[0-9A-Za-z]{3}-[0-9A-Za-z]{3}-[0-9A-Za-z]{3}$', name):
                ts_status_var.set(f"Resolving AUID {name}...")
                root.update_idletasks()
                res = resolve_auid(name, target_hint=ts_target_name_var.get())
                if res:
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    c = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg)
                    # If we don't have a name yet, or it's just the AUID, maybe keep it
                else:
                    ts_status_var.set(f"AUID {name} not found in VSX.")
                    return
            
            # 2. Try Name Resolution if not an AUID or AUID resolution failed
            if not c and name and not name.lower().startswith("star "):
                ts_status_var.set(f"Resolving {name}...")
                root.update_idletasks()
                try:
                    from astropy.coordinates import SkyCoord
                    c = SkyCoord.from_name(name)
                except: pass
                
            # 3. Fallback to manual coords
            if not c:
                if has_manual.get():
                    try:
                        from astropy.coordinates import SkyCoord
                        import astropy.units as u
                        c = SkyCoord(ra=ra_val.get()*u.deg, dec=dec_val.get()*u.deg)
                        ts_status_var.set(f"Fetching from coordinates...")
                    except:
                        ts_status_var.set("Invalid coordinates.")
                        return
                elif name:
                    # Final attempt at name resolution if we skipped it earlier
                    ts_status_var.set(f"Resolving {name}...")
                    root.update_idletasks()
                    try:
                        from astropy.coordinates import SkyCoord
                        c = SkyCoord.from_name(name)
                    except Exception as e:
                        ts_status_var.set(f"Could not resolve '{name}': {e}")
                        return
                else:
                    ts_status_var.set("No name or coordinates provided.")
                    return

            if c:
                # Update UI immediately with resolved coordinates
                ra_val.set(c.ra.deg)
                dec_val.set(c.dec.deg)
                has_manual.set(True)
                
                ra_hms = c.ra.to_string(unit='hour', sep=':', precision=1)
                dec_dms = c.dec.to_string(unit='degree', sep=':', precision=1, alwayssign=True)
                coord_v.set(f"({ra_hms}, {dec_dms}) [Resolving...]")
                
                # Clear stale mags
                mag_v.set(0.0)
                bv_v.set(0.0)
                root.update_idletasks()

            root.update_idletasks()
            try:
                import astropy.units as u
                from photometry.calibration import fetch_online_catalog
                
                cat_name = vars_dict["reference_catalog"][0].get()
                stars = fetch_online_catalog(c.ra.deg, c.dec.deg, radius_arcmin=2.0, catalog_name=cat_name)
                if not stars:
                    ts_status_var.set(f"No catalog match found near coordinates.")
                    return
                
                cat_coords = SkyCoord([s['ra_deg'] for s in stars], [s['dec_deg'] for s in stars], unit=u.deg)
                match_idx, d2d, _ = c.match_to_catalog_sky(cat_coords)
                # Fix: Handle d2d as array or scalar (Astropy returns 1-item array for single coord)
                dist_arcsec = d2d.arcsec[0] if hasattr(d2d.arcsec, "__len__") else d2d.arcsec
                if dist_arcsec > 10.0:
                    ts_status_var.set(f"No match found in {cat_name} (>10\")")
                    return
                    
                star = stars[match_idx]
                filt = ts_filter_var.get().upper()
                mag = star['B_mag'] if filt == 'B' else star['V_mag']
                bv = star['B_mag'] - star['V_mag']
                
                mag_v.set(round(mag, 3))
                bv_v.set(round(bv, 3))
                
                # Update coordinates to precise catalog ones
                ra_val.set(star['ra_deg'])
                dec_val.set(star['dec_deg'])
                has_manual.set(True) # Keep as true so runner uses these coords directly
                
                c_cat = SkyCoord(ra=star['ra_deg']*u.deg, dec=star['dec_deg']*u.deg)
                ra_hms = c_cat.ra.to_string(unit='hour', sep=':', precision=1)
                dec_dms = c_cat.dec.to_string(unit='degree', sep=':', precision=1, alwayssign=True)
                coord_v.set(f"({ra_hms}, {dec_dms}) [Cat]")
                
                if not name_v.get(): name_v.set(star.get('id', 'RefStar'))
                
                # Fix: Ensure dist_arcsec is treated as float for formatting
                ts_status_var.set(f"Updated from {cat_name} (Dist: {float(dist_arcsec):.1f}\")")
            except Exception as e:
                ts_status_var.set(f"Fetch failed: {e}")
                print(f"Fetch error details: {e}")

        # Store the fetch function so it can be called programmatically
        if 'ts_ref_fetch_funcs' not in vars_dict: vars_dict['ts_ref_fetch_funcs'] = {}
        vars_dict['ts_ref_fetch_funcs'][idx] = on_fetch

        ttk.Button(row_f, text="Fetch", command=on_fetch, width=6).pack(side=tk.LEFT, padx=2)
        
        def on_manual():
            pop = tk.Toplevel(root)
            pop.title(f"Manual Coords Star {idx+1}")
            pop.geometry("300x150")
            
            tk.Label(pop, text="Enter RA (HMS) or Deg:").pack(pady=5)
            ra_e = ttk.Entry(pop, width=25)
            ra_e.pack()
            if has_manual.get(): ra_e.insert(0, str(ra_val.get()))
            
            tk.Label(pop, text="Enter Dec (DMS) or Deg:").pack(pady=5)
            dec_e = ttk.Entry(pop, width=25)
            dec_e.pack()
            if has_manual.get(): dec_e.insert(0, str(dec_val.get()))
            
            def save_manual():
                try:
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    # Try to parse
                    c_str = f"{ra_e.get()} {dec_e.get()}"
                    if ":" in c_str or " " in c_str.strip():
                        c = SkyCoord(c_str, unit=(u.hourangle, u.deg))
                    else:
                        c = SkyCoord(ra=float(ra_e.get())*u.deg, dec=float(dec_e.get())*u.deg)
                    
                    ra_val.set(c.ra.deg)
                    dec_val.set(c.dec.deg)
                    has_manual.set(True)
                    
                    ra_hms = c.ra.to_string(unit='hour', sep=':', precision=1)
                    dec_dms = c.dec.to_string(unit='degree', sep=':', precision=1, alwayssign=True)
                    coord_v.set(f"({ra_hms}, {dec_dms}) [M]")
                    if not name_v.get(): name_v.set(f"Star_{idx+1}_Man")
                    
                    pop.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid coordinates: {e}")
            
            ttk.Button(pop, text="Save", command=save_manual).pack(pady=10)

        ttk.Button(row_f, text="Manual", command=on_manual, width=6).pack(side=tk.LEFT, padx=2)

    for i in range(5):
        create_ensemble_row(i, lf_ts_ensemble)
        
    ttk.Radiobutton(lf_ts_ensemble, text="No Check Star", variable=ts_check_star_idx_var, value=-1).pack(anchor=tk.W, padx=10)

    def on_check_star_change(*args):
        # If a check star is selected, uncheck its "Use" box
        idx = ts_check_star_idx_var.get()
        if idx >= 0:
            vars_dict[f"ts_ref_{idx}_use"][0].set(False)
            
    ts_check_star_idx_var.trace_add("write", on_check_star_change)

    # Light Curve Preview
    lf_ts_plot = tk.LabelFrame(ts_container, text="Light Curve Preview")
    
    ts_fig, ts_ax = plt.subplots(figsize=(8, 4))
    ts_canvas = FigureCanvasTkAgg(ts_fig, master=lf_ts_plot)
    ts_canvas.get_tk_widget().pack(fill="both", expand=True)
    ts_toolbar = NavigationToolbar2Tk(ts_canvas, lf_ts_plot)
    ts_toolbar.update()

    # Target Star
    lf_ts_target = tk.LabelFrame(ts_container, text="Target Star (Variable)")
    # Now pack in the requested order
    # 1. Selection (Handled by File Manager)
    lf_ts_target.pack(fill="x", padx=10, pady=5)
    lf_ts_ensemble.pack(fill="x", padx=10, pady=5)
    lf_ts_coeff.pack(fill="x", padx=10, pady=5)
    
    ts_target_mode_var = tk.StringVar(value="name")
    vars_dict["ts_target_mode"] = (ts_target_mode_var, str)
    ttk.Radiobutton(lf_ts_target, text="Variable Name:", variable=ts_target_mode_var, value="name").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
    
    ts_target_name_var = tk.StringVar(value="AE UMa")
    vars_dict["ts_target_name"] = (ts_target_name_var, str)
    ttk.Entry(lf_ts_target, textvariable=ts_target_name_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=2)
    
    ts_target_coord_display_var = tk.StringVar(value="")
    ts_target_coord_label = SelectableLabel(lf_ts_target, textvariable=ts_target_coord_display_var, font=("Arial", 8, "italic"))
    ts_target_coord_label.grid(row=0, column=3, sticky=tk.W, padx=10)

    def on_fetch_target_ts():
        name = ts_target_name_var.get().strip()
        if not name: return
        ts_status_var.set(f"Fetching {name}...")
        root.update_idletasks()
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from photometry.calibration import fetch_online_catalog, resolve_auid
            import re
            
            c = None
            if re.match(r'^[0-9A-Za-z]{3}-[0-9A-Za-z]{3}-[0-9A-Za-z]{3}$', name):
                res = resolve_auid(name)
                if res:
                    c = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg)
                else:
                    ts_status_var.set(f"AUID {name} not found in VSX.")
                    return
            else:
                c = SkyCoord.from_name(name)
            
            if c:
                ra_hms = c.ra.to_string(unit='hour', sep=':', precision=1)
                dec_dms = c.dec.to_string(unit='degree', sep=':', precision=1, alwayssign=True)
                ts_target_coord_display_var.set(f"({ra_hms}, {dec_dms}) [Resolving...]")
                ts_target_ra_var.set(ra_hms)
                ts_target_dec_var.set(dec_dms)
                root.update_idletasks()

            cat_name = vars_dict["reference_catalog"][0].get()
            stars = fetch_online_catalog(c.ra.deg, c.dec.deg, radius_arcmin=2.0, catalog_name=cat_name)
            bv_str = ""
            if stars:
                cat_coords = SkyCoord([s['ra_deg'] for s in stars], [s['dec_deg'] for s in stars], unit=u.deg)
                match_idx, d2d, _ = c.match_to_catalog_sky(cat_coords)
                dist_arcsec = d2d.arcsec[0] if hasattr(d2d.arcsec, "__len__") else d2d.arcsec
                if dist_arcsec < 10.0:
                    star = stars[match_idx]
                    bv = star['B_mag'] - star['V_mag']
                    ts_target_bv_var.set(round(bv, 3))
                    bv_str = f", B-V: {bv:.3f}"
            
            ts_target_coord_display_var.set(f"({ra_hms}, {dec_dms}){bv_str}")
            ts_status_var.set(f"Target {name} resolved.")
        except Exception as e:
            ts_status_var.set(f"Target resolution failed: {e}")

    def on_get_aavso_refs():
        if not hasattr(on_get_aavso_refs, "cache"):
            on_get_aavso_refs.cache = None
            on_get_aavso_refs.last_target = ""
            on_get_aavso_refs.last_cat = ""
            
        target = ts_target_name_var.get().strip()
        cat_name = vars_dict["reference_catalog"][0].get()
        if not target:
            messagebox.showwarning("Warning", "Please enter a target star name first.")
            return
            
        # Check if we can use cached results
        if (on_get_aavso_refs.cache and 
            on_get_aavso_refs.last_target == target and 
            on_get_aavso_refs.last_cat == cat_name):
            stars = on_get_aavso_refs.cache
            ts_status_var.set(f"Showing cached AAVSO sequence for {target}.")
        else:
            try:
                radius = float(vars_dict["catalog_search_radius"][0].get())
            except:
                radius = 45.0
                
            ts_status_var.set(f"Fetching AAVSO sequence for {target}...")
            root.update_idletasks()
            
            from photometry.calibration import fetch_aavso_chart_stars, fetch_online_catalog
            stars = fetch_aavso_chart_stars(target, fov_arcmin=radius)
            
            if not stars:
                messagebox.showinfo("No Stars Found", f"Could not find any comparison stars for {target} in the AAVSO VSP.")
                return

            # Map catalog name for fetch_online_catalog
            cat_map = {
                "ATLAS refcat2": "ATLAS",
                "APASS DR9": "APASS",
                "Landolt Standard Star Catalogue": "LANDOLT",
                "GAIA_DR3": "GAIA_DR3"
            }
            internal_cat_name = cat_map.get(cat_name, "ATLAS")
            
            ts_status_var.set(f"Cross-matching AAVSO stars with {cat_name}...")
            root.update_idletasks()
            
            try:
                center_ra = stars[0]['ra']
                center_dec = stars[0]['dec']
                cat_stars = fetch_online_catalog(center_ra, center_dec, radius_arcmin=radius*1.1, catalog_name=internal_cat_name)
                
                if cat_stars:
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    aavso_coords = SkyCoord([s['ra'] for s in stars], [s['dec'] for s in stars], unit=u.deg)
                    cat_coords = SkyCoord([s['ra_deg'] for s in cat_stars], [s['dec_deg'] for s in cat_stars], unit=u.deg)
                    idx, d2d, _ = aavso_coords.match_to_catalog_sky(cat_coords)
                    for i, s in enumerate(stars):
                        if d2d[i].arcsec < 3.0:
                            match = cat_stars[idx[i]]
                            s['cat_v'] = match.get('V_mag', np.nan)
                            s['cat_b'] = match.get('B_mag', np.nan)
                            s['cat_match'] = True
                        else:
                            s['cat_v'] = np.nan; s['cat_b'] = np.nan; s['cat_match'] = False
            except Exception as e:
                print(f"Catalog cross-match failed: {e}")
                for s in stars: s['cat_v'] = np.nan; s['cat_b'] = np.nan; s['cat_match'] = False
            
            # Update Cache
            on_get_aavso_refs.cache = stars
            on_get_aavso_refs.last_target = target
            on_get_aavso_refs.last_cat = cat_name

        # Create Popup Window
        pop = tk.Toplevel(root)
        pop.title(f"AAVSO Comparison Stars: {target}")
        pop.geometry("1100x550")
        pop.transient(root)
        
        tk.Label(pop, text=f"Select stars for Ensemble (AAVSO values vs {cat_name}):", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Table of stars
        cols = ("AUID", "Name", "V (AAVSO)", "B (AAVSO)", "B-V (AAVSO)", f"V ({cat_name})", f"B ({cat_name})", "RA", "Dec")
        tree_stars = ttk.Treeview(pop, columns=cols, show='headings', selectmode='extended')
        for col in cols:
            tree_stars.heading(col, text=col)
            tree_stars.column(col, width=100 if "Name" not in col and "AUID" not in col else 120, anchor=tk.CENTER)
        
        tree_stars.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Populate Treeview
        for s in stars:
            bv = s['B_mag'] - s['V_mag']
            cat_v_str = f"{s['cat_v']:.3f}" if not np.isnan(s['cat_v']) else "N/A"
            cat_b_str = f"{s['cat_b']:.3f}" if not np.isnan(s['cat_b']) else "N/A"
            tree_stars.insert("", tk.END, values=(
                s['auid'], s['name'], f"{s['V_mag']:.3f}", f"{s['B_mag']:.3f}", f"{bv:.3f}", 
                cat_v_str, cat_b_str, f"{s['ra']:.4f}", f"{s['dec']:.4f}"
            ))
            
        # Source Selection
        source_var = tk.StringVar(value="catalog")
        src_frame = ttk.Frame(pop)
        src_frame.pack(fill="x", padx=20, pady=5)
        tk.Label(src_frame, text="Magnitude Source to Import:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(src_frame, text=f"Use User Catalog ({cat_name})", variable=source_var, value="catalog").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(src_frame, text="Use AAVSO Chart Values", variable=source_var, value="aavso").pack(side=tk.LEFT, padx=10)

        def do_import():
            selected_items = tree_stars.selection()
            if not selected_items:
                messagebox.showwarning("No Selection", "Please select at least one star.")
                return
                
            count = 0
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            use_catalog = (source_var.get() == "catalog")
            
            for item in selected_items:
                if count >= 5: break
                vals = tree_stars.item(item, 'values')
                auid = vals[0]
                star_data = next((s for s in stars if s['auid'] == auid), None)
                if not star_data: continue
                
                v_to_use = star_data['V_mag']
                b_to_use = star_data['B_mag']
                src_label = "AAVSO"
                if use_catalog and star_data.get('cat_match'):
                    v_to_use = star_data['cat_v']
                    b_to_use = star_data['cat_b']
                    src_label = cat_name.split()[0] # Short name
                
                vars_dict[f"ts_ref_{count}_name"][0].set(star_data['auid'])
                vars_dict[f"ts_ref_{count}_mag"][0].set(v_to_use)
                vars_dict[f"ts_ref_{count}_bv"][0].set(round(b_to_use - v_to_use, 3))
                vars_dict[f"ts_ref_{count}_ra"][0].set(star_data['ra'])
                vars_dict[f"ts_ref_{count}_dec"][0].set(star_data['dec'])
                vars_dict[f"ts_ref_{count}_has_manual"][0].set(True)
                
                c = SkyCoord(ra=star_data['ra']*u.deg, dec=star_data['dec']*u.deg)
                ra_hms = c.ra.to_string(unit='hour', sep=':', precision=1)
                dec_dms = c.dec.to_string(unit='degree', sep=':', precision=1, alwayssign=True)
                vars_dict[f"ts_ref_{count}_coord_label"][0].set(f"({ra_hms}, {dec_dms}) [{src_label}]")
                vars_dict[f"ts_ref_{count}_use"][0].set(True)
                count += 1
            
            ts_status_var.set(f"Imported {count} stars to ensemble.")
            pop.destroy()
            
        btn_frame = ttk.Frame(pop)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Add Selected to Ensemble", command=do_import, width=30).pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Button(btn_frame, text="Clear AAVSO Cache & Re-fetch", command=lambda: [pop.destroy(), setattr(on_get_aavso_refs, "cache", None), on_get_aavso_refs()], width=30).pack(side=tk.LEFT, padx=10, expand=True)
        
        ts_status_var.set(f"Found {len(stars)} AAVSO comparison stars.")

    def on_reset_ensemble():
        if not messagebox.askyesno("Confirm Reset", "Are you sure you want to completely clear the Ensemble star list?"):
            return
        for i in range(5):
            vars_dict[f"ts_ref_{i}_name"][0].set("")
            vars_dict[f"ts_ref_{i}_mag"][0].set(0.0)
            vars_dict[f"ts_ref_{i}_bv"][0].set(0.0)
            vars_dict[f"ts_ref_{i}_ra"][0].set(0.0)
            vars_dict[f"ts_ref_{i}_dec"][0].set(0.0)
            vars_dict[f"ts_ref_{i}_has_manual"][0].set(False)
            vars_dict[f"ts_ref_{i}_coord_label"][0].set("(00:00:00, +00:00:00)")
            vars_dict[f"ts_ref_{i}_use"][0].set(False)
        ts_status_var.set("Ensemble list cleared.")

    ttk.Button(lf_ts_target, text="Fetch", command=on_fetch_target_ts, width=6).grid(row=0, column=2, sticky=tk.W, padx=2)
    ttk.Button(lf_ts_target, text="Get AAVSO Ref Stars", command=on_get_aavso_refs).grid(row=0, column=4, sticky=tk.W, padx=10)
    ttk.Button(lf_ts_target, text="Reset Ensemble", command=on_reset_ensemble).grid(row=0, column=5, sticky=tk.W, padx=2)
    vars_dict['ts_target_fetch_func'] = on_fetch_target_ts
    
    ttk.Radiobutton(lf_ts_target, text="Manual RA/Dec", variable=ts_target_mode_var, value="manual").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
    ts_target_ra_var = tk.StringVar(value="14:34:00")
    vars_dict["ts_target_ra"] = (ts_target_ra_var, str)
    ts_target_dec_var = tk.StringVar(value="+43:30:00")
    vars_dict["ts_target_dec"] = (ts_target_dec_var, str)
    ttk.Entry(lf_ts_target, textvariable=ts_target_ra_var, width=12).grid(row=1, column=1, sticky=tk.W, padx=2)
    ttk.Entry(lf_ts_target, textvariable=ts_target_dec_var, width=12).grid(row=1, column=2, sticky=tk.W, padx=2)
    
    tk.Label(lf_ts_target, text="Target (B-V) [assumed]:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
    ts_target_bv_var = tk.DoubleVar(value=0.5)
    vars_dict["ts_target_bv"] = (ts_target_bv_var, float)
    ttk.Entry(lf_ts_target, textvariable=ts_target_bv_var, width=8).grid(row=2, column=1, sticky=tk.W, padx=2)

    # Coefficients
    # Moved to packing section above
    
    # Coefficients & Metadata
    ts_do_trans_var = tk.BooleanVar(value=False)
    vars_dict["ts_do_trans"] = (ts_do_trans_var, bool)
    ttk.Checkbutton(lf_ts_coeff, text="Apply Color Transformation (Set TRANS=YES in report)", variable=ts_do_trans_var).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)

    tk.Label(lf_ts_coeff, text="Color Term (e.g. Tv_bv):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
    ts_coeff_var = tk.DoubleVar(value=0.0)
    vars_dict["ts_coeff"] = (ts_coeff_var, float)
    ttk.Entry(lf_ts_coeff, textvariable=ts_coeff_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=2)
    
    tk.Label(lf_ts_coeff, text="Extinction (k):").grid(row=1, column=2, sticky=tk.W, padx=10, pady=2)
    ts_k_var = tk.DoubleVar(value=0.15)
    vars_dict["ts_k"] = (ts_k_var, float)
    ttk.Entry(lf_ts_coeff, textvariable=ts_k_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=2)
    
    tk.Label(lf_ts_coeff, text="AAVSO Observer Code:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
    # Linked to global settings
    ts_obs_var = vars_dict["aavso_obs_code"][0]
    ttk.Entry(lf_ts_coeff, textvariable=ts_obs_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=2)
    
    tk.Label(lf_ts_coeff, text="Site Lat:").grid(row=2, column=2, sticky=tk.W, padx=10, pady=2)
    ts_lat_var = tk.DoubleVar(value=59.8)
    vars_dict["ts_lat"] = (ts_lat_var, float)
    ttk.Entry(lf_ts_coeff, textvariable=ts_lat_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=2)
    
    tk.Label(lf_ts_coeff, text="Site Long:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=2)
    ts_lon_var = tk.DoubleVar(value=17.6)
    vars_dict["ts_lon"] = (ts_lon_var, float)
    ttk.Entry(lf_ts_coeff, textvariable=ts_lon_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=2)

    def load_ts_coefficients():
        import json
        json_path = os.path.join("photometry_output", "color_coefficients.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    coeffs = json.load(f)
                filt = ts_filter_var.get().upper()
                if filt == 'B':
                    val = coeffs.get('Tb_bv', 0.0)
                    key_name = 'Tb_bv'
                else:
                    val = coeffs.get('Tv_bv', 0.0)
                    key_name = 'Tv_bv'
                ts_coeff_var.set(val)
                ts_status_var.set(f"Loaded {key_name} for {filt} filter.")
            except Exception as e:
                ts_status_var.set(f"Error loading JSON: {e}")
        else:
            ts_status_var.set("No coefficients file found.")
            
    tk.Button(lf_ts_coeff, text="Load Last Coeffs", command=load_ts_coefficients).grid(row=4, column=0, columnspan=4, pady=5)

    ts_status_var = tk.StringVar(value="Ready to process sequence.")
    ts_status_label = SelectableLabel(ts_container, textvariable=ts_status_var, font=("Arial", 9, "italic"), justify=tk.CENTER)
    ts_status_label.pack(pady=5, fill="x")
    
    ts_progress_var = tk.DoubleVar(value=0)
    ts_progress = ttk.Progressbar(ts_container, variable=ts_progress_var, maximum=100, length=400)
    ts_progress.pack(pady=5)
    
    # Analysis Summary Panel (Persistent)
    lf_ts_summary = tk.LabelFrame(ts_container, text="Analysis Summary")
    lf_ts_summary.pack(fill="x", padx=10, pady=5)
    ts_summary_var = tk.StringVar(value="No analysis data yet. Run 'Generate Light Curve' to see results.")
    tk.Label(lf_ts_summary, textvariable=ts_summary_var, font=("Arial", 10), justify=tk.LEFT, anchor="w", fg="#2c3e50").pack(fill="x", padx=15, pady=10)

    cancel_event = threading.Event()
    
    def on_cancel_ts():
        if messagebox.askyesno("Cancel", "Stop processing the sequence?"):
            cancel_event.set()
            ts_status_var.set("Cancelling...")

    cancel_btn = tk.Button(ts_container, text="Cancel Processing", command=on_cancel_ts, 
                           bg="#f44336", fg="white", font=("Arial", 9))
    # Packed later during execution or hidden by default
    
    def on_run_ts():
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager.")
            return
        
        selected_filter = ts_filter_var.get().upper()
        v_key = vars_dict["filter_v_keyword"][0].get().upper()
        b_key = vars_dict["filter_b_keyword"][0].get().upper()
        
        files = []
        for iid in selected_iids:
            file_data = loaded_files[int(iid)]
            filt_str = str(file_data['filter']).upper()
            
            # Determine translated filter of this file
            trans_filt = filt_str
            if v_key and v_key in filt_str:
                trans_filt = "V"
            elif b_key and b_key in filt_str:
                trans_filt = "B"
            
            if selected_filter == trans_filt:
                files.append(file_data['path'])
        
        if not files:
            messagebox.showwarning("No Matching Files", f"None of the checked files match the selected filter: {selected_filter}")
            return
            
        # Collect Ensemble & Check Star
        ensemble_data = []
        check_star_data = None
        cs_idx = ts_check_star_idx_var.get()
        
        for i in range(5):
            name = vars_dict[f"ts_ref_{i}_name"][0].get().strip()
            mag = vars_dict[f"ts_ref_{i}_mag"][0].get()
            bv = vars_dict[f"ts_ref_{i}_bv"][0].get()
            if name or vars_dict[f"ts_ref_{i}_has_manual"][0].get():
                s_dict = {
                    'name': name if name else f"Star_{i+1}", 
                    'mag_std': mag, 
                    'bv_std': bv,
                    'ra_man': vars_dict[f"ts_ref_{i}_ra"][0].get(),
                    'dec_man': vars_dict[f"ts_ref_{i}_dec"][0].get(),
                    'has_manual': vars_dict[f"ts_ref_{i}_has_manual"][0].get()
                }
                if i == cs_idx:
                    check_star_data = s_dict
                if vars_dict[f"ts_ref_{i}_use"][0].get():
                    ensemble_data.append(s_dict)
        
        if not ensemble_data:
            messagebox.showerror("Error", "Please select at least one reference star in the ensemble.")
            return

        ts_status_var.set("Resolving target coordinates...")
        root.update_idletasks()
        
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        try:
            tar_c = get_coords_ts(ts_target_mode_var.get(), ts_target_name_var.get(), ts_target_ra_var.get(), ts_target_dec_var.get())
        except Exception as e:
            messagebox.showerror("Coord Error", f"Target coordinate resolution failed: {e}")
            return

        def ts_thread():
            from photometry.time_series import run_time_series_photometry, save_aavso_report, plot_light_curve
            
            # Resolve Ensemble Coords
            ts_status_var.set("Resolving ensemble coordinates...")
            resolved_ensemble = []
            # Combine ensemble and check star for resolution
            to_resolve = list(ensemble_data)
            if check_star_data and check_star_data not in to_resolve:
                to_resolve.append(check_star_data)
                
            for s in to_resolve:
                try:
                    if s.get('has_manual'):
                        s['ra'] = s['ra_man']
                        s['dec'] = s['dec_man']
                    else:
                        c = SkyCoord.from_name(s['name'])
                        s['ra'] = c.ra.deg
                        s['dec'] = c.dec.deg
                    
                    if s in ensemble_data:
                        resolved_ensemble.append(s)
                except Exception as e:
                    root.after(0, lambda: messagebox.showerror("Ensemble Error", f"Could not resolve star '{s['name']}': {e}"))
                    ts_status_var.set("Failed: Coordinate resolution error.")
                    return

            ts_status_var.set(f"Processing {len(files)} files...")
            
            # Reset Phase 3 state
            root.after(0, lambda: cancel_btn.pack(pady=5))
            root.after(0, lambda: ts_progress_var.set(0))
            cancel_event.clear()
            
            def update_prog(val):
                root.after(0, lambda: ts_progress_var.set(val))

            results, msg = run_time_series_photometry(
                files, target_ra=tar_c.ra.deg, target_dec=tar_c.dec.deg,
                ensemble_stars=resolved_ensemble,
                check_star=check_star_data,
                target_bv=ts_target_bv_var.get(),
                coeff_term=ts_coeff_var.get() if ts_do_trans_var.get() else 0.0,
                coeff_color=0.0, # Not used in current simplified model
                aperture_radius=float(vars_dict["aperture_radius"][0].get()),
                annulus_inner=float(vars_dict["annulus_inner"][0].get()),
                annulus_outer=float(vars_dict["annulus_outer"][0].get()),
                gain=float(vars_dict["ccd_gain"][0].get()),
                k_coeff=ts_k_var.get(),
                filter_name=ts_filter_var.get(),
                site_lat=ts_lat_var.get(),
                site_long=ts_lon_var.get(),
                cancel_event=cancel_event,
                update_progress=update_prog,
                use_flexible_aperture=bool(vars_dict["use_flexible_aperture"][0].get()),
                aperture_fwhm_factor=float(vars_dict["aperture_fwhm_factor"][0].get()),
                annulus_inner_gap=float(vars_dict["annulus_inner_gap"][0].get()),
                annulus_width=float(vars_dict["annulus_width"][0].get()),
                print_psf_fitting=bool(vars_dict.get("print_psf_fitting", [None, None])[0].get()) if "print_psf_fitting" in vars_dict else False
            )
            
            root.after(0, lambda: cancel_btn.pack_forget())
            
            if results:
                out_csv = os.path.join("photometry_output", f"light_curve_{ts_target_name_var.get().replace(' ','_')}.csv")
                out_aavso = os.path.join("photometry_output", f"aavso_{ts_target_name_var.get().replace(' ','_')}.txt")
                out_plot = os.path.join("photometry_output", f"plot_{ts_target_name_var.get().replace(' ','_')}.png")
                
                # Save CSV
                with open(out_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                
                # AAVSO Metadata
                comp_name = "ENSEMBLE" if len(resolved_ensemble) > 1 else resolved_ensemble[0]['name']
                comp_mag = "na" if len(resolved_ensemble) > 1 else resolved_ensemble[0]['mag_std']
                check_name = check_star_data['name'] if check_star_data else "na"
                check_mag = check_star_data['mag_std'] if check_star_data else "na"
                is_trans = "YES" if ts_do_trans_var.get() and abs(ts_coeff_var.get()) > 1e-5 else "NO"
                
                # Construct header comments (Fix)
                cat_name = vars_dict.get("reference_catalog", (tk.StringVar(value="na"), None))[0].get()
                report_comments = f"RefCat: {cat_name}"
                if is_trans == "YES":
                    report_comments += " | Transformed to Johnson-Cousins"
                
                # Construct observation notes (Ensemble details)
                ens_names = " ".join([s['name'] for s in resolved_ensemble])
                report_notes = f"Ensemble of {len(resolved_ensemble)}- {ens_names}"
                
                save_aavso_report(results, out_aavso, ts_target_name_var.get(), ts_filter_var.get(), ts_obs_var.get(),
                                  comp_name=comp_name, comp_mag=comp_mag, 
                                  check_name=check_name, check_mag=check_mag,
                                  trans=is_trans, software_version=APP_VERSION,
                                  comments=report_comments,
                                  notes=report_notes)
                
                # Calculate Check Star Stats if available
                check_stats = ""
                check_mags = [r['check_mag'] for r in results if isinstance(r.get('check_mag'), (int, float)) and not np.isnan(r['check_mag'])]
                if check_mags and check_star_data:
                    avg_check = np.mean(check_mags)
                    std_check = np.std(check_mags)
                    bias = avg_check - check_star_data['mag_std']
                    check_stats = f"Check Star Stats ({check_star_data['name']}):\n"
                    check_stats += f"Average Bias: {bias:+.3f} (Meas - Cat)  |  Precision (1-sigma): {std_check:.3f} mag"
                
                # Update embedded plot
                plot_title = f"{ts_target_name_var.get()} ({ts_filter_var.get()} Filter)"
                def update_viz():
                    plot_light_curve(results, plot_title, out_plot, ax=ts_ax)
                    ts_canvas.draw()
                root.after(0, update_viz)
                
                # Update table
                def update_table():
                    # Clear existing
                    for item in ts_tree.get_children():
                        ts_tree.delete(item)
                    # Add new
                    for r in results:
                        ts_tree.insert("", tk.END, values=(
                            f"{r['jd']:.5f}", 
                            f"{r['hjd']:.5f}", 
                            f"{r['mag']:.3f}", 
                            f"{r['mag_err']:.3f}", 
                            f"{r['check_mag']:.3f}" if isinstance(r.get('check_mag'), (int, float)) and not np.isnan(r.get('check_mag')) else "N/A",
                            f"{r['snr']:.1f}",
                            f"{r.get('fwhm', 0.0):.2f}",
                            r.get('flag', 'OK')
                        ))
                root.after(0, update_table)

                # Update summary
                summary_text = f"Target: {ts_target_name_var.get()}  |  Filter: {ts_filter_var.get()}  |  Files: {len(results)}\n"
                summary_text += f"Saved CSV:    {os.path.abspath(out_csv)}\n"
                summary_text += f"Saved Plot:   {os.path.abspath(out_plot)}\n"
                summary_text += f"AAVSO Report: {os.path.abspath(out_aavso)}\n"
                if check_stats:
                    summary_text += f"--------------------------------------------------------------------------------\n"
                    summary_text += check_stats.strip()
                
                def update_summary():
                    ts_summary_var.set(summary_text)
                root.after(0, update_summary)
                
                ts_status_var.set(f"Complete! Results saved.")
            else:
                ts_status_var.set(f"Failed: {msg}")

        threading.Thread(target=ts_thread, daemon=True).start()
    # --- Light Curve Generation Button Row ---
    def on_clear_cache():
        from photometry.time_series import clear_session_cache
        if messagebox.askyesno("Clear Cache", "This will wipe all saved photometry results for these files. \n\nAre you sure?"):
            clear_session_cache()
            messagebox.showinfo("Cache Cleared", "The photometry cache has been wiped.")

    def on_open_report():
        target_name = ts_target_name_var.get().replace(' ','_')
        report_path = os.path.abspath(os.path.join("photometry_output", f"aavso_{target_name}.txt"))
        if os.path.exists(report_path):
            try:
                if sys.platform == 'win32':
                    os.startfile(report_path)
                elif sys.platform == 'darwin':
                    import subprocess
                    subprocess.call(['open', report_path])
                else:
                    import subprocess
                    subprocess.call(['xdg-open', report_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open report file: {e}")
        else:
            messagebox.showwarning("Not Found", f"No report file found at:\n{report_path}\n\nPlease generate a light curve first.")

    def on_reset_ensemble():
        if messagebox.askyesno("Reset Ensemble", "Are you sure you want to clear all selected reference stars and the check star?"):
            # Clear all TS ref vars
            for i in range(5):
                vars_dict[f"ts_ref_{i}_use"][0].set(False)
                vars_dict[f"ts_ref_{i}_name"][0].set("")
                vars_dict[f"ts_ref_{i}_mag"][0].set(0.0)
                vars_dict[f"ts_ref_{i}_bv"][0].set(0.0)
                vars_dict[f"ts_ref_{i}_ra"][0].set(0.0)
                vars_dict[f"ts_ref_{i}_dec"][0].set(0.0)
                vars_dict[f"ts_ref_{i}_has_manual"][0].set(False)
                if f"ts_ref_{i}_coord_label" in vars_dict:
                    vars_dict[f"ts_ref_{i}_coord_label"][0].set("")
            
            # Clear check star
            vars_dict["ts_check_use"][0].set(False)
            vars_dict["ts_check_name"][0].set("")
            vars_dict["ts_check_mag"][0].set(0.0)
            vars_dict["ts_check_bv"][0].set(0.0)
            vars_dict["ts_check_ra"][0].set(0.0)
            vars_dict["ts_check_dec"][0].set(0.0)
            ts_check_star_idx_var.set(-1)
            
            # Reset target defaults if needed
            ts_target_name_var.set("Target")
            
            messagebox.showinfo("Reset Complete", "Ensemble and check star configuration has been cleared.")

    ts_btn_container = ttk.Frame(ts_container)
    ts_btn_container.pack(pady=10)

    ts_btn_row1 = ttk.Frame(ts_btn_container)
    ts_btn_row1.pack(pady=5)

    ts_btn_row2 = ttk.Frame(ts_btn_container)
    ts_btn_row2.pack(pady=5)

    run_ts_btn = tk.Button(ts_btn_row1, text="Generate Light Curve", command=on_run_ts,
                           bg="#00796b", fg="white", font=("Arial", 11, "bold"), pady=10, width=25)
    run_ts_btn.pack(side=tk.LEFT, padx=10)

    ts_open_viewer_btn = tk.Button(ts_btn_row1, text="🔍 Open Selected FITS in Viewer", command=on_open_viewer_button,
                                   bg="#1a3a5f", fg="white", font=("Arial", 11, "bold"), pady=10, padx=15)
    ts_open_viewer_btn.pack(side=tk.LEFT, padx=10)

    clear_cache_btn = tk.Button(ts_btn_row2, text="Clear Photometry Cache", command=on_clear_cache,
                                bg="#f57c00", fg="white", font=("Arial", 9), pady=10)
    clear_cache_btn.pack(side=tk.LEFT, padx=10)

    open_report_btn = tk.Button(ts_btn_row2, text="Open AAVSO Report", command=on_open_report,
                                bg="#43a047", fg="white", font=("Arial", 9, "bold"), pady=10)
    open_report_btn.pack(side=tk.LEFT, padx=10)

    reset_ens_btn = tk.Button(ts_btn_row2, text="Reset Ensemble", command=on_reset_ensemble,
                              bg="#757575", fg="white", font=("Arial", 9), pady=10)
    reset_ens_btn.pack(side=tk.LEFT, padx=10)

    ts_info_btn = tk.Button(ts_btn_row2, text=f"{get_icon('❓', '?')} What does this do?", command=show_lightcurve_info,
                            bg="#f0f2f5", fg="#00796b", font=("Arial", 9, "bold"), pady=10, padx=15)
    ts_info_btn.pack(side=tk.LEFT, padx=10)

    # 5. Graph
    lf_ts_plot.pack(fill="both", expand=True, padx=10, pady=5)

    # 6. Numerical Data Table
    lf_ts_table = tk.LabelFrame(ts_container, text="Numerical Results")
    lf_ts_table.pack(fill="both", expand=True, padx=10, pady=5)
    
    ts_tree = ttk.Treeview(lf_ts_table, columns=("JD", "HJD", "Mag", "Err", "Check Mag", "SNR", "FWHM", "Flag"), show='headings', height=10)
    ts_tree.heading("JD", text="JD")
    ts_tree.heading("HJD", text="HJD")
    ts_tree.heading("Mag", text="Mag")
    ts_tree.heading("Err", text="Err")
    ts_tree.heading("Check Mag", text="Check Mag")
    ts_tree.heading("SNR", text="SNR")
    ts_tree.heading("FWHM", text="FWHM")
    ts_tree.heading("Flag", text="Flag")
    
    ts_tree.column("JD", width=100, anchor=tk.CENTER)
    ts_tree.column("HJD", width=100, anchor=tk.CENTER)
    ts_tree.column("Mag", width=70, anchor=tk.CENTER)
    ts_tree.column("Err", width=70, anchor=tk.CENTER)
    ts_tree.column("Check Mag", width=80, anchor=tk.CENTER)
    ts_tree.column("SNR", width=60, anchor=tk.CENTER)
    ts_tree.column("FWHM", width=60, anchor=tk.CENTER)
    ts_tree.column("Flag", width=70, anchor=tk.CENTER)
    
    ts_vsb = ttk.Scrollbar(lf_ts_table, orient="vertical", command=ts_tree.yview)
    ts_tree.configure(yscrollcommand=ts_vsb.set)
    
    ts_tree.pack(side=tk.LEFT, fill="both", expand=True)
    ts_vsb.pack(side=tk.RIGHT, fill="y")

    # =========================================================================
    # --- TAB 5: DATA MINING ---
    # =========================================================================
    dm_container = tk.Frame(tab_datamining, bg="white")
    dm_container.pack(fill="both", expand=True, padx=20, pady=10)

    tk.Label(dm_container, text=f"{get_icon('🚀', '')}  Data Mining & Variable Star Discovery", font=("Segoe UI", 16, "bold"), fg="#1a3a5f", bg="white").pack(anchor="w", pady=(0,10))
    
    # 1. Config Frame
    lf_dm_config = tk.LabelFrame(dm_container, text="Data Mining Configuration", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_dm_config.pack(fill="x", pady=5)
    
    add_entry(lf_dm_config, "Detection Sigma Threshold:", "dm_detect_sigma", 5.0, 0)
    add_entry(lf_dm_config, "Saturation Limit (ADU):", "dm_saturation", 55000, 1, vtype=int)
    add_entry(lf_dm_config, "Edge Boundary Buffer (px):", "dm_edge_buffer", 50, 2, vtype=int)
    add_entry(lf_dm_config, "Min Valid Frames Fraction (0.0 - 1.0):", "dm_min_valid_pct", 0.7, 3)
    add_check(lf_dm_config, "Fast Test Mode (limit ensemble to top 100 stars)", "dm_fast_test", True, 4)
    
    dm_status_var = tk.StringVar(value="Ready to discover.")
    tk.Label(dm_container, textvariable=dm_status_var, fg="#d32f2f", bg="white", font=("Arial", 10, "bold")).pack(pady=5)
    
    # 2. Execution Button & Progress
    dm_progress = ttk.Progressbar(dm_container, orient="horizontal", length=400, mode="determinate")
    
    def on_run_datamining():
        from photometry.data_mining import detect_all_stars, track_and_measure, calibrate_and_analyze
        import threading
        import numpy as np
        import os
        
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showerror("No Files", "Please select at least one FITS file in the File Manager.")
            return
            
        fits_files = [loaded_files[int(iid)]['path'] for iid in selected_iids]
        ref_file = fits_files[0]
        
        detect_sig = vars_dict["dm_detect_sigma"][0].get()
        sat_limit = vars_dict["dm_saturation"][0].get()
        edge_buf = vars_dict["dm_edge_buffer"][0].get()
        min_pct = vars_dict["dm_min_valid_pct"][0].get()
        is_fast = vars_dict["dm_fast_test"][0].get()
        
        # Aperture settings are shared with general config
        ap_rad = vars_dict["aperture_radius"][0].get()
        ann_in = vars_dict["annulus_inner"][0].get()
        ann_out = vars_dict["annulus_outer"][0].get()
        
        cancel_event = threading.Event()
        
        dm_progress.pack(pady=5)
        dm_progress["value"] = 0
        
        btn_cancel = tk.Button(dm_container, text="Cancel", command=cancel_event.set, bg="#d32f2f", fg="white", font=("Arial", 9, "bold"))
        btn_cancel.pack(pady=5)
        
        run_dm_btn.config(state=tk.DISABLED)
        
        def dm_thread():
            try:
                dm_status_var.set("Phase 1/3: Detecting stars in reference frame...")
                master_cat = detect_all_stars(ref_file, detect_sigma=detect_sig, saturation_limit=sat_limit, edge_buffer=edge_buf)
                
                if cancel_event.is_set(): return
                
                if is_fast and len(master_cat) > 100:
                    master_cat = master_cat[:100]
                    
                if not master_cat:
                    dm_status_var.set("Error: No stars detected in reference frame.")
                    return
                    
                dm_status_var.set(f"Phase 2/3: Tracking {len(master_cat)} stars across {len(fits_files)} frames...")
                
                def update_prog(val):
                    root.after(0, lambda: dm_progress.configure(value=val))
                    
                track_res, warnings_lst = track_and_measure(
                    fits_files, master_cat, 
                    aperture_radius=ap_rad, annulus_inner=ann_in, annulus_outer=ann_out,
                    saturation_limit=sat_limit, edge_buffer=edge_buf,
                    cancel_event=cancel_event, progress_callback=update_prog
                )
                
                if cancel_event.is_set(): return
                if warnings_lst:
                    warn_msg = "\n".join(warnings_lst)
                    root.after(0, lambda: messagebox.showwarning("Data Mining Warnings", warn_msg))
                    
                dm_status_var.set("Phase 3/3: Calibrating and Analyzing Statistics...")
                
                out_csv = os.path.abspath(os.path.join("photometry_output", "data_mining_suspects.csv"))
                final_stats, exp_func = calibrate_and_analyze(track_res, master_cat, out_csv, min_valid_fraction=min_pct)
                
                if not final_stats:
                    dm_status_var.set("Analysis complete, but no valid stars survived filters.")
                    return
                    
                # Update UI
                def update_ui():
                    dm_tree.final_stats = final_stats
                    
                    # Reset sorting column text highlights (removing / adding arrows)
                    col_headers = {
                        "ID": "ID",
                        "RA": "RA (deg)",
                        "Dec": "Dec (deg)",
                        "Mean Mag": "Mean Mag",
                        "RMS": "RMS",
                        "Excess": "Excess RMS",
                        "Valid": "Valid",
                        "Flag": "Flag"
                    }
                    for c_id, base_text in col_headers.items():
                        if c_id == dm_tree.sort_col:
                            arrow = " ▼" if dm_tree.sort_desc else " ▲"
                            dm_tree.heading(c_id, text=base_text + arrow)
                        else:
                            dm_tree.heading(c_id, text=base_text)
                            
                    populate_dm_tree()
                    
                    dm_fig.clear()
                    ax = dm_fig.add_subplot(111)
                    dm_tree.ax = ax  # Store active axis reference
                    dm_tree.highlight_artists = []  # Clear highlights
                    
                    mags = np.array([s['mean_mag'] for s in final_stats])
                    rms = np.array([s['rms'] for s in final_stats])
                    sus = np.array([s.get('is_suspect', False) for s in final_stats])
                    
                    stable_stars_list = [s for s in final_stats if not s.get('is_suspect')]
                    suspect_stars_list = [s for s in final_stats if s.get('is_suspect')]
                    
                    mags_stable = mags[~sus]
                    rms_stable = rms[~sus]
                    mags_sus = mags[sus]
                    rms_sus = rms[sus]
                    
                    # Task 4: Added picker=5 to enable clicks on points
                    sc_stable = ax.scatter(mags_stable, rms_stable, c='blue', alpha=0.5, s=10, label='Stable', picker=5)
                    sc_suspect = ax.scatter(mags_sus, rms_sus, c='red', alpha=0.9, s=40, label='Suspect', marker='*', picker=5)
                    
                    # Task 4: Store lists on collections to map indexes back to stars
                    sc_stable.star_list = stable_stars_list
                    sc_suspect.star_list = suspect_stars_list
                    
                    if exp_func:
                        x_curve = np.linspace(np.min(mags), np.max(mags), 100)
                        y_curve = exp_func(x_curve)
                        ax.plot(x_curve, y_curve, c='green', lw=2, label='Expected Noise Floor')
                        
                    ax.set_xlabel("Mean Instrumental Mag")
                    ax.set_ylabel("RMS (std dev)")
                    ax.set_title("Stellar Variability Diagram")
                    ax.legend()
                    if not ax.xaxis_inverted():
                        ax.invert_xaxis()
                    dm_canvas.draw()
                    
                    dm_status_var.set(f"Complete! Found {np.sum(sus)} suspect(s). Saved to {out_csv}")
                
                root.after(0, update_ui)
                
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
                dm_status_var.set("Error occurred.")
            finally:
                root.after(0, lambda: dm_progress.pack_forget())
                root.after(0, lambda: btn_cancel.pack_forget())
                root.after(0, lambda: run_dm_btn.config(state=tk.NORMAL))
                
        threading.Thread(target=dm_thread, daemon=True).start()

    dm_btn_frame = tk.Frame(dm_container, bg="white")
    dm_btn_frame.pack(pady=5)
    
    run_dm_btn = tk.Button(dm_btn_frame, text="Run Data Mining", command=on_run_datamining,
                           bg="#1a3a5f", fg="white", font=("Arial", 11, "bold"), pady=10, width=25)
    run_dm_btn.pack(side=tk.LEFT, padx=10)
    
    def on_send_suspect():
        selected = dm_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a star from the table first.")
            return
        vals = dm_tree.item(selected[0], 'values')
        ra_deg, dec_deg = float(vals[1]), float(vals[2])
        
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg))
        ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)
        dec_dms = coord.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)
        
        vars_dict["ts_target_ra"][0].set(ra_hms)
        vars_dict["ts_target_dec"][0].set(dec_dms)
        vars_dict["ts_target_mode"][0].set("manual")
        ts_target_name_var.set(f"Suspect_{vals[0]}")
        messagebox.showinfo("Sent", f"Target {vals[0]} coordinates sent to Light Curves tab!")
        switch_tab("ts")

    send_btn = tk.Button(dm_btn_frame, text="Send Suspect to Light Curves", command=on_send_suspect,
                           bg="#f57c00", fg="white", font=("Arial", 10, "bold"), pady=10, padx=15)
    send_btn.pack(side=tk.LEFT, padx=10)

    # Task 5: Added Preview Light Curve button
    def on_preview_click():
        selected = dm_tree.selection()
        if not selected:
            messagebox.showinfo("Select Star", "Please select a star from the table to preview its light curve.")
            return
        vals = dm_tree.item(selected[0], 'values')
        show_mini_lightcurve(vals[0])

    preview_btn = tk.Button(dm_btn_frame, text="Preview Light Curve", command=on_preview_click,
                             bg="#2e7d32", fg="white", font=("Arial", 10, "bold"), pady=10, padx=15)
    preview_btn.pack(side=tk.LEFT, padx=10)

    def show_dm_help():
        pop = tk.Toplevel(root)
        pop.title("About Data Mining (Variable Star Discovery)")
        pop.geometry("650x550")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 650) // 2
        py = ry + (rh - 550) // 2
        pop.geometry(f"+{px}+{py}")
        
        header = tk.Frame(pop, bg="#1a3a5f", pady=15)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text=f"{get_icon('🚀', '')}  Data Mining & Discovery".strip(), font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT DATA MINING DOES:\n"
            "----------------------------\n"
            "This tool automatically scans your entire FITS sequence to discover unknown variable stars! It operates without requiring any prior knowledge of targets or reference stars.\n\n"
            "HOW IT WORKS:\n"
            "1. Star Detection:\n"
            "   Identifies all point sources in your reference frame using DAOStarFinder.\n\n"
            "2. Bulk Photometry:\n"
            "   Tracks every star across every loaded frame and measures its instrumental magnitude.\n\n"
            "3. Global Calibration:\n"
            "   Identifies the brightest, most stable stars across your sequence and builds an automated 'Global Ensemble'. It uses this ensemble to remove atmospheric variations (clouds, airmass) from all stars.\n\n"
            "4. Statistical Outlier Detection (Flagging):\n"
            "   It calculates the expected photometric 'noise floor' for your specific camera system based on Poisson statistics. Any star that fluctuates significantly above this expected noise curve (and > 0.03 mag absolute) is flagged as a Suspect.\n\n"
            "----------------------------\n"
            "RESULTS:\n"
            "- Variability Diagram: Stable stars are plotted as small blue dots. The green line represents the expected noise floor. Discovered Suspects are highlighted as large red stars.\n"
            "- Send to Light Curves: Click on a suspect in the table and press the 'Send' button to instantly transfer its coordinates to the Light Curves tab for full analysis!\n"
        )
        info_box.insert(tk.END, details)
        info_box.config(state=tk.DISABLED)
        
        btn_close = tk.Button(content_frame, text="Got it!", command=pop.destroy, 
                              bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"), 
                              relief="flat", width=15, pady=8)
        btn_close.pack(pady=(15, 0))
    
    tk.Button(dm_btn_frame, text="❓ What does this do?", command=show_dm_help, bg="#f0f2f5", fg="#00796b", font=("Arial", 9, "bold"), pady=10, padx=15).pack(side=tk.LEFT, padx=10)

    # 3. Plot Frame
    lf_dm_plot = tk.LabelFrame(dm_container, text="Variability Diagram", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_dm_plot.pack(fill="both", expand=True, padx=10, pady=5)
    
    dm_fig, dm_ax = plt.subplots(figsize=(8, 4))
    dm_canvas = FigureCanvasTkAgg(dm_fig, master=lf_dm_plot)
    dm_canvas.get_tk_widget().pack(fill="both", expand=True)
    dm_toolbar = NavigationToolbar2Tk(dm_canvas, lf_dm_plot)
    dm_toolbar.update()
    
    # 4. Table Frame
    lf_dm_table = tk.LabelFrame(dm_container, text="Suspects Table", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_dm_table.pack(fill="both", expand=True, padx=10, pady=5)
    
    # Filter frame above Treeview
    dm_filter_frame = tk.Frame(lf_dm_table, bg="white")
    dm_filter_frame.pack(fill="x", side=tk.TOP, padx=5, pady=5)
    
    show_non_suspects_var = tk.BooleanVar(value=False)
    
    # Task 2: Added "Valid" column to columns tuple
    dm_tree = ttk.Treeview(lf_dm_table, columns=("ID", "RA", "Dec", "Mean Mag", "RMS", "Excess", "Valid", "Flag"), show='headings', height=8)
    
    def populate_dm_tree():
        for item in dm_tree.get_children():
            dm_tree.delete(item)
            
        show_all = show_non_suspects_var.get()
        
        stats_to_show = []
        if hasattr(dm_tree, 'final_stats') and dm_tree.final_stats:
            for s in dm_tree.final_stats:
                if show_all or s.get('is_suspect'):
                    stats_to_show.append(s)
                    
        # Apply sorting
        if hasattr(dm_tree, 'sort_col') and dm_tree.sort_col:
            col = dm_tree.sort_col
            rev = dm_tree.sort_desc
            
            def try_numeric(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return str(val).lower()
                    
            if col == "ID":
                stats_to_show.sort(key=lambda x: try_numeric(x['id']), reverse=rev)
            elif col == "RA":
                stats_to_show.sort(key=lambda x: x['ra_deg'], reverse=rev)
            elif col == "Dec":
                stats_to_show.sort(key=lambda x: x['dec_deg'], reverse=rev)
            elif col == "Mean Mag":
                stats_to_show.sort(key=lambda x: x['mean_mag'], reverse=rev)
            elif col == "RMS":
                stats_to_show.sort(key=lambda x: x['rms'], reverse=rev)
            elif col == "Excess":
                stats_to_show.sort(key=lambda x: x['excess_rms'], reverse=rev)
            elif col == "Valid":
                # Task 2 Sorting on Valid column
                stats_to_show.sort(key=lambda x: x.get('n_valid', 0), reverse=rev)
            elif col == "Flag":
                stats_to_show.sort(key=lambda x: ("SUSPECT" if x.get('is_suspect') else ""), reverse=rev)
                
        for s in stats_to_show:
            flag = "SUSPECT" if s.get('is_suspect') else ""
            # Task 2: Populate Valid count
            dm_tree.insert("", tk.END, values=(
                s['id'], f"{s['ra_deg']:.5f}", f"{s['dec_deg']:.5f}",
                f"{s['mean_mag']:.3f}", f"{s['rms']:.4f}", f"{s['excess_rms']:.4f}",
                f"{s.get('n_valid', 0)}", flag
            ))
            
    def sort_column(col):
        if dm_tree.sort_col == col:
            dm_tree.sort_desc = not dm_tree.sort_desc
        else:
            dm_tree.sort_col = col
            dm_tree.sort_desc = False
            
        col_headers = {
            "ID": "ID",
            "RA": "RA (deg)",
            "Dec": "Dec (deg)",
            "Mean Mag": "Mean Mag",
            "RMS": "RMS",
            "Excess": "Excess RMS",
            "Valid": "Valid",
            "Flag": "Flag"
        }
        for c_id, base_text in col_headers.items():
            if c_id == col:
                arrow = " ▼" if dm_tree.sort_desc else " ▲"
                dm_tree.heading(c_id, text=base_text + arrow)
            else:
                dm_tree.heading(c_id, text=base_text)
                
        populate_dm_tree()
        
    # Task 3: Draw a bright yellow halo and label over selected star in plot
    def highlight_star_in_plot(star):
        if not hasattr(dm_tree, 'ax') or not dm_tree.ax:
            return
        if hasattr(dm_tree, 'highlight_artists') and dm_tree.highlight_artists:
            for artist in dm_tree.highlight_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            dm_tree.highlight_artists = []
            
        x = star['mean_mag']
        y = star['rms']
        
        # Yellow circle around point
        halo = dm_tree.ax.scatter([x], [y], s=120, facecolors='none', edgecolors='#ffd600', linewidths=2.0, zorder=10)
        
        # Text label above point
        ylim = dm_tree.ax.get_ylim()
        y_offset = 0.015 * (ylim[1] - ylim[0])
        label = dm_tree.ax.text(x, y + y_offset, f"{star['id']}", 
                                color='black', fontsize=9, fontweight='bold',
                                bbox=dict(facecolor='yellow', alpha=0.8, boxstyle='round,pad=0.2'),
                                zorder=11, ha='center')
                           
        dm_tree.highlight_artists = [halo, label]
        dm_canvas.draw_idle()
        
    # Task 5: Non-blocking Toplevel popup window for Magnitude vs Frame preview
    def show_mini_lightcurve(star_id):
        star = None
        for s in dm_tree.final_stats:
            if str(s['id']) == str(star_id):
                star = s
                break
        if not star or 'corrected_mags' not in star:
            messagebox.showwarning("No Data", f"No magnitude data available for star {star_id}.")
            return
            
        pop = tk.Toplevel(root)
        pop.title(f"Light Curve Preview: {star_id}")
        pop.geometry("600x400")
        pop.configure(bg="white")
        pop.transient(root)
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 600) // 2
        py = ry + (rh - 400) // 2
        pop.geometry(f"+{px}+{py}")
        
        header = tk.Frame(pop, bg="#1a3a5f", pady=10)
        header.pack(fill="x", side=tk.TOP)
        tk.Label(header, text=f"Raw Light Curve for {star_id}", font=("Segoe UI", 12, "bold"), fg="white", bg="#1a3a5f").pack()
        
        plot_frame = tk.Frame(pop, bg="white")
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        mags = np.array(star['corrected_mags'])
        frames = np.arange(len(mags))
        
        valid = np.isfinite(mags)
        if np.any(valid):
            ax.plot(frames[valid], mags[valid], marker='o', color='#1a3a5f', linestyle='-', markersize=4, alpha=0.8, label="Instrumental Mag")
            ax.invert_yaxis()
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Mag (Calibrated)")
            ax.set_title(f"Mean Mag: {star['mean_mag']:.3f} | RMS: {star['rms']:.4f} | Valid Frames: {star['n_valid']}", fontdict={'fontsize': 10})
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No valid magnitude measurements.", ha='center', va='center')
            
        canvas.draw()
        
    # Task 3: Selection callback for Treeview click
    def on_dm_tree_select(event):
        selected = dm_tree.selection()
        if not selected:
            return
        vals = dm_tree.item(selected[0], 'values')
        star_id = vals[0]
        
        for s in dm_tree.final_stats:
            if str(s['id']) == str(star_id):
                highlight_star_in_plot(s)
                break
                
    # Task 5: Double-click preview binding
    def on_dm_double_click(event):
        selected = dm_tree.selection()
        if not selected:
            return
        vals = dm_tree.item(selected[0], 'values')
        show_mini_lightcurve(vals[0])
        
    # Task 4: Pick event callback for matplotlib canvas click
    def on_canvas_pick(event):
        col = event.artist
        ind = event.ind[0]
        if hasattr(col, 'star_list'):
            star = col.star_list[ind]
            star_id = star['id']
            
            found = False
            for item in dm_tree.get_children():
                if str(dm_tree.item(item, 'values')[0]) == str(star_id):
                    dm_tree.selection_set(item)
                    dm_tree.see(item)
                    found = True
                    break
                    
            highlight_star_in_plot(star)
            
    dm_tree.final_stats = []
    dm_tree.sort_col = "Excess"
    dm_tree.sort_desc = True
    
    chk_show_non = tk.Checkbutton(
        dm_filter_frame, 
        text="Show Non-Suspects (stable stars)", 
        variable=show_non_suspects_var, 
        command=populate_dm_tree,
        bg="white", 
        activebackground="white",
        font=("Segoe UI", 9)
    )
    chk_show_non.pack(side=tk.LEFT, padx=10)

    # Task 2: Setup Table Headings
    dm_tree.heading("ID", text="ID", command=lambda: sort_column("ID"))
    dm_tree.heading("RA", text="RA (deg)", command=lambda: sort_column("RA"))
    dm_tree.heading("Dec", text="Dec (deg)", command=lambda: sort_column("Dec"))
    dm_tree.heading("Mean Mag", text="Mean Mag", command=lambda: sort_column("Mean Mag"))
    dm_tree.heading("RMS", text="RMS", command=lambda: sort_column("RMS"))
    dm_tree.heading("Excess", text="Excess RMS ▼", command=lambda: sort_column("Excess"))
    dm_tree.heading("Valid", text="Valid", command=lambda: sort_column("Valid"))
    dm_tree.heading("Flag", text="Flag", command=lambda: sort_column("Flag"))
    
    # Task 2: Setup Column Widths
    dm_tree.column("ID", width=70, anchor=tk.CENTER)
    dm_tree.column("RA", width=80, anchor=tk.CENTER)
    dm_tree.column("Dec", width=80, anchor=tk.CENTER)
    dm_tree.column("Mean Mag", width=70, anchor=tk.CENTER)
    dm_tree.column("RMS", width=70, anchor=tk.CENTER)
    dm_tree.column("Excess", width=70, anchor=tk.CENTER)
    dm_tree.column("Valid", width=60, anchor=tk.CENTER)
    dm_tree.column("Flag", width=70, anchor=tk.CENTER)
    
    # Bindings
    dm_tree.bind("<<TreeviewSelect>>", on_dm_tree_select)
    dm_tree.bind("<Double-1>", on_dm_double_click)
    dm_canvas.mpl_connect('pick_event', on_canvas_pick)
    
    dm_vsb = ttk.Scrollbar(lf_dm_table, orient="vertical", command=dm_tree.yview)
    dm_tree.configure(yscrollcommand=dm_vsb.set)
    dm_tree.pack(side=tk.LEFT, fill="both", expand=True)
    dm_vsb.pack(side=tk.RIGHT, fill="y")

    # --- END TAB 6 ---

    # Camera Settings (from old TAB 2)
    lf_ccd = tk.LabelFrame(sub_phot, text="CCD Settings (Error Analysis)")
    lf_ccd.pack(fill="x", padx=10, pady=10)
    add_entry(lf_ccd, "Gain (e-/ADU):", "ccd_gain", 1.27, 0)
    add_entry(lf_ccd, "Read Noise (e-):", "ccd_read_noise", 3.3, 1)
    add_entry(lf_ccd, "Dark Current (e-/s/px):", "ccd_dark_current", 0.0007, 2)
    add_entry(lf_ccd, "Saturation Limit (ADU):", "saturation_limit", 63000, 3, vtype=int)

    # Detection (from old TAB 2)
    lf_det = tk.LabelFrame(sub_phot, text="Detection (DAOStarFinder)")
    lf_det.pack(fill="x", padx=10, pady=10)
    add_entry(lf_det, "Detection Sigma:", "detect_sigma", 5.0, 0)
    add_entry(lf_det, "Sharpness Low:", "dao_sharplo", 0.2, 1, col_offset=0)
    add_entry(lf_det, "Sharpness High:", "dao_sharphi", 1.0, 1, col_offset=1)
    add_entry(lf_det, "Roundness Low:", "dao_roundlo", -1.2, 2, col_offset=0)
    add_entry(lf_det, "Roundness High:", "dao_roundhi", 1.2, 2, col_offset=1)


    # Aperture Photometry (from old TAB 3)
    lf_ap = tk.LabelFrame(sub_phot, text="Aperture Photometry")
    lf_ap.pack(fill="x", padx=10, pady=10)
    add_entry(lf_ap, "PSF Box Size (px):", "box_size", 15, 0, vtype=int)
    
    # Flexible Aperture Toggle
    add_check(lf_ap, "Use Flexible Aperture (FWHM-based)", "use_flexible_aperture", False, 1)
    add_entry(lf_ap, "Aperture FWHM Factor:", "aperture_fwhm_factor", 2.0, 2)
    add_entry(lf_ap, "Annulus Inner Gap (px):", "annulus_inner_gap", 2.0, 3)
    add_entry(lf_ap, "Annulus Width (px):", "annulus_width", 5.0, 4)
    
    tk.Label(lf_ap, text="--- OR Fixed Values ---", fg="#555", font=("Arial", 8, "italic")).grid(row=5, column=0, columnspan=2, pady=(10, 0))
    add_entry(lf_ap, "Fixed Aperture Radius (px):", "aperture_radius", 5.0, 6)
    add_entry(lf_ap, "Fixed Annulus Inner (px):", "annulus_inner", 7.0, 7)
    add_entry(lf_ap, "Fixed Annulus Outer (px):", "annulus_outer", 13.0, 8)

    # Zero Point Calibration (from old TAB 3)
    lf_cal = tk.LabelFrame(sub_phot, text="Zero Point Calibration")
    lf_cal.pack(fill="x", padx=10, pady=10)
    add_entry(lf_cal, "Match Tolerance (arcsec):", "match_tolerance_arcsec", 8.0, 0)
    add_entry(lf_cal, "Default Zero Point (V):", "default_zp_v", 24.0, 1, col_offset=0)
    add_entry(lf_cal, "Default Zero Point (B):", "default_zp_b", 24.0, 1, col_offset=1)
    add_entry(lf_cal, "Min SNR for Calib:", "calib_snr_threshold", 10.0, 2)
    add_entry(lf_cal, "Catalog Search Radius (arcmin):", "catalog_search_radius", 15.0, 3)
    # Checkboxes moved to Pipeline Configuration section.

    # Global Extinction (Shared)
    lf_ext = tk.LabelFrame(sub_astro, text="Atmospheric Extinction (Global)")
    lf_ext.pack(fill="x", padx=10, pady=10)
    add_entry(lf_ext, "k_V (Visual):", "extinction_kv", 0.20, 0)
    add_entry(lf_ext, "k_B (Blue):", "extinction_kb", 0.35, 1)

    # Observer Information (Global)
    lf_obs = tk.LabelFrame(sub_ops, text="Observer Information")
    lf_obs.pack(fill="x", padx=10, pady=10)
    add_entry(lf_obs, "AAVSO Observer Code (4 chars):", "aavso_obs_code", "XXXX", 0, vtype=str)
    add_entry(lf_obs, "Observer Name (Report Header):", "observer_name", "Calibra User", 1, vtype=str)


    # Output Toggles (from old TAB 4)
    lf_out = tk.LabelFrame(sub_ops, text="Console & Plot Toggles")
    lf_out.pack(fill="x", padx=10, pady=10)
    add_check(lf_out, "Print Detailed Calibration to Console", "print_detailed_calibration", False, 0)
    add_check(lf_out, "Print Massive Aperture Photometry Table", "print_star_detection_table", False, 1)
    add_check(lf_out, "Print PSF Quality & Fitting Details (Console)", "print_psf_fitting", False, 2)
    add_check(lf_out, "Display Matplotlib Plots (Blocking)", "display_plots", False, 3)
    add_entry(lf_out, "Max PSF plots to show/save per file:", "max_plots_to_show_per_file", 3, 4, vtype=int)


    # --- TAB 5: About ---
    tab_about_outer = ttk.Frame(content_container)
    tab_about_outer.grid(row=0, column=0, sticky="nsew")
    
    about_scroll = ScrollableFrame(tab_about_outer)
    about_scroll.pack(fill="both", expand=True)
    tab_about = about_scroll.scrollable_frame

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
        tk.Label(about_container, text="[ CALIBRA ]", font=("Arial", 24, "bold"), fg=primary_blue, bg="white").pack(pady=(0, 20))
    
    tk.Label(about_container, text="Calibra: An automated photometric analysis & calibration toolkit", font=("Arial", 16, "bold"), fg=primary_blue, bg="white").pack(fill="x")
    
    info_frame = tk.Frame(about_container, bg="white")
    info_frame.pack(fill="x", pady=10)
    tk.Label(info_frame, text="Version: 4.0 \tLatest Update: 2026-05-17", font=("Arial", 10), bg="white").pack(fill="x")
    
    tk.Label(about_container, text="Description:", font=("Arial", 11, "bold"), fg=primary_blue, bg="white").pack(fill="x", pady=(10, 5))
    desc_text = (
        "Calibra uses star detection, sub-pixel PSF fitting, aperture photometry, \n"
        "zero-point calibration, and offers determination of color transformations between filters. \n"
        "For the analysis, Calibra can use the online catalogues ATLAS-RefCat2, APASS DR9, \n"
        "Landolt Standards Catalogue and Gaia DR3, or a user-provided catalogue.\n"
        "\n"
        "Finally, Calibra can produce light curves for variable stars from a series of images and create AAVSO-formatted text files.\n"
    )
    tk.Label(about_container, text=desc_text, justify=tk.LEFT, font=("Arial", 10), bg="white").pack(fill="x")

    # --- TAB 5: Help ---
    tab_help_outer = ttk.Frame(content_container)
    tab_help_outer.grid(row=0, column=0, sticky="nsew")
    
    help_scroll = ScrollableFrame(tab_help_outer)
    help_scroll.pack(fill="both", expand=True)
    tab_help = help_scroll.scrollable_frame
    
    help_frame = tk.Frame(tab_help, padx=20, pady=20)
    help_frame.pack(fill="both", expand=True)
    
    tk.Label(help_frame, text="Documentation & Support", font=("Arial", 12, "bold")).pack(pady=(0,10), ipady=2)
    tk.Label(help_frame, text="For detailed instructions on how to use Calibra, please refer to the documentation files in the program directory:", justify=tk.LEFT).pack(pady=5)
    
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
    tk.Label(help_frame, text=help_info, justify=tk.LEFT, font=("Arial", 9, "italic")).pack(pady=10)

    # Developer Info
    dev_info = (
        "\nDevelopers & Contact:\n"
        "Developed by Stephan Pomp & Google DeepMind Antigravity\n"
        "Contact: stephan.pomp@gmail.com"
    )
    tk.Label(help_frame, text=dev_info, justify=tk.LEFT, font=("Arial", 9), fg="#555").pack(side=tk.BOTTOM, anchor="w")

    # Create Sidebar buttons
    create_sidebar_button(f"{get_icon('📂', '')}  File Manager".strip(), "files")
    create_sidebar_button(f"{get_icon('⚙', '')}  Pre-processing".strip(), "pre")
    create_sidebar_button(f"{get_icon('🔍', '')}  Analysis & Calib".strip(), "analysis")
    create_sidebar_button(f"{get_icon('📈', '')}  Light Curves".strip(), "ts")
    create_sidebar_button(f"{get_icon('🚀', '')}  Data Mining".strip(), "datamining")
    create_sidebar_button(f"{get_icon('🔧', '')}  Settings".strip(), "settings")
    create_sidebar_button(f"{get_icon('ℹ', '')}  About Calibra".strip(), "about")
    create_sidebar_button(f"{get_icon('❓', '')}  User Help".strip(), "help")
    
    # Select first tab by default
    switch_tab("files")

    # --- OUTPUT CONSOLE (Separate Window) ---
    console_win = tk.Toplevel(root)
    console_win.title("Calibra: Process Console")
    console_win.geometry("800x500")
    console_win.configure(bg="#f0f2f5")
    
    console_frame = tk.LabelFrame(console_win, text="Log Output", bg="#f0f2f5", font=("Arial", 10, "bold"))
    console_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    console = scrolledtext.ScrolledText(console_frame, font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
    console.pack(fill="both", expand=True, padx=5, pady=5)
    add_copy_context_menu(console)
    
    # Redirect stdout and stderr
    sys.stdout = StdoutRedirector(console)
    sys.stderr = StdoutRedirector(console)

    # Ensure closing main window closes everything and saves session
    def on_closing():
        try:
            save_session()
        except:
            pass
        root.destroy()
        sys.exit(0)

    footer_frame = tk.Frame(root, bg="#f0f2f5")
    footer_frame.pack(side=tk.BOTTOM, fill="x", pady=2)
    
    exit_btn = tk.Button(footer_frame, text="Exit Calibra", command=on_closing, width=15, 
                           font=("Arial", 9), relief="flat", bg="#f44336", fg="white")
    exit_btn.pack(side=tk.LEFT, padx=10, pady=5)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    console_win.protocol("WM_DELETE_WINDOW", lambda: None) 

    # Auto-load previous session
    load_session()

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
