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
from photometry.filters import classify_filter
from photometry.names import from_name as _resolve_name   # v4.30: local GCVS list, then SIMBAD

# Platform-agnostic safe icons to prevent broken emoji boxes on Linux/Ubuntu
IS_LINUX = sys.platform.startswith("linux")

def get_icon(emoji_win, fallback_linux):
    return fallback_linux if IS_LINUX else emoji_win

class StdoutRedirector:
    """
    Redirects print() output into Tk text widgets (and optionally a log file).

    v4.15: tkinter is not thread-safe. Text written from worker threads is now
    put on a queue and inserted into the widgets by the GUI thread
    (see start_polling), instead of touching the widgets directly.
    """
    def __init__(self, text_widget):
        import queue
        self.text_widgets = [text_widget]
        self.log_file = None
        self._queue = queue.Queue()
        self._main_thread = threading.main_thread()

    def start_polling(self, root, interval_ms=50):
        def _drain():
            try:
                while True:
                    widgets, text = self._queue.get_nowait()
                    self._insert(text, widgets)
            except Exception:
                pass
            root.after(interval_ms, _drain)
        root.after(interval_ms, _drain)

    def _insert(self, string, widgets=None):
        for w in list(widgets if widgets is not None else self.text_widgets):
            try:
                w.insert(tk.END, string)
                w.see(tk.END)
            except Exception:
                pass

    def add_widget(self, text_widget):
        if text_widget not in self.text_widgets:
            self.text_widgets.append(text_widget)

    def remove_widget(self, text_widget):
        if text_widget in self.text_widgets:
            self.text_widgets.remove(text_widget)

    def set_log_file(self, file_path):
        self.log_file = file_path

    def write(self, string):
        if threading.current_thread() is self._main_thread:
            self._insert(string)
        else:
            # remember the target widgets now (the pipeline terminal may be
            # detached before the queue is drained)
            self._queue.put((tuple(self.text_widgets), string))
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(string)
            except Exception:
                pass

    def flush(self):
        pass

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="#f0f2f5")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        # v4.39: horizontal scrollbar, shown only when the content is wider than the window
        self.xscrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self._x_shown = False
        self.scrollable_frame = ttk.Frame(self.canvas)

        def update_scrollregion(e):
            self._update_width()
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
            
        def _on_shift_wheel(event):
            if self._x_shown:
                self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")

        def _enter(_):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.canvas.bind_all("<Shift-MouseWheel>", _on_shift_wheel)

        def _leave(_):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Shift-MouseWheel>")
        self.canvas.bind("<Enter>", _enter)
        self.canvas.bind("<Leave>", _leave)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=self.xscrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def _update_width(self, canvas_width=None):
        """Inner frame as wide as the window, or wider (then with a horizontal scrollbar)."""
        cw = canvas_width or self.canvas.winfo_width()
        rw = self.scrollable_frame.winfo_reqwidth()
        self.canvas.itemconfig(self.canvas_window, width=max(cw, rw))
        need = rw > cw + 2
        if need and not self._x_shown:
            self.xscrollbar.pack(side="bottom", fill="x", before=self.canvas)
            self._x_shown = True
        elif not need and self._x_shown:
            self.xscrollbar.pack_forget()
            self.canvas.xview_moveto(0)
            self._x_shown = False
        
    def on_canvas_configure(self, event):
        self._update_width(event.width)
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

# v4.19: file extensions recognised as FITS (compared in lower case)
FITS_EXTENSIONS = ('.fits', '.fit', '.fts')


def read_tk_var(var, vtype):
    """
    v4.15: strict, user-friendly conversion of a Tk variable.
    Tk raises TclError (not ValueError) for empty/invalid numeric fields, and
    IntVar silently truncated '3.7' to 3. Numbers are therefore parsed from the
    raw text here; any problem is raised as ValueError with a clear message.
    """
    if vtype in (int, float):
        try:
            raw = str(var._tk.globalgetvar(var._name)).strip()
        except Exception:
            raw = str(var.get()).strip()
        if raw == "":
            raise ValueError("field is empty")
        try:
            num = float(raw)
        except ValueError:
            raise ValueError(f"'{raw}' is not a number")
        if vtype is int:
            if not float(num).is_integer():
                raise ValueError(f"'{raw}' is not a whole number")
            return int(num)
        return num
    try:
        return vtype(var.get())
    except tk.TclError as e:
        raise ValueError(str(e))


class AdvancedPlotWindow(tk.Toplevel):
    def __init__(self, parent, results, target_name, filter_name, check_name="Check Star", y_quantity=None):
        super().__init__(parent)
        self.title(f"Advanced Plot Customizer: {target_name}")
        self.geometry("1100x700")
        self.minsize(900, 600)
        
        self.results = results
        self.target_name = target_name
        self.filter_name = filter_name
        
        # Light-curve mode (calibrated / relative / raw) decides the y quantity
        self.lc_mode = results[0].get('lc_mode', 'calibrated') if results else 'calibrated'
        if self.lc_mode == 'calibrated':
            self.y_quantity = 'mag'
        else:
            self.y_quantity = y_quantity or 'flux'
        is_flux = (self.y_quantity == 'flux')
        unit = results[0].get('flux_unit', 'ADU') if results else 'ADU'
        if self.lc_mode == 'calibrated':
            default_ylabel = "Magnitude"
        elif self.lc_mode == 'raw':
            default_ylabel = f"Net flux [{unit}]" if is_flux else "Instrumental magnitude"
        else:
            default_ylabel = "Relative flux (target / \u03a3 ref)" if is_flux else "Differential instrumental magnitude"
        
        self.visibility_state = {"Target": tk.BooleanVar(value=True), "Check Star": tk.BooleanVar(value=True)}
        
        all_refs = set()
        for r in results:
            if 'ref_mags' in r and isinstance(r['ref_mags'], dict):
                all_refs.update(r['ref_mags'].keys())
                
        for ref in sorted(all_refs):
            self.visibility_state[f"Ref: {ref}"] = tk.BooleanVar(value=False)
            
        self.custom_label_vars = {}
        for key in self.visibility_state.keys():
            val = target_name if key == "Target" else check_name if key == "Check Star" else key
            self.custom_label_vars[key] = tk.StringVar(value=val)
            
        self.custom_color_vars = {}
        self.custom_symbol_vars = {}
        
        def_colors = ['orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray']
        r_idx = 0
        for key in self.visibility_state.keys():
            if key == "Target":
                c, s = "darkblue", "o"
            elif key == "Check Star":
                c, s = "green", "o"
            else:
                c, s = def_colors[r_idx % len(def_colors)], "x"
                r_idx += 1
            self.custom_color_vars[key] = tk.StringVar(value=c)
            self.custom_symbol_vars[key] = tk.StringVar(value=s)
            
        self.plot_options = {
            "title": tk.StringVar(value=f"{target_name} ({filter_name} Filter)"),
            "xlabel": tk.StringVar(value=""), 
            "ylabel": tk.StringVar(value=default_ylabel),
            "y_min": tk.StringVar(value=""),
            "y_max": tk.StringVar(value=""),
            "invert_y": tk.BooleanVar(value=not is_flux)
        }
        
        self.build_ui()
        self.redraw_plot()
        
    def build_ui(self):
        left_panel = tk.Frame(self, width=420, bg="#f8f9fa", padx=10, pady=10)
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)
        
        right_panel = tk.Frame(self, bg="white")
        right_panel.pack(side="right", fill="both", expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.update()
        
        lf_vis = tk.LabelFrame(left_panel, text="Visibility, Labels & Styles", bg="#f8f9fa", padx=5, pady=5)
        lf_vis.pack(fill="both", expand=True, pady=(0, 10))
        
        vis_sf = ScrollableFrame(lf_vis)
        vis_sf.pack(fill="both", expand=True)
        
        keys = ["Target", "Check Star"] + sorted([k for k in self.visibility_state.keys() if k.startswith("Ref:")])
        
        colors = ['darkblue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'olive', 'gray', 'black']
        symbols = ['o', 'x', 's', '^', 'v', '<', '>', 'D', '*', 'p', 'h', '+']
        
        for r_idx, k in enumerate(keys):
            f = tk.Frame(vis_sf.scrollable_frame, bg="#f8f9fa")
            f.pack(fill="x", pady=2)
            ttk.Checkbutton(f, text="", variable=self.visibility_state[k], command=self.redraw_plot).pack(side="left")
            e = ttk.Entry(f, textvariable=self.custom_label_vars[k], width=15)
            e.pack(side="left", fill="x", expand=True, padx=2)
            e.bind("<Return>", lambda e: self.redraw_plot())
            
            cb_color = ttk.Combobox(f, textvariable=self.custom_color_vars[k], values=colors, width=8, state="readonly")
            cb_color.pack(side="left", padx=1)
            cb_color.bind("<<ComboboxSelected>>", lambda e: self.redraw_plot())
            
            cb_sym = ttk.Combobox(f, textvariable=self.custom_symbol_vars[k], values=symbols, width=3, state="readonly")
            cb_sym.pack(side="left", padx=1)
            cb_sym.bind("<<ComboboxSelected>>", lambda e: self.redraw_plot())
            
        lf_axes = tk.LabelFrame(left_panel, text="Axis Settings", bg="#f8f9fa", padx=5, pady=5)
        lf_axes.pack(fill="x", pady=(0, 10))
        
        tk.Label(lf_axes, text="Y-Axis Min:", bg="#f8f9fa").grid(row=0, column=0, sticky="w", pady=2)
        e_ymin = ttk.Entry(lf_axes, textvariable=self.plot_options["y_min"], width=10)
        e_ymin.grid(row=0, column=1, sticky="w", pady=2)
        e_ymin.bind("<Return>", lambda e: self.redraw_plot())
        
        tk.Label(lf_axes, text="Y-Axis Max:", bg="#f8f9fa").grid(row=1, column=0, sticky="w", pady=2)
        e_ymax = ttk.Entry(lf_axes, textvariable=self.plot_options["y_max"], width=10)
        e_ymax.grid(row=1, column=1, sticky="w", pady=2)
        e_ymax.bind("<Return>", lambda e: self.redraw_plot())
        
        ttk.Checkbutton(lf_axes, text="Invert Y-Axis (use for magnitudes)", variable=self.plot_options["invert_y"], command=self.redraw_plot).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        lf_labels = tk.LabelFrame(left_panel, text="Labels & Title", bg="#f8f9fa", padx=5, pady=5)
        lf_labels.pack(fill="x", pady=(0, 10))
        
        tk.Label(lf_labels, text="Title:", bg="#f8f9fa").pack(anchor="w")
        e_title = ttk.Entry(lf_labels, textvariable=self.plot_options["title"], width=25)
        e_title.pack(anchor="w", pady=(0, 5))
        e_title.bind("<Return>", lambda e: self.redraw_plot())
        
        tk.Label(lf_labels, text="X Label (blank=auto):", bg="#f8f9fa").pack(anchor="w")
        e_xlabel = ttk.Entry(lf_labels, textvariable=self.plot_options["xlabel"], width=25)
        e_xlabel.pack(anchor="w", pady=(0, 5))
        e_xlabel.bind("<Return>", lambda e: self.redraw_plot())
        
        tk.Label(lf_labels, text="Y Label:", bg="#f8f9fa").pack(anchor="w")
        e_ylabel = ttk.Entry(lf_labels, textvariable=self.plot_options["ylabel"], width=25)
        e_ylabel.pack(anchor="w", pady=(0, 10))
        e_ylabel.bind("<Return>", lambda e: self.redraw_plot())
        
        btn_apply = tk.Button(left_panel, text="Apply & Redraw", command=self.redraw_plot, bg="#1a3a5f", fg="white", font=("Arial", 10, "bold"), pady=5)
        btn_apply.pack(fill="x", pady=(10, 5))
        
        btn_export = tk.Button(left_panel, text="Save High-Res Image", command=self.export_plot, bg="#2e7d32", fg="white", font=("Arial", 10), pady=5)
        btn_export.pack(fill="x")
        
    def redraw_plot(self):
        from photometry.time_series import plot_light_curve
        v_state = {k: v.get() for k, v in self.visibility_state.items()}
        opts = {k: v.get() for k, v in self.plot_options.items()}
        
        opts["custom_labels"] = {k: v.get() for k, v in self.custom_label_vars.items()}
        opts["custom_colors"] = {k: v.get() for k, v in self.custom_color_vars.items()}
        opts["custom_symbols"] = {k: v.get() for k, v in self.custom_symbol_vars.items()}
        opts["y_quantity"] = self.y_quantity
        
        if not str(opts["xlabel"]).strip():
            del opts["xlabel"]
            
        plot_light_curve(self.results, self.target_name, "", ax=self.ax, visibility_state=v_state, plot_options=opts)
        self.canvas.draw()
        
    def export_plot(self):
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(defaultextension=".png", 
                                                initialfile=f"custom_plot_{self.target_name.replace(' ','_')}.png",
                                                filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Graphics", "*.svg")])
        if filepath:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")

def run_config_gui(pipeline_callback=None):
    """
    Launches a persistent Tkinter GUI for pipeline configuration.
    pipeline_callback: A function that takes (config) and runs the analysis.
    """
    APP_VERSION = "4.45"
    APP_DATE = "2026-11-15"          # update together with APP_VERSION (shown in the About tab)
    root = tk.Tk()
    root.title(f"Calibra v{APP_VERSION} - Astro Analysis & Photometry Suite")

    # v4.15: show uncaught errors in button callbacks instead of only printing
    # a traceback to the console (e.g. an empty or non-numeric settings field).
    def _report_callback_exception(exc, val, tb):
        import traceback
        traceback.print_exception(exc, val, tb)
        msg = str(val)
        if isinstance(val, tk.TclError) and "expected" in msg:
            msg = (f"A numeric field contains an invalid value:\n\n{msg}\n\n"
                   "Please check the entry fields (Settings and the current tab).")
        try:
            messagebox.showerror("Calibra - Error", msg)
        except Exception:
            pass
    root.report_callback_exception = _report_callback_exception
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
    unchecked_paths = set()  # v4.15: files the user un-ticked (survives table refreshes & sessions)
    vars_dict = {}    # Global-like storage for variable access

    def get_catalog_radius(file_path=None):
        """v4.16: configured catalogue search radius [arcmin]; 0 or negative =
        automatic: half-diagonal of the image (from its WCS) + 1'."""
        try:
            r = float(vars_dict["catalog_search_radius"][0].get())
        except (KeyError, tk.TclError, ValueError):
            r = 15.0
        if r > 0:
            return r
        paths = [file_path] if file_path else [f['path'] for f in loaded_files]
        for pth in paths:
            try:
                from astropy.io import fits as _fits
                from astropy.wcs import WCS as _WCS
                from astropy.wcs.utils import proj_plane_pixel_scales
                h = _fits.getheader(pth)
                w = _WCS(h)
                if w.has_celestial and h.get('NAXIS1') and h.get('NAXIS2'):
                    scale = float(np.mean(proj_plane_pixel_scales(w.celestial))) * 60.0
                    return round(0.5 * np.hypot(h['NAXIS1'], h['NAXIS2']) * scale + 1.0, 1)
            except Exception:
                continue
        return 60.0

    def open_path(path):
        """v4.18: open a file or folder with the system's default application."""
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True) if not os.path.splitext(path)[1] else None
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                import subprocess; subprocess.Popen(['open', path])
            else:
                import subprocess; subprocess.Popen(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open:\n{path}\n\n{e}")

    def tidy_numbers():
        """v4.18: round values shown in entry fields (catalogue values were shown with
        up to 16 digits, e.g. 10.256653035127442) - magnitudes/colours 3, coefficients 4 decimals."""
        import re as _re
        rules = [(r'^ts_ref_\d_(mag|bv)$', 3), (r'^ts_target_bv$', 3), (r'^default_zp_[vb]$', 3),
                 (r'^ts_coeff$', 4), (r'^diff_(tbv|tbbv|tvbv)$', 4)]
        for key, val in list(vars_dict.items()):
            if not isinstance(val, (tuple, list)) or len(val) < 2 or val[1] is not float:
                continue
            for pat, nd in rules:
                if _re.match(pat, key):
                    try:
                        v = float(val[0].get())
                        if np.isfinite(v) and round(v, nd) != v:
                            val[0].set(round(v, nd))
                    except (tk.TclError, ValueError):
                        pass
                    break

    def band_of(filt):
        """v4.15: single B/V classification used everywhere in the GUI ('V', 'B' or None)."""
        try:
            v_key = vars_dict["filter_v_keyword"][0].get() if "filter_v_keyword" in vars_dict else "Vmag"
            b_key = vars_dict["filter_b_keyword"][0].get() if "filter_b_keyword" in vars_dict else "Bmag"
        except tk.TclError:
            v_key, b_key = "Vmag", "Bmag"
        return classify_filter(filt, v_key, b_key)

    def lband_of(filt):
        """v4.23: 'V', 'B', 'L' (luminance) or None - used where light curves are made."""
        from photometry.filters import classify_band
        try:
            v_key = vars_dict["filter_v_keyword"][0].get() if "filter_v_keyword" in vars_dict else "Vmag"
            b_key = vars_dict["filter_b_keyword"][0].get() if "filter_b_keyword" in vars_dict else "Bmag"
            l_key = vars_dict["filter_l_keyword"][0].get() if "filter_l_keyword" in vars_dict else "L"
        except tk.TclError:
            v_key, b_key, l_key = "Vmag", "Bmag", "L"
        return classify_band(filt, v_key, b_key, l_key)

    def aavso_band(filt):
        """AAVSO filter code: L (luminance) photometry reduced with V comparison magnitudes is 'CV'."""
        f = str(filt).upper()
        return "CV" if f == "L" else f
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
        elif tab_name == "fieldvars":
            tab_fieldvars_outer.tkraise()
            if 'fv_auto_gcvs' in vars_dict:
                root.after(100, vars_dict['fv_auto_gcvs'])
        elif tab_name == "period":
            tab_period_outer.tkraise()
        elif tab_name == "planner":
            tab_planner_outer.tkraise()
            if 'planner_refresh' in vars_dict:
                vars_dict['planner_refresh']()
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
        pop.geometry("620x650")
        pop.resizable(False, False)
        pop.transient(root)
        pop.configure(bg="white")
        
        pop.update_idletasks()
        rx, ry = root.winfo_x(), root.winfo_y()
        rw, rh = root.winfo_width(), root.winfo_height()
        px = rx + (rw - 620) // 2
        py = ry + (rh - 650) // 2
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
            "- Airmass values (X_B, X_V) of your observations (global extinction coefficients k_B and k_V from the Settings tab are also applied to correct for atmospheric extinction). If FITS headers contain airmass, they are loaded automatically unless the override checkbox is checked.\n\n"
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
        pop.title("About Differential Photometry & Standard Calibration")
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
        tk.Label(header, text=f"{get_icon('📊', '')}  Differential Photometry & Standard Calibration".strip(), font=("Segoe UI", 13, "bold"), fg="white", bg="#1a3a5f").pack()
        
        content_frame = tk.Frame(pop, bg="white", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)
        
        info_box = scrolledtext.ScrolledText(content_frame, font=("Segoe UI", 9), bg="white", fg="#333333", relief="flat", wrap=tk.WORD)
        info_box.pack(fill="both", expand=True)
        
        details = (
            "WHAT THIS TAB DOES:\n"
            "-------------------\n"
            "This tab calibrates a single pair of B and V photometry result files (CSVs generated during the Detection & Zero-Point step) using a reference star of known standard magnitude.\n\n"
            "It solves for the zero points (Z_B, Z_V, Z_BV) of the field, applies extinction and color corrections (using color coefficients Tbv, Tb_bv, Tv_bv), and outputs standardized B, V, and B-V magnitudes for all matched stars in the field.\n\n"
            "It also evaluates the accuracy of the photometric calibration by comparing the resulting magnitudes of all other non-variable catalog stars in the image against their true standard catalog values.\n\n"
            "How it works:\n"
            "1. Select the B-Filter and V-Filter CSV result files generated from your calibrated frames.\n"
            "2. Enter the Color Transformation Coefficients (Tbv, Tb_bv, Tv_bv) or click 'Load Last Coefficients' to load them from 'color_coefficients.json'.\n"
            "3. Zero points: by default (v4.16) from an ENSEMBLE of all suitable catalogue stars (good SNR, below the peak limit, not blended, not variable); alternatively one automatically chosen star or a star given by name/coordinates. Results include uncertainties; an optional check star is compared with the catalogue.\n"
            "4. Select target stars to analyze (either 'all stars' or a specific target star).\n"
            "5. Click 'Execute Differential Photometry' to compute standardized magnitudes, save the results, and plot the Gaussian precision evaluation of the field.\n\n"
            "-------------------\n"
            "REQUIRED INPUTS:\n"
            "- B-filter and V-filter photometry CSV result files.\n"
            "- Atmospheric extinction coefficients (k_B, k_V) from the Settings tab.\n"
            "- Color transformation coefficients (Tbv, Tb_bv, Tv_bv).\n\n"
            "-------------------\n"
            "OUTPUTS:\n"
            "- 'differential_photometry_results.csv': A table listing the standardized B, V, and B-V magnitudes of all matched stars.\n"
            "- 'differential_photometry_report.md': A summary report including calculated zero points and reference star parameters.\n"
            "- An accuracy evaluation plot displaying a Gaussian fit of calibration errors (Delta-B, Delta-V, Delta-BV) across all matched stars."
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
            "- The engine automatically maps and applies the correct Master Flat corresponding to the filter specified inside the FITS header (V, B, L or any other filter).\n\n"
            "Master frames (v4.27):\n"
            "- Choose the folder with your raw bias / flat (/ dark) frames and press 'Build / Update Masters'. Frames are sorted by IMAGETYP (or file/folder name), grouped by size, binning, gain, offset and filter (darks: exposure, temperature), combined (sigma-clipped mean or median) and saved as master_*.fits in the same folder. Existing masters made from the same files are reused.\n"
            "- Automatic mode picks the matching masters for every light frame; 'Show Masters for Checked Frames' shows the choice.\n"
            "- 'Calibrate on the fly' corrects frames in memory when measuring (no extra files). 'Write Calibrated Files' writes cal_*.fits into a 'calibrated' folder next to the raw files.\n"
            "- Darks are optional ('Use darks'). Already calibrated frames are never calibrated twice.\n\n"
            "Separate folders (v4.29):\n"
            "- Give a bias folder and a flat folder (and a dark folder if you use darks). Every FITS file in the bias folder is used as a bias, every file in the flat folder as a flat - the file names do not matter. The filter of the flats comes from the FITS header (FILTER) or, if missing, from the file or folder name (V, B, L, Lum, Vmag ...). Files whose header says something else (e.g. IMAGETYP = Light) are still used as the type of their folder; a warning is shown (v4.33).\n"
            "- The masters are written into the same folders. Order: bias first, then the flats are corrected with this master bias (or a dark of the flats' exposure) and normalised.\n\n"
            "No double counting:\n"
            "- Calibrated light = (light - master bias) / normalised master flat.\n"
            "- The bias is removed from the flats when the master flat is built, and once from every light. The master flat itself contains no bias; it only describes the relative sensitivity (median = 1).\n"
            "- So both masters are needed when calibrating lights: the bias (additive offset of the light) and the flat (multiplicative response).\n"
            "- If a master flat was made by other software, tell Calibra whether it is already bias-subtracted (checkbox); if not, Calibra subtracts the bias from it before normalising.\n\n"
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
            "POSITION GUESS (v4.22):\n"
            "ASTAP is fast when it knows roughly where the image points. The guess comes from:\n"
            "- the FITS header (RA/DEC or OBJCTRA/OBJCTDEC written by the capture software), or\n"
            "- an object name (looked up in SIMBAD / AAVSO VSX - internet needed), or\n"
            "- RA/Dec typed in (hh:mm:ss +dd:mm:ss or degrees).\n"
            "The search radius (Solver Settings) is the area searched around the guess. With 'Use the solution "
            "of each image as the guess for the next image', every solved image guides the next one, so a series "
            "only needs a guess for its first image. If a guess fails, Calibra tries your guess, then the header, "
            "and finally (if allowed) a blind whole-sky solve, which can take minutes. 'Field height' tells ASTAP "
            "the image height in degrees if it cannot derive it from the header (0 = automatic).\n\n"
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
            "- Photometry results saved to standard CSV and AAVSO-format text reports ready for upload!\n\n"
            "-----------------------------\n"
            "LIGHT CURVE MODES (v4.14):\n"
            "- Calibrated magnitudes: the classic ensemble mode described above (needs reference stars with catalogue magnitudes). Only this mode writes an AAVSO report.\n"
            "- Relative flux: uses only the measured ADU of the reference stars. The light curve is F_target / sum(F_ref); catalogue magnitudes, B-V and extinction are ignored. "
            "Cancels first-order transparency/airmass changes. Each reference star is also shown relative to the sum of the other references as a stability diagnostic.\n"
            "- Raw ADU: the target's background-subtracted net flux in ADU (optionally ADU/s) with CCD-equation errors. No reference or check stars needed - ideal for a quick look, "
            "for eclipse timing on photometric nights, or for fields without good comparison stars. Selected reference/check stars are measured for display only.\n"
            "If no reference star is selected, Calibra switches to Raw ADU automatically (can be disabled)."
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
            
            if band_of(filt) == 'B':
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
                    'filter_v_keyword': vars_dict["filter_v_keyword"][0].get() if "filter_v_keyword" in vars_dict else "Vmag",
                    'filter_b_keyword': vars_dict["filter_b_keyword"][0].get() if "filter_b_keyword" in vars_dict else "Bmag",
                }
                
                # 2. Gather INITIAL STARS from Light Curves tab
                initial_stars = {'variable': None, 'check': None, 'refs': []}
                
                # Target
                t_mode = vars_dict["ts_target_mode"][0].get()
                t_name = vars_dict["ts_target_name"][0].get().strip()
                if t_mode == "name" and t_name:
                    # v4.34: names are resolved by the viewer in the background (local GCVS list first)
                    initial_stars['variable'] = {'ra': None, 'dec': None, 'name': t_name}
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
                        s_data = {'ra': None, 'dec': None, 'name': name}     # v4.34: resolved in the viewer
                        if i == cs_idx: initial_stars['check'] = s_data
                        elif vars_dict[f"ts_ref_{i}_use"][0].get(): initial_stars['refs'].append(s_data)

                # 3. Define EXPORT Callbacks
                def update_light_curve_stars(data):
                    # v4.15: protect existing Light-Curve data from being replaced by
                    # viewer placeholders ('Target', 'Uncataloged_*'), NaN magnitudes
                    # or catalogue values when the same star is already configured.
                    def _clean(val):
                        try:
                            f = float(val)
                            return f if np.isfinite(f) else None
                        except (TypeError, ValueError):
                            return None
                    def _is_placeholder(name):
                        n = str(name or "").strip()
                        return (not n) or n in ("Target", "Star") or n.startswith("Uncataloged")
                    def _sep_arcsec(ra1, dec1, ra2, dec2):
                        try:
                            dra = (float(ra1) - float(ra2)) * np.cos(np.radians(float(dec1)))
                            return 3600.0 * np.hypot(dra, float(dec1) - float(dec2))
                        except (TypeError, ValueError):
                            return 1e9
                    # Snapshot of the currently configured comparison-star slots
                    old_slots = []
                    for k in range(5):
                        if vars_dict[f"ts_ref_{k}_has_manual"][0].get():
                            try:
                                old_mag = _clean(vars_dict[f"ts_ref_{k}_mag"][0].get())
                                old_bv = _clean(vars_dict[f"ts_ref_{k}_bv"][0].get())
                            except tk.TclError:
                                old_mag, old_bv = None, None
                            old_slots.append({
                                'name': vars_dict[f"ts_ref_{k}_name"][0].get(),
                                'ra': vars_dict[f"ts_ref_{k}_ra"][0].get(),
                                'dec': vars_dict[f"ts_ref_{k}_dec"][0].get(),
                                'mag': old_mag if old_mag else None,   # 0.0 = "not fetched"
                                'bv': old_bv,
                                'label': vars_dict[f"ts_ref_{k}_coord_label"][0].get() if f"ts_ref_{k}_coord_label" in vars_dict else ""
                            })
                    def _merge_with_existing(star):
                        """Return (name, mag, bv, label) preferring an already configured slot
                        at the same position (within 3 arcsec)."""
                        mag, bv = _clean(star.get('mag')), _clean(star.get('bv'))
                        name = star.get('name')
                        for o in old_slots:
                            if _sep_arcsec(star['ra'], star['dec'], o['ra'], o['dec']) < 3.0:
                                return (o['name'] or name, o['mag'] if o['mag'] is not None else mag,
                                        o['bv'] if o['mag'] is not None else bv, o['label'])
                        if _is_placeholder(name):
                            c_tmp = SkyCoord(star['ra'], star['dec'], unit=u.deg)
                            name = "J" + c_tmp.to_string('hmsdms', sep='', precision=0).replace(' ', '')
                        return name, mag, bv, None
                    for key in ('variable', 'check'):
                        if data.get(key):
                            data[key]['mag'] = _clean(data[key].get('mag'))
                            data[key]['bv'] = _clean(data[key].get('bv'))
                    if data['variable']:
                        v = data['variable']
                        # Keep the current target name if the exported one is a placeholder
                        # or the star is the one already configured.
                        old_name = vars_dict["ts_target_name"][0].get().strip()
                        same_target = False
                        try:
                            if vars_dict["ts_target_mode"][0].get() == "manual":
                                c_old = SkyCoord(vars_dict["ts_target_ra"][0].get(), vars_dict["ts_target_dec"][0].get(), unit=(u.hourangle, u.deg))
                                same_target = _sep_arcsec(v['ra'], v['dec'], c_old.ra.deg, c_old.dec.deg) < 5.0
                        except Exception:
                            same_target = False
                        new_name = v['name']
                        if old_name and (_is_placeholder(new_name) or same_target):
                            new_name = old_name
                        vars_dict["ts_target_mode"][0].set("manual")
                        vars_dict["ts_target_name"][0].set(new_name)
                        c = SkyCoord(v['ra'], v['dec'], unit=u.deg)
                        vars_dict["ts_target_ra"][0].set(c.ra.to_string(unit='hour', sep=':', precision=2))
                        vars_dict["ts_target_dec"][0].set(c.dec.to_string(unit='degree', sep=':', precision=2, alwayssign=True))
                        
                        # Use exported mag/bv if available to skip re-fetch
                        if v.get('bv') is not None: vars_dict["ts_target_bv"][0].set(v['bv'])
                    
                    # Check & Refs
                    slots_filled = 0
                    if data['check']:
                        c = data['check']
                        c_name, c_mag, c_bv, c_label = _merge_with_existing(c)
                        c['mag'], c['bv'] = c_mag, c_bv
                        vars_dict["ts_ref_0_name"][0].set(c_name)
                        vars_dict["ts_ref_0_ra"][0].set(c['ra'])
                        vars_dict["ts_ref_0_dec"][0].set(c['dec'])
                        vars_dict["ts_ref_0_has_manual"][0].set(True)
                        vars_dict["ts_ref_0_use"][0].set(False)
                        ts_check_star_idx_var.set(0)
                        if c_label and "ts_ref_0_coord_label" in vars_dict:
                            vars_dict["ts_ref_0_coord_label"][0].set(c_label)
                        
                        vars_dict["ts_ref_0_mag"][0].set(c_mag if c_mag is not None else 0.0)
                        vars_dict["ts_ref_0_bv"][0].set(c_bv if c_bv is not None else 0.0)
                        slots_filled = 1
                    else:
                        ts_check_star_idx_var.set(-1)
                    
                    current_idx = slots_filled
                    for r in data['refs']:
                        if current_idx >= 5: break
                        r_name, r_mag, r_bv, r_label = _merge_with_existing(r)
                        r['mag'], r['bv'] = r_mag, r_bv
                        vars_dict[f"ts_ref_{current_idx}_name"][0].set(r_name)
                        vars_dict[f"ts_ref_{current_idx}_ra"][0].set(r['ra'])
                        vars_dict[f"ts_ref_{current_idx}_dec"][0].set(r['dec'])
                        vars_dict[f"ts_ref_{current_idx}_has_manual"][0].set(True)
                        vars_dict[f"ts_ref_{current_idx}_use"][0].set(True)
                        if r_label and f"ts_ref_{current_idx}_coord_label" in vars_dict:
                            vars_dict[f"ts_ref_{current_idx}_coord_label"][0].set(r_label)
                        
                        vars_dict[f"ts_ref_{current_idx}_mag"][0].set(r_mag if r_mag is not None else 0.0)
                        vars_dict[f"ts_ref_{current_idx}_bv"][0].set(r_bv if r_bv is not None else 0.0)
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
                    
                    tidy_numbers()
                    # Autosave session to disk (Fix)
                    save_session()

                def update_apertures(ap_data):
                    vars_dict["aperture_radius"][0].set(ap_data['aperture'])
                    vars_dict["annulus_inner"][0].set(ap_data['annulus_in'])
                    vars_dict["annulus_outer"][0].set(ap_data['annulus_out'])
                    ts_status_var.set("Aperture settings updated from FITS Viewer.")
                    
                    # Autosave session to disk (Fix)
                    save_session()

                # Pass the AAVSO cache if available - v4.35: the sequence of the CURRENT target,
                # from memory or from the stored file (no query)
                aavso_cache = getattr(on_get_aavso_refs, 'cache', None)
                try:
                    from photometry.local_catalogs import name_key as _nk
                    if t_name and _nk(getattr(on_get_aavso_refs, 'last_target', '')) != _nk(t_name):
                        aavso_cache = stored_aavso_sequence(t_name)
                except Exception:
                    pass
                
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
                    vars_vals[k] = read_tk_var(var, vtype)
                except ValueError as ve:
                    raise ValueError(f"Setting '{k}' must be a valid {vtype.__name__} ({ve}).") from ve
            
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
                # v4.16
                'stack_mode': vars_vals.pop('stack_mode', 'average'),
                'detect_fwhm': vars_vals.pop('detect_fwhm', 3.5),
                'sky_noise_factor': vars_vals.pop('sky_noise_factor', 1.0),
                'calib_peak_limit': vars_vals.pop('calib_peak_limit', 30000),
                'calib_exclude_blended': vars_vals.pop('calib_exclude_blended', True),
                'flat_already_bias_subtracted': vars_vals.pop('flat_already_bias_subtracted', True),
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
                
                pipeline_terminal.delete("1.0", tk.END)  # v4.15: done on the GUI thread
                def thread_target():
                    try:
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
                                
                                band = classify_filter(filt, config_run.get('filter_v_keyword', 'Vmag'), config_run.get('filter_b_keyword', 'Bmag'))
                                if band == 'B':
                                    root.after(0, lambda p=csv_path: vars_dict['color_b_csv'][0].set(p))
                                    last_zp_b = zp_val
                                elif band == 'V':
                                    root.after(0, lambda p=csv_path: vars_dict['color_v_csv'][0].set(p))
                                    last_zp_v = zp_val
                                else:
                                    print(f"Note: filter '{filt}' is neither B nor V (see Settings > Filter mapping); "
                                          f"its zero point is not stored as a default.")
                                
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
                    except Exception as e:
                        # v4.15: pipeline errors used to disappear silently
                        import traceback
                        traceback.print_exc()
                        root.after(0, lambda err=e: messagebox.showerror("Pipeline Error", f"The analysis pipeline stopped with an error:\n\n{err}"))
                    finally:
                        if hasattr(sys.stdout, "remove_widget"):
                            sys.stdout.remove_widget(pipeline_terminal)
                        if hasattr(sys.stderr, "remove_widget"):
                            sys.stderr.remove_widget(pipeline_terminal)
                        if hasattr(sys.stdout, 'set_log_file'):
                            try: sys.stdout.set_log_file(None)
                            except Exception: pass
                        root.after(0, lambda: run_btn.config(state=tk.NORMAL, text="Run Analysis Pipeline on Selected"))
                
                thread = threading.Thread(target=thread_target)
                thread.daemon = True
                thread.start()
            else:
                print("No pipeline callback provided.")
                
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please ensure all numerical fields contain valid numbers.\n\nDetail: {e}")

    def on_list_variables():
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager.")
            return
            
        file_list = [loaded_files[int(iid)]['path'] for iid in selected_iids]
        
        # Show a temporary please wait popup
        wait_pop = tk.Toplevel(root)
        wait_pop.title("Querying VSX...")
        wait_pop.geometry("320x110")
        wait_pop.resizable(False, False)
        wait_pop.transient(root)
        wait_pop.grab_set()
        
        # Center the wait dialog
        wx = root.winfo_x() + (root.winfo_width() - 320) // 2
        wy = root.winfo_y() + (root.winfo_height() - 110) // 2
        wait_pop.geometry(f"+{wx}+{wy}")
        
        tk.Label(wait_pop, text="Querying VSX Database...", font=("Arial", 10, "bold"), pady=10).pack()
        status_lbl = tk.Label(wait_pop, text="Opening FITS files...", font=("Arial", 9, "italic"))
        status_lbl.pack()
        wait_pop.update()
        
        all_vsx_stars = []
        
        def run_query_thread():
            try:
                import os
                import csv
                import numpy as np
                from astropy.io import fits
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                from photometry.calibration import get_vsx_stars
                
                radius = get_catalog_radius()
                    
                seen_names = set()
                queried_coords = []
                
                for idx, fits_path in enumerate(file_list):
                    if not os.path.exists(fits_path):
                        continue
                    
                    filename_only = os.path.basename(fits_path)
                    root.after(0, lambda f=filename_only: status_lbl.config(text=f"Reading {f}..."))
                    
                    with fits.open(fits_path) as hdul:
                        header = hdul[0].header
                    
                    ra_val = header.get('RA') or header.get('OBJCTRA') or header.get('CRVAL1')
                    dec_val = header.get('DEC') or header.get('OBJCTDEC') or header.get('CRVAL2')
                    
                    center_ra, center_dec = None, None
                    if ra_val is not None and dec_val is not None:
                        try:
                            if isinstance(ra_val, str) and (':' in ra_val or ' ' in ra_val):
                                c = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg))
                                center_ra = c.ra.deg
                                center_dec = c.dec.deg
                            else:
                                center_ra = float(ra_val)
                                center_dec = float(dec_val)
                        except:
                            pass
                    
                    if center_ra is None or center_dec is None:
                        # Try to read catalog centers
                        base_name = os.path.splitext(filename_only)[0]
                        csv_path = os.path.join('photometry_output', f'targets_auto_{base_name}.csv')
                        if os.path.exists(csv_path):
                            with open(csv_path, 'r') as f:
                                reader = csv.DictReader(f)
                                first_row = next(reader, None)
                                if first_row and first_row.get('ra_deg') and first_row.get('dec_deg'):
                                    center_ra = float(first_row['ra_deg'])
                                    center_dec = float(first_row['dec_deg'])
                                    
                    if center_ra is not None and center_dec is not None:
                        queried_coords.append((center_ra, center_dec))
                        root.after(0, lambda f=filename_only: status_lbl.config(text=f"Querying VSX near {center_ra:.2f}, {center_dec:.2f}..."))
                        vsx_list = get_vsx_stars(center_ra, center_dec, radius_arcmin=radius, verbose=False)
                        for star in vsx_list:
                            # Exclude constant stars (CST)
                            star_type = str(star.get('Type', '')).upper().strip()
                            if 'CST' in star_type:
                                continue
                            name = star['id']
                            if name not in seen_names:
                                seen_names.add(name)
                                all_vsx_stars.append(star)
                                
                root.after(0, wait_pop.destroy)
                root.after(0, lambda: show_vsx_results_window(all_vsx_stars, queried_coords))
            except Exception as e:
                root.after(0, wait_pop.destroy)
                root.after(0, lambda err=e: messagebox.showerror("Query Error", f"Failed to retrieve VSX variables: {err}"))
                
        import threading
        threading.Thread(target=run_query_thread, daemon=True).start()

    def show_vsx_results_window(all_vsx_stars, queried_coords):
        if not queried_coords:
            messagebox.showwarning("No Coordinates", 
                                   "Could not resolve celestial coordinates for any of the selected FITS file(s).\n\n"
                                   "Please ensure the files have RA/Dec header keywords or have been plate-solved (WCS) first.")
            return
            
        if not all_vsx_stars:
            coord_str = ", ".join([f"({ra:.2f}, {dec:.2f})" for ra, dec in queried_coords[:3]])
            if len(queried_coords) > 3:
                coord_str += "..."
            messagebox.showinfo("No Variables Found", 
                                f"No variable stars matching the VSX catalog were found in the searched fields near {coord_str}.")
            return
            
        pop = tk.Toplevel(root)
        pop.title("VSX Variable Stars List")
        pop.geometry("900x600")
        pop.transient(root)
        
        # Center window
        px = root.winfo_x() + (root.winfo_width() - 900) // 2
        py = root.winfo_y() + (root.winfo_height() - 600) // 2
        pop.geometry(f"+{px}+{py}")
        
        main_frame = ttk.Frame(pop, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        title_lbl = tk.Label(main_frame, text=f"VSX Variable Stars Found ({len(all_vsx_stars)} stars)", font=("Arial", 11, "bold"), fg="#1a3a5f")
        title_lbl.pack(pady=(0, 5), anchor=tk.W)
        
        tk.Label(main_frame, text="* Tip: Click on any column header to sort the list.", fg="#666", font=("Arial", 8, "italic")).pack(anchor=tk.W, pady=(0, 5))
        
        # Create a frame for the Treeview and its scrollbars to prevent scrollbars from overlapping the content
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(side=tk.TOP, fill="both", expand=True, pady=(5, 0))
        
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        
        # Create Treeview table
        cols = ("Name", "Type", "RA", "Dec", "Max Mag", "Min Mag", "Amp", "Period")
        tree_var = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)
        tree_var.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree_var.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree_var.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_var.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Column setup and header sort bindings
        for col in cols:
            width = 150 if col in ["Name", "RA", "Dec"] else 80
            if col == "Period": width = 100
            tree_var.column(col, width=width, anchor=tk.CENTER if col != "Name" else tk.W)
            tree_var.heading(col, text=col, command=lambda _col=col: sort_treeview(_col, False))
            
        def sort_treeview(col, reverse):
            item_list = []
            is_numeric = col in ["Max Mag", "Min Mag", "Amp", "Period"]
            for k in tree_var.get_children(''):
                val = tree_var.set(k, col)
                if is_numeric:
                    try:
                        val_float = float(val)
                    except ValueError:
                        val_float = float('inf') if not reverse else float('-inf')
                    item_list.append((val_float, k))
                else:
                    item_list.append((val.lower(), k))
                    
            item_list.sort(key=lambda t: t[0], reverse=reverse)
            
            for index, (_, k) in enumerate(item_list):
                tree_var.move(k, '', index)
                
            tree_var.heading(col, command=lambda _col=col: sort_treeview(_col, not reverse))
            
        # Populating treeview
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        import numpy as np
        
        for idx, s in enumerate(all_vsx_stars):
            try:
                c_star = SkyCoord(ra=s['ra_deg']*u.deg, dec=s['dec_deg']*u.deg)
                ra_str = c_star.ra.to_string(unit='hour', sep=':', precision=2, pad=True)
                dec_str = c_star.dec.to_string(unit='degree', sep=':', precision=1, pad=True, alwayssign=True)
            except:
                ra_str = f"{s['ra_deg']:.5f}"
                dec_str = f"{s['dec_deg']:.5f}"
                
            max_mag = s.get('max', np.nan)
            min_mag = s.get('min', np.nan)
            period = s.get('Period', np.nan)
            
            max_str = f"{max_mag:.2f}" if not np.isnan(max_mag) else ""
            min_str = f"{min_mag:.2f}" if not np.isnan(min_mag) else ""
            period_str = f"{period:.5f}" if not np.isnan(period) else ""
            
            if not np.isnan(max_mag) and not np.isnan(min_mag):
                amp_str = f"{min_mag - max_mag:.2f}"
            else:
                amp_str = ""
                
            tree_var.insert("", tk.END, iid=str(idx), values=(
                s['id'], s.get('Type', ''), ra_str, dec_str, max_str, min_str, amp_str, period_str
            ))
            
        # Bottom area for Copy/Selection details
        detail_frame = ttk.LabelFrame(main_frame, text="Selected Star Details (Selectable & Copyable)", padding=10)
        detail_frame.pack(fill="x", pady=10)
        
        detail_text = tk.Text(detail_frame, font=("Consolas", 9), bg="#f8f9fa", fg="#1a3a5f", height=5, relief="flat", borderwidth=0, wrap="word")
        detail_text.pack(fill="x", expand=True)
        detail_text.insert("1.0", "Click on a star in the table above to view and copy its full details.")
        detail_text.config(state="disabled")
        
        def on_tree_select(event):
            selected = tree_var.selection()
            if not selected:
                return
            idx_str = selected[0]
            row_vals = tree_var.item(idx_str, 'values')
            
            msg = f"VSX Name:   {row_vals[0]}\n"
            msg += f"Type:       {row_vals[1]}\n"
            msg += f"Coords:     RA {row_vals[2]} / Dec {row_vals[3]}\n"
            msg += f"Magnitudes: Max {row_vals[4]} / Min {row_vals[5]} (Amp: {row_vals[6]} mag)\n"
            msg += f"Period:     {row_vals[7]} days"
            
            detail_text.config(state="normal")
            detail_text.delete("1.0", tk.END)
            detail_text.insert("1.0", msg)
            detail_text.config(state="disabled")
            
        tree_var.bind("<<TreeviewSelect>>", on_tree_select)
        
        # Add buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(5, 0))
        
        def copy_selected():
            selected = tree_var.selection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select a star from the table first.")
                return
            idx_str = selected[0]
            row_vals = tree_var.item(idx_str, 'values')
            summary = ", ".join([f"{cols[i]}: {row_vals[i]}" for i in range(len(cols))])
            pop.clipboard_clear()
            pop.clipboard_append(summary)
            messagebox.showinfo("Copied", "Selected star details copied to clipboard!")
            
        def copy_all():
            lines = [",".join(cols)]
            for k in tree_var.get_children(''):
                row_vals = tree_var.item(k, 'values')
                lines.append(",".join([f'"{val}"' for val in row_vals]))
            all_text = "\n".join(lines)
            pop.clipboard_clear()
            pop.clipboard_append(all_text)
            messagebox.showinfo("Copied All", "All variable stars copied to clipboard as CSV!")
            
        ttk.Button(btn_frame, text="Copy Selected Star Details", command=copy_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Copy All as CSV Table", command=copy_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close Window", command=pop.destroy).pack(side=tk.RIGHT, padx=5)

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
            search_radius = get_catalog_radius()
            from photometry.color_calibration import derive_color_terms
            from photometry.calibration import get_ref_stars
            
            catalog = get_ref_stars(cat_type, ra_c, dec_c, radius_arcmin=search_radius, verbose=True)
            
            if not catalog:
                color_status_var.set("Error: Could not fetch online catalog.")
                return
                
            max_v_txt = color_max_v_var.get().strip()
            try:
                max_v = float(max_v_txt) if max_v_txt else None
            except ValueError:
                messagebox.showerror("Input Error", f"'Faintest catalogue V' must be a number or empty (got '{max_v_txt}').")
                return
            res = derive_color_terms(data_b, data_v, 
                                     catalog, "photometry_output", 
                                     airmass_b=air_b_var.get(), airmass_v=air_v_var.get(),
                                     k_b=vars_dict["extinction_kb"][0].get(), k_v=vars_dict["extinction_kv"][0].get(),
                                     axes=color_coeff_axes,
                                     min_snr=float(color_min_snr_var.get()),
                                     peak_limit=float(vars_dict["calib_peak_limit"][0].get()),
                                     max_v_mag=max_v,
                                     exclude_blended=bool(vars_dict["calib_exclude_blended"][0].get()))
            color_coeff_canvas.draw()
            color_status_var.set(res)
            
        except Exception as e:
            color_status_var.set(f"Error: {e}")
            messagebox.showerror("Analysis Error", str(e))

    last_diff_args = {}

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
                diff_status_var.set(f"Resolving '{star_name}' (local GCVS list, else Simbad)...")
                root.update_idletasks()
                try:
                    c = _resolve_name(star_name)
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
                        messagebox.showerror("Resolution Error", f"Could not resolve name '{star_name}' via the local GCVS list, Simbad or the local AAVSO catalogue.\n\nNote: Make sure your AAVSO CSV is loaded in the Analysis tab if you are using AUIDs like 000-BJS-555.")
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
                diff_status_var.set(f"Resolving '{star_name}' (local GCVS list, else Simbad)...")
                root.update_idletasks()
                try:
                    c = _resolve_name(star_name)
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
                        messagebox.showerror("Resolution Error", f"Could not resolve target name '{star_name}' via the local GCVS list, Simbad or the local AAVSO catalogue.\n\nNote: Make sure your AAVSO CSV is loaded in the Analysis tab if you are using AUIDs like 000-BJS-555.")
                        diff_status_var.set(f"Error: Could not resolve target '{star_name}'.")
                        return
                except Exception as e:
                    messagebox.showerror("Resolution Error", f"Error looking up target '{star_name}':\n{e}")
                    diff_status_var.set("Error during target name resolution.")
                    return

            # v4.16: optional check star (name or "RA Dec")
            check_coord = None
            chk_txt = diff_check_var.get().strip()
            if chk_txt:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                try:
                    parts = chk_txt.replace(',', ' ').split()
                    if len(parts) == 2:
                        try:
                            c = SkyCoord(float(parts[0]) * u.deg, float(parts[1]) * u.deg)
                        except ValueError:
                            c = SkyCoord(parts[0], parts[1], unit=(u.hourangle, u.deg))
                    elif len(parts) == 6:
                        c = SkyCoord(" ".join(parts), unit=(u.hourangle, u.deg))
                    else:
                        c = _resolve_name(chk_txt)
                    check_coord = (c.ra.deg, c.dec.deg)
                except Exception as e:
                    messagebox.showwarning("Check Star", f"Could not resolve the check star '{chk_txt}': {e}\nContinuing without check star.")
            comp_mode = 'single' if ref_mode_var.get() == "auto_single" else 'ensemble'
            diff_extra = dict(comp_mode=comp_mode, min_snr=float(diff_min_snr_var.get()),
                              peak_limit=float(vars_dict["calib_peak_limit"][0].get()),
                              exclude_blended=bool(vars_dict["calib_exclude_blended"][0].get()),
                              Tbv_err=diff_coeff_err['Tbv'], Tb_bv_err=diff_coeff_err['Tb_bv'], Tv_bv_err=diff_coeff_err['Tv_bv'],
                              check_coord=check_coord)

            res = run_differential_photometry(
                csv_b=b_csv, csv_v=v_csv, ref_catalog=cat_type,
                k_b=vars_dict["extinction_kb"][0].get(), k_v=vars_dict["extinction_kv"][0].get(),
                Tbv=diff_tbv_var.get(), Tb_bv=diff_tbbv_var.get(), Tv_bv=diff_tvbv_var.get(),
                radius_arcmin=get_catalog_radius(),
                manual_ref_coord=manual_coord,
                target_mode=target_mode,
                manual_target_coord=manual_target_coord,
                axes=accuracy_axes,
                **diff_extra
            )
            accuracy_canvas.draw()
            diff_status_var.set(res)
            
            if "Success" in res:
                last_diff_args.clear()
                last_diff_args.update({
                    'csv_b': b_csv,
                    'csv_v': v_csv,
                    'ref_catalog': cat_type,
                    'k_b': vars_dict["extinction_kb"][0].get(),
                    'k_v': vars_dict["extinction_kv"][0].get(),
                    'Tbv': diff_tbv_var.get(),
                    'Tb_bv': diff_tbbv_var.get(),
                    'Tv_bv': diff_tvbv_var.get(),
                    'radius_arcmin': get_catalog_radius(),
                    'manual_ref_coord': manual_coord,
                    'target_mode': target_mode,
                    'manual_target_coord': manual_target_coord,
                    'extra': diff_extra
                })
        except Exception as e:
            diff_status_var.set(f"Error: {e}")
            messagebox.showerror("Analysis Error", str(e))

    def save_session():
        import json
        try:
            vars_dict['instrument_autosave']()     # v4.36: current settings -> active instrument setup
        except Exception:
            pass
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
        
        # Add loaded file paths (+ which ones are un-ticked, v4.15)
        data['loaded_file_paths'] = [f['path'] for f in loaded_files]
        data['unchecked_file_paths'] = sorted(p for p in unchecked_paths if any(f['path'] == p for f in loaded_files))
        
        # Add AAVSO sequence cache
        try:
            if hasattr(on_get_aavso_refs, 'cache'):
                data['aavso_cache'] = on_get_aavso_refs.cache
                data['aavso_last_target'] = getattr(on_get_aavso_refs, 'last_target', "")
                data['aavso_last_cat'] = getattr(on_get_aavso_refs, 'last_cat', "")
        except NameError:
            pass # on_get_aavso_refs not bound yet
        # v4.19: Field Variables target list
        try:
            data['fv_target_list'] = [{k: t.get(k) for k in ('name', 'ra', 'dec', 'type', 'period', 'max', 'min', 'source', 'use')}
                                      for t in fv_targets]
        except NameError:
            pass
        
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
                        # v4.15: no duplicates when a session is loaded twice
                        if os.path.exists(path) and not any(f['path'] == path for f in loaded_files):
                            loaded_files.append(scan_fits_header(path))
                    continue
                if key == 'unchecked_file_paths':
                    unchecked_paths.update(value or [])
                    continue
                if key == 'fv_target_list':
                    try:
                        fv_targets[:] = [dict(t) for t in (value or []) if isinstance(t, dict) and t.get('name')]
                        fv_refresh_tree()
                    except NameError:
                        pass
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
                else:
                    try:
                        if key == 'aavso_cache':
                            on_get_aavso_refs.cache = value
                        elif key == 'aavso_last_target':
                            on_get_aavso_refs.last_target = value
                        elif key == 'aavso_last_cat':
                            on_get_aavso_refs.last_cat = value
                    except NameError:
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
    
    def update_file_table(select_iid=None):
        # Remember selection if select_iid is not explicitly passed
        current_selection = tree.selection()
        prev_selected_iid = current_selection[0] if current_selection else None
        
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Populate with loaded_files
        for idx, file_data in enumerate(loaded_files):
            tag = 'even' if idx % 2 == 0 else 'odd'
            # v4.15: keep the user's tick state across refreshes (default: checked)
            tree.insert('', tk.END, iid=str(idx), values=(
                '[ ]' if file_data.get('path') in unchecked_paths else '[X]',
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
        l_count = 0
        for f in loaded_files:
            band = lband_of(f.get('filter', ''))
            if band == 'V':
                v_count += 1
            elif band == 'B':
                b_count += 1
            elif band == 'L':
                l_count += 1

        status_text = f"Loaded: {len(loaded_files)} files ({v_count}× V, {b_count}× B" + (f", {l_count}× L)" if l_count else ")")
        file_manager_status.set(status_text)
        
        # Update light curve filter dropdown with unique translated filters
        unique_translated = set()
        for f in loaded_files:
            filt_str = str(f.get('filter', '')).upper().strip()
            if filt_str:
                band = lband_of(filt_str)
                unique_translated.add(band if band else filt_str)
        
        unique_filters = sorted(list(unique_translated))
        if 'filter_cb' in ts_widgets:
            ts_widgets['filter_cb']['values'] = unique_filters
            # If current selection is not in new list and list is not empty, select first
            if unique_filters and vars_dict.get("ts_filter", [None])[0]:
                curr = vars_dict["ts_filter"][0].get()
                if curr not in unique_filters:
                    vars_dict["ts_filter"][0].set(unique_filters[0])
        
        # Select target item, or re-select previous, or select first item if nothing selected
        target_iid = select_iid if select_iid is not None else prev_selected_iid
        if target_iid is not None and tree.exists(str(target_iid)):
            tree.selection_set(str(target_iid))
            tree.see(str(target_iid))
            on_tree_select(None)
        elif tree.get_children():
            first_iid = tree.get_children()[0]
            tree.selection_set(first_iid)
            tree.see(first_iid)
            # Manually trigger header update
            on_tree_select(None)

    def on_load_files():
        from tkinter import filedialog
        files = filedialog.askopenfilenames(title="Select FITS Files", filetypes=(("FITS files", "*.fits *.fit *.fts *.FITS *.FIT *.FTS"), ("all files", "*.*")))
        if files:
            for f in files:
                # Avoid duplicates
                if not any(existing['path'] == f for existing in loaded_files):
                    loaded_files.append(scan_fits_header(f))
            update_file_table()
            try:
                vars_dict['instrument_check'](list(files))      # v4.36
            except Exception:
                pass

    def on_load_dir():
        from tkinter import filedialog
        import glob
        dirname = filedialog.askdirectory(title="Select FITS Directory")
        if dirname:
            # v4.19: .fits, .fit and .fts (any letter case), sorted by name
            files = sorted(f for f in glob.glob(os.path.join(dirname, "*"))
                           if os.path.splitext(f)[1].lower() in FITS_EXTENSIONS)
            for f in files:
                if not any(existing['path'] == f for existing in loaded_files):
                    loaded_files.append(scan_fits_header(f))
            update_file_table()
            try:
                vars_dict['instrument_check'](list(files))      # v4.36
            except Exception:
                pass

    def on_remove_selected():
        # v4.15: remove the HIGHLIGHTED rows. Only if nothing is highlighted are the
        # checked files removed - and only after confirmation (previously all checked
        # files - i.e. usually the whole list - were removed first).
        selected = list(tree.selection())
        if not selected:
            selected = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
            if not selected: return
            if not messagebox.askyesno("Remove Files", f"No rows are highlighted.\n\nRemove all {len(selected)} CHECKED file(s) from the list?"):
                return
        
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
        unchecked_paths.clear()
        for iid in tree.get_children():
            vals = list(tree.item(iid, 'values'))
            vals[0] = '[X]'
            tree.item(iid, values=vals)

    def on_uncheck_all():
        unchecked_paths.update(f['path'] for f in loaded_files)
        for iid in tree.get_children():
            vals = list(tree.item(iid, 'values'))
            vals[0] = '[ ]'
            tree.item(iid, values=vals)

    ttk.Button(toolbar_frame, text="Load Files...", command=on_load_files).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Load Directory...", command=on_load_dir).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Check All", command=on_check_all).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Uncheck All", command=on_uncheck_all).pack(side=tk.LEFT, padx=5)
    ttk.Button(toolbar_frame, text="Remove Highlighted", command=on_remove_selected).pack(side=tk.LEFT, padx=5)
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
    
    # Right-click context menu for the treeview
    tree_menu = tk.Menu(tree, tearoff=0)
    
    def on_edit_header_context():
        on_edit_header()
        
    def copy_to_clipboard(iids):
        if not iids:
            return
        lines = []
        cols = tree["columns"]
        header = "\t".join([tree.heading(c)["text"] for c in cols if tree.heading(c)["text"]])
        if header:
            lines.append(header)
        for iid in iids:
            values = tree.item(iid, "values")
            lines.append("\t".join([str(v) for v in values]))
        data = "\n".join(lines)
        tree.clipboard_clear()
        tree.clipboard_append(data)

    def copy_selected():
        copy_to_clipboard(tree.selection())

    def copy_all():
        copy_to_clipboard(tree.get_children())

    tree_menu.add_command(label="✏️ Edit FITS Header...", command=on_edit_header_context)
    tree_menu.add_separator()
    tree_menu.add_command(label="Copy Selected Rows (Excel-ready)", command=copy_selected)
    tree_menu.add_command(label="Copy Entire List (Excel-ready)", command=copy_all)

    def show_tree_menu(event):
        item = tree.identify_row(event.y)
        if item and item not in tree.selection():
            tree.selection_set(item)
        tree_menu.post(event.x_root, event.y_root)

    tree.bind("<Button-3>", show_tree_menu)
    
    def open_header_editor(file_path, file_idx):
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
            
        editor_win = tk.Toplevel(root)
        editor_win.title(f"✏️ Edit FITS Header - {os.path.basename(file_path)}")
        editor_win.geometry("750x650")
        editor_win.minsize(650, 550)
        editor_win.transient(root)
        editor_win.grab_set()
        editor_win.configure(bg="#f0f2f5")

        # Center popup over root window
        editor_win.update_idletasks()
        rx = root.winfo_x()
        ry = root.winfo_y()
        rw = root.winfo_width()
        rh = root.winfo_height()
        px = rx + (rw - 750) // 2
        py = ry + (rh - 650) // 2
        editor_win.geometry(f"+{px}+{py}")

        # Top Header Frame
        top_frame = tk.Frame(editor_win, bg="#1a3a5f", pady=12)
        top_frame.pack(fill="x", side=tk.TOP)
        tk.Label(top_frame, text="✏️ FITS Header Editor", font=("Segoe UI", 12, "bold"), fg="white", bg="#1a3a5f").pack()
        tk.Label(top_frame, text=file_path, font=("Segoe UI", 8, "italic"), fg="#cbd5e1", bg="#1a3a5f").pack()

        # Structural keywords definition
        STRUCTURAL_KEYWORDS = {
            'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 
            'BSCALE', 'BZERO', 'PCOUNT', 'GCOUNT', 'TFIELDS', 'XTENSION'
        }
        COMMENTARY_KEYWORDS = {'COMMENT', 'HISTORY', ''}

        # Load cards in memory
        from astropy.io import fits
        all_cards = []
        orig_values = {}  # v4.15: KEY -> (original python value, its string form) to preserve types
        try:
            with fits.open(file_path) as hdul:
                hdr = hdul[0].header
                for card in hdr.cards:
                    if card.keyword:
                        val_str = str(card.value) if card.value is not None else ''
                        comment = card.comment or ''
                        all_cards.append([card.keyword.upper(), val_str, comment])
                        if card.keyword.upper() not in COMMENTARY_KEYWORDS:
                            orig_values[card.keyword.upper()] = (card.value, val_str)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read FITS header: {e}")
            editor_win.destroy()
            return

        # Main body container
        body_frame = tk.Frame(editor_win, bg="#f0f2f5", padx=10, pady=10)
        body_frame.pack(fill="both", expand=True)

        # Search Bar
        search_frame = ttk.Frame(body_frame)
        search_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Label(search_frame, text="🔍 Search Card:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        search_entry.focus_set()

        # Cards Treeview Frame
        table_frame = ttk.Frame(body_frame)
        table_frame.pack(fill="both", expand=True)

        editor_tree = ttk.Treeview(table_frame, columns=("key", "val", "comment"), show="headings", selectmode="browse")
        editor_tree.heading("key", text="Keyword")
        editor_tree.heading("val", text="Value")
        editor_tree.heading("comment", text="Comment")
        
        editor_tree.column("key", width=120, anchor=tk.W)
        editor_tree.column("val", width=200, anchor=tk.W)
        editor_tree.column("comment", width=350, anchor=tk.W)

        # Scrollbars for table
        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=editor_tree.yview)
        editor_tree.configure(yscrollcommand=v_scroll.set)
        h_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=editor_tree.xview)
        editor_tree.configure(xscrollcommand=h_scroll.set)

        editor_tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Style tag for structural keys to show them in light grey
        editor_tree.tag_configure('structural', foreground='#94a3b8', background='#f8fafc')
        editor_tree.tag_configure('even', background='#ffffff')
        editor_tree.tag_configure('odd', background='#f1f5f9')

        # Filter function
        def refresh_tree(event=None):
            for item in editor_tree.get_children():
                editor_tree.delete(item)
            
            query = search_var.get().lower().strip()
            for idx, (key, val, comment) in enumerate(all_cards):
                if not query or query in key.lower() or query in val.lower() or query in comment.lower():
                    tag = 'even' if idx % 2 == 0 else 'odd'
                    if key in STRUCTURAL_KEYWORDS:
                        tag = 'structural'
                    editor_tree.insert('', tk.END, iid=str(idx), values=(key, val, comment), tags=(tag,))

        search_var.trace_add("write", lambda *args: refresh_tree())

        # Inputs Form Frame
        form_frame = ttk.LabelFrame(body_frame, text="Edit / Add Card")
        form_frame.pack(fill="x", pady=10)

        for i in range(3):
            form_frame.columnconfigure(i, weight=1)

        ttk.Label(form_frame, text="Keyword (max 8 chars):", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ttk.Label(form_frame, text="Value:", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(form_frame, text="Comment:", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, sticky="w", padx=10, pady=5)

        key_var = tk.StringVar()
        val_var = tk.StringVar()
        comment_var = tk.StringVar()

        def limit_key(*args):
            s = key_var.get().upper().strip()
            if len(s) > 8:
                s = s[:8]
            key_var.set(s)
        key_var.trace_add("write", limit_key)

        key_entry = ttk.Entry(form_frame, textvariable=key_var)
        val_entry = ttk.Entry(form_frame, textvariable=val_var)
        comment_entry = ttk.Entry(form_frame, textvariable=comment_var)

        key_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        val_entry.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        comment_entry.grid(row=1, column=2, sticky="ew", padx=10, pady=5)

        # Form Buttons
        form_btn_frame = ttk.Frame(form_frame)
        form_btn_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

        def on_select_card(event):
            selected = editor_tree.selection()
            if not selected:
                return
            idx = int(selected[0])
            k, v, c = all_cards[idx]
            key_var.set(k)
            val_var.set(v)
            comment_var.set(c)

        editor_tree.bind("<<TreeviewSelect>>", on_select_card)

        def clear_form():
            editor_tree.selection_remove(editor_tree.selection())
            key_var.set("")
            val_var.set("")
            comment_var.set("")

        def apply_changes():
            key = key_var.get().strip().upper()
            val = val_var.get()
            comment = comment_var.get().strip()

            if not key:
                messagebox.showwarning("Validation Error", "Keyword cannot be empty.")
                return

            if key in STRUCTURAL_KEYWORDS:
                if not messagebox.askyesno("Warning", f"'{key}' is a structural FITS keyword. Modifying it can corrupt the FITS file layout. Are you sure you want to proceed?"):
                    return

            selected = editor_tree.selection()
            if selected:
                idx = int(selected[0])
                all_cards[idx] = [key, val, comment]
            else:
                existing_idx = None
                for idx, (k, _, _) in enumerate(all_cards):
                    if k == key and key not in COMMENTARY_KEYWORDS:
                        existing_idx = idx
                        break
                if existing_idx is not None:
                    all_cards[existing_idx] = [key, val, comment]
                else:
                    all_cards.append([key, val, comment])
            
            refresh_tree()
            clear_form()

        def delete_card():
            selected = editor_tree.selection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select a card to delete.")
                return
            
            idx = int(selected[0])
            key, _, _ = all_cards[idx]
            
            if key in STRUCTURAL_KEYWORDS:
                if not messagebox.askyesno("Warning", f"'{key}' is a structural FITS keyword. Deleting it can corrupt the FITS file layout. Are you sure you want to proceed?"):
                    return

            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the '{key}' card?"):
                all_cards.pop(idx)
                refresh_tree()
                clear_form()

        ttk.Button(form_btn_frame, text="Apply / Add Card", command=apply_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(form_btn_frame, text="Delete Card", command=delete_card).pack(side=tk.LEFT, padx=5)
        ttk.Button(form_btn_frame, text="Clear Fields", command=clear_form).pack(side=tk.LEFT, padx=5)

        # Bottom Window Controls
        bottom_frame = ttk.Frame(editor_win)
        bottom_frame.pack(fill="x", side=tk.BOTTOM, padx=10, pady=10)

        def save_and_close():
            # Automatically apply any active edits before saving
            key = key_var.get().strip().upper()
            if key:
                val = val_var.get()
                comment = comment_var.get().strip()
                selected = editor_tree.selection()
                if selected:
                    idx = int(selected[0])
                    all_cards[idx] = [key, val, comment]
                else:
                    existing_idx = None
                    for idx, (k, _, _) in enumerate(all_cards):
                        if k == key and key not in COMMENTARY_KEYWORDS:
                            existing_idx = idx
                            break
                    if existing_idx is not None:
                        all_cards[existing_idx] = [key, val, comment]
                    else:
                        all_cards.append([key, val, comment])

            def parse_fits_value(val_str):
                val_str_stripped = val_str.strip()
                if not val_str_stripped:
                    return None
                if val_str_stripped.upper() == 'TRUE':
                    return True
                if val_str_stripped.upper() == 'FALSE':
                    return False
                try:
                    if '.' not in val_str_stripped and (val_str_stripped.isdigit() or (val_str_stripped.startswith('-') and val_str_stripped[1:].isdigit())):
                        return int(val_str_stripped)
                except ValueError:
                    pass
                try:
                    return float(val_str_stripped)
                except ValueError:
                    pass
                if (val_str_stripped.startswith("'") and val_str_stripped.endswith("'")) or (val_str_stripped.startswith('"') and val_str_stripped.endswith('"')):
                    return val_str_stripped[1:-1]
                return val_str_stripped

            def value_for_card(key_upper, val_str):
                """v4.15: unchanged cards keep their original value AND type
                (e.g. the string '007' no longer becomes the integer 7);
                edited cards whose original value was a string stay strings;
                non-finite numbers (nan/inf) are stored as strings."""
                if key_upper in orig_values:
                    orig_val, orig_str = orig_values[key_upper]
                    if val_str == orig_str:
                        return orig_val
                    if isinstance(orig_val, str):
                        v = val_str.strip()
                        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                            v = v[1:-1]
                        return v
                parsed = parse_fits_value(val_str)
                if isinstance(parsed, float) and not np.isfinite(parsed):
                    return val_str.strip()
                return parsed

            # v4.15: keep a backup of the original file (created once; never overwritten)
            backup_path = file_path + ".bak"
            try:
                if not os.path.exists(backup_path):
                    shutil.copy2(file_path, backup_path)
            except Exception as e:
                if not messagebox.askyesno("Backup Failed", f"Could not create a backup copy:\n{e}\n\nSave the header anyway?"):
                    return

            try:
                with fits.open(file_path, mode='update') as hdul:
                    hdr = hdul[0].header
                    
                    new_std_keys = {card[0].upper() for card in all_cards if card[0].upper() not in STRUCTURAL_KEYWORDS and card[0].upper() not in COMMENTARY_KEYWORDS}
                    
                    for key in list(hdr.keys()):
                        key_upper = key.upper()
                        if key_upper not in STRUCTURAL_KEYWORDS and key_upper not in COMMENTARY_KEYWORDS:
                            if key_upper not in new_std_keys:
                                try:
                                    del hdr[key]
                                except KeyError:
                                    pass
                                    
                    for key, val_str, comment in all_cards:
                        key_upper = key.upper()
                        if key_upper in COMMENTARY_KEYWORDS:
                            continue
                        hdr.set(key, value_for_card(key_upper, val_str), comment)

                    # v4.15: COMMENT / HISTORY cards are rewritten from the editor list
                    # (previously added/edited/deleted ones were silently ignored)
                    for ck in ('COMMENT', 'HISTORY'):
                        while ck in hdr:
                            del hdr[ck]
                    for key, val_str, comment in all_cards:
                        if key.upper() == 'COMMENT':
                            hdr.add_comment(val_str)
                        elif key.upper() == 'HISTORY':
                            hdr.add_history(val_str)
                    
                    hdul.flush()
                
                loaded_files[file_idx] = scan_fits_header(file_path)
                update_file_table(select_iid=str(file_idx))
                on_tree_select(None)
                
                messagebox.showinfo("Success", f"FITS header saved successfully.\n\nThe original file is kept as:\n{backup_path}")
                editor_win.destroy()
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save FITS header: {e}")

        # Save button styled with accent green
        style = ttk.Style()
        style.configure("Save.TButton", background="#2e7d32", foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Save.TButton", background=[("active", "#1b5e20")])

        ttk.Button(bottom_frame, text="Save to FITS File", style="Save.TButton", command=save_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Cancel", command=editor_win.destroy).pack(side=tk.RIGHT, padx=5)

        refresh_tree()

    # Right side: Header Viewer
    right_side = tk.LabelFrame(paned, text=f"{get_icon('📄', '')} FITS Header Preview".strip())
    paned.add(right_side, width=350)
    
    # Edit button frame at the top of the preview pane
    hdr_btn_frame = ttk.Frame(right_side)
    hdr_btn_frame.pack(side=tk.TOP, fill="x", padx=5, pady=5)
    
    def on_edit_header():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a FITS file in the table first.")
            return
        iid = selected[0]
        file_path = loaded_files[int(iid)]['path']
        open_header_editor(file_path, int(iid))
        
    edit_hdr_btn = ttk.Button(hdr_btn_frame, text="✏️ Edit FITS Header...", command=on_edit_header)
    edit_hdr_btn.pack(fill="x", expand=True)
    
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
                    path = loaded_files[int(iid)]['path'] if int(iid) < len(loaded_files) else None
                    if vals[0] == '[X]':
                        vals[0] = '[ ]'
                        if path: unchecked_paths.add(path)
                    else:
                        vals[0] = '[X]'
                        if path: unchecked_paths.discard(path)
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
    
    sub_osc_scroll = ScrollableFrame(pre_notebook)
    tab_pre_osc = sub_osc_scroll.scrollable_frame
    
    pre_notebook.add(sub_calib_scroll, text=f"{get_icon('🧪', '')}  FITS Calibration".strip())
    pre_notebook.add(sub_solve_scroll, text=f"{get_icon('🧭', '')}  WCS Plate Solving".strip())
    pre_notebook.add(sub_osc_scroll, text=f"{get_icon('🌈', '')}  OSC Channel Split".strip())
    
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

    # --- TAB 5b: Field Variables (v4.19) ---
    tab_fieldvars_outer = ttk.Frame(content_container)
    tab_fieldvars_outer.grid(row=0, column=0, sticky="nsew")
    fv_scroll = ScrollableFrame(tab_fieldvars_outer)
    fv_scroll.pack(fill="both", expand=True)
    tab_fieldvars = fv_scroll.scrollable_frame

    # --- TAB 5c: Period Analysis (v4.21) ---
    tab_period_outer = ttk.Frame(content_container)
    tab_period_outer.grid(row=0, column=0, sticky="nsew")
    period_scroll = ScrollableFrame(tab_period_outer)
    period_scroll.pack(fill="both", expand=True)
    tab_period = period_scroll.scrollable_frame

    # --- TAB 5d: Observation Planner (v4.37) ---
    tab_planner_outer = ttk.Frame(content_container)
    tab_planner_outer.grid(row=0, column=0, sticky="nsew")
    planner_scroll = ScrollableFrame(tab_planner_outer)
    planner_scroll.pack(fill="both", expand=True)
    tab_planner = planner_scroll.scrollable_frame

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
    
    # v4.26: local copies of the catalogues. The user only points to the folder that holds
    # them; the subfolders (ATLAS, Gaia, APASS, VSX, GCVS list ...) are found automatically.
    from photometry import local_catalogs as _lcat
    _lcfg = _lcat.get_config()
    tk.Label(lf_files, text="Local catalogue folder:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
    localcat_dir_var = tk.StringVar(value=_lcfg.get('root') or "")
    vars_dict["local_catalog_dir"] = (localcat_dir_var, str)
    ttk.Entry(lf_files, textvariable=localcat_dir_var, width=64).grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
    localcat_use_var = tk.BooleanVar(value=_lcfg.get('use_local', True))
    vars_dict["local_catalogs_use"] = (localcat_use_var, bool)
    localcat_online_var = tk.BooleanVar(value=_lcfg.get('allow_online', True))
    vars_dict["local_catalogs_online"] = (localcat_online_var, bool)
    localcat_status_var = tk.StringVar(value="")

    def localcat_show():
        cfg = _lcat.get_config()
        if not cfg.get('root'):
            localcat_status_var.set("No local catalogue folder set - all catalogues are queried online (VizieR).")
        else:
            localcat_status_var.set("Found in " + cfg['root'] + ":\n   " + "\n   ".join(_lcat.summary_lines()))

    def localcat_scan(path):
        localcat_status_var.set(f"Scanning {path} ...")
        root.update_idletasks()

        def worker():
            try:
                found = _lcat.set_root(path)
                tables = [c for c, i in found.items() if i.get('layout') == 'table']
                if tables:
                    root.after(0, lambda: localcat_status_var.set(
                        "Indexing table catalogues (only once; large files take a while) ..."))
                    _lcat.build_indexes(progress=lambda c, k, n, p_: root.after(
                        0, lambda c=c, k=k, n=n: localcat_status_var.set(f"Indexing {c}: file {k}/{n} ...")))
                root.after(0, localcat_show)
            except Exception as e:
                root.after(0, lambda err=e: localcat_status_var.set(f"Scan failed: {err}"))
        threading.Thread(target=worker, daemon=True).start()

    def browse_catalog_dir():
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select the folder with the local catalogues (ATLAS, Gaia, APASS ... subfolders)",
                                    initialdir=localcat_dir_var.get() or os.path.expanduser("~"))
        if d:
            localcat_dir_var.set(d)
            localcat_scan(d)

    def localcat_rescan():
        d = localcat_dir_var.get().strip()
        if not d:
            _lcat.set_root(None); localcat_show(); return
        if not os.path.isdir(d):
            messagebox.showerror("Local catalogues", f"Folder not found:\n{d}"); return
        localcat_scan(d)

    def localcat_options(*_):
        try:
            _lcat.set_options(use_local=bool(localcat_use_var.get()), allow_online=bool(localcat_online_var.get()))
        except tk.TclError:
            pass
    localcat_use_var.trace_add("write", localcat_options)
    localcat_online_var.trace_add("write", localcat_options)

    ttk.Button(lf_files, text="Browse Folder...", command=browse_catalog_dir).grid(row=1, column=2, padx=5)
    ttk.Button(lf_files, text="Rescan", command=localcat_rescan).grid(row=1, column=3, padx=5)

    def localcat_overview():
        """v4.28: which catalogue files can be deleted (tools/catalog_overview.py) - read-only."""
        d = localcat_dir_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showinfo("Catalogue Overview", "Choose the local catalogue folder first."); return
        pop = tk.Toplevel(root); pop.title("Catalogue overview - which files can be deleted?"); pop.geometry("950x560")
        top = tk.Frame(pop); top.pack(fill="x", padx=8, pady=6)
        dec_v = tk.StringVar(value="-30"); mag_v = tk.StringVar(value="18")
        tk.Label(top, text="Keep Dec >=").pack(side=tk.LEFT); ttk.Entry(top, textvariable=dec_v, width=6).pack(side=tk.LEFT, padx=(2, 10))
        tk.Label(top, text="and magnitude <=").pack(side=tk.LEFT); ttk.Entry(top, textvariable=mag_v, width=6).pack(side=tk.LEFT, padx=(2, 10))
        txt = scrolledtext.ScrolledText(pop, font=("Consolas", 9)); txt.pack(fill="both", expand=True, padx=8, pady=4)
        tk.Label(pop, text="Nothing is deleted. The full list is written to catalog_overview.csv in the Calibra folder. "
                           "Large Gaia/APASS files are read once and can take a while. For reduced copies of partly useful "
                           "files use the command-line version (tools/catalog_overview.py --reduce, see manual 3.6a).",
                 fg="#555", font=("Arial", 8), wraplength=900, justify=tk.LEFT).pack(anchor="w", padx=8, pady=(0, 6))

        def run():
            try:
                dec_min, mag_max = float(dec_v.get()), float(mag_v.get())
            except ValueError:
                messagebox.showerror("Catalogue Overview", "Limits must be numbers.", parent=pop); return
            txt.delete("1.0", tk.END); txt.insert(tk.END, "Scanning ...\n")
            def emit(*a_, **k_):
                s_ = " ".join(str(x) for x in a_) + "\n"
                root.after(0, lambda s_=s_: (txt.insert(tk.END, s_), txt.see(tk.END)) if txt.winfo_exists() else None)

            def worker():
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
                import catalog_overview as _co
                try:
                    _co.main([d, '--dec-min', str(dec_min), '--mag-max', str(mag_max), '--report', 'catalog_overview.csv'],
                             emit=emit)
                except Exception as e:
                    emit(f"Error: {e}")
            threading.Thread(target=worker, daemon=True).start()
        ttk.Button(top, text="Scan", command=run).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Open Report (CSV)", command=lambda: open_path(os.path.abspath('catalog_overview.csv'))).pack(side=tk.LEFT, padx=4)
    ttk.Button(lf_files, text="Which files can be deleted?...", command=localcat_overview).grid(row=2, column=3, padx=5, sticky=tk.E)
    _lc_opts = tk.Frame(lf_files); _lc_opts.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=10)
    ttk.Checkbutton(_lc_opts, text="Use the local copies", variable=localcat_use_var).pack(side=tk.LEFT, padx=(0, 15))
    ttk.Checkbutton(_lc_opts, text="Go online (VizieR) for catalogues or sky areas that are not available locally",
                    variable=localcat_online_var).pack(side=tk.LEFT)
    # v4.32: APASS DR10 from the AAVSO archive - button and automatic option
    apass_auto_var = tk.BooleanVar(value=_lcfg.get('apass_auto', False))
    vars_dict["apass_dr10_auto"] = (apass_auto_var, bool)
    apass_auto_var.trace_add("write", lambda *_: _lcat.set_options(apass_auto=bool(apass_auto_var.get())))
    _apass_row = tk.Frame(lf_files); _apass_row.grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(0, 4))
    ttk.Checkbutton(_apass_row, text="APASS: download DR10 automatically from the AAVSO archive for fields not "
                                     "available locally (saved for next time; DR9 from VizieR if it fails)",
                    variable=apass_auto_var).pack(side=tk.LEFT)

    def checked_image_paths():
        try:
            return [loaded_files[int(iid)]['path'] for iid in tree.get_children()
                    if tree.item(iid, 'values')[0] == '[X]']
        except Exception:
            return []

    def apass_dr10_dialog(*_):
        """v4.32: download APASS DR10 for the checked images, a star name or RA/Dec."""
        from photometry import apass_dr10 as _ad
        pop = tk.Toplevel(root); pop.title("Download APASS DR10 (AAVSO archive)"); pop.geometry("820x520")
        mode = tk.StringVar(value="images")
        name_v = tk.StringVar(value=""); ra_v = tk.StringVar(value=""); dec_v = tk.StringVar(value="")
        rad_v = tk.StringVar(value="0.5")
        top = tk.Frame(pop); top.pack(fill="x", padx=10, pady=6)
        ttk.Radiobutton(top, text="Field(s) of the checked images (plate-solved; radius from the image size)",
                        variable=mode, value="images").grid(row=0, column=0, columnspan=6, sticky=tk.W)
        ttk.Radiobutton(top, text="Star name:", variable=mode, value="name").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=name_v, width=18).grid(row=1, column=1, sticky=tk.W, padx=4)
        ttk.Radiobutton(top, text="RA / Dec:", variable=mode, value="coords").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=ra_v, width=14).grid(row=2, column=1, sticky=tk.W, padx=4)
        ttk.Entry(top, textvariable=dec_v, width=14).grid(row=2, column=2, sticky=tk.W, padx=4)
        tk.Label(top, text="(hh:mm:ss / ±dd:mm:ss or degrees)", fg="#555").grid(row=2, column=3, sticky=tk.W)
        tk.Label(top, text="Radius [deg] (name, RA/Dec):").grid(row=3, column=0, sticky=tk.W, pady=(4, 0))
        ttk.Entry(top, textvariable=rad_v, width=8).grid(row=3, column=1, sticky=tk.W, padx=4, pady=(4, 0))
        tk.Label(top, text="max. 2; the download is a square of 2 x radius", fg="#555").grid(row=3, column=2, columnspan=3, sticky=tk.W)
        tk.Label(pop, text="Saved unchanged in: " + os.path.abspath(_ad.target_folder()) + "\nEach field is downloaded only "
                           "once. AAVSO asks to use the service with moderation; for large areas contact AAVSO.",
                 fg="#555", font=("Arial", 8), justify=tk.LEFT).pack(anchor="w", padx=10)
        txt = scrolledtext.ScrolledText(pop, font=("Consolas", 9), height=14); txt.pack(fill="both", expand=True, padx=10, pady=6)

        def emit(s_):
            root.after(0, lambda s_=s_: (txt.insert(tk.END, s_ + "\n"), txt.see(tk.END)) if txt.winfo_exists() else None)

        def go():
            m_ = mode.get()
            jobs = []
            try:
                if m_ == "images":
                    paths = checked_image_paths()
                    if not paths:
                        messagebox.showinfo("APASS DR10", "Please check images in the File Manager.", parent=pop); return
                    fields, no_wcs = _ad.fields_from_images(paths)
                    if no_wcs:
                        emit(f"{len(no_wcs)} image(s) without WCS skipped (plate-solve them first).")
                    jobs = [(ra_, de_, r_) for ra_, de_, r_, n_ in fields]
                    for ra_, de_, r_, n_ in fields:
                        emit(f"Field: RA {ra_:.4f}, Dec {de_:+.4f}, radius {r_:g} deg ({n_} image(s))")
                else:
                    r_ = float(rad_v.get())
                    if m_ == "name":
                        c_ = _resolve_name(name_v.get().strip())
                    else:
                        from astropy.coordinates import SkyCoord
                        import astropy.units as u
                        import re
                        rs, ds = ra_v.get().strip(), dec_v.get().strip()
                        c_ = (SkyCoord(float(rs) * u.deg, float(ds) * u.deg)
                              if re.fullmatch(r'[+-]?\d+(\.\d*)?', rs) else SkyCoord(rs, ds, unit=(u.hourangle, u.deg)))
                    jobs = [(float(c_.ra.deg), float(c_.dec.deg), r_)]
            except Exception as e:
                messagebox.showerror("APASS DR10", f"Position could not be determined: {e}", parent=pop); return
            if not jobs:
                emit("Nothing to download."); return
            go_btn.config(state=tk.DISABLED)

            def worker():
                import time as _t
                for k, (ra_, de_, r_) in enumerate(jobs):
                    if k:
                        _t.sleep(_ad.MIN_INTERVAL)
                    try:
                        path_, n_, msg_ = _ad.download_field(ra_, de_, r_, log=emit)
                        if n_ is None:
                            emit(msg_)
                    except Exception as e:
                        emit(str(e))
                root.after(0, lambda: (go_btn.config(state=tk.NORMAL) if go_btn.winfo_exists() else None, localcat_show()))
            threading.Thread(target=worker, daemon=True).start()
        bot = tk.Frame(pop); bot.pack(fill="x", padx=10, pady=(0, 8))
        go_btn = ttk.Button(bot, text="Download", command=go); go_btn.pack(side=tk.LEFT)
        ttk.Button(bot, text="Close", command=pop.destroy).pack(side=tk.RIGHT)
    vars_dict['apass_dr10_dialog'] = apass_dr10_dialog
    ttk.Button(_apass_row, text="Download APASS DR10...", command=apass_dr10_dialog).pack(side=tk.LEFT, padx=10)
    tk.Label(lf_files, textvariable=localcat_status_var, justify=tk.LEFT, fg="#1a3a5f",
             font=("Consolas", 8)).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(2, 6))
    vars_dict['localcat_after_load'] = lambda: (
        localcat_scan(localcat_dir_var.get().strip())
        if localcat_dir_var.get().strip() and os.path.abspath(localcat_dir_var.get().strip()) != (_lcat.get_config().get('root') or '')
        else localcat_show())
    localcat_show()

    # Filter Keyword Mapping
    lf_map = tk.LabelFrame(sub_astro, text="Filter Keyword Mapping")
    lf_map.pack(fill="x", padx=10, pady=10)
    tk.Label(lf_map, text="Define keywords in your FITS headers that identify the B, V and L filters.", 
              fg="#555", font=("Arial", 8, "italic")).grid(row=0, column=0, columnspan=4, padx=10, pady=(0, 10))
    add_entry(lf_map, "B-Filter Keyword:", "filter_b_keyword", "Bmag", 1, col_offset=0, vtype=str)
    add_entry(lf_map, "V-Filter Keyword:", "filter_v_keyword", "Vmag", 1, col_offset=1, vtype=str)
    add_entry(lf_map, "L-Filter Keyword:", "filter_l_keyword", "L", 2, col_offset=0, vtype=str)
    tk.Label(lf_map, text="L (luminance) is used for light curves (Light Curves, Field Variables): magnitudes on the V scale "
                          "of the comparison stars; 'L', 'Lum', 'Luminance' are recognised automatically.",
             fg="#555", font=("Arial", 8, "italic"), wraplength=600, justify=tk.LEFT).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=10)

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
    
    # 1. Calibration (Bias & Flats) - v4.27: master frames built by Calibra, automatic
    # selection of the matching masters, calibration on the fly or into files
    from photometry import master_frames as _mf
    _mcfg = _mf.get_config()

    lf_mast = tk.LabelFrame(tab_pre, text="Master Frames (bias, flats, optional darks)")
    lf_mast.pack(fill="x", padx=10, pady=(10, 4))
    # v4.29: one folder per type - every FITS file in the bias folder is a bias, etc.; the masters
    # are written into the same folder. The mixed folder (type from header / names) is still possible.
    calbias_var = tk.StringVar(value=_mcfg.get('bias_folder', ''))
    calflat_var = tk.StringVar(value=_mcfg.get('flat_folder', ''))
    caldark_var = tk.StringVar(value=_mcfg.get('dark_folder', ''))
    calfolder_var = tk.StringVar(value=_mcfg.get('folder', ''))
    vars_dict["cal_bias_folder"] = (calbias_var, str)
    vars_dict["cal_flat_folder"] = (calflat_var, str)
    vars_dict["cal_dark_folder"] = (caldark_var, str)
    vars_dict["cal_frames_folder"] = (calfolder_var, str)

    def _cal_folder_row(r, label, var, button, title):
        tk.Label(lf_mast, text=label).grid(row=r, column=0, sticky=tk.W, padx=10, pady=2)
        ttk.Entry(lf_mast, textvariable=var, width=60).grid(row=r, column=1, columnspan=3, sticky=tk.W, padx=10)

        def browse():
            from tkinter import filedialog
            d = filedialog.askdirectory(title=title, initialdir=var.get() or calbias_var.get() or os.path.expanduser("~"))
            if d:
                var.set(d)
                cal_sync(); cal_show_masters()
        ttk.Button(lf_mast, text=button, command=browse).grid(row=r, column=4, padx=5, sticky="ew")
    _cal_folder_row(0, "Bias folder:", calbias_var, "Browse Bias Folder...",
                    "Folder with bias frames (every FITS file in it is used as a bias)")
    _cal_folder_row(1, "Flat folder:", calflat_var, "Browse Flat Folder...",
                    "Folder with flat frames (every FITS file in it is a flat; filter from header or name)")
    _cal_folder_row(2, "Dark folder (optional):", caldark_var, "Browse Dark Folder...",
                    "Folder with dark frames (only needed if you use darks)")
    _cal_folder_row(3, "or mixed folder:", calfolder_var, "Browse Calibration Folder...",
                    "Folder with bias / flat (/ dark) frames mixed - type from FITS header or file names")
    tk.Label(lf_mast, text="Bias/flat/dark folder: every FITS file in it is taken as that type, names do not matter. "
                           "Mixed folder: type from the FITS header (IMAGETYP) or the file names. "
                           "Masters are written into the same folder.",
             fg="#555", font=("Arial", 8, "italic"), justify=tk.LEFT, wraplength=760).grid(
        row=4, column=0, columnspan=5, sticky=tk.W, padx=10)
    cal_method_var = tk.StringVar(value="sigma-clipped mean")
    vars_dict["cal_combine_method"] = (cal_method_var, str)
    cal_minframes_var = tk.IntVar(value=3)
    vars_dict["cal_min_frames"] = (cal_minframes_var, int)
    tk.Label(lf_mast, text="Combine:").grid(row=5, column=0, sticky=tk.W, padx=10)
    ttk.Combobox(lf_mast, textvariable=cal_method_var, values=["sigma-clipped mean", "median"], width=18,
                 state="readonly").grid(row=5, column=1, sticky=tk.W, padx=10)
    tk.Label(lf_mast, text="min. frames:").grid(row=5, column=2, sticky=tk.E)
    ttk.Spinbox(lf_mast, from_=2, to=500, textvariable=cal_minframes_var, width=5).grid(row=5, column=3, sticky=tk.W, padx=4)
    cal_build_btn = ttk.Button(lf_mast, text="Build / Update Masters")
    cal_build_btn.grid(row=5, column=4, padx=5, pady=4)
    cal_mast_status = tk.StringVar(value="")
    tk.Label(lf_mast, textvariable=cal_mast_status, fg="#1a3a5f", justify=tk.LEFT, wraplength=760,
             font=("Arial", 9)).grid(row=6, column=0, columnspan=5, sticky=tk.W, padx=10)
    cal_tree = ttk.Treeview(lf_mast, columns=("type", "filter", "size", "gain", "exp", "n", "file"), show="headings", height=5)
    for c_, t_, w_ in (("type", "Type", 70), ("filter", "Filter", 60), ("size", "Size / bin", 110), ("gain", "Gain / offset", 90),
                       ("exp", "Exp / temp", 90), ("n", "N", 40), ("file", "File", 330)):
        cal_tree.heading(c_, text=t_); cal_tree.column(c_, width=w_, anchor="w" if c_ == "file" else "center")
    cal_tree.grid(row=7, column=0, columnspan=5, sticky="ew", padx=10, pady=(2, 6))

    def _cal_folders():
        return {k: v.get().strip() for k, v in (('bias_folder', calbias_var), ('flat_folder', calflat_var),
                                                ('dark_folder', caldark_var), ('folder', calfolder_var))}

    def cal_show_masters():
        cal_tree.delete(*cal_tree.get_children())
        dirs = [d for d in _cal_folders().values() if d and os.path.isdir(d)]
        if not dirs:
            cal_mast_status.set("Choose the folder with your bias frames and the folder with your flat frames. "
                                "'Build / Update Masters' creates the master frames in these folders.")
            return
        try:
            folder = os.path.commonpath([os.path.abspath(d) for d in dirs]) if len(dirs) > 1 else dirs[0]
        except ValueError:                       # e.g. different drives on Windows
            folder = None
        ms = _mf.find_masters(*dirs)
        from astropy.io import fits as _fits
        for m in ms:
            g_ = "" if m['gain'] is None else f"{m['gain']:g}"
            o_ = "" if m['offset'] is None else f"{m['offset']:g}"
            et = ""
            if m['mtype'] in ('DARK', 'DARKFLAT'):
                et = f"{m['exptime']:g} s" + (f" / {m['temp']:g} C" if m['temp'] is not None else "")
            try:
                n_ = _fits.getheader(m['path']).get('NCOMBINE', '')
            except Exception:
                n_ = ''
            cal_tree.insert("", "end", values=(m['mtype'], m['filter'] if m['mtype'] == 'FLAT' else "",
                                               f"{m['shape'][1]}x{m['shape'][0]} / {m['xbin']}", f"{g_} / {o_}", et, n_,
                                               os.path.relpath(m['path'], folder) if folder else m['path']))
        where = folder if len(dirs) == 1 else "the calibration folders"
        cal_mast_status.set(f"{len(ms)} master frame(s) in {where}." if ms else
                            f"No master frames yet in {where} - press 'Build / Update Masters'.")

    def on_build_masters():
        dirs = _cal_folders()
        bad = [d for d in dirs.values() if d and not os.path.isdir(d)]
        if bad:
            messagebox.showerror("Master Frames", "Folder not found:\n" + "\n".join(bad)); return
        if not any(dirs.values()):
            messagebox.showerror("Master Frames", "Please choose the bias folder and the flat folder "
                                                  "(or one mixed calibration folder)."); return
        method = 'median' if cal_method_var.get().startswith('median') else 'sigma'
        try:
            minf = max(2, int(cal_minframes_var.get()))
        except (tk.TclError, ValueError):
            minf = 3
        cal_build_btn.config(state=tk.DISABLED)
        cal_mast_status.set("Scanning calibration frames ...")

        def worker():
            try:
                mistyped = []

                def _log(msg):
                    print(msg)
                    if msg.lstrip().startswith("WARNING"):
                        mistyped.append(msg.strip())
                res = _mf.build_masters(dirs['folder'] or None, method=method, min_frames=minf,
                                        bias_folder=dirs['bias_folder'] or None, flat_folder=dirs['flat_folder'] or None,
                                        dark_folder=dirs['dark_folder'] or None,
                                        progress=lambda x, name: root.after(0, lambda: cal_mast_status.set(
                                            f"Building {name} ... {100 * x:.0f} %")),
                                        log=_log)
                nb = sum(r['status'] == 'built' for r in res); nr = sum(r['status'] == 'reused' for r in res)
                ns = [r for r in res if r['status'] == 'skipped']
                warn = [r for r in res if r['note'].startswith('WARNING')]
                msg = f"Masters: {nb} built, {nr} up to date" + (f", {len(ns)} group(s) skipped (too few frames)" if ns else "")
                if warn:
                    msg += "\n" + "\n".join(f"{os.path.basename(r['path'])}: {r['note']}" for r in warn)
                if mistyped:
                    msg += (f"\nWarning: {len(mistyped)} file(s) have another frame type in the header (e.g. Light) - "
                            "used anyway as the type of their folder (details in the console).")
                if not res:
                    msg = ("No raw bias/flat/dark frames found (bias/flat/dark folders: all FITS files; mixed folder: "
                           "type from IMAGETYP, or 'bias'/'flat'/'dark' in file or folder names).")
                def done():
                    cal_sync(); cal_show_masters()
                    cal_mast_status.set(msg)
                root.after(0, done)
            except Exception as e:
                root.after(0, lambda err=e: (cal_mast_status.set(f"Error: {err}"),
                                             messagebox.showerror("Master Frames", str(err))))
            finally:
                root.after(0, lambda: cal_build_btn.config(state=tk.NORMAL))
        threading.Thread(target=worker, daemon=True).start()
    cal_build_btn.config(command=on_build_masters)

    lf_calib = tk.LabelFrame(tab_pre, text="Calibration of Light Frames")
    lf_calib.pack(fill="x", padx=10, pady=4)
    cal_mode_var = tk.StringVar(value=_mcfg.get('mode', 'auto'))
    vars_dict["cal_mode"] = (cal_mode_var, str)
    ttk.Radiobutton(lf_calib, text="Automatic: matching masters from the calibration folders (filter, size, binning, gain, offset)",
                    variable=cal_mode_var, value="auto").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(4, 0))
    ttk.Radiobutton(lf_calib, text="Manual master files:", variable=cal_mode_var, value="manual").grid(
        row=1, column=0, columnspan=3, sticky=tk.W, padx=10)
    lf_man = tk.Frame(lf_calib); lf_man.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=20)
    add_file_selector(lf_man, "Master Bias:", "bias_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\Master_Bias_1x1_gain_0.fits", 0, initial_dir="bias_and_flats")
    add_file_selector(lf_man, "Master Flat (V):", "flat_v_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\FLAT_Vmag_1x1_gain_0.fits", 1, initial_dir="bias_and_flats")
    add_file_selector(lf_man, "Master Flat (B):", "flat_b_path", r"C:\Astro\Photometry_Calibra\bias_and_flats\FLAT_Bmag_1x1_gain_0.fits", 2, initial_dir="bias_and_flats")
    add_file_selector(lf_man, "Master Flat (L):", "flat_l_path", "", 3, initial_dir="bias_and_flats")
    add_file_selector(lf_man, "Master Dark (optional):", "dark_path", "", 4, initial_dir="bias_and_flats")
    # v4.15: option to subtract the bias from the master flat before normalising.
    flat_bias_sub_var = tk.BooleanVar(value=True)
    vars_dict["flat_already_bias_subtracted"] = (flat_bias_sub_var, bool)
    ttk.Checkbutton(lf_man, text="Master flats from other software (e.g. Siril) are already bias/dark-subtracted (Calibra's own flats always are)",
                    variable=flat_bias_sub_var).grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(0, 5))
    cal_darks_var = tk.BooleanVar(value=_mcfg.get('use_darks', False))
    vars_dict["cal_use_darks"] = (cal_darks_var, bool)
    ttk.Checkbutton(lf_calib, text="Use darks (master dark of the same exposure/temperature, or scaled) - otherwise bias only",
                    variable=cal_darks_var).grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=10)
    cal_fly_var = tk.BooleanVar(value=_mcfg.get('on_the_fly', False))
    vars_dict["cal_on_the_fly"] = (cal_fly_var, bool)
    ttk.Checkbutton(lf_calib, text="Calibrate on the fly when measuring (Light Curves, Field Variables, Data Mining, Analysis) "
                                   "- no calibrated files needed", variable=cal_fly_var).grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=10)
    cal_info_var = tk.StringVar(value="")
    tk.Label(lf_calib, textvariable=cal_info_var, fg="#555", font=("Arial", 8, "italic"), justify=tk.LEFT,
             wraplength=760).grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(0, 6))

    def cal_sync(*_):
        """Push the GUI settings into photometry.master_frames (also saved for the pipeline)."""
        try:
            sat = float(vars_dict["saturation_limit"][0].get()) if "saturation_limit" in vars_dict else 63000.0
        except (tk.TclError, ValueError):
            sat = 63000.0
        try:
            _mf.configure(on_the_fly=bool(cal_fly_var.get()), mode=cal_mode_var.get(), **_cal_folders(),
                          use_darks=bool(cal_darks_var.get()), flat_already_sub=bool(flat_bias_sub_var.get()), sat_limit=sat,
                          manual={'bias': vars_dict["bias_path"][0].get(), 'dark': vars_dict["dark_path"][0].get(),
                                  'flat_V': vars_dict["flat_v_path"][0].get(), 'flat_B': vars_dict["flat_b_path"][0].get(),
                                  'flat_L': vars_dict["flat_l_path"][0].get()})
        except tk.TclError:
            return
        cal_info_var.set("Calibration on the fly is ON: every measured frame is corrected in memory with the selected masters "
                         "(already calibrated frames are left as they are). Raw files stay unchanged."
                         if cal_fly_var.get() else
                         "Calibration on the fly is OFF: frames are measured as loaded (raw, or cal_ files written below).")
    for _v in (cal_fly_var, cal_mode_var, cal_darks_var, flat_bias_sub_var, calfolder_var, calbias_var, calflat_var, caldark_var):
        _v.trace_add("write", cal_sync)
    for _k in ("bias_path", "dark_path", "flat_v_path", "flat_b_path", "flat_l_path"):
        vars_dict[_k][0].trace_add("write", cal_sync)
    vars_dict['cal_after_load'] = lambda: (cal_sync(), cal_show_masters())

    def on_run_calibration():
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager.")
            return
        cal_sync()
        jobs = [(int(iid), loaded_files[int(iid)]['path']) for iid in selected_iids]

        def cal_thread():
            try:
                print(f"\n--- Calibration of {len(jobs)} frame(s) (written to 'calibrated' next to the raw files) ---")
                ok = 0
                for idx, (iid_int, orig_path) in enumerate(jobs):
                    out, msg = _mf.calibrate_to_file(orig_path)
                    print(f"[{idx + 1}/{len(jobs)}] {os.path.basename(orig_path)}: {msg}")
                    if out and os.path.exists(out):
                        ok += 1
                        new_info = scan_fits_header(out)
                        def _replace(orig=orig_path, info=new_info):
                            for k, f in enumerate(loaded_files):
                                if f['path'] == orig:
                                    loaded_files[k] = info
                                    break
                            update_file_table()
                        root.after(0, _replace)
                msg = f"Calibration complete.\n{ok} of {len(jobs)} files calibrated (or already up to date).\nDetails in the console."
                print(msg)
                root.after(0, lambda: messagebox.showinfo("Batch Complete", msg))
            except Exception as e:
                print(f"Calibration error: {e}")
                root.after(0, lambda err=e: messagebox.showerror("Error", f"Calibration failed: {err}"))
        threading.Thread(target=cal_thread, daemon=True).start()

    def on_check_calibration():
        """Show which masters would be used for the checked frames."""
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager."); return
        cal_sync()
        from astropy.io import fits
        lines = []
        for iid in selected_iids[:40]:
            p_ = loaded_files[int(iid)]['path']
            try:
                h_ = fits.getheader(p_)
                why = _mf.skip_reason(h_, p_)
                if why:
                    lines.append(f"{os.path.basename(p_)}: not calibrated - {why}"); continue
                b_, d_, f_, notes = _mf._CAL.choose(_mf.frame_info(p_, h_))
                lines.append(f"{os.path.basename(p_)}:  bias={os.path.basename(b_) if b_ else '-'}"
                             + (f"  dark={os.path.basename(d_[0])}{' (scaled)' if d_[1] else ''}" if d_ else "")
                             + f"  flat={os.path.basename(f_) if f_ else '-'}"
                             + (("   [" + "; ".join(notes) + "]") if notes else "")
                             + (f"   NOTE: {_mf.stack_note(h_)}" if _mf.stack_note(h_) else ""))
            except Exception as e:
                lines.append(f"{os.path.basename(p_)}: {e}")
        pop = tk.Toplevel(root); pop.title("Masters for the checked frames"); pop.geometry("1000x400")
        txt = scrolledtext.ScrolledText(pop, font=("Consolas", 9)); txt.pack(fill="both", expand=True)
        txt.insert("1.0", "\n".join(lines) + ("\n..." if len(selected_iids) > 40 else ""))
        txt.config(state=tk.DISABLED)

    cal_btn_frame = ttk.Frame(tab_pre)
    cal_btn_frame.pack(pady=(10, 20))
    
    ttk.Button(cal_btn_frame, text="Show Masters for Checked Frames", command=on_check_calibration).pack(side=tk.LEFT, padx=5)
    run_cal_btn = tk.Button(cal_btn_frame, text="Write Calibrated Files for Checked Frames", command=on_run_calibration,
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
    add_entry(lf_plate_settings, "Field height (deg, 0 = auto):", "plate_fov", 0.0, 3, col_offset=0)

    # v4.22: position guess for images without (usable) coordinates in the header
    lf_plate_guess = tk.LabelFrame(tab_pre_right, text="Position Guess")
    lf_plate_guess.pack(fill="x", padx=10, pady=(0, 10))
    plate_guess_mode = tk.StringVar(value="header")
    vars_dict["plate_guess_mode"] = (plate_guess_mode, str)
    plate_obj_var = tk.StringVar(value="")
    vars_dict["plate_guess_object"] = (plate_obj_var, str)
    plate_ra_var = tk.StringVar(value="")
    vars_dict["plate_guess_ra"] = (plate_ra_var, str)
    plate_dec_var = tk.StringVar(value="")
    vars_dict["plate_guess_dec"] = (plate_dec_var, str)
    plate_chain_var = tk.BooleanVar(value=True)
    vars_dict["plate_chain"] = (plate_chain_var, bool)
    plate_blind_var = tk.BooleanVar(value=True)
    vars_dict["plate_allow_blind"] = (plate_blind_var, bool)
    plate_obj_coord_var = tk.StringVar(value="")
    plate_obj_resolved = {}

    ttk.Radiobutton(lf_plate_guess, text="From the FITS header (blind solve if there is none)",
                    variable=plate_guess_mode, value="header").grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=10, pady=2)
    ttk.Radiobutton(lf_plate_guess, text="Object name:", variable=plate_guess_mode, value="name").grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
    ttk.Entry(lf_plate_guess, textvariable=plate_obj_var, width=18).grid(row=1, column=1, sticky=tk.W, padx=2)

    def plate_lookup_name(name):
        """SIMBAD, then AAVSO VSX. Returns (ra, dec) in degrees; raises on failure."""
        key = name.strip().lower()
        if key in plate_obj_resolved:
            return plate_obj_resolved[key]
        from astropy.coordinates import SkyCoord as _SC
        try:
            c = _resolve_name(name)
            pos = (float(c.ra.deg), float(c.dec.deg))
        except Exception:
            from photometry.calibration import resolve_auid
            r = resolve_auid(name, verbose=False)
            if not r:
                raise ValueError(f"'{name}' could not be resolved (local GCVS list, SIMBAD, VSX).")
            pos = (float(r['ra']), float(r['dec']))
        plate_obj_resolved[key] = pos
        return pos

    def on_plate_lookup():
        name = plate_obj_var.get().strip()
        if not name:
            return
        plate_guess_mode.set("name")
        plate_obj_coord_var.set("looking up...")
        def worker():
            try:
                ra_, dec_ = plate_lookup_name(name)
                c = SkyCoord(ra_ * u.deg, dec_ * u.deg)
                txt = f"{c.ra.to_string(unit=u.hour, sep=':', precision=1, pad=True)}  {c.dec.to_string(sep=':', precision=0, pad=True, alwayssign=True)}"
            except Exception as ex:
                txt = f"not found: {ex}"
            root.after(0, lambda: plate_obj_coord_var.set(txt))
        threading.Thread(target=worker, daemon=True).start()
    ttk.Button(lf_plate_guess, text="Look up", command=on_plate_lookup).grid(row=1, column=2, sticky=tk.W, padx=2)
    tk.Label(lf_plate_guess, textvariable=plate_obj_coord_var, font=("Arial", 8, "italic")).grid(row=1, column=3, sticky=tk.W, padx=6)
    ttk.Radiobutton(lf_plate_guess, text="RA / Dec:", variable=plate_guess_mode, value="manual").grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
    ttk.Entry(lf_plate_guess, textvariable=plate_ra_var, width=18).grid(row=2, column=1, sticky=tk.W, padx=2)
    ttk.Entry(lf_plate_guess, textvariable=plate_dec_var, width=14).grid(row=2, column=2, sticky=tk.W, padx=2)
    tk.Label(lf_plate_guess, text="e.g. 09:36:53 +44:04:00 or 144.22 44.07 (deg)",
             font=("Arial", 8, "italic")).grid(row=2, column=3, sticky=tk.W, padx=6)
    ttk.Checkbutton(lf_plate_guess, text="Use the solution of each image as the guess for the next image of the series",
                    variable=plate_chain_var).grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=10, pady=2)
    ttk.Checkbutton(lf_plate_guess, text="Allow blind solving (whole sky, slow) if all guesses fail",
                    variable=plate_blind_var).grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=10, pady=(0, 6))

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
        try:
            fov = float(vars_dict["plate_fov"][0].get() or 0)
        except (tk.TclError, ValueError):
            fov = 0.0
        # v4.22: position guess
        mode = plate_guess_mode.get()
        chain = bool(plate_chain_var.get())
        allow_blind = bool(plate_blind_var.get())
        guess = None
        if mode == "manual":
            try:
                from photometry.field_variables import parse_coordinates
                guess = parse_coordinates(plate_ra_var.get(), plate_dec_var.get())
            except Exception as ex:
                messagebox.showerror("Position Guess", f"RA/Dec could not be read:\n'{plate_ra_var.get()}'  "
                                     f"'{plate_dec_var.get()}'\n({ex})\n\nUse e.g. 09:36:53 +44:04:00 or 144.22 44.07.")
                return
        obj_name = plate_obj_var.get().strip() if mode == "name" else ""
        if mode == "name" and not obj_name:
            messagebox.showerror("Position Guess", "Please enter an object name (or choose another option).")
            return
        
        plate_status_var.set(f"Solving {len(files_to_solve)} files...")
        run_plate_btn.config(state=tk.DISABLED, text="Solving...")
        
        def plate_thread():
            try:
                from photometry.plate_solve import solve_series
                print(f"\n--- Starting Batch Plate Solve ---")
                print(f"Target files: {len(files_to_solve)}")
                print(f"Suffix: {suffix}")
                g = guess
                if obj_name:
                    try:
                        g = plate_lookup_name(obj_name)
                        print(f"Position guess from '{obj_name}': RA={g[0]:.4f}, Dec={g[1]:+.4f} deg")
                    except Exception as ex:
                        print(f"Object name not resolved ({ex}) - using the header / blind solve instead.")
                        g = None
                
                # copy files first (same order as the File Manager)
                todo = []
                for idx, iid in enumerate(files_to_solve):
                    iid_int = int(iid)
                    orig_path = loaded_files[iid_int]['path']
                    base, ext = os.path.splitext(orig_path)
                    new_filename = f"{base}_{suffix}{ext}" if suffix else orig_path
                    if new_filename != orig_path:
                        print(f"[{idx+1}/{len(files_to_solve)}] Copying {orig_path} to {new_filename}...")
                        shutil.copy2(orig_path, new_filename)
                    todo.append((iid_int, new_filename))

                counts = {'ok': 0}
                def on_result(i, path, res, used):
                    iid_int = todo[i][0]
                    if res is not None:
                        counts['ok'] += 1
                        print(f"[{i+1}/{len(todo)}] Solved (guess: {used}).")
                        loaded_files[iid_int] = scan_fits_header(path)
                        root.after(0, update_file_table)
                    else:
                        print(f"[{i+1}/{len(todo)}] Solve failed for {path}")
                    root.after(0, lambda k=i + 1, n=counts['ok']: plate_status_var.set(
                        f"Solving... {k}/{len(todo)} done, {n} solved"))

                solve_series([p for _, p in todo], astap_exe=exe, search_radius=radius, annotate=annotate,
                             ra_hint_deg=g[0] if g else None, dec_hint_deg=g[1] if g else None,
                             chain=chain, fov=fov, allow_blind=allow_blind, on_result=on_result)
                solved_count = counts['ok']
                
                msg = f"Plate solve complete.\n{solved_count} of {len(files_to_solve)} files were successfully solved."
                print(f"\n{msg}")
                root.after(0, lambda: messagebox.showinfo("Batch Complete", msg))
                root.after(0, lambda: plate_status_var.set("Complete."))
            except Exception as e:
                print(f"Plate solve error: {e}")
                root.after(0, lambda err=e: messagebox.showerror("Error", f"Plate solve failed: {err}"))
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


    # 3. OSC Channel Splitter Tab CONTENT
    lf_osc = tk.LabelFrame(tab_pre_osc, text="One-Shot Color (OSC) Demultiplexing")
    lf_osc.pack(fill="x", padx=10, pady=10)
    
    tk.Label(lf_osc, text="Extract monochrome Red, Green, and/or Blue FITS channels from OSC Bayered files.\nSuperpixel mode reduces resolution by 2x to avoid interpolation artifacts.", justify=tk.LEFT).pack(padx=10, pady=10)

    lf_osc_settings = tk.LabelFrame(tab_pre_osc, text="Splitter Settings")
    lf_osc_settings.pack(fill="x", padx=10, pady=10)
    
    add_dropdown(lf_osc_settings, "Bayer Pattern:", "osc_pattern", ["RGGB", "BGGR", "GRBG", "GBRG"], "RGGB", 0, col_offset=0)
    add_dropdown(lf_osc_settings, "Extraction Method:", "osc_method", ["Superpixel (2x2 -> 1px)", "Bilinear (Interpolated)"], "Superpixel (2x2 -> 1px)", 0, col_offset=1)
    
    add_check(lf_osc_settings, "Extract Red Channel", "osc_chan_r", True, 1, col_offset=0)
    add_check(lf_osc_settings, "Extract Green Channel", "osc_chan_g", True, 1, col_offset=1)
    add_check(lf_osc_settings, "Extract Blue Channel", "osc_chan_b", True, 2, col_offset=0)

    osc_status_var = tk.StringVar(value="Ready")
    osc_status_label = SelectableLabel(tab_pre_osc, textvariable=osc_status_var, font=("Arial", 9, "italic"), justify=tk.CENTER)
    osc_status_label.pack(pady=5, fill="x")

    def show_osc_info():
        info_text = (
            "One-Shot Color (OSC) Channel Splitter\n\n"
            "This tool extracts the individual Red, Green, and Blue channels from your color raw FITS images, "
            "converting them into separate monochrome FITS files that Calibra can analyze.\n\n"
            "Methods:\n"
            "1. Superpixel (2x2 -> 1px) [Highly Recommended for Photometry]:\n"
            "   Combines a 2x2 grid of pixels (R, G, G, B) into a single pixel at half resolution. "
            "   This avoids any pixel interpolation or color 'bleeding' between channels, maintaining pure "
            "   physical measurements and reducing noise (especially in Green). It scales the WCS header automatically.\n\n"
            "2. Bilinear (Interpolated):\n"
            "   Estimates missing color data at every pixel using standard bilinear interpolation. "
            "   Keeps the original image resolution, but correlates noise and causes color bleeding near high-contrast stars.\n\n"
            "Settings Alignment:\n"
            "Once split, you will see new files in your File Manager (e.g. ending in _SP_G.fits). "
            "Go to the 'Settings' tab and map the B-Filter and V-Filter keywords to 'B' and 'G' respectively "
            "to run calibration on them."
        )
        messagebox.showinfo("About OSC Channel Splitter", info_text)

    def on_run_osc_split():
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showwarning("No Selection", "Please check at least one FITS file in the File Manager.")
            return
            
        bayer_pat = vars_dict["osc_pattern"][0].get()
        method_str = vars_dict["osc_method"][0].get()
        method = "superpixel" if "Superpixel" in method_str else "bilinear"
        
        chans = []
        if vars_dict["osc_chan_r"][0].get(): chans.append('R')
        if vars_dict["osc_chan_g"][0].get(): chans.append('G')
        if vars_dict["osc_chan_b"][0].get(): chans.append('B')
        
        if not chans:
            messagebox.showwarning("No Channels", "Please check at least one color channel (Red, Green, or Blue) to extract.")
            return
            
        osc_status_var.set(f"Splitting {len(selected_iids)} OSC files...")
        run_osc_btn.config(state=tk.DISABLED, text="Splitting...")
        
        def osc_thread():
            try:
                from photometry.osc_splitter import split_osc_fits
                print(f"\n--- Starting Batch OSC Channel Split ---")
                print(f"Target files: {len(selected_iids)}")
                print(f"Pattern: {bayer_pat} | Method: {method} | Channels: {chans}")
                
                success_count = 0
                new_files_added = 0
                
                for idx, iid in enumerate(selected_iids):
                    iid_int = int(iid)
                    orig_path = loaded_files[iid_int]['path']
                    print(f"[{idx+1}/{len(selected_iids)}] Splitting {os.path.basename(orig_path)}...")
                    
                    try:
                        out_dir = os.path.dirname(orig_path)
                        split_paths = split_osc_fits(orig_path, output_dir=out_dir, pattern=bayer_pat, method=method, channels=chans)
                        
                        existing_paths = {f['path'] for f in loaded_files}
                        for sp in split_paths:
                            if sp not in existing_paths:
                                loaded_files.append(scan_fits_header(sp))
                                new_files_added += 1
                        success_count += 1
                    except Exception as fe:
                        print(f"[{idx+1}/{len(selected_iids)}] Error splitting {os.path.basename(orig_path)}: {fe}")
                
                msg = f"OSC Channel Split complete.\nSuccessfully split {success_count} of {len(selected_iids)} files.\nImported {new_files_added} new monochrome frames into File Manager."
                print(f"\n{msg}")
                root.after(0, lambda: messagebox.showinfo("Batch Complete", msg))
                root.after(0, lambda: osc_status_var.set("Complete."))
                root.after(0, update_file_table)
            except Exception as e:
                print(f"OSC Split error: {e}")
                root.after(0, lambda err=e: messagebox.showerror("Error", f"OSC Split failed: {err}"))
                root.after(0, lambda: osc_status_var.set("Error occurred."))
            finally:
                root.after(0, lambda: run_osc_btn.config(state=tk.NORMAL, text="Run OSC Channel Split on Selected"))
                
        threading.Thread(target=osc_thread, daemon=True).start()

    osc_btn_frame = ttk.Frame(tab_pre_osc)
    osc_btn_frame.pack(pady=20)
    
    run_osc_btn = tk.Button(osc_btn_frame, text="Run OSC Channel Split on Selected", command=on_run_osc_split,
                             bg="#8e24aa", fg="white", font=("Arial", 11, "bold"), pady=10, width=35)
    run_osc_btn.pack(side=tk.LEFT, padx=5)
    
    osc_info_btn = tk.Button(osc_btn_frame, text=f"{get_icon('❓', '?')} What does this do?", command=show_osc_info,
                              bg="#f0f2f5", fg="#8e24aa", font=("Arial", 11, "bold"), pady=10, padx=15)
    osc_info_btn.pack(side=tk.LEFT, padx=5)



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

    var_list_btn = tk.Button(run_btn_frame, text=f"{get_icon('⭐', '')} List Variable Stars".strip(), command=on_list_variables,
                             bg="#f0f2f5", fg="#1a3a5f", font=("Arial", 10, "bold"),
                             relief="flat", pady=10, padx=15)
    var_list_btn.pack(side=tk.LEFT, padx=5)

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

    # v4.16: star selection for the colour fit (faint stars biased the old fit)
    tk.Label(lf_color, text="Min SNR (colour fit):").grid(row=6, column=0, sticky=tk.W, padx=10, pady=5)
    color_min_snr_var = tk.DoubleVar(value=50.0)
    vars_dict["color_min_snr"] = (color_min_snr_var, float)
    ttk.Entry(lf_color, textvariable=color_min_snr_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=10, pady=5)
    tk.Label(lf_color, text="Faintest catalogue V (empty = no limit):").grid(row=7, column=0, sticky=tk.W, padx=10, pady=5)
    color_max_v_var = tk.StringVar(value="")
    vars_dict["color_max_v"] = (color_max_v_var, str)
    ttk.Entry(lf_color, textvariable=color_max_v_var, width=10).grid(row=7, column=1, sticky=tk.W, padx=10, pady=5)
    tk.Label(lf_color, text="* Stars brighter than 'Max Peak for Calib Stars' (Settings) and blends are also excluded.",
             fg="#555", font=("Arial", 8, "italic")).grid(row=6, column=2, columnspan=2, sticky=tk.W, padx=10, pady=5)

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
            band = band_of(finfo.get('filter', ''))
            if band == 'B': b_count += 1
            elif band == 'V': v_count += 1
        return b_count, v_count

    # --- Differential Photometry Section ---
    lf_diff = tk.LabelFrame(sub_diff, text="1. Select Photometry Results & Color Coefficients")
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
    
    lf_ref = tk.LabelFrame(sub_diff, text="2. Reference Star Selection")
    lf_ref.pack(fill="x", padx=10, pady=10)
    
    ref_mode_var = tk.StringVar(value="auto")
    vars_dict["ref_mode"] = (ref_mode_var, str)
    # v4.16: 'auto' = ensemble of all suitable comparison stars; 'auto_single' = one star
    ttk.Radiobutton(lf_ref, text="Automatic ensemble (all suitable catalogue stars - recommended)", variable=ref_mode_var, value="auto").grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    ttk.Radiobutton(lf_ref, text="Automatic single star (typical brightness/colour of the field)", variable=ref_mode_var, value="auto_single").grid(row=0, column=4, columnspan=6, sticky=tk.W, padx=10, pady=5)
    ttk.Radiobutton(lf_ref, text="Resolve Star by Name (local GCVS list, else Simbad)", variable=ref_mode_var, value="name").grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    
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
            from photometry.calibration import resolve_auid
            c = None
            src = ''
            try:
                c = _resolve_name(star_name)
                from photometry.names import LAST_SOURCE as _ls
                src = _ls['source']
            except Exception:
                pass
                
            if not c:
                res = resolve_auid(star_name, verbose=False)
                if res:
                    c = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg)
                    src = 'AAVSO VSX'
            
            if c:
                ra_hms = c.ra.to_string(unit='hour', sep='hms', precision=1)
                dec_dms = c.dec.to_string(unit='degree', sep='dms', precision=1)
                root.after(0, lambda: name_resolve_status_var.set(f"Found ({src}): {ra_hms}, {dec_dms}"))
            else:
                root.after(0, lambda: name_resolve_status_var.set("Not found in the local GCVS list, Simbad or VSX."))
        
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

    # v4.16: ensemble selection and optional check star
    tk.Label(lf_ref, text="Min SNR (comp stars):").grid(row=6, column=0, sticky=tk.E, padx=(10, 2), pady=5)
    diff_min_snr_var = tk.DoubleVar(value=30.0)
    vars_dict["diff_min_snr"] = (diff_min_snr_var, float)
    ttk.Entry(lf_ref, textvariable=diff_min_snr_var, width=6).grid(row=6, column=1, columnspan=2, sticky=tk.W, padx=2)
    tk.Label(lf_ref, text="Check star (name or RA Dec, optional):").grid(row=7, column=0, columnspan=3, sticky=tk.E, padx=(10, 2), pady=5)
    diff_check_var = tk.StringVar(value="")
    vars_dict["diff_check_star"] = (diff_check_var, str)
    ttk.Entry(lf_ref, textvariable=diff_check_var, width=28).grid(row=7, column=3, columnspan=6, sticky=tk.W, padx=2)
    
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

    lf_target = tk.LabelFrame(sub_diff, text="3. Target Star Selection")
    lf_target.pack(fill="x", padx=10, pady=10)
    
    target_mode_var = tk.StringVar(value="all")
    vars_dict["target_mode"] = (target_mode_var, str)
    ttk.Radiobutton(lf_target, text="Analyze all stars", variable=target_mode_var, value="all").grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    ttk.Radiobutton(lf_target, text="Resolve Target by Name (local GCVS list, else Simbad)", variable=target_mode_var, value="name").grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=10, pady=5)
    
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
            from photometry.calibration import resolve_auid
            c = None
            src = ''
            try:
                c = _resolve_name(star_name)
                from photometry.names import LAST_SOURCE as _ls
                src = _ls['source']
            except Exception:
                pass
                
            if not c:
                res = resolve_auid(star_name, verbose=False)
                if res:
                    c = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg)
                    src = 'AAVSO VSX'
            
            if c:
                ra_hms = c.ra.to_string(unit='hour', sep='hms', precision=1)
                dec_dms = c.dec.to_string(unit='degree', sep='dms', precision=1)
                root.after(0, lambda: target_resolve_status_var.set(f"Found ({src}): {ra_hms}, {dec_dms}"))
            else:
                root.after(0, lambda: target_resolve_status_var.set("Not found in the local GCVS list, Simbad or VSX."))
        
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

    diff_coeff_err = {'Tbv': 0.0, 'Tb_bv': 0.0, 'Tv_bv': 0.0}

    def load_color_coefficients():
        import json
        json_path = os.path.join("photometry_output", "color_coefficients.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    coeffs = json.load(f)
                diff_tbv_var.set(round(float(coeffs.get('Tbv', 1.0)), 4))
                diff_tbbv_var.set(round(float(coeffs.get('Tb_bv', 0.0)), 4))
                diff_tvbv_var.set(round(float(coeffs.get('Tv_bv', 0.0)), 4))
                # v4.16: coefficient uncertainties (0 if the file is from an older version)
                diff_coeff_err['Tbv'] = float(coeffs.get('Tbv_err', 0.0) or 0.0)
                diff_coeff_err['Tb_bv'] = float(coeffs.get('Tb_bv_err', 0.0) or 0.0)
                diff_coeff_err['Tv_bv'] = float(coeffs.get('Tv_bv_err', 0.0) or 0.0)
                extra = "" if coeffs.get('consistent', True) else "  WARNING: coefficients flagged INCONSISTENT in the colour report."
                diff_status_var.set(f"Loaded coefficients (±{diff_coeff_err['Tbv']:.3f}, ±{diff_coeff_err['Tb_bv']:.3f}, ±{diff_coeff_err['Tv_bv']:.3f}).{extra}")
            except Exception as e:
                diff_status_var.set(f"Error loading JSON: {e}")
        else:
            diff_status_var.set("No previous coefficients found. Enter manually.")
            
    tk.Button(lf_diff, text="Load Last Coefficients", command=load_color_coefficients, bg="#f0f2f5", relief="flat").grid(row=5, column=0, columnspan=2, pady=5)
            
    def popout_accuracy_plot():
        if not last_diff_args:
            messagebox.showwarning("No Data", "Please run 'Execute Differential Photometry' first to generate results.")
            return
            
        pop = tk.Toplevel(root)
        pop.title("Differential Photometry Accuracy Plot")
        pop.geometry("1100x700")
        
        # Build canvas in popup
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        fig.suptitle('Differential Photometry Accuracy vs Catalog', fontsize=14)
        
        canvas = FigureCanvasTkAgg(fig, master=pop)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, pop)
        toolbar.update()
        
        # Run differential photometry on the new axes to populate the plot
        from photometry.diff_photometry import run_differential_photometry
        run_differential_photometry(
            csv_b=last_diff_args['csv_b'],
            csv_v=last_diff_args['csv_v'],
            ref_catalog=last_diff_args['ref_catalog'],
            k_b=last_diff_args['k_b'],
            k_v=last_diff_args['k_v'],
            Tbv=last_diff_args['Tbv'],
            Tb_bv=last_diff_args['Tb_bv'],
            Tv_bv=last_diff_args['Tv_bv'],
            radius_arcmin=last_diff_args['radius_arcmin'],
            manual_ref_coord=last_diff_args['manual_ref_coord'],
            target_mode=last_diff_args['target_mode'],
            manual_target_coord=last_diff_args['manual_target_coord'],
            axes=axes,
            **last_diff_args.get('extra', {})
        )
        canvas.draw()

    diff_btn_frame = ttk.Frame(sub_diff)
    diff_btn_frame.pack(pady=5)
    
    tk.Button(diff_btn_frame, text="Execute Differential Photometry", command=on_run_diff,
              bg="#1a3a5f", fg="white", font=("Arial", 10, "bold"), width=35, relief="flat", pady=10).pack(side=tk.LEFT, padx=5)
              
    diff_info_btn = tk.Button(diff_btn_frame, text=f"{get_icon('❓', '?')} What does this do?", command=show_diff_info,
                              bg="#f0f2f5", fg="#1a3a5f", font=("Arial", 10, "bold"),
                              relief="flat", pady=10, padx=15)
    diff_info_btn.pack(side=tk.LEFT, padx=5)

    popout_btn = tk.Button(diff_btn_frame, text=f"{get_icon('📈', '')} Open Plot Window".strip(), command=popout_accuracy_plot,
                            bg="#f0f2f5", fg="#1a3a5f", font=("Arial", 10, "bold"),
                            relief="flat", pady=10, padx=15)
    popout_btn.pack(side=tk.LEFT, padx=5)

    diff_status_label = SelectableLabel(sub_diff, textvariable=diff_status_var, font=("Arial", 9, "italic"), justify=tk.CENTER)
    diff_status_label.pack(pady=5, fill="x")

    # Preview for Accuracy
    lf_accuracy_preview = tk.LabelFrame(sub_diff, text="4. Accuracy Evaluation Preview")
    lf_accuracy_preview.pack(fill="x", padx=10, pady=5)
    
    accuracy_fig, accuracy_axes = plt.subplots(2, 3, figsize=(11, 6.0))
    accuracy_canvas = FigureCanvasTkAgg(accuracy_fig, master=lf_accuracy_preview)
    accuracy_canvas.get_tk_widget().pack(fill="x", expand=True)
    accuracy_toolbar = NavigationToolbar2Tk(accuracy_canvas, lf_accuracy_preview)
    accuracy_toolbar.update()

    ts_container = tab_ts

    # v4.18: fixed (non-scrolling) action bar at the top of the Light Curves tab:
    # Generate / Viewer / setups / output folder, status line and progress bar.
    ts_topbar = tk.Frame(tab_ts_outer, bg="#e8eef5", bd=1, relief="groove")
    ts_topbar.pack(side=tk.TOP, fill="x", before=ts_scroll)
    
    # Filter selection for Light Curves
    filter_frame = ttk.Frame(ts_container)
    filter_frame.pack(fill="x", padx=10, pady=5)
    tk.Label(filter_frame, text="Light Curve Filter:").pack(side=tk.LEFT, padx=5)
    ts_filter_var = tk.StringVar(value="V")
    vars_dict["ts_filter"] = (ts_filter_var, str)
    ts_filter_cb = ttk.Combobox(filter_frame, textvariable=ts_filter_var, values=["V", "B", "L"], state="readonly", width=5)
    ts_filter_cb.pack(side=tk.LEFT, padx=5)
    ts_widgets['filter_cb'] = ts_filter_cb
    ts_filter_note = tk.Label(filter_frame, text="", fg="#555", font=("Arial", 8, "italic"), justify=tk.LEFT)
    ts_filter_note.pack(side=tk.LEFT, padx=8)

    def _ts_filter_changed(*_):
        if ts_filter_var.get().upper() == "L":
            ts_filter_note.config(text="L (luminance): calibrated magnitudes use the V magnitudes of the comparison stars "
                                       "(AAVSO filter code CV).\nUse comparison stars of similar colour as the target; "
                                       "relative flux / raw ADU work as usual.")
        else:
            ts_filter_note.config(text="")
    ts_filter_var.trace_add("write", _ts_filter_changed)

    # --- Light Curve Mode (v4.14) ---
    lf_ts_mode = tk.LabelFrame(ts_container, text="Light Curve Mode")
    lf_ts_mode.pack(fill="x", padx=10, pady=5)

    ts_lc_mode_var = tk.StringVar(value="calibrated")
    vars_dict["ts_lc_mode"] = (ts_lc_mode_var, str)
    ttk.Radiobutton(lf_ts_mode, text="Calibrated magnitudes  (ensemble + catalogue magnitudes; AAVSO report)",
                    variable=ts_lc_mode_var, value="calibrated").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=10, pady=1)
    ttk.Radiobutton(lf_ts_mode, text="Relative flux  (target ADU / \u03a3 reference-star ADU; catalogue magnitudes ignored)",
                    variable=ts_lc_mode_var, value="relative").grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=10, pady=1)
    ttk.Radiobutton(lf_ts_mode, text="Raw ADU  (target net flux only; no reference or check stars needed)",
                    variable=ts_lc_mode_var, value="raw").grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=10, pady=1)

    ts_auto_raw_var = tk.BooleanVar(value=True)
    vars_dict["ts_auto_raw"] = (ts_auto_raw_var, bool)
    ttk.Checkbutton(lf_ts_mode, text="Automatically use Raw ADU when no reference stars are selected",
                    variable=ts_auto_raw_var).grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(6, 1))

    ts_per_second_var = tk.BooleanVar(value=False)
    vars_dict["ts_flux_per_second"] = (ts_per_second_var, bool)
    ttk.Checkbutton(lf_ts_mode, text="Raw ADU: normalise to ADU/s (divide by EXPTIME)",
                    variable=ts_per_second_var).grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=10, pady=1)

    ts_plot_flux_var = tk.BooleanVar(value=True)
    vars_dict["ts_plot_flux"] = (ts_plot_flux_var, bool)

    def get_ts_y_quantity():
        return 'flux' if ts_plot_flux_var.get() else 'mag'

    def on_plot_quantity_change():
        try:
            if vars_dict.get("ts_last_results"):
                on_inline_toggle()
        except NameError:
            pass
    ttk.Checkbutton(lf_ts_mode, text="Raw/Relative: plot flux (uncheck to plot instrumental magnitude)",
                    variable=ts_plot_flux_var, command=on_plot_quantity_change).grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(1, 5))

    # --- Faint target (v4.44) ---
    lf_ts_faint = tk.LabelFrame(ts_container, text="Faint Target  (forced photometry at a fixed position, combine frames)")
    lf_ts_faint.pack(fill="x", padx=10, pady=5)
    ts_faint_var = tk.BooleanVar(value=False)
    vars_dict["ts_faint"] = (ts_faint_var, bool)
    ts_faint_fwhm_var = tk.DoubleVar(value=1.5)
    vars_dict["ts_faint_fwhm_factor"] = (ts_faint_fwhm_var, float)
    ts_faint_group_var = tk.IntVar(value=1)
    vars_dict["ts_faint_group"] = (ts_faint_group_var, int)
    ts_faint_snr_var = tk.DoubleVar(value=3.0)
    vars_dict["ts_faint_snr_limit"] = (ts_faint_snr_var, float)
    ttk.Checkbutton(lf_ts_faint, text="Faint target: no centroiding - aperture at the WCS position (corrected by the "
                    "reference-star offset), same radius in all frames",
                    variable=ts_faint_var).grid(row=0, column=0, columnspan=8, sticky=tk.W, padx=10, pady=(2, 1))
    tk.Label(lf_ts_faint, text="Aperture radius:").grid(row=1, column=0, sticky=tk.W, padx=(28, 2))
    ttk.Entry(lf_ts_faint, textvariable=ts_faint_fwhm_var, width=5).grid(row=1, column=1, sticky=tk.W)
    tk.Label(lf_ts_faint, text="x median FWHM (0 = radius from Settings)").grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=2)
    tk.Label(lf_ts_faint, text="Upper limit when S/N <").grid(row=1, column=4, sticky=tk.E, padx=(15, 2))
    ttk.Entry(lf_ts_faint, textvariable=ts_faint_snr_var, width=5).grid(row=1, column=5, sticky=tk.W)
    tk.Label(lf_ts_faint, text="Combine in groups of").grid(row=2, column=0, sticky=tk.W, padx=(28, 2), pady=(4, 2))
    ttk.Spinbox(lf_ts_faint, from_=1, to=999, textvariable=ts_faint_group_var, width=5).grid(row=2, column=1, sticky=tk.W, pady=(4, 2))
    tk.Label(lf_ts_faint, text="consecutive frames (1 = single frames)").grid(row=2, column=2, sticky=tk.W, padx=2, pady=(4, 2))
    ts_faint_btns = tk.Frame(lf_ts_faint)
    ts_faint_btns.grid(row=3, column=0, columnspan=8, sticky=tk.W, padx=(24, 2), pady=(2, 4))
    tk.Button(ts_faint_btns, text="Apply Groups", font=("Arial", 9), command=lambda: on_faint_apply_groups()).pack(side=tk.LEFT, padx=3)
    tk.Button(ts_faint_btns, text="Combine Selected Rows", font=("Arial", 9), command=lambda: on_faint_combine_selected()).pack(side=tk.LEFT, padx=3)
    tk.Button(ts_faint_btns, text="Split Selected Rows", font=("Arial", 9), command=lambda: on_faint_split_selected()).pack(side=tk.LEFT, padx=3)
    tk.Button(ts_faint_btns, text="Single Frames", font=("Arial", 9), command=lambda: on_faint_single()).pack(side=tk.LEFT, padx=3)
    tk.Label(ts_faint_btns, text="(uses the frames of the last run - no new measurement)", fg="#555").pack(side=tk.LEFT, padx=6)
    def get_coords_ts(mode, name, ra_s, dec_s):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        if mode == "name":
            try:
                return _resolve_name(name)
            except Exception:
                from photometry.calibration import resolve_auid
                res = resolve_auid(name, verbose=False)
                if res:
                    return SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg)
                raise Exception(f"Could not resolve '{name}' via the local GCVS list, Simbad or VSX.")
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
                    c = _resolve_name(name)
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
                        c = _resolve_name(name)
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
    
    # Inline Toggles Frame
    ts_inline_toggles_frame = tk.Frame(lf_ts_plot)
    ts_inline_toggles_frame.pack(fill="x", padx=5, pady=2)
    
    vars_dict["ts_visibility_vars"] = {}
    vars_dict["ts_last_results"] = []
    
    def on_inline_toggle():
        from photometry.time_series import plot_light_curve
        v_state = {k: v.get() for k, v in vars_dict["ts_visibility_vars"].items()}
        results = vars_dict["ts_last_results"]
        mode_lbl = ""
        if results:
            m = results[0].get('lc_mode', 'calibrated')
            mode_lbl = {"raw": " - Raw ADU", "relative": " - Relative flux"}.get(m, "")
        plot_title = f"{ts_target_name_var.get()} ({ts_filter_var.get()} Filter){mode_lbl}"
        plot_light_curve(results, plot_title, "", ax=ts_ax, visibility_state=v_state,
                         plot_options={"y_quantity": get_ts_y_quantity()})
        ts_canvas.draw()
        
    def open_advanced_plot_window():
        check_name = "Check Star"
        c_idx = ts_check_star_idx_var.get()
        if c_idx >= 0:
            if hasattr(vars_dict.get(f"ts_ref_{c_idx}_name"), "__getitem__"):
                check_name = vars_dict[f"ts_ref_{c_idx}_name"][0].get()
        if not vars_dict["ts_last_results"]:
            messagebox.showinfo("No Data", "Generate a light curve first.")
            return
        AdvancedPlotWindow(root, vars_dict["ts_last_results"], ts_target_name_var.get(), ts_filter_var.get(), check_name,
                           y_quantity=get_ts_y_quantity())
        
    def rebuild_inline_toggles(results):
        vars_dict["ts_last_results"] = results
        for widget in ts_inline_toggles_frame.winfo_children():
            widget.destroy()
            
        vars_dict["ts_visibility_vars"].clear()
        v_vars = vars_dict["ts_visibility_vars"]
        
        v_vars["Target"] = tk.BooleanVar(value=True)
        v_vars["Check Star"] = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(ts_inline_toggles_frame, text="Target", variable=v_vars["Target"], command=on_inline_toggle).pack(side="left", padx=5)
        ttk.Checkbutton(ts_inline_toggles_frame, text="Check Star", variable=v_vars["Check Star"], command=on_inline_toggle).pack(side="left", padx=5)
        
        all_refs = set()
        for r in results:
            if 'ref_mags' in r:
                all_refs.update(r['ref_mags'].keys())
                
        for ref in sorted(all_refs):
            key = f"Ref: {ref}"
            v_vars[key] = tk.BooleanVar(value=False)
            ttk.Checkbutton(ts_inline_toggles_frame, text=key, variable=v_vars[key], command=on_inline_toggle).pack(side="left", padx=5)
            
        btn = tk.Button(ts_inline_toggles_frame, text="⚙️ Customize Plot", command=open_advanced_plot_window, bg="#1a3a5f", fg="white", font=("Arial", 9, "bold"))
        btn.pack(side="right", padx=10, pady=2)

    # Target Star
    lf_ts_target = tk.LabelFrame(ts_container, text="Target Star (Variable)")
    # Now pack in the requested order
    # 1. Selection (Handled by File Manager)
    lf_ts_target.pack(fill="x", padx=10, pady=5)
    lf_ts_ensemble.pack(fill="x", padx=10, pady=5)
    # v4.18: rarely changed section can be collapsed
    ts_show_coeff_var = tk.BooleanVar(value=False)
    vars_dict["ts_show_coeff"] = (ts_show_coeff_var, bool)
    ts_coeff_toggle = tk.Button(ts_container, relief="flat", anchor="w", fg="#1a3a5f", font=("Arial", 9, "bold"))
    def _apply_coeff_visibility(*_):
        if ts_show_coeff_var.get():
            ts_coeff_toggle.config(text="▾ Coefficients, AAVSO code & chart (click to hide)")
            lf_ts_coeff.pack(fill="x", padx=10, pady=5, after=ts_coeff_toggle)
        else:
            ts_coeff_toggle.config(text="▸ Coefficients, AAVSO code & chart (click to show)")
            lf_ts_coeff.pack_forget()
    ts_coeff_toggle.config(command=lambda: (ts_show_coeff_var.set(not ts_show_coeff_var.get()), _apply_coeff_visibility()))
    ts_coeff_toggle.pack(fill="x", padx=10, pady=(5, 0))
    ts_show_coeff_var.trace_add("write", _apply_coeff_visibility)
    _apply_coeff_visibility()
    
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
            try:
                c = _resolve_name(name)
            except Exception:
                pass
                
            if not c:
                res = resolve_auid(name, verbose=False)
                if res:
                    c = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg)
                else:
                    ts_status_var.set(f"Could not resolve '{name}' via the local GCVS list, Simbad or VSX.")
                    return
            
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

    def aavso_crossmatch(stars, cat_name, radius):
        """Catalogue V/B for the AAVSO sequence stars (cat_v, cat_b, cat_match) - local copy first."""
        from photometry.calibration import fetch_online_catalog
        cat_map = {"ATLAS refcat2": "ATLAS", "APASS DR9": "APASS",
                   "Landolt Standard Star Catalogue": "LANDOLT", "GAIA_DR3": "GAIA_DR3"}
        internal_cat_name = cat_map.get(cat_name, "ATLAS")
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
                return True
            for s in stars: s['cat_v'] = np.nan; s['cat_b'] = np.nan; s['cat_match'] = False
        except Exception as e:
            print(f"Catalog cross-match failed: {e}")
            for s in stars: s['cat_v'] = np.nan; s['cat_b'] = np.nan; s['cat_match'] = False
        return False

    def stored_aavso_sequence(target, cat_name=None):
        """v4.35: stored sequence of target (list of stars) or None - no query."""
        from photometry import aavso_sequences as _aseq
        d = _aseq.load(target)
        return d['stars'] if d else None

    def on_get_aavso_refs(refresh=False):
        if not hasattr(on_get_aavso_refs, "cache"):
            on_get_aavso_refs.cache = None
            on_get_aavso_refs.last_target = ""
            on_get_aavso_refs.last_cat = ""
            
        target = ts_target_name_var.get().strip()
        cat_name = vars_dict["reference_catalog"][0].get()
        if not target:
            messagebox.showwarning("Warning", "Please enter a target star name first.")
            return
        try:
            radius = get_catalog_radius()
        except:
            radius = 45.0
        from photometry import aavso_sequences as _aseq
        from photometry.local_catalogs import name_key as _nk
        stored = None if refresh else _aseq.load(target)
        if stored and float(stored.get('fov_arcmin', 0)) < 0.95 * radius:
            stored = None                    # stored for a smaller field - fetch the larger one
            
        # Check if we can use cached results
        if (not refresh and on_get_aavso_refs.cache and 
            _nk(on_get_aavso_refs.last_target) == _nk(target) and 
            on_get_aavso_refs.last_cat == cat_name):
            stars = on_get_aavso_refs.cache
            ts_status_var.set(f"Showing cached AAVSO sequence for {target}.")
        elif stored:
            # v4.35: sequence stored on disk - no query to AAVSO
            stars = stored['stars']
            if stored.get('catalog') != cat_name or not all('cat_match' in s_ for s_ in stars):
                ts_status_var.set(f"Cross-matching the stored AAVSO sequence with {cat_name}...")
                root.update_idletasks()
                aavso_crossmatch(stars, cat_name, radius)
                _aseq.update_crossmatch(target, stars, cat_name)
            age = _aseq.age_days(stored)
            msg = f"AAVSO sequence for {target} from file (fetched {stored.get('fetched', '?')[:10]}, {len(stars)} stars)."
            if age is not None and age > _aseq.OLD_DAYS:
                msg += " Older than a year - 'Refresh from AAVSO' gets the current sequence."
            ts_status_var.set(msg)
            on_get_aavso_refs.cache = stars
            on_get_aavso_refs.last_target = target
            on_get_aavso_refs.last_cat = cat_name
        else:
            ts_status_var.set(f"Fetching AAVSO sequence for {target}...")
            root.update_idletasks()
            
            from photometry.calibration import fetch_aavso_chart_stars, fetch_online_catalog
            stars = fetch_aavso_chart_stars(target, fov_arcmin=radius)
            
            if not stars:
                messagebox.showinfo("No Stars Found", f"Could not find any comparison stars for {target} in the AAVSO VSP.")
                return

            ts_status_var.set(f"Cross-matching AAVSO stars with {cat_name}...")
            root.update_idletasks()
            aavso_crossmatch(stars, cat_name, radius)
            try:
                p_ = _aseq.save(target, stars, radius, cat_name)
                ts_status_var.set(f"AAVSO sequence for {target}: {len(stars)} stars, stored in {p_} for later use.")
            except Exception as e:
                print(f"AAVSO sequence could not be stored: {e}")

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
        _st_ = _aseq.load(target)
        _hdr = tk.Frame(pop); _hdr.pack(fill="x", padx=10)
        tk.Label(_hdr, text=(f"Stored sequence, fetched {_st_.get('fetched', '?')[:16]} (field {float(_st_.get('fov_arcmin', 0)):g}')"
                             if _st_ else "Sequence not stored"), fg="#555").pack(side=tk.LEFT)

        def _refresh():
            pop.destroy()
            on_get_aavso_refs(refresh=True)
        ttk.Button(_hdr, text="Refresh from AAVSO", command=_refresh).pack(side=tk.RIGHT)
        
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
                # v4.23: B light curves need the B magnitudes of the comparison stars (V and L use V)
                use_b = ts_filter_var.get().upper() == 'B'
                vars_dict[f"ts_ref_{count}_mag"][0].set(b_to_use if use_b else v_to_use)
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
            
            tidy_numbers()
            ts_status_var.set(f"Imported {count} stars to ensemble.")
            pop.destroy()
            
        btn_frame = ttk.Frame(pop)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Add Selected to Ensemble", command=do_import, width=30).pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Button(btn_frame, text="Clear AAVSO Cache & Re-fetch", command=lambda: [pop.destroy(), setattr(on_get_aavso_refs, "cache", None), on_get_aavso_refs()], width=30).pack(side=tk.LEFT, padx=10, expand=True)
        
        _st2 = _aseq.load(target)
        ts_status_var.set(f"Found {len(stars)} AAVSO comparison stars"
                          + (f" (stored sequence, fetched {_st2.get('fetched', '?')[:10]})." if _st2 else "."))

    def on_reset_ensemble():
        # v4.15: single implementation used by both "Reset Ensemble" buttons
        # (previously two different versions; one left the check-star index set).
        # The target star is intentionally left unchanged.
        if not messagebox.askyesno("Reset Ensemble", "Clear all reference stars and the check star?\n\n(The target star is not changed.)"):
            return
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
        ts_check_star_idx_var.set(-1)
        ts_status_var.set("Ensemble and check star cleared.")

    ttk.Button(lf_ts_target, text="Fetch", command=on_fetch_target_ts, width=6).grid(row=0, column=2, sticky=tk.W, padx=2)
    ttk.Button(lf_ts_target, text="Get AAVSO Ref Stars", command=on_get_aavso_refs).grid(row=0, column=4, sticky=tk.W, padx=10)
    ttk.Button(lf_ts_target, text="Reset Ensemble", command=on_reset_ensemble).grid(row=0, column=5, sticky=tk.W, padx=2)
    ttk.Button(lf_ts_target, text="APASS DR10...", command=lambda: vars_dict['apass_dr10_dialog']()).grid(
        row=0, column=6, sticky=tk.W, padx=2)
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
    
    # v4.18: extinction is taken from Settings > Astro & Catalog (k_V or k_B by filter);
    # the separate Light-Curve value (default 0.15) was removed.
    tk.Label(lf_ts_coeff, text="Extinction k: from Settings (k_V / k_B)", fg="#555",
             font=("Arial", 8, "italic")).grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=10, pady=2)
    
    tk.Label(lf_ts_coeff, text="AAVSO Observer Code:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
    # Linked to global settings
    ts_obs_var = vars_dict["aavso_obs_code"][0]
    ttk.Entry(lf_ts_coeff, textvariable=ts_obs_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=2)
    
    # v4.18: observing site lives in Settings > Operator & Session (same variables);
    # it is only used when a FITS header has no SITELAT/SITELONG.
    ts_lat_var = tk.DoubleVar(value=59.8)
    vars_dict["ts_lat"] = (ts_lat_var, float)
    ts_lon_var = tk.DoubleVar(value=17.6)
    vars_dict["ts_lon"] = (ts_lon_var, float)
    tk.Label(lf_ts_coeff, text="Site: from FITS header (SITELAT/SITELONG), else Settings > Operator & Session",
             fg="#555", font=("Arial", 8, "italic")).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=10, pady=2)

    tk.Label(lf_ts_coeff, text="AAVSO Chart ID:").grid(row=3, column=2, sticky=tk.W, padx=10, pady=2)
    ts_chart_var = tk.StringVar(value="na")
    vars_dict["ts_chart"] = (ts_chart_var, str)
    ttk.Entry(lf_ts_coeff, textvariable=ts_chart_var, width=10).grid(row=3, column=3, sticky=tk.W, padx=2)

    def load_ts_coefficients():
        import json
        json_path = os.path.join("photometry_output", "color_coefficients.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    coeffs = json.load(f)
                filt = ts_filter_var.get().upper()
                if filt == 'L':
                    ts_coeff_var.set(0.0)
                    ts_status_var.set("L filter: no colour transformation available (coefficient set to 0).")
                    return
                if filt == 'B':
                    val = coeffs.get('Tb_bv', 0.0)
                    key_name = 'Tb_bv'
                else:
                    val = coeffs.get('Tv_bv', 0.0)
                    key_name = 'Tv_bv'
                ts_coeff_var.set(round(float(val), 4))
                ts_status_var.set(f"Loaded {key_name} for {filt} filter.")
            except Exception as e:
                ts_status_var.set(f"Error loading JSON: {e}")
        else:
            ts_status_var.set("No coefficients file found.")
            
    tk.Button(lf_ts_coeff, text="Load Last Coeffs", command=load_ts_coefficients).grid(row=4, column=0, columnspan=4, pady=5)

    ts_status_var = tk.StringVar(value="Ready to process sequence.")
    ts_status_label = SelectableLabel(ts_topbar, textvariable=ts_status_var, font=("Arial", 9, "italic"), justify=tk.CENTER)
    ts_status_label.pack(side=tk.BOTTOM, pady=2, fill="x")
    
    ts_progress_var = tk.DoubleVar(value=0)
    ts_progress = ttk.Progressbar(ts_topbar, variable=ts_progress_var, maximum=100, length=400)
    ts_progress.pack(side=tk.BOTTOM, pady=2)
    
    # Analysis Summary Panel (Persistent)
    lf_ts_summary = tk.LabelFrame(ts_container, text="Analysis Summary")
    lf_ts_summary.pack(fill="x", padx=10, pady=5)
    
    ts_summary_scroll = ttk.Scrollbar(lf_ts_summary)
    ts_summary_scroll.pack(side="right", fill="y")
    
    ts_summary_text = tk.Text(lf_ts_summary, height=8, font=("Courier", 9), bg="#f0f2f5", fg="#2c3e50", wrap="word", relief="flat", yscrollcommand=ts_summary_scroll.set)
    ts_summary_text.pack(side="left", fill="both", expand=True, padx=(15, 0), pady=10)
    ts_summary_scroll.config(command=ts_summary_text.yview)
    
    ts_summary_text.insert("1.0", "No analysis data yet. Run 'Generate Light Curve' to see results.")
    ts_summary_text.config(state="disabled")
    
    add_copy_context_menu(ts_summary_text)

    cancel_event = threading.Event()
    
    def on_cancel_ts():
        if messagebox.askyesno("Cancel", "Stop processing the sequence?"):
            cancel_event.set()
            ts_status_var.set("Cancelling...")

    cancel_btn = tk.Button(ts_topbar, text="Cancel Processing", command=on_cancel_ts, 
                           bg="#f44336", fg="white", font=("Arial", 9))
    # Packed later during execution or hidden by default
    
    # ---------------------------------------------------------------- v4.44: faint target
    def _faint_params():
        def f(var, default):
            try:
                return float(var.get())
            except (tk.TclError, ValueError):
                return default
        return {'fwhm_factor': f(ts_faint_fwhm_var, 1.5), 'group': max(int(f(ts_faint_group_var, 1)), 1),
                'snr_limit': f(ts_faint_snr_var, 3.0)}

    def _faint_summary(state, groups):
        info, prm = state['info'], state['params']
        n_comb = sum(1 for g in groups if len(g) > 1)
        txt = ("Faint target: forced photometry at the WCS position corrected by the reference-star offset; "
               "fixed aperture for all frames and stars\n")
        txt += (f"  Aperture r = {info['r_ap']:.2f} px, sky annulus {info['r_in']:.2f}-{info['r_out']:.2f} px"
                + (f"  ({prm['fwhm_factor']:g} x median FWHM {info['median_fwhm']:.2f} px)" if info.get('median_fwhm') else "") + "\n")
        dx, dy = info.get('offset_mean', (np.nan, np.nan))
        sx, sy = info.get('offset_scatter', (np.nan, np.nan))
        if np.isfinite(dx):
            txt += (f"  Reference stars vs. WCS: mean offset dx={dx:+.2f} dy={dy:+.2f} px, "
                    f"frame-to-frame scatter {sx:.2f}/{sy:.2f} px\n")
        txt += (f"  Frames: {info['n_used']} of {info['n_files']} -> {len(groups)} points "
                f"({n_comb} combined); upper limit (fainter than) when S/N < {prm['snr_limit']:g}\n")
        return txt

    def faint_publish(groups, state=None):
        """(Re-)combine the frames of the last faint-target run into the given groups and show/save the result."""
        from photometry.faint_target import combine
        state = state or vars_dict.get("ts_faint_state")
        if not state:
            messagebox.showinfo("Faint Target", "Generate a light curve with 'Faint target' switched on first.")
            return None
        state['params']['snr_limit'] = _faint_params()['snr_limit']
        a = state['args']
        results = combine(state['frames'], groups, a['lc_mode'], a['ensemble'], a['check'], a['target_bv'],
                          a['coeff_term'], a['flux_per_second'], state['params']['snr_limit'])
        state['groups'] = [list(g) for g in groups]
        ctx = {'state': state, 'summary': _faint_summary(state, groups)}
        return results, ctx

    def run_faint_target(files, tar_c, resolved_ensemble, check_star_data, lc_mode, flux_per_second,
                         stack_mode, sky_factor, update_prog):
        from photometry.faint_target import measure_frames, groups_consecutive
        prm = _faint_params()
        frames, info = measure_frames(
            files, {'ra': tar_c.ra.deg, 'dec': tar_c.dec.deg}, resolved_ensemble, check_star_data,
            aperture_radius=float(vars_dict["aperture_radius"][0].get()),
            annulus_inner=float(vars_dict["annulus_inner"][0].get()),
            annulus_outer=float(vars_dict["annulus_outer"][0].get()),
            fwhm_factor=prm['fwhm_factor'],
            annulus_inner_gap=float(vars_dict["annulus_inner_gap"][0].get()),
            annulus_width=float(vars_dict["annulus_width"][0].get()),
            gain=float(vars_dict["ccd_gain"][0].get()), stack_mode=stack_mode, sky_noise_factor=sky_factor,
            max_shift=float(vars_dict['centroid_tracking_shift'][0].get()) if 'centroid_tracking_shift' in vars_dict else 4.0,
            initial_radius=float(vars_dict['centroid_initial_radius'][0].get()) if 'centroid_initial_radius' in vars_dict else 15.0,
            site_lat=ts_lat_var.get(), site_long=ts_lon_var.get(),
            cancel_event=cancel_event, update_progress=update_prog)
        if not frames:
            return None, "No frame could be measured (WCS? target on the images?)", None
        state = {'frames': frames, 'info': info, 'params': prm,
                 'args': {'lc_mode': lc_mode, 'ensemble': resolved_ensemble, 'check': check_star_data,
                          'target_bv': ts_target_bv_var.get(),
                          'coeff_term': ts_coeff_var.get() if ts_do_trans_var.get() else 0.0,
                          'flux_per_second': flux_per_second},
                 'ensemble': resolved_ensemble, 'check_star': check_star_data}
        results, ctx = faint_publish(groups_consecutive(frames, prm['group']), state)
        return results, "Success", ctx

    def _faint_regroup(make_groups):
        state = vars_dict.get("ts_faint_state")
        if not state:
            messagebox.showinfo("Faint Target", "Generate a light curve with 'Faint target' switched on first.\n\n"
                                "Combining works on the frames measured in that run (no new measurement needed).")
            return
        groups = make_groups(state)
        if groups is None:
            return
        out = faint_publish(groups, state)
        if out:
            results, ctx = out
            ts_publish(results, state['args']['lc_mode'], state['ensemble'], state['check_star'], ctx)

    def on_faint_apply_groups():
        from photometry.faint_target import groups_consecutive
        _faint_regroup(lambda st: groups_consecutive(st['frames'], _faint_params()['group']))

    def on_faint_single():
        from photometry.faint_target import groups_single
        _faint_regroup(lambda st: groups_single(st['frames']))

    def _selected_rows():
        return sorted(ts_tree.index(it) for it in ts_tree.selection())

    def on_faint_combine_selected():
        from photometry.faint_target import merge_groups
        rows = _selected_rows()
        if len(rows) < 2:
            messagebox.showinfo("Combine Selected", "Select at least two rows in the result table "
                                "(Ctrl+click or Shift+click), then press 'Combine Selected'.")
            return
        _faint_regroup(lambda st: merge_groups(st['groups'], rows))

    def on_faint_split_selected():
        from photometry.faint_target import split_groups
        rows = _selected_rows()
        if not rows:
            messagebox.showinfo("Split Selected", "Select the combined row(s) in the result table first.")
            return
        _faint_regroup(lambda st: split_groups(st['groups'], rows))

    def ts_publish(results, lc_mode, resolved_ensemble, check_star_data, faint=None):
        """v4.44: write CSV / AAVSO report / plot and fill table + summary for a finished light curve
        (normal run, or faint-target run / re-combination; `faint` = dict with info for the summary)."""
        from photometry.time_series import save_aavso_report, plot_light_curve
        base_name = ts_target_name_var.get().replace(' ','_')
        suffix = {"calibrated": "", "relative": "_relflux", "raw": "_rawADU"}.get(lc_mode, "")
        if faint:
            suffix += "_faint"          # v4.44: own files for faint-target light curves
        # v4.18: UT date of the first observation in the file names, so that
        # light curves from different nights no longer overwrite each other
        # (a re-run of the same night and mode replaces its own files).
        # v4.20: data from several nights -> 'first_to_last'
        try:
            from photometry.time_series import date_label
            night = date_label([r['jd'] for r in results])
        except Exception:
            night = "nodate"
        out_csv = os.path.join("photometry_output", f"light_curve_{base_name}{suffix}_{night}.csv")
        out_aavso = os.path.join("photometry_output", f"aavso_{base_name}{'_faint' if faint else ''}_{night}.txt")
        out_plot = os.path.join("photometry_output", f"plot_{base_name}{suffix}_{night}.png")
        vars_dict["ts_last_csv_path"] = os.path.abspath(out_csv)   # v4.21: for Period Analysis
        if lc_mode == "calibrated":
            vars_dict["ts_last_report_path"] = os.path.abspath(out_aavso)

        # Save CSV (union of all keys, in first-seen order; v4.42: UT of mid-exposure instead of JD)
        from photometry.time_series import csv_fields as _csv_fields
        fieldnames = _csv_fields(results)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        check_stats = ""
        if lc_mode == "calibrated":
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
            if faint:
                report_notes += " | forced photometry at fixed position; fluxes of the frames of a point summed (see n_frames in CSV)"

            save_aavso_report(results, out_aavso, ts_target_name_var.get(), aavso_band(ts_filter_var.get()), ts_obs_var.get(),
                              comp_name=comp_name, comp_mag=comp_mag, 
                              check_name=check_name, check_mag=check_mag,
                              trans=is_trans, software_version=APP_VERSION,
                              comments=report_comments,
                              notes=report_notes,
                              chart=ts_chart_var.get())

            # Calculate Stats for Check Star & Reference Stars
            check_mags = [r['check_mag'] for r in results if isinstance(r.get('check_mag'), (int, float)) and not np.isnan(r['check_mag'])]
            if check_mags and check_star_data:
                avg_check = np.mean(check_mags)
                std_check = np.std(check_mags)
                bias = avg_check - check_star_data['mag_std']
                check_stats += f"Check Star Stats ({check_star_data['name']}):\n"
                check_stats += f"Average Bias: {bias:+.3f} (Meas - Cat)  |  Precision (1-sigma): {std_check:.3f} mag\n\n"

            for ref_star in resolved_ensemble:
                ref_name = ref_star['name']
                cat_mag = ref_star['mag_std']
                mags = [r['ref_mags'][ref_name] for r in results if 'ref_mags' in r and ref_name in r['ref_mags'] and not np.isnan(r['ref_mags'][ref_name])]
                if mags:
                    avg_mag = np.mean(mags)
                    std_mag = np.std(mags)
                    bias = avg_mag - cat_mag
                    check_stats += f"Reference Star Stats ({ref_name}):\n"
                    check_stats += f"Average Bias: {bias:+.3f} (Meas - Cat)  |  Precision (1-sigma): {std_mag:.3f} mag\n\n"
        else:
            # Uncalibrated modes: statistics in flux units (no catalogue comparison)
            unit = results[0].get('flux_unit', 'ADU')
            def flux_stats(vals):
                v = np.array([x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)], dtype=float)
                if len(v) < 2: return None
                mean = np.mean(v); std = np.std(v, ddof=1)
                rel = 100.0 * std / abs(mean) if mean != 0 else np.nan
                return mean, std, rel, len(v)
            ok_res = [r for r in results if r.get('flag', 'OK') == 'OK']
            st = flux_stats([r['flux'] for r in ok_res])
            errs = [r['flux_err'] for r in ok_res if np.isfinite(r.get('flux_err', np.nan))]
            if st:
                check_stats += f"Target ({'raw net flux' if lc_mode == 'raw' else 'target / ensemble'}, OK points):\n"
                check_stats += (f"Mean: {st[0]:.6g} {unit}  |  Std: {st[1]:.4g} ({st[2]:.2f} %)  |  "
                                f"Median formal error: {np.median(errs):.4g} {unit}  |  N={st[3]}\n"
                                f"Peak-to-peak: {np.nanmax([r['flux'] for r in ok_res]) - np.nanmin([r['flux'] for r in ok_res]):.6g} {unit}\n\n")
            if check_star_data:
                st = flux_stats([r.get('check_flux') for r in results])
                if st:
                    check_stats += f"Check Star Stats ({check_star_data['name']}):\n"
                    check_stats += f"Mean: {st[0]:.6g} {unit}  |  Std: {st[1]:.4g}  |  Scatter: {st[2]:.2f} % (~{1.0857*st[2]/100:.3f} mag)\n\n"
            ref_names = sorted({k for r in results for k in (r.get('ref_fluxes') or {}).keys()})
            for ref_name in ref_names:
                st = flux_stats([(r.get('ref_fluxes') or {}).get(ref_name, np.nan) for r in results])
                if st:
                    lbl = "flux" if lc_mode == "raw" else "flux / sum of other refs"
                    check_stats += f"Reference Star Stats ({ref_name}, {lbl}):\n"
                    check_stats += f"Mean: {st[0]:.6g}  |  Scatter: {st[2]:.2f} % (~{1.0857*st[2]/100:.3f} mag)\n\n"
            if lc_mode == "raw" and resolved_ensemble:
                check_stats += "Note: reference stars were measured for display only; they are NOT used in Raw ADU mode.\n"

        check_stats = check_stats.strip()

        # Save a plot file (thread-safe, object-oriented matplotlib)
        try:
            mode_lbl = {"raw": " - Raw ADU", "relative": " - Relative flux"}.get(lc_mode, "")
            plot_light_curve(results, f"{ts_target_name_var.get()} ({ts_filter_var.get()} Filter){mode_lbl}", out_plot,
                             plot_options={"y_quantity": get_ts_y_quantity()})
        except Exception as e:
            print(f"Warning: could not save plot: {e}")

        # Update embedded plot
        plot_title = f"{ts_target_name_var.get()} ({ts_filter_var.get()} Filter)"
        def update_viz():
            rebuild_inline_toggles(results)
            on_inline_toggle()
        root.after(0, update_viz)

        # Update table
        def update_table():
            # Clear existing
            for item in ts_tree.get_children():
                ts_tree.delete(item)
            def fmt(v, f):
                return format(v, f) if isinstance(v, (int, float)) and np.isfinite(v) else "N/A"
            if lc_mode == "calibrated":
                ts_tree.heading("Mag", text="Mag"); ts_tree.heading("Err", text="Err"); ts_tree.heading("Check Mag", text="Check Mag")
                v_key, e_key, c_key, vf = 'mag', 'mag_err', 'check_mag', ".3f"
            else:
                unit = results[0].get('flux_unit', 'ADU')
                lbl = f"Flux [{unit}]" if lc_mode == "raw" else "Rel. Flux"
                ts_tree.heading("Mag", text=lbl); ts_tree.heading("Err", text="Err"); ts_tree.heading("Check Mag", text="Check Flux")
                v_key, e_key, c_key = 'flux', 'flux_err', 'check_flux'
                vf = ".1f" if lc_mode == "raw" else ".5g"
            # Add new
            for r in results:
                val_txt = fmt(r.get(v_key), vf)
                if r.get('upper_limit') and v_key == 'mag':
                    val_txt = "<" + fmt(r.get('mag_limit'), ".3f")      # v4.44: fainter than
                ts_tree.insert("", tk.END, values=(
                    f"{r.get('date_ut', '')} {r.get('ut_mid', '')}".strip(),
                    f"{r['hjd']:.5f}", 
                    r.get('n_frames', 1),
                    val_txt, 
                    fmt(r.get(e_key), vf if lc_mode == "raw" else (".3f" if lc_mode == "calibrated" else ".3g")), 
                    fmt(r.get(c_key), vf),
                    f"{r['snr']:.1f}",
                    fmt(r.get('fwhm', np.nan), ".2f"),
                    r.get('flag', 'OK')
                ))
        root.after(0, update_table)

        # Update summary
        mode_names = {"calibrated": "Calibrated magnitudes", "relative": "Relative flux (ensemble ADU, no catalogue mags)", "raw": "Raw ADU (target only)"}
        summary_text = f"Target: {ts_target_name_var.get()}  |  Filter: {ts_filter_var.get()}  |  " + (f"Points: {len(results)} (from {sum(r.get('n_frames', 1) for r in results)} frames)\n" if faint else f"Files: {len(results)}\n")
        summary_text += f"Mode:         {mode_names.get(lc_mode, lc_mode)}\n"
        summary_text += f"Saved CSV:    {os.path.abspath(out_csv)}\n"
        summary_text += f"Saved Plot:   {os.path.abspath(out_plot)}\n"
        if faint:
            summary_text += faint.get('summary', '')
        if lc_mode == "calibrated":
            summary_text += f"AAVSO Report: {os.path.abspath(out_aavso)}\n"
        else:
            summary_text += "AAVSO Report: not written (only available in Calibrated mode)\n"
        if check_stats:
            summary_text += f"--------------------------------------------------------------------------------\n"
            summary_text += check_stats

        def update_summary():
            ts_summary_text.config(state="normal")
            ts_summary_text.delete("1.0", tk.END)
            ts_summary_text.insert("1.0", summary_text)
            ts_summary_text.config(state="disabled")
        root.after(0, update_summary)

        vars_dict["ts_faint_state"] = faint.get('state') if faint else None
        ts_status_var.set(f"Complete! Results saved." + (f"  ({len(results)} points)" if faint else ""))

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
            trans_filt = lband_of(filt_str) or filt_str.strip()
            
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
        
        # --- Light curve mode selection (v4.14) ---
        lc_mode = ts_lc_mode_var.get() or "calibrated"
        if not ensemble_data and lc_mode in ("calibrated", "relative"):
            if ts_auto_raw_var.get():
                prev = "Calibrated" if lc_mode == "calibrated" else "Relative flux"
                lc_mode = "raw"
                ts_lc_mode_var.set("raw")
                ts_status_var.set(f"No reference stars selected -> switched from {prev} to Raw ADU mode.")
                messagebox.showinfo("Raw ADU Mode",
                                    f"No reference stars are selected, so the {prev} mode cannot be used.\n\n"
                                    "Calibra switched to 'Raw ADU' mode: the light curve will be the target's "
                                    "background-subtracted net flux (ADU) with CCD-equation uncertainties.\n\n"
                                    "(Untick 'Automatically use Raw ADU...' in 'Light Curve Mode' to disable this.)")
            else:
                messagebox.showerror("Error", "Please select at least one reference star in the ensemble,\n"
                                     "or choose 'Raw ADU' mode (or enable automatic switching).")
                return
        flux_per_second = bool(ts_per_second_var.get()) and lc_mode == "raw"
        lc_stack_mode = vars_dict["stack_mode"][0].get() if "stack_mode" in vars_dict else "average"
        try:
            lc_k_coeff = float(vars_dict["extinction_kb" if ts_filter_var.get().upper() == "B" else "extinction_kv"][0].get())
        except (KeyError, tk.TclError, ValueError):
            lc_k_coeff = 0.2
        try:
            lc_sky_factor = float(vars_dict["sky_noise_factor"][0].get()) if "sky_noise_factor" in vars_dict else 1.0
        except (tk.TclError, ValueError):
            lc_sky_factor = 1.0

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
                        try:
                            c = _resolve_name(s['name'])
                        except Exception:
                            from photometry.calibration import resolve_auid
                            res = resolve_auid(s['name'], verbose=False)
                            if res:
                                c = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg)
                            else:
                                raise Exception(f"Could not resolve '{s['name']}' via the local GCVS list, Simbad or VSX.")
                        s['ra'] = c.ra.deg
                        s['dec'] = c.dec.deg
                    
                    if s in ensemble_data:
                        resolved_ensemble.append(s)
                except Exception as e:
                    root.after(0, lambda err=e, nm=s['name']: messagebox.showerror("Ensemble Error", f"Could not resolve star '{nm}': {err}"))
                    ts_status_var.set("Failed: Coordinate resolution error.")
                    return

            # v4.20: are target and comparison stars on the images? (several nights may
            # have different pointings; a star off an image cannot be measured there)
            try:
                from photometry.field_variables import image_footprints, in_footprints
                fps = image_footprints(files)
                good = [f for f in fps if f[1] is not None]
                if len(good) < len(fps):
                    print(f"Light Curves: {len(fps) - len(good)} image(s) without WCS will give no measurement.")
                if good:
                    t_in = in_footprints(tar_c.ra.deg, tar_c.dec.deg, good, margin_px=0)
                    if not t_in.any():
                        root.after(0, lambda: messagebox.showerror("Target not on the images",
                            f"The target ({tar_c.ra.deg:.4f}, {tar_c.dec.deg:+.4f}) is not on any of the "
                            f"{len(good)} selected images.\nCheck the coordinates/name and the selected files."))
                        ts_status_var.set("Failed: target not on the images.")
                        return
                    notes = []
                    if not t_in.all():
                        notes.append(f"target outside {int((~t_in).sum())} of {len(good)} image(s) (skipped there)")
                    for s in resolved_ensemble + ([check_star_data] if check_star_data else []):
                        s_in = in_footprints(float(s['ra']), float(s['dec']), [f for f, k in zip(good, t_in) if k], margin_px=0)
                        if not s_in.all():
                            notes.append(f"{s['name']} outside {int((~s_in).sum())} image(s)")
                    if notes:
                        print("Light Curves - field check: " + "; ".join(notes))
                        ts_status_var.set("Note: " + "; ".join(notes)[:150])
            except Exception as e:
                print(f"Field check skipped: {e}")

            ts_status_var.set(f"Processing {len(files)} files...")
            
            # Reset Phase 3 state
            root.after(0, lambda: cancel_btn.pack(side=tk.BOTTOM, pady=2))
            root.after(0, lambda: ts_progress_var.set(0))
            cancel_event.clear()
            
            def update_prog(val):
                root.after(0, lambda: ts_progress_var.set(val))

            faint_ctx = None
            if ts_faint_var.get():
                # v4.44: faint target - forced photometry at a fixed position, frames combined afterwards
                results, msg, faint_ctx = run_faint_target(files, tar_c, resolved_ensemble, check_star_data, lc_mode,
                                                           flux_per_second, lc_stack_mode, lc_sky_factor, update_prog)
                root.after(0, lambda: cancel_btn.pack_forget())
                if results:
                    ts_publish(results, lc_mode, resolved_ensemble, check_star_data, faint_ctx)
                else:
                    ts_status_var.set(f"Failed: {msg}")
                return

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
                k_coeff=lc_k_coeff,
                filter_name=ts_filter_var.get(),
                site_lat=ts_lat_var.get(),
                site_long=ts_lon_var.get(),
                centroid_initial_radius=vars_dict.get('centroid_initial_radius', [None])[0].get() if 'centroid_initial_radius' in vars_dict else 15.0,
                centroid_tracking_shift=vars_dict.get('centroid_tracking_shift', [None])[0].get() if 'centroid_tracking_shift' in vars_dict else 4.0,
                cancel_event=cancel_event,
                update_progress=update_prog,
                use_flexible_aperture=bool(vars_dict["use_flexible_aperture"][0].get()),
                aperture_fwhm_factor=float(vars_dict["aperture_fwhm_factor"][0].get()),
                annulus_inner_gap=float(vars_dict["annulus_inner_gap"][0].get()),
                annulus_width=float(vars_dict["annulus_width"][0].get()),
                print_psf_fitting=bool(vars_dict.get("print_psf_fitting", [None, None])[0].get()) if "print_psf_fitting" in vars_dict else False,
                lc_mode=lc_mode,
                flux_per_second=flux_per_second,
                stack_mode=lc_stack_mode,
                sky_noise_factor=lc_sky_factor
            )
            
            root.after(0, lambda: cancel_btn.pack_forget())
            
            if results:
                ts_publish(results, lc_mode, resolved_ensemble, check_star_data, faint_ctx)
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
        # v4.18: the report of the last run; otherwise the newest report for this target
        import glob as _glob
        target_name = ts_target_name_var.get().replace(' ','_')
        report_path = vars_dict.get("ts_last_report_path")
        if not (isinstance(report_path, str) and os.path.exists(report_path)):
            cands = sorted(_glob.glob(os.path.join("photometry_output", f"aavso_{target_name}*.txt")), key=os.path.getmtime)
            report_path = os.path.abspath(cands[-1]) if cands else os.path.abspath(os.path.join("photometry_output", f"aavso_{target_name}.txt"))
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

    # (second on_reset_ensemble definition removed in v4.15 - see above)

    # v4.18: named Light-Curve setups (target, comparison/check stars, mode, coefficients)
    SETUP_DIR = "setups"
    def _lc_setup_keys():
        exclude = {"ts_lat", "ts_lon", "ts_show_coeff"}   # site and layout are global, not per target
        return [k for k, v in vars_dict.items()
                if k.startswith("ts_") and k not in exclude
                and isinstance(v, (tuple, list)) and len(v) >= 2 and isinstance(v[0], tk.Variable)]

    def save_lc_setup():
        import json
        from tkinter import filedialog
        os.makedirs(SETUP_DIR, exist_ok=True)
        default = ts_target_name_var.get().strip().replace(' ', '_') or "setup"
        path = filedialog.asksaveasfilename(title="Save Light-Curve Setup", initialdir=os.path.abspath(SETUP_DIR),
                                            initialfile=f"{default}.json", defaultextension=".json",
                                            filetypes=[("Calibra setup", "*.json")])
        if not path:
            return
        data = {'calibra_setup_version': 1, 'values': {}}
        for k in _lc_setup_keys():
            try:
                data['values'][k] = vars_dict[k][0].get()
            except tk.TclError:
                pass
        cache = getattr(on_get_aavso_refs, 'cache', None)
        if cache and getattr(on_get_aavso_refs, 'last_target', '') == ts_target_name_var.get().strip():
            data['aavso_cache'] = cache
            data['aavso_last_target'] = on_get_aavso_refs.last_target
            data['aavso_last_cat'] = getattr(on_get_aavso_refs, 'last_cat', '')
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=float)
            ts_status_var.set(f"Setup saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Setup", f"Could not save the setup:\n{e}")

    def open_lc_setup():
        import json
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Open Light-Curve Setup",
                                          initialdir=os.path.abspath(SETUP_DIR) if os.path.isdir(SETUP_DIR) else ".",
                                          filetypes=[("Calibra setup", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            values = data.get('values', {})
            n = 0
            for k, v in values.items():
                if k in vars_dict and isinstance(vars_dict[k], (tuple, list)) and hasattr(vars_dict[k][0], 'set'):
                    try:
                        vars_dict[k][0].set(v); n += 1
                    except tk.TclError:
                        pass
            if data.get('aavso_cache'):
                on_get_aavso_refs.cache = data['aavso_cache']
                on_get_aavso_refs.last_target = data.get('aavso_last_target', '')
                on_get_aavso_refs.last_cat = data.get('aavso_last_cat', '')
            tidy_numbers()
            ts_status_var.set(f"Setup loaded: {os.path.basename(path)} ({n} values)")
        except Exception as e:
            messagebox.showerror("Open Setup", f"Could not read the setup:\n{e}")

    ts_btn_row1 = tk.Frame(ts_topbar, bg="#e8eef5")
    ts_btn_row1.pack(side=tk.TOP, pady=4, before=ts_progress)

    ts_btn_container = ttk.Frame(ts_container)
    ts_btn_container.pack(pady=10)
    ts_btn_row2 = ttk.Frame(ts_btn_container)
    ts_btn_row2.pack(pady=5)

    run_ts_btn = tk.Button(ts_btn_row1, text="Generate Light Curve", command=on_run_ts,
                           bg="#00796b", fg="white", font=("Arial", 11, "bold"), pady=6, width=22)
    run_ts_btn.pack(side=tk.LEFT, padx=8)

    ts_open_viewer_btn = tk.Button(ts_btn_row1, text=f"{get_icon('🔍', '')} Open Selected FITS in Viewer".strip(), command=on_open_viewer_button,
                                   bg="#1a3a5f", fg="white", font=("Arial", 10, "bold"), pady=6, padx=10)
    ts_open_viewer_btn.pack(side=tk.LEFT, padx=8)

    tk.Button(ts_btn_row1, text="Save Target Setup...", command=lambda: save_lc_setup(), font=("Arial", 9), pady=6).pack(side=tk.LEFT, padx=4)
    tk.Button(ts_btn_row1, text="Open Target Setup...", command=lambda: open_lc_setup(), font=("Arial", 9), pady=6).pack(side=tk.LEFT, padx=4)
    tk.Button(ts_btn_row1, text="Open Output Folder", command=lambda: open_path(os.path.abspath("photometry_output")),
              font=("Arial", 9), pady=6).pack(side=tk.LEFT, padx=4)

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
    
    ts_tree = ttk.Treeview(lf_ts_table, columns=("JD", "HJD", "N", "Mag", "Err", "Check Mag", "SNR", "FWHM", "Flag"), show='headings', height=10,
                           selectmode="extended")
    ts_tree.heading("JD", text="UT (mid)")          # v4.42/4.43: UT date + time of mid-exposure instead of JD
    ts_tree.heading("HJD", text="HJD")
    ts_tree.heading("N", text="N")                  # v4.44: frames per point (faint-target combining)
    ts_tree.heading("Mag", text="Mag")
    ts_tree.heading("Err", text="Err")
    ts_tree.heading("Check Mag", text="Check Mag")
    ts_tree.heading("SNR", text="SNR")
    ts_tree.heading("FWHM", text="FWHM")
    ts_tree.heading("Flag", text="Flag")
    
    ts_tree.column("JD", width=150, anchor=tk.CENTER)
    ts_tree.column("HJD", width=100, anchor=tk.CENTER)
    ts_tree.column("N", width=35, anchor=tk.CENTER)
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
    
    def on_ts_tree_right_click(event):
        region = ts_tree.identify("region", event.x, event.y)
        if region != "cell": return
        row_id = ts_tree.identify_row(event.y)
        if not row_id: return
        if row_id not in ts_tree.selection():
            ts_tree.selection_set(row_id)
        
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Toggle SUSPECT / OK Flag", command=toggle_ts_flag)
        menu.tk_popup(event.x_root, event.y_root)

    def toggle_ts_flag():
        selected = ts_tree.selection()
        if not selected: return
        results = vars_dict.get("ts_last_results", [])
        for item in selected:
            idx = ts_tree.index(item)
            if idx < len(results):
                r = results[idx]
                new_flag = 'OK' if r.get('flag', 'OK') == 'SUSPECT' else 'SUSPECT'
                r['flag'] = new_flag
                vals = list(ts_tree.item(item, 'values'))
                vals[-1] = new_flag
                ts_tree.item(item, values=vals)
        on_inline_toggle()

    ts_tree.bind("<Button-3>", on_ts_tree_right_click)


    # =========================================================================
    # --- TAB 5: DATA MINING ---
    # =========================================================================
    dm_container = tk.Frame(tab_datamining, bg="white")
    dm_container.pack(fill="both", expand=True, padx=20, pady=10)

    tk.Label(dm_container, text=f"{get_icon('🚀', '')}  Data Mining & Variable Star Discovery", font=("Segoe UI", 16, "bold"), fg="#1a3a5f", bg="white").pack(anchor="w", pady=(0,10))
    
    # 1. Config Frame
    lf_dm_config = tk.LabelFrame(dm_container, text="Data Mining Configuration", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_dm_config.pack(fill="x", pady=5)
    
    def show_quiet_tooltip(title, msg):
        top = tk.Toplevel(root)
        top.title(title)
        top.geometry("350x140")
        top.resizable(False, False)
        x = root.winfo_rootx() + root.winfo_width()//2 - 175
        y = root.winfo_rooty() + root.winfo_height()//2 - 70
        top.geometry(f"+{x}+{y}")
        tk.Label(top, text=msg, wraplength=310, justify="left", font=("Segoe UI", 10)).pack(padx=20, pady=(15, 10))
        tk.Button(top, text="OK", command=top.destroy, width=8, bg="#e0e0e0").pack(pady=5)
        top.transient(root)
        top.grab_set()

    def add_dm_setting(label_text, var_name, default_val, row, col, tooltip, vtype=float):
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
        
        ttk.Label(lf_dm_config, text=label_text).grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
        ttk.Entry(lf_dm_config, textvariable=var, width=15).grid(row=row, column=col+1, sticky=tk.W, padx=10, pady=5)
        
        btn = tk.Button(lf_dm_config, text="?", relief="flat", fg="#1a3a5f", bg="#e0e0e0", 
                        command=lambda msg=tooltip, t=label_text: show_quiet_tooltip(t, msg),
                        font=("Arial", 8, "bold"), padx=5, pady=0)
        btn.grid(row=row, column=col+2, padx=(0, 20), sticky="w")

    # Column 1
    add_dm_setting("Detection Sigma Threshold:", "dm_detect_sigma", 5.0, 0, 0, "How many times above the background noise a star must be to be detected.")
    add_dm_setting("Detection FWHM (px):", "detect_fwhm", 3.5, 1, 0, "The assumed Full Width at Half Maximum of stars in pixels for the DAOStarFinder algorithm. Same value as Settings > Detection (one setting since v4.18).")
    add_dm_setting("Edge Boundary Buffer (px):", "dm_edge_buffer", 50, 2, 0, "Ignore stars closer to the image edge than this many pixels.", vtype=int)
    add_dm_setting("Min Valid Frames Fraction:", "dm_min_valid_pct", 0.7, 3, 0, "A star must be successfully measured in at least this fraction (0.0 - 1.0) of frames to be kept.")
    
    # Column 2
    add_dm_setting("Ensemble Size (Stars):", "dm_ensemble_size", 100, 0, 3, "Number of brightest, most stable stars to use for calculating frame-to-frame zero-point offsets.", vtype=int)
    add_dm_setting("Suspect Sigma Threshold:", "dm_suspect_sigma", 3.0, 1, 3, "A star's RMS scatter must be this many times higher than the expected noise floor to be flagged as a Suspect.")
    add_dm_setting("Suspect Minimum Excess (Mag):", "dm_suspect_excess", 0.03, 2, 3, "A star's absolute RMS must exceed the expected noise floor by at least this many magnitudes to be flagged as a Suspect. Prevents flagging micro-variations in ultra-bright stars.")
    
    # Fast Test Mode Checkbox (placed under Column 2)
    var_fast = tk.BooleanVar(value=False)  # v4.16: default off (was on and silently limited the analysis)
    vars_dict["dm_fast_test"] = (var_fast, bool)
    ttk.Checkbutton(lf_dm_config, text="Fast Test Mode (only the 100 brightest stars - quick check, not a full analysis)", variable=var_fast).grid(row=3, column=3, columnspan=3, sticky=tk.W, padx=10, pady=5)

    # v4.17: reuse stored per-frame measurements (photometry_output/dm_cache)
    var_dm_cache = tk.BooleanVar(value=True)
    vars_dict["dm_use_cache"] = (var_dm_cache, bool)
    ttk.Checkbutton(lf_dm_config, text="Reuse stored measurements (re-analysis without re-reading the FITS files)",
                    variable=var_dm_cache).grid(row=4, column=3, columnspan=3, sticky=tk.W, padx=10, pady=5)

    def on_clear_dm_cache():
        import shutil as _sh
        from photometry.data_mining import DM_CACHE_DIR
        if os.path.isdir(DM_CACHE_DIR) and messagebox.askyesno("Clear Data Mining Cache",
                f"Delete all stored Data Mining measurements in\n{os.path.abspath(DM_CACHE_DIR)} ?"):
            _sh.rmtree(DM_CACHE_DIR, ignore_errors=True)
            messagebox.showinfo("Cache Cleared", "Data Mining measurement cache deleted.")
    ttk.Button(lf_dm_config, text="Clear Measurement Cache", command=on_clear_dm_cache).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
    
    dm_status_var = tk.StringVar(value="Ready to discover.")
    tk.Label(dm_container, textvariable=dm_status_var, fg="#d32f2f", bg="white", font=("Arial", 10, "bold")).pack(pady=5)
    
    # Summary Statistics Banner (hidden until results are available)
    dm_summary_frame = tk.Frame(dm_container, bg="#e8f5e9", bd=1, relief="solid")
    # Don't pack yet - packed after results arrive
    
    dm_summary_labels = {}
    for col, key in enumerate(["total_stars", "total_suspects", "total_noisy", "known_vsx", "new_discoveries", "median_rms"]):
        nice_names = {
            "total_stars": "Stars Analyzed", 
            "total_suspects": "Suspects", 
            "total_noisy": "Noisy (Rejected)", 
            "known_vsx": "Known (VSX)", 
            "new_discoveries": "New Discoveries", 
            "median_rms": "Median RMS"
        }
        tk.Label(dm_summary_frame, text=nice_names[key], font=("Segoe UI", 8), fg="#555", bg="#e8f5e9").grid(row=0, column=col, padx=15, pady=(5,0))
        lbl = tk.Label(dm_summary_frame, text="—", font=("Segoe UI", 14, "bold"), fg="#1a3a5f", bg="#e8f5e9")
        lbl.grid(row=1, column=col, padx=15, pady=(0,5))
        dm_summary_labels[key] = lbl
    
    # 2. Execution Button & Progress
    dm_progress = ttk.Progressbar(dm_container, orient="horizontal", length=400, mode="determinate")
    
    def on_run_datamining():
        from photometry.data_mining import detect_all_stars, track_and_measure, calibrate_and_analyze, cross_match_vsx_suspects
        import threading
        import numpy as np
        import os
        from astropy.io import fits as dm_fits
        from astropy.wcs import WCS as dm_WCS
        
        selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
        if not selected_iids:
            messagebox.showerror("No Files", "Please select at least one FITS file in the File Manager.")
            return
            
        fits_files = [loaded_files[int(iid)]['path'] for iid in selected_iids]
        ref_file = fits_files[0]
        
        detect_sig = float(vars_dict["dm_detect_sigma"][0].get())
        dm_fwhm_val = float(vars_dict["detect_fwhm"][0].get())
        sat_limit = int(vars_dict["saturation_limit"][0].get())
        edge_buf = int(vars_dict["dm_edge_buffer"][0].get())
        min_pct = float(vars_dict["dm_min_valid_pct"][0].get())
        ens_size = int(vars_dict["dm_ensemble_size"][0].get())
        susp_sig = float(vars_dict["dm_suspect_sigma"][0].get())
        susp_exc = float(vars_dict["dm_suspect_excess"][0].get())
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
        
        # v4.16: read all Tk settings on the GUI thread (tkinter is not thread-safe)
        sharplo_val = float(vars_dict["dao_sharplo"][0].get())
        sharphi_val = float(vars_dict["dao_sharphi"][0].get())
        roundlo_val = float(vars_dict["dao_roundlo"][0].get())
        roundhi_val = float(vars_dict["dao_roundhi"][0].get())
        dm_centroid_shift = vars_dict['centroid_tracking_shift'][0].get() if 'centroid_tracking_shift' in vars_dict else 4.0
        dm_gain = float(vars_dict["ccd_gain"][0].get())
        dm_stack_mode = vars_dict["stack_mode"][0].get() if "stack_mode" in vars_dict else "average"
        dm_sky_factor = float(vars_dict["sky_noise_factor"][0].get()) if "sky_noise_factor" in vars_dict else 1.0
        dm_use_cache = bool(var_dm_cache.get())

        class _ThreadSafeStatus:
            """Forwards .set() to the GUI thread."""
            def set(self, text):
                root.after(0, lambda t=text: dm_status_var.set(t))

        def dm_thread():
            dm_status = _ThreadSafeStatus()
            try:
                # DAOStarFinder parameters were read from the Settings tab above
                fwhm_val = dm_fwhm_val
                
                dm_status.set("Phase 1/3: Detecting stars in reference frame...")
                master_cat = detect_all_stars(
                    ref_file, detect_sigma=detect_sig, saturation_limit=sat_limit, edge_buffer=edge_buf,
                    fwhm=fwhm_val, sharplo=sharplo_val, sharphi=sharphi_val, 
                    roundlo=roundlo_val, roundhi=roundhi_val, use_cache=dm_use_cache
                )
                
                if cancel_event.is_set(): return
                
                if is_fast and len(master_cat) > 100:
                    master_cat = master_cat[:100]
                    
                if not master_cat:
                    dm_status.set("Error: No stars detected in reference frame.")
                    return
                    
                dm_status.set(f"Phase 2/3: Tracking {len(master_cat)} stars across {len(fits_files)} frames...")
                
                def update_prog(val):
                    root.after(0, lambda: dm_progress.configure(value=val))
                    
                track_res, warnings_lst = track_and_measure(
                    fits_files, master_cat, 
                    aperture_radius=ap_rad, annulus_inner=ann_in, annulus_outer=ann_out,
                    saturation_limit=sat_limit, edge_buffer=edge_buf,
                    centroid_tracking_shift=dm_centroid_shift,
                    cancel_event=cancel_event, progress_callback=update_prog,
                    gain=dm_gain, stack_mode=dm_stack_mode, sky_noise_factor=dm_sky_factor,
                    fwhm=dm_fwhm_val, use_cache=dm_use_cache
                )
                
                if cancel_event.is_set(): return
                if warnings_lst:
                    warn_msg = "\n".join(warnings_lst)
                    root.after(0, lambda: messagebox.showwarning("Data Mining Warnings", warn_msg))
                    
                # VSX Cross-Match: extract center RA/Dec from reference frame
                center_ra_dm, center_dec_dm = None, None
                try:
                    with dm_fits.open(ref_file) as hdul:
                        hdr = hdul[0].header
                        w = dm_WCS(hdr)
                        if w.has_celestial:
                            ny, nx = hdul[0].data.shape[:2] if hdul[0].data is not None else (1000, 1000)
                            center_ra_dm, center_dec_dm = w.all_pix2world(nx/2, ny/2, 0)
                            center_ra_dm, center_dec_dm = float(center_ra_dm), float(center_dec_dm)
                except Exception:
                    pass

                dm_status.set("Phase 3/3: Calibrating and Analyzing Statistics...")
                
                out_csv = os.path.abspath(os.path.join("photometry_output", "data_mining_suspects.csv"))
                final_stats, exp_func = calibrate_and_analyze(
                    track_res, master_cat, out_csv, min_valid_fraction=min_pct, 
                    ensemble_size=ens_size, center_ra=center_ra_dm, center_dec=center_dec_dm,
                    suspect_sigma=susp_sig, min_excess=susp_exc
                )
                
                if not final_stats:
                    dm_status.set("Analysis complete, but no valid stars survived filters.")
                    return
                
                dm_status.set("Cross-matching suspects with AAVSO VSX catalog...")
                known_count = cross_match_vsx_suspects(final_stats, center_ra_dm, center_dec_dm, radius_arcmin=30.0, verbose=True)
                    
                # Update UI
                def update_ui():
                    dm_tree.final_stats = final_stats
                    
                    # Reset sorting column text highlights (removing / adding arrows)
                    col_headers = {
                        "ID": "ID",
                        "RA (hms)": "RA (hms)",
                        "Dec (dms)": "Dec (dms)",
                        "RA (deg)": "RA (deg)",
                        "Dec (deg)": "Dec (deg)",
                        "Mean Mag": "Mean Mag",
                        "RMS": "RMS",
                        "Excess": "Excess RMS",
                        "Valid": "Valid",
                        "Flag": "Flag",
                        "VSX Name": "VSX Name",
                        "VSX Type": "VSX Type",
                        "Catalog ID": "Catalog ID"
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
                    
                    stable_stars_list = [s for s in final_stats if not s.get('is_suspect') and not s.get('is_noisy')]
                    suspect_stars_list = [s for s in final_stats if s.get('is_suspect')]
                    noisy_stars_list = [s for s in final_stats if s.get('is_noisy')]
                    
                    mags_stable = np.array([s['mean_mag'] for s in stable_stars_list])
                    rms_stable = np.array([s['rms'] for s in stable_stars_list])
                    
                    mags_sus = np.array([s['mean_mag'] for s in suspect_stars_list])
                    rms_sus = np.array([s['rms'] for s in suspect_stars_list])
                    
                    mags_noisy = np.array([s['mean_mag'] for s in noisy_stars_list])
                    rms_noisy = np.array([s['rms'] for s in noisy_stars_list])
                    
                    # Task 4: Added picker=5 to enable clicks on points
                    sc_stable = ax.scatter(mags_stable, rms_stable, c='blue', alpha=0.5, s=10, label='Stable', picker=5)
                    sc_suspect = ax.scatter(mags_sus, rms_sus, c='red', alpha=0.9, s=40, label='Suspect', marker='*', picker=5)
                    sc_noisy = ax.scatter(mags_noisy, rms_noisy, c='#ff9100', alpha=0.8, s=30, label='Noisy (outliers)', marker='^', picker=5)
                    
                    # Task 4: Store lists on collections to map indexes back to stars
                    sc_stable.star_list = stable_stars_list
                    sc_suspect.star_list = suspect_stars_list
                    sc_noisy.star_list = noisy_stars_list
                    
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
                    
                    # Update summary statistics banner
                    n_suspects = len(suspect_stars_list)
                    n_noisy = len(noisy_stars_list)
                    n_total = len(final_stats)
                    n_known = sum(1 for s in final_stats if s.get('is_suspect') and s.get('vsx_name'))
                    n_new = max(0, n_suspects - n_known)
                    med_rms = float(np.median(rms))
                    
                    dm_summary_labels["total_stars"].config(text=str(n_total))
                    dm_summary_labels["total_suspects"].config(text=str(n_suspects), fg="#d32f2f" if n_suspects > 0 else "#1a3a5f")
                    dm_summary_labels["total_noisy"].config(text=str(n_noisy), fg="#7b1fa2" if n_noisy > 0 else "#1a3a5f")
                    dm_summary_labels["known_vsx"].config(text=str(n_known))
                    dm_summary_labels["new_discoveries"].config(text=str(n_new), fg="#e65100" if n_new > 0 else "#1a3a5f")
                    dm_summary_labels["median_rms"].config(text=f"{med_rms:.4f}")
                    dm_summary_frame.pack(fill="x", padx=10, pady=(0, 5), before=dm_progress)
                    
                    status_parts = [f"Complete! {n_suspects} suspect(s)"]
                    if n_noisy > 0:
                        status_parts.append(f"{n_noisy} noisy star(s) rejected")
                    if n_known > 0:
                        status_parts.append(f"{n_known} known VSX")
                    if n_new > 0:
                        status_parts.append(f"{n_new} potential new discovery!")
                    status_parts.append(f"Saved to {out_csv}")
                    dm_status.set(" | ".join(status_parts))
                
                root.after(0, update_ui)
                
            except Exception as e:
                root.after(0, lambda err=e: messagebox.showerror("Error", str(err)))
                dm_status.set("Error occurred.")
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
    
    def choose_dm_comparisons(target, stats, n_comp=4, n_check=1):
        """
        v4.18: pick comparison and check stars for a Data-Mining star from the stored
        Data-Mining statistics: constant stars (not suspect/noisy, no disturbed
        background, RMS close to their own photometric error), measured in (almost)
        all frames, similar brightness (within 1 mag, widened if necessary) and
        close on the sky. Returns (comps, checks) lists of stat dicts.
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        n_max = max((int(s_.get('n_valid', 0)) for s_ in stats), default=0)
        pool = []
        for s_ in stats:
            if s_ is target or s_.get('id') == target.get('id'):
                continue
            if s_.get('is_suspect') or s_.get('is_noisy') or s_.get('high_bg_var'):
                continue
            if str(s_.get('vsx_name', '')).strip():
                continue                                  # known variable
            if int(s_.get('n_valid', 0)) < 0.95 * n_max:
                continue
            nr = s_.get('noise_ratio', np.nan)
            if np.isfinite(nr) and nr > 1.5:
                continue
            if not np.isfinite(s_.get('mean_mag', np.nan)):
                continue
            pool.append(s_)
        if not pool:
            return [], []
        tc = SkyCoord(target['ra_deg'] * u.deg, target['dec_deg'] * u.deg)
        pc = SkyCoord([p_['ra_deg'] for p_ in pool] * u.deg, [p_['dec_deg'] for p_ in pool] * u.deg)
        dist = tc.separation(pc).arcmin
        dmag = np.abs(np.array([p_['mean_mag'] for p_ in pool]) - target['mean_mag'])
        need = n_comp + n_check
        for window in (1.0, 1.5, 2.5, 99.0):
            ok = np.where(dmag <= window)[0]
            if len(ok) >= need:
                break
        score = dmag[ok] + dist[ok] / 30.0               # 1 mag ~ 30 arcmin
        order = ok[np.argsort(score)]
        chosen = [pool[i] for i in order[:need]]
        return chosen[:n_comp], chosen[n_comp:need]

    def send_dm_star_to_lc(star_id, auto_comps=True):
        """v4.18: send a Data-Mining star to Light Curves; optionally also fill the
        comparison/check star slots automatically (Relative-flux mode)."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        stats = getattr(dm_tree, 'final_stats', []) or []
        target = next((s_ for s_ in stats if str(s_['id']) == str(star_id)), None)
        if target is None:
            messagebox.showwarning("Not Found", f"Star {star_id} not found in the Data Mining results.")
            return
        coord = SkyCoord(ra=target['ra_deg'] * u.deg, dec=target['dec_deg'] * u.deg)
        vars_dict["ts_target_ra"][0].set(coord.ra.to_string(unit=u.hourangle, sep=':', precision=2))
        vars_dict["ts_target_dec"][0].set(coord.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True))
        vars_dict["ts_target_mode"][0].set("manual")
        def _dm_name(st_):
            sid = str(st_['id'])
            return sid if sid.startswith("DM_") else f"DM_{sid}"
        name = str(target.get('vsx_name', '')).strip() or str(target.get('catalog_id', '')).strip() or _dm_name(target)
        ts_target_name_var.set(name)

        msg = f"Target {target['id']} ({name}) sent to the Light Curves tab."
        if auto_comps:
            comps, checks = choose_dm_comparisons(target, stats)
            if comps:
                slots = comps + checks
                for i in range(5):
                    if i < len(slots):
                        c_ = slots[i]
                        cn = str(c_.get('catalog_id', '')).strip() or _dm_name(c_)
                        cc = SkyCoord(c_['ra_deg'] * u.deg, c_['dec_deg'] * u.deg)
                        vars_dict[f"ts_ref_{i}_name"][0].set(cn)
                        vars_dict[f"ts_ref_{i}_ra"][0].set(float(c_['ra_deg']))
                        vars_dict[f"ts_ref_{i}_dec"][0].set(float(c_['dec_deg']))
                        vars_dict[f"ts_ref_{i}_has_manual"][0].set(True)
                        vars_dict[f"ts_ref_{i}_mag"][0].set(0.0)
                        vars_dict[f"ts_ref_{i}_bv"][0].set(0.0)
                        vars_dict[f"ts_ref_{i}_use"][0].set(i < len(comps))
                        if f"ts_ref_{i}_coord_label" in vars_dict:
                            vars_dict[f"ts_ref_{i}_coord_label"][0].set(
                                f"({cc.ra.to_string(unit='hour', sep=':', precision=1)}, "
                                f"{cc.dec.to_string(unit='degree', sep=':', precision=1, alwayssign=True)}) [DM, dm={c_['mean_mag'] - target['mean_mag']:+.2f}]")
                    else:
                        vars_dict[f"ts_ref_{i}_name"][0].set("")
                        vars_dict[f"ts_ref_{i}_has_manual"][0].set(False)
                        vars_dict[f"ts_ref_{i}_use"][0].set(False)
                        if f"ts_ref_{i}_coord_label" in vars_dict:
                            vars_dict[f"ts_ref_{i}_coord_label"][0].set("")
                ts_check_star_idx_var.set(len(comps) if checks else -1)
                vars_dict["ts_lc_mode"][0].set("relative")
                msg += (f"\n\n{len(comps)} comparison star(s){' and 1 check star' if checks else ''} were chosen automatically "
                        f"(constant in the Data Mining analysis, similar brightness, close on the sky) and the mode was set to "
                        f"'Relative flux'.\n\nFor calibrated magnitudes use 'Fetch' on each star (catalogue values) or "
                        f"'Get AAVSO Ref Stars', then switch to 'Calibrated magnitudes'.")
            else:
                msg += "\n\nNo suitable comparison stars were found automatically - please add them manually."
        messagebox.showinfo("Sent to Light Curves", msg)
        switch_tab("ts")

    def on_send_suspect():
        selected = dm_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a star from the table first.")
            return
        vals = dm_tree.item(selected[0], 'values')
        send_dm_star_to_lc(vals[0], auto_comps=True)

    send_btn = tk.Button(dm_btn_frame, text="Send to Light Curves (+ comp stars)", command=on_send_suspect,
                           bg="#f57c00", fg="white", font=("Arial", 10, "bold"), pady=10, padx=15)
    send_btn.pack(side=tk.LEFT, padx=10)
    tk.Button(dm_btn_frame, text="Open Output Folder", command=lambda: open_path(os.path.abspath("photometry_output")),
              font=("Arial", 9), pady=10).pack(side=tk.RIGHT, padx=10)

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

    def on_export_all_click():
        if not hasattr(dm_tree, 'final_stats') or not dm_tree.final_stats:
            messagebox.showwarning("No Data", "No results to export.")
            return
        from tkinter import filedialog
        import csv
        path = filedialog.asksaveasfilename(initialdir="photometry_output", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID", "RA (deg)", "Dec (deg)", "Mean Mag", "RMS", "Expected RMS", "Excess RMS", "Valid Frames", "Flag", "VSX Name", "VSX Type"])
                    for s in dm_tree.final_stats:
                        writer.writerow([
                            s['id'], f"{s['ra_deg']:.6f}", f"{s['dec_deg']:.6f}", 
                            f"{s['mean_mag']:.4f}", f"{s['rms']:.5f}", f"{s.get('expected_rms', 0):.5f}", 
                            f"{s['excess_rms']:.5f}", s.get('n_valid', 0),
                            'SUSPECT' if s.get('is_suspect') else ('NOISY' if s.get('is_noisy') else ''),
                            s.get('vsx_name', ''), s.get('vsx_type', '')
                        ])
                messagebox.showinfo("Export Complete", f"Exported {len(dm_tree.final_stats)} stars to {path}")
            except Exception as e:
                messagebox.showerror("Export Failed", str(e))

    export_all_btn = tk.Button(dm_btn_frame, text="Export All Results", command=on_export_all_click,
                             bg="#f0f2f5", fg="#1a3a5f", font=("Arial", 10, "bold"), pady=10, padx=15)
    export_all_btn.pack(side=tk.LEFT, padx=10)

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
            "5. VSX Cross-Matching:\n"
            "   Automatically cross-matches suspects against the AAVSO International Variable Star Index (VSX) within 5 arcseconds. This flags known variables with their VSX Name and Type, making it easy to identify truly new discoveries.\n\n"
            "----------------------------\n"
            "RESULTS & INSPECTION FEATURES:\n"
            "- Summary Banner: Real-time statistics showing Stars Analyzed, Suspects found, Known VSX matches, New Discoveries, and Median RMS.\n"
            "- Row Coloring: Suspects passing the threshold are highlighted in red. 'Borderline' candidates (high excess scatter but not meeting full threshold criteria) are highlighted in amber for manual review.\n"
            "- Right-Click Context Menu: Right-click any suspect to copy its RA/Dec, send it to the Time-Series/Light Curves tab, or copy the row data as CSV.\n"
            "- Variability Diagram: Stable stars are plotted as small blue dots. The green line represents the expected noise floor. Discovered Suspects are highlighted as large red stars.\n"
            "- Exporting: Use 'Export All Results' to output coordinates and photometry stats for every single tracked star in the field (both stable and variable) to a CSV for external analysis.\n"
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
    
    # Task 2: Added "Valid" and "Catalog ID" column to columns tuple
    dm_tree = ttk.Treeview(lf_dm_table, columns=("ID", "RA (hms)", "Dec (dms)", "RA (deg)", "Dec (deg)", "Mean Mag", "RMS", "Excess", "Valid", "Flag", "VSX Name", "VSX Type", "Catalog ID"), show='headings', height=8)
    
    # Configure row tags for coloring
    dm_tree.tag_configure("suspect", background="#ffcccc")
    dm_tree.tag_configure("borderline", background="#fff3cd")
    dm_tree.tag_configure("noisy", background="#e8eaf6")
    
    def populate_dm_tree():
        for item in dm_tree.get_children():
            dm_tree.delete(item)
            
        show_all = show_non_suspects_var.get()
        
        stats_to_show = []
        if hasattr(dm_tree, 'final_stats') and dm_tree.final_stats:
            for s in dm_tree.final_stats:
                if show_all or s.get('is_suspect') or s.get('is_noisy'):
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
            elif col == "RA (deg)":
                stats_to_show.sort(key=lambda x: x['ra_deg'], reverse=rev)
            elif col == "Dec (deg)":
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
                stats_to_show.sort(key=lambda x: ("SUSPECT" if x.get('is_suspect') else ("NOISY" if x.get('is_noisy') else "")), reverse=rev)
            elif col == "VSX Name":
                stats_to_show.sort(key=lambda x: x.get('vsx_name', ''), reverse=rev)
            elif col == "VSX Type":
                stats_to_show.sort(key=lambda x: x.get('vsx_type', ''), reverse=rev)
            elif col == "Catalog ID":
                stats_to_show.sort(key=lambda x: x.get('catalog_id', ''), reverse=rev)
                
        for s in stats_to_show:
            flag = "SUSPECT" if s.get('is_suspect') else ("NOISY" if s.get('is_noisy') else "")
            if s.get('high_bg_var'):
                flag = flag + " (High Bg Var)" if flag else "(High Bg Var)"
                
            vsx_name = s.get('vsx_name', '')
            vsx_type = s.get('vsx_type', '')
            
            # Row coloring based on flag
            tag = ""
            if "SUSPECT" in flag:
                tag = "suspect"
            elif "NOISY" in flag:
                tag = "noisy"
            elif s['excess_rms'] > 0.02 and s['rms'] > 2.0 * s.get('expected_rms', 0):
                tag = "borderline"
                
            # Populate Valid count and Catalog ID
            catalog_id = s.get('catalog_id', '')
            
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(ra=s['ra_deg']*u.deg, dec=s['dec_deg']*u.deg)
            ra_hms = c.ra.to_string(unit=u.hourangle, sep=':', precision=2)
            dec_dms = c.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)
            
            dm_tree.insert("", tk.END, values=(
                s['id'], ra_hms, dec_dms, f"{s['ra_deg']:.5f}", f"{s['dec_deg']:.5f}",
                f"{s['mean_mag']:.3f}", f"{s['rms']:.4f}", f"{s['excess_rms']:.4f}",
                f"{s.get('n_valid', 0)}", flag, vsx_name, vsx_type, catalog_id
            ), tags=(tag,) if tag else ())
            
    def sort_column(col):
        if dm_tree.sort_col == col:
            dm_tree.sort_desc = not dm_tree.sort_desc
        else:
            dm_tree.sort_col = col
            dm_tree.sort_desc = False
            
        col_headers = {
            "ID": "ID",
            "RA (hms)": "RA (hms)",
            "Dec (dms)": "Dec (dms)",
            "RA (deg)": "RA (deg)",
            "Dec (deg)": "Dec (deg)",
            "Mean Mag": "Mean Mag",
            "RMS": "RMS",
            "Excess": "Excess RMS",
            "Valid": "Valid",
            "Flag": "Flag",
            "VSX Name": "VSX Name",
            "VSX Type": "VSX Type",
            "Catalog ID": "Catalog ID"
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
        
    def show_dm_context_menu(item, x_root, y_root, star_stat=None):
        if item is not None:
            dm_tree.selection_set(item)
            vals = dm_tree.item(item, 'values')
        else:
            if not star_stat: return
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(ra=star_stat['ra_deg']*u.deg, dec=star_stat['dec_deg']*u.deg)
            ra_hms = c.ra.to_string(unit=u.hourangle, sep=':', precision=2)
            dec_dms = c.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)
            flag = "SUSPECT" if star_stat.get('is_suspect') else "NOISY" if star_stat.get('is_noisy') else ""
            vals = (
                star_stat['id'], ra_hms, dec_dms, f"{star_stat['ra_deg']:.5f}", f"{star_stat['dec_deg']:.5f}",
                f"{star_stat['mean_mag']:.3f}", f"{star_stat['rms']:.4f}", f"{star_stat['excess_rms']:.4f}",
                f"{star_stat.get('n_valid', 0)}", flag, star_stat.get('vsx_name', ''), star_stat.get('vsx_type', ''), star_stat.get('catalog_id', '')
            )

        menu = tk.Menu(root, tearoff=0)
        ra, dec = vals[3], vals[4]
        
        menu.add_command(label="Copy RA/Dec", command=lambda: root.clipboard_clear() or root.clipboard_append(f"{ra}, {dec}"))
        
        def local_send_to_lc():
            send_dm_star_to_lc(vals[0], auto_comps=True)
            
        menu.add_command(label="Send to Light Curves", command=local_send_to_lc)
        
        def open_in_viewer():
            selected_iids = [iid for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
            if not selected_iids:
                messagebox.showwarning("No Selection", "Please check a FITS file in the File Manager to view it.")
                return
            file_path = loaded_files[int(selected_iids[0])]['path']
            
            star_id = vals[0]
            target_stat = None
            if hasattr(dm_tree, 'final_stats'):
                for s in dm_tree.final_stats:
                    if str(s['id']) == str(star_id):
                        target_stat = s
                        break
            
            if target_stat:
                viewer_win = tk.Toplevel(root)
                from photometry.fits_viewer import FITSViewer
                
                viewer_config = {
                    'aperture_radius': float(vars_dict["aperture_radius"][0].get()),
                    'annulus_inner': float(vars_dict["annulus_inner"][0].get()),
                    'annulus_outer': float(vars_dict["annulus_outer"][0].get()),
                    'use_flexible_aperture': vars_dict["use_flexible_aperture"][0].get(),
                    'aperture_fwhm_factor': float(vars_dict["aperture_fwhm_factor"][0].get()),
                    'annulus_inner_gap': float(vars_dict["annulus_inner_gap"][0].get()),
                    'annulus_width': float(vars_dict["annulus_width"][0].get()),
                    'filter_v_keyword': vars_dict["filter_v_keyword"][0].get(),
                    'filter_b_keyword': vars_dict["filter_b_keyword"][0].get(),
                }
                # v4.15: pass the configured catalogue and the matching default ZP
                dm_band = band_of(loaded_files[int(selected_iids[0])].get('filter', ''))
                dm_zp_key = "default_zp_b" if dm_band == 'B' else "default_zp_v"
                try:
                    dm_def_zp = float(vars_dict[dm_zp_key][0].get())
                except (tk.TclError, ValueError, KeyError):
                    dm_def_zp = 23.399
                
                FITSViewer(viewer_win, file_path, config=viewer_config,
                           ref_catalog=vars_dict["reference_catalog"][0].get(), default_zp=dm_def_zp,
                           initial_stars={'variable': {'ra': target_stat['ra_deg'], 'dec': target_stat['dec_deg'], 'name': star_id, 'zoom': True}, 'check': None, 'refs': []})
        
        menu.add_command(label="Open in FITS Viewer", command=open_in_viewer)
        
        def copy_csv():
            root.clipboard_clear()
            root.clipboard_append(",".join(map(str, vals)))
            
        def remove_suspect():
            star_id = vals[0]
            if hasattr(dm_tree, 'final_stats'):
                for s in dm_tree.final_stats:
                    if str(s['id']) == str(star_id):
                        s['is_suspect'] = False
                        break
                populate_dm_tree()
                
                # Refresh plot and banner
                if not hasattr(dm_tree, 'ax') or not dm_tree.ax: return
                ax = dm_tree.ax
                ax.clear()
                dm_tree.highlight_artists = []
                
                final_stats = dm_tree.final_stats
                stable_stars_list = [s for s in final_stats if not s.get('is_suspect') and not s.get('is_noisy')]
                suspect_stars_list = [s for s in final_stats if s.get('is_suspect')]
                noisy_stars_list = [s for s in final_stats if s.get('is_noisy')]
                
                mags_stable = np.array([s['mean_mag'] for s in stable_stars_list])
                rms_stable = np.array([s['rms'] for s in stable_stars_list])
                
                mags_sus = np.array([s['mean_mag'] for s in suspect_stars_list])
                rms_sus = np.array([s['rms'] for s in suspect_stars_list])
                
                mags_noisy = np.array([s['mean_mag'] for s in noisy_stars_list])
                rms_noisy = np.array([s['rms'] for s in noisy_stars_list])
                
                if len(mags_stable) > 0:
                    sc_stable = ax.scatter(mags_stable, rms_stable, c='blue', alpha=0.5, s=10, label='Stable', picker=5)
                    sc_stable.star_list = stable_stars_list
                if len(mags_sus) > 0:
                    sc_suspect = ax.scatter(mags_sus, rms_sus, c='red', alpha=0.9, s=40, label='Suspect', marker='*', picker=5)
                    sc_suspect.star_list = suspect_stars_list
                if len(mags_noisy) > 0:
                    sc_noisy = ax.scatter(mags_noisy, rms_noisy, c='#ff9100', alpha=0.8, s=30, label='Noisy (outliers)', marker='^', picker=5)
                    sc_noisy.star_list = noisy_stars_list
                    
                ax.set_xlabel("Mean Instrumental Mag")
                ax.set_ylabel("RMS (std dev)")
                ax.set_title("Stellar Variability Diagram")
                ax.legend()
                if not ax.xaxis_inverted():
                    ax.invert_xaxis()
                dm_canvas.draw()
                
                # Update summary banner
                n_suspects = len(suspect_stars_list)
                n_known = sum(1 for s in final_stats if s.get('is_suspect') and s.get('vsx_name'))
                n_new = max(0, n_suspects - n_known)
                
                dm_summary_labels["total_suspects"].config(text=str(n_suspects), fg="#d32f2f" if n_suspects > 0 else "#1a3a5f")
                dm_summary_labels["new_discoveries"].config(text=str(n_new), fg="#e65100" if n_new > 0 else "#1a3a5f")
                dm_summary_labels["known_vsx"].config(text=str(n_known))

        if "SUSPECT" in vals[9]:
            menu.add_command(label="Remove from Suspects", command=remove_suspect)
        menu.add_command(label="Copy Row as CSV", command=copy_csv)
        
        menu.post(x_root, y_root)

    def on_dm_right_click(event):
        item = dm_tree.identify_row(event.y)
        if item:
            show_dm_context_menu(item, event.x_root, event.y_root)
            
    # Task 4: Pick event callback for matplotlib canvas click
    def on_canvas_pick(event):
        col = event.artist
        ind = event.ind[0]
        if hasattr(col, 'star_list'):
            star = col.star_list[ind]
            star_id = star['id']
            
            found = False
            item_id = None
            for item in dm_tree.get_children():
                if str(dm_tree.item(item, 'values')[0]) == str(star_id):
                    dm_tree.selection_set(item)
                    dm_tree.see(item)
                    found = True
                    item_id = item
                    break
                    
            highlight_star_in_plot(star)
            
            if getattr(event.mouseevent, 'button', 1) == 3:
                # Get screen coordinates for the right click from the mouse event
                x_root = event.mouseevent.guiEvent.x_root
                y_root = event.mouseevent.guiEvent.y_root
                show_dm_context_menu(item_id, x_root, y_root, star_stat=star)
            
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
    dm_tree.heading("RA (hms)", text="RA (hms)")
    dm_tree.heading("Dec (dms)", text="Dec (dms)")
    dm_tree.heading("RA (deg)", text="RA (deg)", command=lambda: sort_column("RA (deg)"))
    dm_tree.heading("Dec (deg)", text="Dec (deg)", command=lambda: sort_column("Dec (deg)"))
    dm_tree.heading("Mean Mag", text="Mean Mag", command=lambda: sort_column("Mean Mag"))
    dm_tree.heading("RMS", text="RMS", command=lambda: sort_column("RMS"))
    dm_tree.heading("Excess", text="Excess RMS ▼", command=lambda: sort_column("Excess"))
    dm_tree.heading("Valid", text="Valid", command=lambda: sort_column("Valid"))
    dm_tree.heading("Flag", text="Flag", command=lambda: sort_column("Flag"))
    dm_tree.heading("VSX Name", text="VSX Name", command=lambda: sort_column("VSX Name"))
    dm_tree.heading("VSX Type", text="VSX Type", command=lambda: sort_column("VSX Type"))
    dm_tree.heading("Catalog ID", text="Catalog ID", command=lambda: sort_column("Catalog ID"))
    
    # Task 2: Setup Column Widths
    dm_tree.column("ID", width=70, anchor=tk.CENTER)
    dm_tree.column("RA (hms)", width=90, anchor=tk.CENTER)
    dm_tree.column("Dec (dms)", width=90, anchor=tk.CENTER)
    dm_tree.column("RA (deg)", width=80, anchor=tk.CENTER)
    dm_tree.column("Dec (deg)", width=80, anchor=tk.CENTER)
    dm_tree.column("Mean Mag", width=70, anchor=tk.CENTER)
    dm_tree.column("RMS", width=70, anchor=tk.CENTER)
    dm_tree.column("Excess", width=70, anchor=tk.CENTER)
    dm_tree.column("Valid", width=60, anchor=tk.CENTER)
    dm_tree.column("Flag", width=70, anchor=tk.CENTER)
    dm_tree.column("VSX Name", width=120, anchor=tk.W)
    dm_tree.column("VSX Type", width=80, anchor=tk.W)
    dm_tree.column("Catalog ID", width=120, anchor=tk.CENTER)
    
    # Bindings
    dm_tree.bind("<<TreeviewSelect>>", on_dm_tree_select)
    dm_tree.bind("<Double-1>", on_dm_double_click)
    dm_tree.bind("<Button-3>", on_dm_right_click)
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
    # v4.16 noise model (see photometry/noise.py and manual section 2.3)
    add_dropdown(lf_ccd, "Stacked images (STACKCNT):", "stack_mode", ["average", "sum", "none"], "average", 4)
    add_entry(lf_ccd, "Sky Noise Factor (>=1):", "sky_noise_factor", 1.0, 5)
    tk.Label(lf_ccd, text="'average': effective gain = gain x STACKCNT. Sky noise factor: correlated noise in stacks "
                          "(the pipeline log prints a measured value).", fg="#555", font=("Arial", 8, "italic"),
             wraplength=600, justify=tk.LEFT).grid(row=6, column=0, columnspan=4, sticky=tk.W, padx=10)

    # Detection (from old TAB 2)
    lf_det = tk.LabelFrame(sub_phot, text="Detection (DAOStarFinder)")
    lf_det.pack(fill="x", padx=10, pady=10)
    add_entry(lf_det, "Detection Sigma:", "detect_sigma", 5.0, 0)
    add_entry(lf_det, "Detection FWHM (px):", "detect_fwhm", 3.5, 0, col_offset=1)
    add_entry(lf_det, "Sharpness Low:", "dao_sharplo", 0.2, 1, col_offset=0)
    add_entry(lf_det, "Sharpness High:", "dao_sharphi", 1.0, 1, col_offset=1)
    add_entry(lf_det, "Roundness Low:", "dao_roundlo", -1.2, 2, col_offset=0)
    add_entry(lf_det, "Roundness High:", "dao_roundhi", 1.2, 2, col_offset=1)

    # Centroid Tracking
    lf_centroid = tk.LabelFrame(sub_phot, text="Centroid Tracking")
    lf_centroid.pack(fill="x", padx=10, pady=10)
    add_entry(lf_centroid, "Initial Search Radius (px):", "centroid_initial_radius", 15.0, 0, col_offset=0)
    add_entry(lf_centroid, "Max Tracking Shift (px):", "centroid_tracking_shift", 4.0, 0, col_offset=1)

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
    tk.Label(lf_cal, text="0 = automatic (image half-diagonal)", fg="#555", font=("Arial", 8, "italic")).grid(row=3, column=2, sticky=tk.W)
    # v4.16: calibration-star selection
    add_entry(lf_cal, "Max Peak for Calib Stars (ADU):", "calib_peak_limit", 30000, 4, vtype=int)
    add_check(lf_cal, "Exclude stars with a neighbour inside the annulus (blends) from calibration", "calib_exclude_blended", True, 5)
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
    # v4.18: observing site (single place; used only if the FITS header lacks SITELAT/SITELONG)
    add_entry(lf_obs, "Site Latitude (deg, fallback):", "ts_lat", 59.8, 2)
    add_entry(lf_obs, "Site Longitude (deg E, fallback):", "ts_lon", 17.6, 3)


    # Output Toggles (from old TAB 4)
    lf_out = tk.LabelFrame(sub_ops, text="Console & Plot Toggles")
    lf_out.pack(fill="x", padx=10, pady=10)
    add_check(lf_out, "Print Detailed Calibration to Console", "print_detailed_calibration", False, 0)
    add_check(lf_out, "Print Massive Aperture Photometry Table", "print_star_detection_table", False, 1)
    add_check(lf_out, "Print PSF Quality & Fitting Details (Console)", "print_psf_fitting", False, 2)
    add_check(lf_out, "Display Matplotlib Plots (Blocking)", "display_plots", False, 3)
    add_entry(lf_out, "Max PSF plots to show/save per file:", "max_plots_to_show_per_file", 3, 4, vtype=int)


    # =========================================================================
    # --- TAB: FIELD VARIABLES (v4.19) ---
    # Light curves of several known variables in the field (long and short
    # periods): targets from a list file or from VSX / GCVS for the image field.
    # =========================================================================
    import re as _fv_re
    from tkinter import filedialog as _fv_fd
    fv_container = tk.Frame(tab_fieldvars, bg="white")
    fv_container.pack(fill="both", expand=True, padx=20, pady=10)
    tk.Label(fv_container, text=f"{get_icon('✨', '')}  Field Variables - light curves of several known variables".strip(),
             font=("Segoe UI", 16, "bold"), fg="#1a3a5f", bg="white").pack(anchor="w", pady=(0, 4))
    tk.Label(fv_container, bg="white", justify="left", fg="#444", font=("Segoe UI", 9),
             text="Measures light curves for many variables at once - e.g. long-period variables over several nights.\n"
                  "Targets: load a list (name + RA/Dec) or query AAVSO VSX / GCVS for the field of the checked images. "
                  "Uses the checked File Manager images of the chosen filter and the aperture settings.",
             wraplength=800).pack(anchor="w", pady=(0, 6))

    fv_targets = []          # list of target dicts (name, ra, dec, type, period, max, min, source, use, x, y, stats)
    fv_last = {}             # paths of the last run

    def _fv_var(name, default, vtype=str):
        if name in vars_dict:
            return vars_dict[name][0]
        var = {str: tk.StringVar, int: tk.IntVar, bool: tk.BooleanVar}.get(vtype, tk.DoubleVar)(value=default)
        vars_dict[name] = (var, vtype)
        return var

    # ---- 1. Targets -------------------------------------------------------
    lf_fv_tg = tk.LabelFrame(fv_container, text="1. Targets", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_fv_tg.pack(fill="x", pady=5)

    fv_filter_var = _fv_var("fv_filter", "V")
    fv_faint_var = _fv_var("fv_faintest_max", "14.0")
    fv_amp_var = _fv_var("fv_min_amplitude", "0.1")
    fv_pmin_var = _fv_var("fv_min_period", "")
    fv_pmax_var = _fv_var("fv_max_period", "")
    fv_types_var = _fv_var("fv_types", "")
    fv_unknown_p_var = _fv_var("fv_include_unknown_period", True, bool)
    fv_edge_var = _fv_var("fv_edge_px", "30")

    row_q = tk.Frame(lf_fv_tg, bg="white"); row_q.pack(fill="x", padx=8, pady=(6, 2))
    tk.Label(row_q, text="Filter:", bg="white").pack(side=tk.LEFT)
    ttk.Combobox(row_q, textvariable=fv_filter_var, values=["V", "B", "L"], width=4, state="readonly").pack(side=tk.LEFT, padx=(2, 12))
    for lbl, var, w in (("Brighter than (at max):", fv_faint_var, 6), ("Min. amplitude (mag):", fv_amp_var, 5),
                        ("Period from (d):", fv_pmin_var, 7), ("to (d):", fv_pmax_var, 7)):
        tk.Label(row_q, text=lbl, bg="white").pack(side=tk.LEFT)
        ttk.Entry(row_q, textvariable=var, width=w).pack(side=tk.LEFT, padx=(2, 10))
    row_q2 = tk.Frame(lf_fv_tg, bg="white"); row_q2.pack(fill="x", padx=8, pady=2)
    tk.Label(row_q2, text="Types (e.g. 'M SR L', empty = all):", bg="white").pack(side=tk.LEFT)
    ttk.Entry(row_q2, textvariable=fv_types_var, width=14).pack(side=tk.LEFT, padx=(2, 10))
    ttk.Checkbutton(row_q2, text="keep stars without catalogued period", variable=fv_unknown_p_var).pack(side=tk.LEFT, padx=(0, 10))
    tk.Label(row_q2, text="Edge margin (px):", bg="white").pack(side=tk.LEFT)
    ttk.Entry(row_q2, textvariable=fv_edge_var, width=5).pack(side=tk.LEFT, padx=2)

    row_b = tk.Frame(lf_fv_tg, bg="white"); row_b.pack(fill="x", padx=8, pady=(4, 4))

    fv_cols = ("use", "name", "type", "period", "range", "source", "imgs", "n", "mean", "std", "p2p", "chi2")
    fv_heads = {"use": "Use", "name": "Name", "type": "Type", "period": "Period (d)", "range": "Range (mag)",
                "source": "Source", "imgs": "On images", "n": "N", "mean": "Mean", "std": "Std", "p2p": "5-95% range",
                "chi2": "chi2_red"}
    fv_widths = {"use": 36, "name": 140, "type": 60, "period": 70, "range": 100, "source": 75, "imgs": 80, "n": 35,
                 "mean": 65, "std": 55, "p2p": 75, "chi2": 60}
    fv_tree_frame = tk.Frame(lf_fv_tg, bg="white"); fv_tree_frame.pack(fill="x", padx=8, pady=4)
    fv_tree = ttk.Treeview(fv_tree_frame, columns=fv_cols, show="headings", height=10, selectmode="extended")
    for c in fv_cols:
        fv_tree.heading(c, text=fv_heads[c])
        fv_tree.column(c, width=fv_widths[c], minwidth=fv_widths[c], anchor="w" if c in ("name", "type", "source") else "center", stretch=False)
    fv_sb = ttk.Scrollbar(fv_tree_frame, orient="vertical", command=fv_tree.yview)
    fv_tree.configure(yscrollcommand=fv_sb.set)
    fv_tree.pack(side=tk.LEFT, fill="x", expand=True)
    fv_sb.pack(side=tk.LEFT, fill="y")

    fv_status_var = tk.StringVar(value="Load a target list or query a catalogue.")

    def _fmt(v, nd=3):
        try:
            v = float(v)
            return f"{v:.{nd}f}" if np.isfinite(v) else ""
        except (TypeError, ValueError):
            return ""

    def fv_refresh_tree():
        fv_tree.delete(*fv_tree.get_children())
        for i, t in enumerate(fv_targets):
            mx, mn = t.get('max'), t.get('min')
            rng = f"{_fmt(mx, 2)} - {_fmt(mn, 2)}" if (_fmt(mx) or _fmt(mn)) else ""
            xy = f"{t['n_in']}/{t['n_img']}" if t.get('n_img') else ""
            st = t.get('stats') or {}
            nd = 3 if st.get('mode', 'calibrated') == 'calibrated' else 4
            fv_tree.insert("", "end", iid=str(i), values=(
                "[X]" if t.get('use', True) else "[ ]", t.get('name', ''), t.get('type') or '',
                _fmt(t.get('period'), 4), rng, t.get('source', ''), xy,
                st.get('n', '') if st else '', _fmt(st.get('mean'), nd) if st else '',
                _fmt(st.get('std'), nd) if st else '', _fmt(st.get('p2p'), nd) if st else '',
                _fmt(st.get('chi2_red'), 1) if st else ''))
        n_use = sum(1 for t in fv_targets if t.get('use', True))
        fv_status_var.set(f"{len(fv_targets)} target(s) in the list, {n_use} selected.")

    def fv_toggle(event):
        if fv_tree.identify_column(event.x) != "#1":
            return
        iid = fv_tree.identify_row(event.y)
        if iid:
            t = fv_targets[int(iid)]
            t['use'] = not t.get('use', True)
            fv_refresh_tree()
    fv_tree.bind("<Button-1>", fv_toggle)

    def fv_checked_files():
        """Checked File Manager files of the chosen filter (paths)."""
        want = fv_filter_var.get().upper()
        out = []
        for iid in tree.get_children():
            if tree.item(iid, 'values')[0] != '[X]':
                continue
            fd = loaded_files[int(iid)]
            filt = str(fd['filter']).upper()
            if (lband_of(filt) or filt.strip()) == want:
                out.append(fd['path'])
        return out

    fv_fp_cache = {}

    def fv_footprints(files):
        """WCS footprints of the images (headers only; cached per file and mtime)."""
        from photometry.field_variables import image_footprints
        out = []
        for p in files:
            try:
                key = (p, os.path.getmtime(p))
            except OSError:
                key = (p, None)
            if key not in fv_fp_cache:
                fv_fp_cache[key] = image_footprints([p])[0]
            out.append(fv_fp_cache[key])
        return out

    def fv_field(quiet=False):
        """Sky region covered by ALL checked images of the chosen filter (several nights
        may have different pointings). Returns (field, files, footprints); field =
        (centre_ra, centre_dec, radius_arcmin) or None."""
        files = fv_checked_files()
        if not files:
            if not quiet:
                messagebox.showwarning("No Images", f"Please check images of filter {fv_filter_var.get()} in the File Manager.")
            return None, files, []
        fps = fv_footprints(files)
        good = [f for f in fps if f[1] is not None]
        if not good:
            if not quiet:
                messagebox.showerror("No WCS", "None of the checked images has an astrometric solution (WCS).\n"
                                               "Please plate-solve them first (Pre-processing).")
            return None, files, fps
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        cen, corners = [], []
        for _, w, (ny, nx) in good:
            cen.append(w.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0))
            corners += [w.pixel_to_world(x, y) for x, y in ((0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1))]
        cc = SkyCoord(cen)
        v = cc.cartesian.xyz.value.mean(axis=1)
        centre = SkyCoord(x=v[0], y=v[1], z=v[2], representation_type='cartesian').spherical
        c0 = SkyCoord(centre.lon.to(u.deg), centre.lat.to(u.deg))
        radius = max(c0.separation(c).arcmin for c in corners)
        n_nowcs = len(fps) - len(good)
        if n_nowcs and not quiet:
            messagebox.showwarning("Images without WCS", f"{n_nowcs} of the {len(fps)} checked images have no WCS "
                                   "and are ignored. Plate-solve them to include them.")
        return (float(c0.ra.deg), float(c0.dec.deg), float(radius)), files, fps

    def _opt_float(var):
        txt = str(var.get()).strip()
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            raise ValueError(f"'{txt}' is not a number")

    def fv_add_targets(new, replace=False):
        """Add targets; a star already in the list (same name, or within 5 arcsec)
        is not added twice - missing catalogue data (type, period, range) is merged."""
        if replace:
            fv_targets.clear()
        for t in new:
            dup = None
            for o in fv_targets:
                if o['name'].strip().lower() == t['name'].strip().lower():
                    dup = o
                elif None not in (o.get('ra'), t.get('ra')):
                    dra = (o['ra'] - t['ra']) * np.cos(np.radians(t['dec']))
                    if np.hypot(dra, o['dec'] - t['dec']) * 3600.0 < 5.0:
                        dup = o
                if dup is not None:
                    break
            if dup is None:
                t.setdefault('use', True)
                fv_targets.append(t)
            else:
                for k in ('type', 'period', 'max', 'min', 'ra', 'dec'):
                    if dup.get(k) in (None, '') and t.get(k) not in (None, ''):
                        dup[k] = t[k]
        fv_refresh_tree()

    def fv_mark_positions(fps, margin=None):
        """Count on how many of the images each target lies (>= edge margin from the
        border); targets on none of them are deselected. Returns that number."""
        if not fps:
            return 0
        from photometry.field_variables import in_footprints
        if margin is None:
            try:
                margin = float(fv_edge_var.get() or 0)
            except ValueError:
                margin = 0.0
        good = [f for f in fps if f[1] is not None]
        off = 0
        for t in fv_targets:
            if t.get('ra') is None:
                continue
            inside = in_footprints(t['ra'], t['dec'], good, margin_px=margin) if good else np.zeros(0, bool)
            t['n_in'], t['n_img'] = int(inside.sum()), len(fps)
            if t['n_in'] == 0 and t.get('use', True):
                t['use'] = False
                off += 1
        return off

    def on_fv_load_list():
        path = _fv_fd.askopenfilename(title="Load target list",
                                          filetypes=(("Lists", "*.csv *.txt *.dat *.lst"), ("all files", "*.*")))
        if not path:
            return
        from photometry.field_variables import parse_target_list, resolve_missing_coordinates
        try:
            new, problems = parse_target_list(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not read {path}:\n{e}")
            return
        if any(t['ra'] is None for t in new):
            fv_status_var.set("Resolving names without coordinates (SIMBAD / VSX)...")
            root.update_idletasks()
            problems += resolve_missing_coordinates(new)
            new = [t for t in new if t.get('ra') is not None]
        fv_add_targets(new)
        if loaded_files:
            try:
                _, _, fps = fv_field(quiet=True)
                off = fv_mark_positions(fps)
                fv_refresh_tree()
                if off:
                    problems.append(f"{off} target(s) are not on any of the checked images and were deselected.")
            except Exception:
                pass
        if problems:
            messagebox.showwarning("Target List", f"{len(new)} target(s) read.\n\n" + "\n".join(problems[:15]) +
                                   ("\n..." if len(problems) > 15 else ""))

    def on_fv_query(source, quiet=False):
        field, files, fps = fv_field(quiet=quiet)
        if not field:
            return
        try:
            filt_kw = dict(faintest_max=_opt_float(fv_faint_var), min_amplitude=_opt_float(fv_amp_var),
                           min_period=_opt_float(fv_pmin_var), max_period=_opt_float(fv_pmax_var),
                           include_unknown_period=bool(fv_unknown_p_var.get()), type_filter=fv_types_var.get(),
                           edge_px=_opt_float(fv_edge_var) or 0.0)
        except ValueError as e:
            if not quiet:
                messagebox.showerror("Input Error", str(e))
            return
        ra_c, dec_c, rad = field
        edge = filt_kw.pop('edge_px')
        fv_status_var.set(f"Querying {source} within {rad:.1f}' of the field centre...")
        root.update_idletasks()

        def worker():
            from photometry.field_variables import query_field_variables, filter_variables
            try:
                stars = query_field_variables(source, ra_c, dec_c, rad)
            except Exception as e:
                if quiet:
                    root.after(0, lambda err=e: fv_status_var.set(f"{source} query failed: {err}"))
                else:
                    root.after(0, lambda err=e: (fv_status_var.set(f"{source} query failed."),
                                                 messagebox.showerror("Query Error", f"{source} query failed:\n{err}\n\n"
                                                                      "(An internet connection is needed for the first query of a field.)")))
                return
            from photometry.field_variables import in_footprints
            sel = filter_variables(stars, None, None, **filt_kw)
            good = [f for f in fps if f[1] is not None]
            if sel and good:
                ins = in_footprints(np.array([x['ra'] for x in sel]), np.array([x['dec'] for x in sel]), good, margin_px=edge)
                sel = [x for x, k in zip(sel, ins.any(axis=0)) if k]

            def done():
                n_before = len(fv_targets)
                fv_add_targets(sel)
                fv_mark_positions(fps)
                fv_refresh_tree()
                src_lbl = "GCVS (local list)" if (stars and stars[0].get('source') == 'GCVS (local)') else source
                fv_status_var.set(f"{src_lbl}: {len(stars)} variable(s) in the field, {len(sel)} pass the filters, "
                                  f"{len(fv_targets) - n_before} new ({len(fv_targets)} in the list)."
                                  + ("  [added automatically]" if quiet else ""))
                if not quiet and stars and not any(s.get('type') for s in stars):
                    messagebox.showinfo("Limited catalogue data",
                                        "Only names and positions are available (old cache, no internet?) - "
                                        "the type/brightness/period filters could not be applied.")
            root.after(0, done)
        import threading as _th
        _th.Thread(target=worker, daemon=True).start()

    def on_fv_add_manual():
        pop = tk.Toplevel(root); pop.title("Add target"); pop.transient(root)
        nv, rv, dv = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for r, (lbl, var) in enumerate((("Name:", nv), ("RA (hh:mm:ss or deg, empty = resolve name):", rv), ("Dec (dd:mm:ss or deg):", dv))):
            ttk.Label(pop, text=lbl).grid(row=r, column=0, sticky="w", padx=8, pady=4)
            ttk.Entry(pop, textvariable=var, width=20).grid(row=r, column=1, padx=8, pady=4)

        def ok():
            from photometry.field_variables import parse_coordinates, resolve_missing_coordinates
            t = {'name': nv.get().strip() or "Target", 'ra': None, 'dec': None, 'source': 'manual',
                 'type': None, 'period': None, 'max': None, 'min': None}
            try:
                if rv.get().strip():
                    t['ra'], t['dec'] = parse_coordinates(rv.get(), dv.get())
                else:
                    pr = resolve_missing_coordinates([t])
                    if pr:
                        raise ValueError(pr[0])
            except Exception as e:
                messagebox.showerror("Add target", str(e), parent=pop)
                return
            fv_add_targets([t])
            pop.destroy()
        ttk.Button(pop, text="Add", command=ok).grid(row=3, column=0, columnspan=2, pady=8)

    def on_fv_remove():
        sel = sorted((int(i) for i in fv_tree.selection()), reverse=True)
        for i in sel:
            del fv_targets[i]
        fv_refresh_tree()

    def on_fv_select(flag):
        for t in fv_targets:
            t['use'] = flag and t.get('ra') is not None
        fv_refresh_tree()

    def on_fv_save_list():
        if not fv_targets:
            return
        path = _fv_fd.asksaveasfilename(title="Save target list", defaultextension=".csv",
                                            initialfile="field_variables.csv", filetypes=(("CSV", "*.csv"),))
        if path:
            from photometry.field_variables import save_target_list
            save_target_list(path, fv_targets)
            fv_status_var.set(f"Target list saved: {path}")

    row_b2 = tk.Frame(lf_fv_tg, bg="white"); row_b2.pack(fill="x", padx=8, pady=(0, 4), after=row_b)
    for frm, items in ((row_b, (("Load List...", on_fv_load_list), ("Query VSX (incl. GCVS)", lambda: on_fv_query("VSX")),
                                ("GCVS (local list / online)", lambda: on_fv_query("GCVS")), ("Add...", on_fv_add_manual),
                                ("APASS DR10...", lambda: vars_dict['apass_dr10_dialog']()))),
                       (row_b2, (("Remove Selected", on_fv_remove), ("Select All", lambda: on_fv_select(True)),
                                 ("Select None", lambda: on_fv_select(False)), ("Save List...", on_fv_save_list)))):
        for txt, cmd in items:
            ttk.Button(frm, text=txt, command=cmd).pack(side=tk.LEFT, padx=2)
    # v4.26: automatically add the variables of the local GCVS list (catalogue folder)
    fv_autogcvs_var = _fv_var("fv_auto_gcvs", True, bool)
    fv_auto_state = {'key': None}
    ttk.Checkbutton(lf_fv_tg, text="Add variables from the local GCVS list automatically (field of the checked images; "
                                   "filters above apply)", variable=fv_autogcvs_var).pack(anchor="w", padx=8)

    def fv_auto_gcvs():
        """Called when the tab is shown: adds GCVS stars in the field once per set of checked images."""
        try:
            from photometry import local_catalogs as _lc
            if not fv_autogcvs_var.get() or not _lc.gcvs_available():
                return
        except Exception:
            return
        files = fv_checked_files()
        key = (fv_filter_var.get(), tuple(files))
        if not files or key == fv_auto_state['key']:
            return
        fv_auto_state['key'] = key
        on_fv_query("GCVS", quiet=True)
    vars_dict['fv_auto_gcvs'] = fv_auto_gcvs

    tk.Label(lf_fv_tg, text="Click in the 'Use' column to include/exclude a target. "
                            "List format: one star per line - name, RA, Dec (sexagesimal or degrees), "
                            "optionally with a header line (name,ra,dec,type,period,...).",
             bg="white", fg="#666", font=("Segoe UI", 8), wraplength=800, justify="left").pack(anchor="w", padx=8, pady=(0, 6))

    # ---- 2. Light curves --------------------------------------------------
    lf_fv_run = tk.LabelFrame(fv_container, text="2. Light Curves", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_fv_run.pack(fill="x", pady=5)
    fv_mode_var = _fv_var("fv_lc_mode", "calibrated")
    fv_ncomp_var = _fv_var("fv_n_comp", 5, int)
    fv_comp_rad_var = _fv_var("fv_comp_radius", "12")
    fv_win_var = _fv_var("fv_mag_window", "1.5")
    fv_aavso_var = _fv_var("fv_aavso_reports", False, bool)

    row_m = tk.Frame(lf_fv_run, bg="white"); row_m.pack(fill="x", padx=8, pady=(6, 2))
    for txt, val in (("Calibrated magnitudes (comparison stars from the reference catalogue)", "calibrated"),
                     ("Relative flux (same comparison stars, no calibration)", "relative"),
                     ("Raw ADU (no comparison stars)", "raw")):
        ttk.Radiobutton(row_m, text=txt, variable=fv_mode_var, value=val).pack(anchor="w")
    row_c = tk.Frame(lf_fv_run, bg="white"); row_c.pack(fill="x", padx=8, pady=2)
    tk.Label(row_c, text="Comparison stars per target:", bg="white").pack(side=tk.LEFT)
    ttk.Spinbox(row_c, from_=1, to=10, textvariable=fv_ncomp_var, width=4).pack(side=tk.LEFT, padx=(2, 10))
    tk.Label(row_c, text="within (arcmin):", bg="white").pack(side=tk.LEFT)
    ttk.Entry(row_c, textvariable=fv_comp_rad_var, width=5).pack(side=tk.LEFT, padx=(2, 10))
    tk.Label(row_c, text="brightness window (+/- mag):", bg="white").pack(side=tk.LEFT)
    ttk.Entry(row_c, textvariable=fv_win_var, width=5).pack(side=tk.LEFT, padx=(2, 10))
    ttk.Checkbutton(lf_fv_run, text="Also write AAVSO reports (calibrated mode)", variable=fv_aavso_var).pack(anchor="w", padx=8)
    tk.Label(lf_fv_run, bg="white", fg="#666", font=("Segoe UI", 8), justify="left", wraplength=800,
             text="Comparison stars are chosen automatically for every target: isolated catalogue stars "
                  "(Settings > reference catalogue) close in brightness and position, not known variables; "
                  "one further star serves as check star. Light curves of several nights: check the images of "
                  "all nights in the File Manager.").pack(anchor="w", padx=8, pady=(0, 4))

    row_r = tk.Frame(lf_fv_run, bg="white"); row_r.pack(fill="x", padx=8, pady=6)
    fv_progress_var = tk.DoubleVar(value=0)
    fv_cancel = threading.Event()
    fv_run_btn = tk.Button(row_r, text="Run Light Curves for Selected Targets", bg="#2e7d32", fg="white",
                           font=("Segoe UI", 10, "bold"), padx=10)
    fv_run_btn.pack(side=tk.LEFT)
    fv_cancel_btn = ttk.Button(row_r, text="Cancel", command=fv_cancel.set)
    fv_cancel_btn.pack(side=tk.LEFT, padx=6)
    ttk.Progressbar(row_r, variable=fv_progress_var, maximum=100, length=250).pack(side=tk.LEFT, padx=8)
    tk.Label(fv_container, textvariable=fv_status_var, fg="#d32f2f", bg="white", font=("Arial", 10, "bold"),
             wraplength=800, justify="left").pack(anchor="w", pady=4)

    def on_fv_run():
        use = [t for t in fv_targets if t.get('use', True) and t.get('ra') is not None]
        if not use:
            messagebox.showwarning("No Targets", "No targets selected.")
            return
        field, files, fps = fv_field()
        if not field:
            return
        # v4.20: only images with WCS; every target must be on at least one of them
        good_fp = [f for f in fps if f[1] is not None]
        files = [f[0] for f in good_fp]
        fv_mark_positions(fps)
        off = [t for t in use if t.get('n_in', 1) == 0]
        use = [t for t in use if t.get('n_in', 1) > 0]
        fv_refresh_tree()
        if not use:
            messagebox.showwarning("Field Variables", "None of the selected targets lies on the checked images.")
            return
        mode = fv_mode_var.get()
        filt = fv_filter_var.get().upper()
        try:
            n_comp = max(1, int(fv_ncomp_var.get()))
            comp_rad = float(fv_comp_rad_var.get())
            win = float(fv_win_var.get())
            settings = dict(
                aperture_radius=float(vars_dict["aperture_radius"][0].get()),
                annulus_inner=float(vars_dict["annulus_inner"][0].get()),
                annulus_outer=float(vars_dict["annulus_outer"][0].get()),
                gain=float(vars_dict["ccd_gain"][0].get()),
                filter_name=filt, site_lat=float(ts_lat_var.get()), site_long=float(ts_lon_var.get()),
                use_flexible_aperture=bool(vars_dict["use_flexible_aperture"][0].get()),
                aperture_fwhm_factor=float(vars_dict["aperture_fwhm_factor"][0].get()),
                annulus_inner_gap=float(vars_dict["annulus_inner_gap"][0].get()),
                annulus_width=float(vars_dict["annulus_width"][0].get()),
                centroid_initial_radius=float(vars_dict["centroid_initial_radius"][0].get()) if "centroid_initial_radius" in vars_dict else 15.0,
                centroid_tracking_shift=float(vars_dict["centroid_tracking_shift"][0].get()) if "centroid_tracking_shift" in vars_dict else 4.0,
                stack_mode=vars_dict["stack_mode"][0].get() if "stack_mode" in vars_dict else "average",
                sky_noise_factor=float(vars_dict["sky_noise_factor"][0].get()) if "sky_noise_factor" in vars_dict else 1.0,
                coeff_term=float(ts_coeff_var.get()) if ts_do_trans_var.get() else 0.0)
            try:
                settings['k_coeff'] = float(vars_dict["extinction_kb" if filt == "B" else "extinction_kv"][0].get())
            except (KeyError, tk.TclError, ValueError):
                settings['k_coeff'] = 0.2
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Input Error", f"Invalid setting: {e}")
            return
        cat_type = vars_dict["reference_catalog"][0].get() if "reference_catalog" in vars_dict else "ATLAS"
        obs_code, chart = ts_obs_var.get(), ts_chart_var.get()
        write_aavso = bool(fv_aavso_var.get())
        is_trans = "YES" if ts_do_trans_var.get() and abs(float(ts_coeff_var.get())) > 1e-5 else "NO"
        ra_c, dec_c, rad = field
        w0, shape0 = good_fp[0][1], good_fp[0][2]
        fv_cancel.clear()
        fv_progress_var.set(0)
        fv_run_btn.config(state=tk.DISABLED)

        def status(txt):
            root.after(0, lambda: fv_status_var.set(txt))

        def worker():
            import csv as _csv
            from photometry import field_variables as fvm
            from photometry.time_series import run_multi_target_photometry, plot_light_curve, save_aavso_report
            from photometry.calibration import get_ref_stars
            try:
                targets = []
                skipped = [f"{t['name']}: not on any of the checked images" for t in off]
                known_vars = [t for t in fv_targets if t.get('ra') is not None]
                if mode in ("calibrated", "relative"):
                    # reference catalogue: one query for a small field, else one per target
                    cat_cache = {}
                    if rad <= 45.0:
                        status(f"Loading {cat_type} catalogue for the field ({rad + 2:.0f}')...")
                        cat_all = get_ref_stars(cat_type, ra_c, dec_c, radius_arcmin=round(rad + 2.0, 1), verbose=False) or []
                    for k, t in enumerate(use):
                        if fv_cancel.is_set():
                            status("Cancelled."); return
                        if rad <= 45.0:
                            cat = cat_all
                        else:
                            status(f"Loading {cat_type} catalogue around {t['name']} ({k + 1}/{len(use)})...")
                            cat = get_ref_stars(cat_type, t['ra'], t['dec'], radius_arcmin=round(2 * comp_rad + 3, 1), verbose=False) or []
                        comps, chk, tmag, tbv = fvm.choose_catalog_comparisons(
                            t, cat, known_vars, w0, shape0, band=filt, n_comp=n_comp,
                            radius_arcmin=comp_rad, mag_window=win, footprints=good_fp)
                        if not comps:
                            skipped.append(f"{t['name']}: no suitable comparison stars in the catalogue")
                            continue
                        t['comps'] = comps
                        targets.append({'name': t['name'], 'ra': t['ra'], 'dec': t['dec'], 'ensemble': comps,
                                        'check': chk, 'target_bv': tbv if tbv is not None else 0.5,
                                        'lc_mode': mode, '_t': t})
                else:
                    targets = [{'name': t['name'], 'ra': t['ra'], 'dec': t['dec'], 'ensemble': [], 'check': None,
                                'target_bv': 0.5, 'lc_mode': 'raw', '_t': t} for t in use]
                if not targets:
                    status("No target could be prepared.")
                    root.after(0, lambda: messagebox.showwarning("Field Variables", "\n".join(skipped[:20])))
                    return

                status(f"Measuring {len(targets)} target(s) in {len(files)} image(s)...")
                out = run_multi_target_photometry(
                    files, [{k: v for k, v in t.items() if k != '_t'} for t in targets],
                    cancel_event=fv_cancel,
                    update_progress=lambda v: root.after(0, lambda: fv_progress_var.set(v)),
                    verbose=False, **settings)
                if fv_cancel.is_set():
                    status("Cancelled."); return

                # ---- output ----
                from photometry.time_series import date_label, count_nights
                all_jd = [r['jd'] for res, _ in out if res for r in res if 'jd' in r]
                night = date_label(all_jd)
                n_nights = count_nights(all_jd)
                outdir = os.path.join("photometry_output", "field_variables", f"{night}_{filt}")
                os.makedirs(outdir, exist_ok=True)
                suffix = {"calibrated": "", "relative": "_relflux", "raw": "_rawADU"}[mode]
                curves, summary, used_names = [], [], set()
                for tg, (res, msg) in zip(targets, out):
                    t = tg['_t']
                    safe = _fv_re.sub(r'[^A-Za-z0-9_.+-]+', '_', t['name']).strip('_') or "target"
                    base_safe, n_dup = safe, 2
                    while safe in used_names:
                        safe = f"{base_safe}_{n_dup}"; n_dup += 1
                    used_names.add(safe)
                    if not res:
                        t['stats'] = None
                        skipped.append(f"{t['name']}: {msg}")
                        continue
                    st = fvm.summarize_light_curve(res) or {}
                    t['stats'] = st
                    from photometry.time_series import csv_fields as _csv_fields
                    keys = _csv_fields(res)                 # v4.42: UT (mid-exposure) instead of JD
                    t['csv'] = os.path.join(outdir, f"lc_{safe}{suffix}.csv")
                    with open(t['csv'], 'w', newline='') as f:
                        wr = _csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
                        wr.writeheader(); wr.writerows(res)
                    t['plot'] = os.path.join(outdir, f"plot_{safe}{suffix}.png")
                    plot_light_curve(res, t['name'], t['plot'])
                    if write_aavso and mode == "calibrated":
                        ens = tg['ensemble']
                        save_aavso_report(res, os.path.join(outdir, f"aavso_{safe}.txt"), t['name'], aavso_band(filt), obs_code,
                                          comp_name="ENSEMBLE" if len(ens) > 1 else ens[0]['name'],
                                          comp_mag="na" if len(ens) > 1 else ens[0]['mag_std'],
                                          check_name=tg['check']['name'] if tg['check'] else "na",
                                          check_mag=tg['check']['mag_std'] if tg['check'] else "na",
                                          trans=is_trans, software_version=APP_VERSION,
                                          comments=f"RefCat: {cat_type}",
                                          notes=f"Ensemble of {len(ens)}- " + " ".join(s['name'] for s in ens),
                                          chart=chart)
                    curves.append((t['name'], res))
                    summary.append([t['name'], t.get('type') or '', t.get('period') or '', t['ra'], t['dec'], mode,
                                    st.get('n', 0), st.get('mean', ''), st.get('std', ''), st.get('p2p', ''),
                                    st.get('med_err', ''), st.get('chi2_red', ''), st.get('t0', ''), st.get('t1', ''),
                                    len(tg['ensemble']), os.path.basename(t['csv'])])
                sum_path = os.path.join(outdir, f"summary{suffix}.csv")
                with open(sum_path, 'w', newline='') as f:
                    wr = _csv.writer(f)
                    wr.writerow(['name', 'type', 'period_d', 'ra_deg', 'dec_deg', 'mode', 'n_ok', 'mean', 'std',
                                 'range_5_95', 'median_err', 'chi2_red', 'hjd_first', 'hjd_last', 'n_comp', 'file'])
                    wr.writerows(summary)
                ov = os.path.join(outdir, f"overview{suffix}.png")
                fvm.plot_overview(curves, ov, f"Field variables - {filt} - {night}  ({len(files)} images"
                                  + (f", {n_nights} nights)" if n_nights > 1 else ")"))
                fv_last.update(outdir=outdir, overview=ov if curves else None, summary=sum_path)

                def done():
                    fv_refresh_tree()
                    fv_progress_var.set(100)
                    fv_status_var.set(f"Done: {len(curves)} light curve(s) in {os.path.abspath(outdir)}"
                                      + (f"  ({len(skipped)} target(s) skipped)" if skipped else ""))
                    if skipped:
                        messagebox.showwarning("Field Variables", "Skipped:\n" + "\n".join(skipped[:20]) +
                                               ("\n..." if len(skipped) > 20 else ""))
                root.after(0, done)
            except Exception as e:
                import traceback; traceback.print_exc()
                root.after(0, lambda err=e: (fv_status_var.set(f"Error: {err}"),
                                             messagebox.showerror("Field Variables", f"Error: {err}")))
            finally:
                root.after(0, lambda: fv_run_btn.config(state=tk.NORMAL))

        import threading as _th
        _th.Thread(target=worker, daemon=True).start()
    fv_run_btn.config(command=on_fv_run)

    # ---- 3. Results ------------------------------------------------------
    lf_fv_res = tk.LabelFrame(fv_container, text="3. Results", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_fv_res.pack(fill="x", pady=5)

    def _sel_target():
        sel = fv_tree.selection()
        if not sel:
            messagebox.showinfo("Field Variables", "Please select a target in the list first.")
            return None
        return fv_targets[int(sel[0])]

    def on_fv_open_selected():
        t = _sel_target()
        if t is None:
            return
        if t.get('plot') and os.path.exists(t['plot']):
            open_path(os.path.abspath(t['plot']))
        else:
            messagebox.showinfo("Field Variables", f"No light curve for {t['name']} yet.")

    def on_fv_to_period():
        t = _sel_target()
        if t is None:
            return
        if not (t.get('csv') and os.path.exists(t['csv'])):
            messagebox.showinfo("Field Variables", f"No light curve for {t['name']} yet."); return
        vars_dict['pa_open_files']([t['csv']])

    def on_fv_to_lc():
        t = _sel_target()
        if t is None or t.get('ra') is None:
            return
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(t['ra'] * u.deg, t['dec'] * u.deg)
        ts_target_name_var.set(t['name'])
        ts_target_mode_var.set("manual")
        ts_target_ra_var.set(c.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True))
        ts_target_dec_var.set(c.dec.to_string(unit=u.deg, sep=':', precision=1, pad=True, alwayssign=True))
        ts_filter_var.set(fv_filter_var.get())
        switch_tab("ts")
        messagebox.showinfo("Light Curves", f"{t['name']} was set as the target of the Light Curves tab "
                                            "(manual RA/Dec). Choose its comparison stars there for a detailed analysis.")

    for _k, (txt, cmd) in enumerate((("Open Overview Plot", lambda: open_path(os.path.abspath(fv_last['overview'])) if fv_last.get('overview')
                      else messagebox.showinfo("Field Variables", "No results yet.")),
                     ("Open Plot of Selected", on_fv_open_selected),
                     ("Open Summary (CSV)", lambda: open_path(os.path.abspath(fv_last['summary'])) if fv_last.get('summary')
                      else messagebox.showinfo("Field Variables", "No results yet.")),
                     ("Open Output Folder", lambda: open_path(os.path.abspath(fv_last.get('outdir') or os.path.join("photometry_output", "field_variables")))),
                     ("Use Selected in Light Curves Tab", on_fv_to_lc),
                     ("Period Analysis of Selected", on_fv_to_period))):
        ttk.Button(lf_fv_res, text=txt, command=cmd).grid(row=_k // 3, column=_k % 3, padx=4, pady=3, sticky="ew")

    # =========================================================================
    # --- TAB: PERIOD ANALYSIS (v4.21) ---
    # Periodogram (Lomb-Scargle / PDM / BLS), spectral window, prewhitening with
    # a multi-frequency fit, phase plots. Input: light-curve CSV files.
    # =========================================================================
    from matplotlib.figure import Figure as _PA_Figure
    from tkinter import filedialog as _pa_fd
    pa_container = tk.Frame(tab_period, bg="white")
    pa_container.pack(fill="both", expand=True, padx=20, pady=10)
    tk.Label(pa_container, text=f"{get_icon('〰', '')}  Period Analysis".strip(), font=("Segoe UI", 16, "bold"),
             fg="#1a3a5f", bg="white").pack(anchor="w", pady=(0, 4))
    tk.Label(pa_container, bg="white", justify="left", fg="#444", font=("Segoe UI", 9), wraplength=820,
             text="Find periods in light curves (one or several CSV files, e.g. several nights or seasons): periodogram, "
                  "spectral window (aliases), prewhitening of further frequencies and phase plots. "
                  "Time in days (HJD), frequency in cycles/day.").pack(anchor="w", pady=(0, 2))
    ttk.Button(pa_container, text="Open Period Analysis Guide",
               command=lambda: open_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "period_analysis_guide.md"))
               ).pack(anchor="w", pady=(0, 6))

    pa = {'sets': [], 't': None, 'y': None, 'e': None, 'idx': None, 'freq': None, 'spec': None,
          'spec_kind': None, 'window': None, 'fit': None, 'freqs': [], 'nharm': [], 'preview': None,
          'busy': False, 'unit': ''}

    def _pa_var(name, default, vtype=str):
        if name in vars_dict:
            return vars_dict[name][0]
        var = {str: tk.StringVar, int: tk.IntVar, bool: tk.BooleanVar}.get(vtype, tk.DoubleVar)(value=default)
        vars_dict[name] = (var, vtype)
        return var

    pa_status_var = tk.StringVar(value="Load one or more light-curve CSV files.")
    pa_info_var = tk.StringVar(value="")

    # ---- 1. Data ----------------------------------------------------------
    lf_pa_data = tk.LabelFrame(pa_container, text="1. Data", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_pa_data.pack(fill="x", pady=5)
    row_d = tk.Frame(lf_pa_data, bg="white"); row_d.pack(fill="x", padx=8, pady=(6, 2))
    pa_quant_var = tk.StringVar(value="")
    pa_only_ok_var = _pa_var("pa_only_ok", True, bool)
    pa_submean_var = _pa_var("pa_subtract_mean", False, bool)
    pa_files_lb = tk.Listbox(lf_pa_data, height=4, font=("Consolas", 9), activestyle="none")

    def pa_prepare():
        """Combine the loaded sets into t, y, e with the chosen options; reset fits."""
        from photometry import period_analysis as pam
        if not pa['sets']:
            pa.update(t=None, y=None, e=None)
            pa_info_var.set("")
            return False
        col = pa_quant_var.get() or pa['sets'][0]['value_col']
        series = []
        for lc in pa['sets']:
            c = col if col in lc['columns'] else lc['value_col']
            series.append(pam.extract_series(lc, c, 'auto', only_ok=bool(pa_only_ok_var.get())))
        t, y, e, idx = pam.combine_series(series, subtract_mean=bool(pa_submean_var.get()))
        if t.size < 5:
            pa_status_var.set("Too few data points (< 5).")
            return False
        pa.update(t=t, y=y, e=e, idx=idx, unit=('mag' if 'mag' in col.lower() else col),
                  fit=None, freqs=[], nharm=[], spec=None, window=None, preview=None, oc=None)
        s = pam.data_summary(t)
        pa_info_var.set(f"N = {s['n']}   span T = {s['span']:.2f} d   nights = {s['nights']}   "
                        f"median spacing = {s['dt_med'] * 1440:.1f} min   "
                        f"resolution 1/T = {s['resolution']:.4g} c/d   pseudo-Nyquist = {s['nyquist']:.4g} c/d"
                        + ("" if s['span'] <= 0 else f"   (longest useful period ~ {s['span'] / 2:.3g} d)"))
        pa_refresh_table(); pa_draw()
        return True

    def pa_refresh_files():
        pa_files_lb.delete(0, tk.END)
        for lc in pa['sets']:
            pa_files_lb.insert(tk.END, f"{lc['name']}   ({len(lc['rows'])} rows, mode: {lc.get('lc_mode') or '?'})")
        cols = []
        for lc in pa['sets']:
            cols += [c for c in lc['columns'] if c not in cols]
        pa_quant_cb['values'] = cols
        if pa['sets'] and (pa_quant_var.get() not in cols):
            pa_quant_var.set(pa['sets'][0]['value_col'] or (cols[0] if cols else ""))

    def pa_load_paths(paths, replace=False):
        from photometry import period_analysis as pam
        if replace:
            pa['sets'].clear()
        errs = []
        for p in paths:
            try:
                pa['sets'].append(pam.read_light_curve(p))
            except Exception as ex:
                errs.append(f"{os.path.basename(p)}: {ex}")
        pa_refresh_files()
        ok = pa_prepare()
        if errs:
            messagebox.showwarning("Period Analysis", "Some files could not be read:\n" + "\n".join(errs))
        if ok:
            pa_status_var.set(f"{len(pa['sets'])} data set(s) loaded. Press 'Compute Periodogram'.")
        return ok

    def on_pa_load():
        paths = _pa_fd.askopenfilenames(title="Load light curve(s)", initialdir=os.path.abspath("photometry_output"),
                                        filetypes=(("CSV", "*.csv"), ("Text", "*.txt *.dat"), ("all files", "*.*")))
        if paths:
            pa_load_paths(list(paths))

    def on_pa_load_last():
        p = vars_dict.get("ts_last_csv_path")
        if isinstance(p, str) and os.path.exists(p):
            pa_load_paths([p])
        else:
            messagebox.showinfo("Period Analysis", "No light curve from the Light Curves tab in this session yet.")

    def on_pa_clear():
        pa['sets'].clear(); pa_refresh_files(); pa_prepare(); pa_refresh_table(); pa_draw()
        pa_status_var.set("Load one or more light-curve CSV files.")

    for txt, cmd in (("Load Light Curve(s)...", on_pa_load), ("Load Last Light Curve", on_pa_load_last),
                     ("Clear", on_pa_clear)):
        ttk.Button(row_d, text=txt, command=cmd).pack(side=tk.LEFT, padx=2)
    tk.Label(row_d, text="   Quantity:", bg="white").pack(side=tk.LEFT)
    pa_quant_cb = ttk.Combobox(row_d, textvariable=pa_quant_var, width=14, state="readonly")
    pa_quant_cb.pack(side=tk.LEFT, padx=2)
    pa_quant_cb.bind("<<ComboboxSelected>>", lambda e: pa_prepare())
    row_d2 = tk.Frame(lf_pa_data, bg="white"); row_d2.pack(fill="x", padx=8, pady=2)
    ttk.Checkbutton(row_d2, text="only OK points (flag)", variable=pa_only_ok_var, command=pa_prepare).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Checkbutton(row_d2, text="subtract the mean of each file (different zero points, e.g. relative flux or other comparison stars)",
                    variable=pa_submean_var, command=pa_prepare).pack(side=tk.LEFT, padx=2)
    pa_files_lb.pack(fill="x", padx=8, pady=2)
    tk.Label(lf_pa_data, textvariable=pa_info_var, bg="white", fg="#1a3a5f", font=("Segoe UI", 9),
             wraplength=820, justify="left").pack(anchor="w", padx=8, pady=(0, 6))

    # ---- 2. Periodogram ---------------------------------------------------
    lf_pa_per = tk.LabelFrame(pa_container, text="2. Periodogram", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_pa_per.pack(fill="x", pady=5)
    row_p = tk.Frame(lf_pa_per, bg="white"); row_p.pack(fill="x", padx=8, pady=6)
    pa_method_var = _pa_var("pa_method", "Lomb-Scargle")
    pa_terms_var = _pa_var("pa_nterms", 1, int)
    pa_pmin_var = _pa_var("pa_period_min", "")
    pa_pmax_var = _pa_var("pa_period_max", "")
    pa_over_var = _pa_var("pa_oversample", 10, int)
    pa_xper_var = _pa_var("pa_axis_period", False, bool)
    tk.Label(row_p, text="Method:", bg="white").pack(side=tk.LEFT)
    ttk.Combobox(row_p, textvariable=pa_method_var, values=["Lomb-Scargle", "PDM", "BLS (eclipses)"],
                 width=14, state="readonly").pack(side=tk.LEFT, padx=(2, 8))
    tk.Label(row_p, text="harmonics:", bg="white").pack(side=tk.LEFT)
    ttk.Spinbox(row_p, from_=1, to=6, textvariable=pa_terms_var, width=3).pack(side=tk.LEFT, padx=(2, 8))
    for lbl, var, w_ in (("Period from (d):", pa_pmin_var, 8), ("to (d):", pa_pmax_var, 8)):
        tk.Label(row_p, text=lbl, bg="white").pack(side=tk.LEFT)
        ttk.Entry(row_p, textvariable=var, width=w_).pack(side=tk.LEFT, padx=(2, 8))
    tk.Label(row_p, text="oversampling:", bg="white").pack(side=tk.LEFT)
    ttk.Spinbox(row_p, from_=2, to=50, textvariable=pa_over_var, width=4).pack(side=tk.LEFT, padx=(2, 8))
    row_p2 = tk.Frame(lf_pa_per, bg="white"); row_p2.pack(fill="x", padx=8, pady=(0, 6))
    pa_compute_btn = ttk.Button(row_p2, text="Compute Periodogram")
    pa_compute_btn.pack(side=tk.LEFT, padx=2)
    pa_window_btn = ttk.Button(row_p2, text="Show Spectral Window")
    pa_window_btn.pack(side=tk.LEFT, padx=2)
    ttk.Checkbutton(row_p2, text="x-axis as period (log)", variable=pa_xper_var, command=lambda: pa_draw()).pack(side=tk.LEFT, padx=8)
    tk.Label(row_p2, text="   Curve in phase plot:", bg="white").pack(side=tk.LEFT)
    pa_phasecurve_var = _pa_var("pa_phase_curve", "Fit (sine + harmonics)")
    _pc_cb = ttk.Combobox(row_p2, textvariable=pa_phasecurve_var, width=22, state="readonly",
                          values=["Fit (sine + harmonics)", "Binned mean (no model)", "None"])
    _pc_cb.pack(side=tk.LEFT, padx=2)
    pa_phasecurve_var.trace_add("write", lambda *_: pa_draw())
    pa_bottom_var = tk.StringVar(value="Phase plot")      # v4.25: radio buttons directly above the plot
    pa_bottom_var.trace_add("write", lambda *_: pa_draw())
    tk.Label(lf_pa_per, bg="white", fg="#666", font=("Segoe UI", 8), wraplength=820, justify="left",
             text="Empty period limits = automatic (from 2 x the median spacing up to 2 x the time span). "
                  "Lomb-Scargle is shown as amplitude spectrum of the current residuals (after subtracting the "
                  "frequencies in the table); 'harmonics' > 1 fits non-sinusoidal shapes. PDM: 1 - theta is shown "
                  "(maxima = good periods). Click into the periodogram to preview the phase plot at that frequency."
             ).pack(anchor="w", padx=8, pady=(0, 6))

    # ---- 3. Frequencies -----------------------------------------------------
    lf_pa_fr = tk.LabelFrame(pa_container, text="3. Frequencies (prewhitening)", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_pa_fr.pack(fill="x", pady=5)
    row_f = tk.Frame(lf_pa_fr, bg="white"); row_f.pack(fill="x", padx=8, pady=6)
    pa_newharm_var = _pa_var("pa_new_harmonics", 1, int)
    pa_cols = ("id", "freq", "ferr", "period", "perr", "amp", "aerr", "harm", "snr", "note")
    pa_heads = {"id": "", "freq": "Freq (c/d)", "ferr": "+/-", "period": "Period (d)", "perr": "+/-",
                "amp": "Amplitude", "aerr": "+/-", "harm": "Harm.", "snr": "S/N", "note": "Note"}
    pa_widths = {"id": 34, "freq": 95, "ferr": 75, "period": 95, "perr": 75, "amp": 75, "aerr": 60,
                 "harm": 45, "snr": 50, "note": 190}
    pa_tree = ttk.Treeview(lf_pa_fr, columns=pa_cols, show="headings", height=5, selectmode="browse")
    for c in pa_cols:
        pa_tree.heading(c, text=pa_heads[c])
        pa_tree.column(c, width=pa_widths[c], minwidth=pa_widths[c], stretch=(c == "note"),
                       anchor="w" if c == "note" else "center")

    def pa_refresh_table():
        pa_tree.delete(*pa_tree.get_children())
        fit = pa['fit']
        pa_update_quality()
        if not fit:
            return
        for i, (f, fe, comps) in enumerate(zip(fit['freqs'], fit['freq_err'], fit['components'])):
            amp, sa = comps[0][1], comps[0][2]
            snr = fit.get('snr', [np.nan] * len(fit['freqs']))[i]
            pa_tree.insert("", "end", iid=str(i), values=(
                f"f{i + 1}", f"{f:.6f}", f"{fe:.2g}", f"{1 / f:.6g}", f"{fe / f ** 2:.2g}",
                f"{amp:.4f}", f"{sa:.4f}", fit['nharm'][i], f"{snr:.1f}" if np.isfinite(snr) else "",
                fit.get('notes', [''] * len(fit['freqs']))[i]))

    def pa_selected_freq():
        sel = pa_tree.selection()
        if sel and pa['fit']:
            return int(sel[0])
        return None

    def _pa_on_select(e=None):
        pa['preview'] = None
        if pa_bottom_var.get() != "Phase plot":
            pa_bottom_var.set("Phase plot")        # redraws via the trace
        else:
            pa_draw()
    pa_tree.bind("<<TreeviewSelect>>", _pa_on_select)

    pa_quality_var = tk.StringVar(value="")

    def pa_update_quality():
        """v4.24: how well do the fitted frequencies describe the data?"""
        fit = pa['fit']
        if not fit or pa['t'] is None:
            pa_quality_var.set("")
            return
        from photometry.period_analysis import fit_quality
        q = fit_quality(fit, pa['t'], pa['e'])
        u_ = pa['unit']
        txt = f"Fit quality:  residual rms = {q['rms']:.4g} {u_}"
        if q['med_err'] is not None:
            txt += (f"   median photometric error = {q['med_err']:.4g} {u_}   rms/error = {q['ratio']:.2f}"
                    f"   reduced chi2 = {q['chi2_red']:.2f}")
            r = q['ratio']
            if r < 0.8:
                verdict = "rms below the errors: errors probably overestimated (or over-fitting)."
            elif r <= 1.3:
                verdict = "the fitted frequencies describe the data to within the noise."
            elif r <= 2.0:
                verdict = "some signal left: look for further peaks (S/N >= 4), more harmonics, or night offsets."
            else:
                verdict = "clearly more signal than noise left: further frequencies, harmonics or zero-point problems."
            txt += "\n  -> " + verdict
        if np.isfinite(q['night_scatter']) and len(q['nights']) > 1:
            txt += (f"\n  Night-to-night scatter of the mean residual: {q['night_scatter']:.4g} {u_} "
                    f"({len(q['nights'])} nights) - compare with the rms; large values point to zero-point "
                    "differences or a slow variation (see 'Residuals vs time').")
        pa_quality_var.set(txt)

    # ---- plots -------------------------------------------------------------
    pa_fig = _PA_Figure(figsize=(9.5, 8.0), dpi=90)
    pa_plot_frame = tk.Frame(pa_container, bg="white")
    pa_canvas = FigureCanvasTkAgg(pa_fig, master=pa_plot_frame)

    def _fmt_period_axis(ax):
        if pa_xper_var.get():
            ax.set_xscale('log')
            ax.set_xlabel("Period (d)", fontsize=8)
        else:
            ax.set_xlabel("Frequency (c/d)", fontsize=8)

    def pa_phase_data(i=None, use_preview=True):
        """v4.25: data for a phase plot. Returns dict with f, t0, phase, value (data minus all
        other, non-harmonic frequencies), err, set index, label, 'only' (sub-fit of the frequency
        and its harmonic frequencies, or None), times. None if there is nothing to fold."""
        from photometry.period_analysis import model_curve, phase_fold, harmonic_family
        t, y, fit = pa['t'], pa['y'], pa['fit']
        if t is None:
            return None
        only = None
        if use_preview and pa['preview'] is not None:
            f = pa['preview']
            val = (fit['residuals'] + fit['offset']) if fit else y
            label = f"f = {f:.6f} c/d (P = {1 / f:.6g} d)" + (" - residuals of the fit" if fit else "")
        elif fit:
            i = (pa_selected_freq() if i is None else i)
            i = 0 if i is None else i
            f = fit['freqs'][i]
            keep = harmonic_family(fit, i)
            only = dict(fit, freqs=[fit['freqs'][j] for j in keep], components=[fit['components'][j] for j in keep],
                        nharm=[fit['nharm'][j] for j in keep], offset=fit['offset'])
            val = fit['residuals'] + model_curve(only, t)
            extra = [f"f{j + 1}" for j in keep if j != i]
            others = len(fit['freqs']) - len(keep)
            label = (f"f{i + 1} = {f:.6f} c/d (P = {1 / f:.6g} d)"
                     + (f", {others} other frequenc{'y' if others == 1 else 'ies'} subtracted" if others else "")
                     + (f", harmonics {', '.join(extra)} kept" if extra else ""))
        else:
            return None
        t0 = fit['t0'] if fit else float(np.round(np.mean(t), 1))
        return {'f': f, 't0': t0, 'phase': phase_fold(t, f, t0), 'value': val, 'err': pa['e'], 'idx': pa['idx'],
                'label': label, 'only': only, 't': t, 'i': (i if only is not None else None)}

    def pa_open_phase_window():
        """v4.25: separate, editable phase plot (labels, ranges, symbols, curve, cycles) that
        can be saved as PNG/PDF/SVG, plus the folded data as CSV."""
        if pa['t'] is None or (not pa['fit'] and pa['preview'] is None):
            messagebox.showinfo("Phase plot", "Fit a frequency (or click a peak in the periodogram) first.")
            return
        from photometry.period_analysis import model_curve
        sel_i = pa_selected_freq()
        d0 = pa_phase_data(sel_i)
        if d0 is None:
            return
        mag = 'mag' in pa['unit'].lower()
        star = pa['sets'][0]['name'] if pa['sets'] else ""
        win = tk.Toplevel(root)
        win.title("Phase plot")
        win.geometry("1000x760")
        ctrl = tk.Frame(win); ctrl.pack(side=tk.TOP, fill="x", padx=6, pady=4)
        V = {}
        def sv(name, val, kind=str):
            V[name] = {str: tk.StringVar, bool: tk.BooleanVar, int: tk.IntVar}[kind](value=val)
            return V[name]
        freq_choices = []
        if pa['fit']:
            freq_choices = [f"f{k + 1} = {f:.6f} c/d" for k, f in enumerate(pa['fit']['freqs'])]
        if pa['preview'] is not None:
            freq_choices.append(f"preview {pa['preview']:.6f} c/d")
        cur = (f"preview {pa['preview']:.6f} c/d" if pa['preview'] is not None
               else freq_choices[sel_i if sel_i is not None else 0])
        sv('freq', cur)
        sv('title', f"{star}   {d0['label']}")
        sv('xlabel', f"Phase (T0 = HJD {d0['t0']})")
        sv('ylabel', pa['unit'] if mag else f"{pa['unit']}")
        sv('ymin', ""); sv('ymax', ""); sv('xmin', "0"); sv('xmax', "2")
        sv('curve', pa_phasecurve_var.get() if pa_phasecurve_var.get() else "Fit (sine + harmonics)")
        sv('bins', "0")
        sv('ms', "3"); sv('errbars', False, bool); sv('colors', True, bool); sv('invert', mag, bool)
        sv('grid', True, bool); sv('legend', True, bool); sv('width', "8"); sv('height', "5"); sv('dpi', "150")
        sv('fontsize', "10"); sv('phase0max', True, bool)
        rows = [
            [("Frequency:", 'freq', 'combo', freq_choices, 26), ("Curve:", 'curve', 'combo',
              ["Fit (sine + harmonics)", "Binned mean (no model)", "None"], 22), ("bins (0 = auto):", 'bins', 'entry', None, 4)],
            [("Title:", 'title', 'entry', None, 70)],
            [("x label:", 'xlabel', 'entry', None, 30), ("y label:", 'ylabel', 'entry', None, 22)],
            [("phase from:", 'xmin', 'entry', None, 5), ("to:", 'xmax', 'entry', None, 5),
             ("y from:", 'ymin', 'entry', None, 8), ("to:", 'ymax', 'entry', None, 8), ("(empty = auto)", None, 'label', None, 0)],
            [("point size:", 'ms', 'entry', None, 4), ("font size:", 'fontsize', 'entry', None, 4),
             ("error bars", 'errbars', 'check', None, 0), ("colour per file", 'colors', 'check', None, 0),
             ("invert y (magnitudes)", 'invert', 'check', None, 0), ("grid", 'grid', 'check', None, 0),
             ("legend", 'legend', 'check', None, 0), ("phase 0 = maximum light", 'phase0max', 'check', None, 0)],
            [("figure size (inch) w:", 'width', 'entry', None, 4), ("h:", 'height', 'entry', None, 4),
             ("dpi (PNG):", 'dpi', 'entry', None, 5)],
        ]
        for r_, items in enumerate(rows):
            fr_ = tk.Frame(ctrl); fr_.pack(fill="x", pady=1)
            for lbl, key, kind, vals, w_ in items:
                if kind == 'check':
                    ttk.Checkbutton(fr_, text=lbl, variable=V[key]).pack(side=tk.LEFT, padx=(0, 8))
                    continue
                tk.Label(fr_, text=lbl).pack(side=tk.LEFT, padx=(0, 2))
                if kind == 'entry':
                    ttk.Entry(fr_, textvariable=V[key], width=w_).pack(side=tk.LEFT, padx=(0, 8))
                elif kind == 'combo':
                    ttk.Combobox(fr_, textvariable=V[key], values=vals, width=w_, state="readonly").pack(side=tk.LEFT, padx=(0, 8))
        fig = _PA_Figure(figsize=(8, 5), dpi=90)
        cv = FigureCanvasTkAgg(fig, master=win)
        state = {}

        def current_data():
            from photometry.period_analysis import ephemeris, phase_fold
            txt = V['freq'].get()
            if txt.startswith("preview"):
                d = pa_phase_data(None, use_preview=True)
            else:
                k = int(txt.split()[0][1:]) - 1
                d = pa_phase_data(k, use_preview=False)
            if d is not None and V['phase0max'].get() and d['i'] is not None:
                # phase 0 at maximum light (ephemeris from the fitted light-curve shape)
                T0max, _, _ = ephemeris(pa['fit'], d['i'], mag)
                d['t0'] = round(T0max, 5)
                d['phase'] = phase_fold(d['t'], d['f'], d['t0'])
            return d

        def num(key, default=None):
            try:
                return float(V[key].get())
            except (ValueError, tk.TclError):
                return default

        def redraw(*_):
            d = current_data()
            if d is None:
                return
            state['d'] = d
            fs = num('fontsize', 10)
            fig.clear()
            ax = fig.add_subplot(111)
            x0, x1 = num('xmin', 0.0), num('xmax', 2.0)
            ph, v_, e_ = d['phase'], d['value'], d['err']
            cols = ['#1f4e8c', '#c45c00', '#2e7d32', '#7b1fa2', '#00838f', '#6d4c41']
            ms_ = num('ms', 3)
            # repeat the cycle to cover [x0, x1]
            shifts = range(int(np.floor(x0)) - 1, int(np.ceil(x1)) + 1)
            for k in np.unique(d['idx']):
                m = d['idx'] == k
                X = np.concatenate([ph[m] + s_ for s_ in shifts]); Y = np.concatenate([v_[m] for _ in shifts])
                keep_ = (X >= x0) & (X <= x1)
                c_ = cols[int(k) % len(cols)] if V['colors'].get() else '#1f4e8c'
                lab = (pa['sets'][int(k)]['name'] if (len(pa['sets']) > 1 and V['colors'].get()) else
                       ("data" if k == np.unique(d['idx'])[0] else None))
                if V['errbars'].get() and e_ is not None:
                    E_ = np.concatenate([e_[m] for _ in shifts])
                    ax.errorbar(X[keep_], Y[keep_], yerr=E_[keep_], fmt='o', ms=ms_, color=c_, ecolor='#bbbbbb',
                                elinewidth=0.6, label=lab)
                else:
                    ax.plot(X[keep_], Y[keep_], 'o', ms=ms_, color=c_, label=lab)
            curve = V['curve'].get()
            if curve.startswith("Fit") and d['only'] is not None:
                pp = np.linspace(x0, x1, 1000)
                mm = model_curve(d['only'], d['t0'] + pp / d['f'])
                nh_ = sum(len(c) for c in d['only']['components'])
                ax.plot(pp, mm, '-', color='red', lw=1.4, zorder=5,
                        label=f"fit (sine + {nh_ - 1} harmonics)" if nh_ > 1 else "fit (sine)")
            elif curve.startswith("Binned"):
                nb = int(num('bins', 0) or 0)
                if nb <= 0:
                    nb = int(max(8, min(50, round(np.sqrt(len(ph)) * 1.5))))
                edges = np.linspace(0, 1, nb + 1)
                cen, med = [], []
                for a_, b_ in zip(edges[:-1], edges[1:]):
                    m_ = (ph >= a_) & (ph < b_)
                    if m_.sum() >= 2:
                        cen.append(0.5 * (a_ + b_)); med.append(np.median(v_[m_]))
                if cen:
                    cen = np.array(cen); med = np.array(med)
                    X = np.concatenate([cen + s_ for s_ in shifts]); Y = np.concatenate([med for _ in shifts])
                    o_ = np.argsort(X); X, Y = X[o_], Y[o_]
                    k_ = (X >= x0) & (X <= x1)
                    ax.plot(X[k_], Y[k_], '-o', ms=3, color='black', lw=1.4, zorder=5, label=f"median of {nb} phase bins")
                state['binned'] = (cen, med)
            ax.set_xlim(x0, x1)
            y0, y1 = num('ymin'), num('ymax')
            if y0 is not None and y1 is not None:
                ax.set_ylim(y0, y1)
            if V['invert'].get():
                if not ax.yaxis_inverted():
                    ax.invert_yaxis()
            ax.set_title(V['title'].get(), fontsize=fs)
            ax.set_xlabel(V['xlabel'].get(), fontsize=fs); ax.set_ylabel(V['ylabel'].get(), fontsize=fs)
            ax.tick_params(labelsize=fs * 0.85)
            if V['grid'].get():
                ax.grid(alpha=0.3)
            if V['legend'].get():
                ax.legend(fontsize=fs * 0.8, loc='best')
            fig.tight_layout()
            cv.draw_idle()

        def on_freq_change(*_):
            d = current_data()
            if d is not None:
                V['title'].set(f"{star}   {d['label']}")
                V['xlabel'].set(f"Phase (T0 = HJD {d['t0']}" + (", maximum light)" if (V['phase0max'].get() and d['i'] is not None) else ")"))
            redraw()
        V['freq'].trace_add("write", on_freq_change)

        def save_fig():
            outdir = os.path.join("photometry_output", "period_analysis"); os.makedirs(outdir, exist_ok=True)
            path = _pa_fd.asksaveasfilename(parent=win, title="Save phase plot", initialdir=os.path.abspath(outdir),
                                            initialfile=f"phase_{star}.png", defaultextension=".png",
                                            filetypes=(("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("all", "*.*")))
            if not path:
                return
            w0, h0 = fig.get_size_inches()
            try:
                fig.set_size_inches(num('width', 8), num('height', 5))
                fig.tight_layout()
                fig.savefig(path, dpi=num('dpi', 150))
            finally:
                fig.set_size_inches(w0, h0)
                fig.tight_layout(); cv.draw_idle()
            pa_status_var.set(f"Phase plot saved: {path}")

        def save_data():
            d = state.get('d')
            if d is None:
                return
            outdir = os.path.join("photometry_output", "period_analysis"); os.makedirs(outdir, exist_ok=True)
            path = _pa_fd.asksaveasfilename(parent=win, title="Save phase data", initialdir=os.path.abspath(outdir),
                                            initialfile=f"phase_{star}.csv", defaultextension=".csv",
                                            filetypes=(("CSV", "*.csv"),))
            if not path:
                return
            with open(path, 'w', newline='') as fh:
                fh.write(f"# phase plot of {star}: {d['label']}\n# frequency = {d['f']:.8f} c/d, period = {1 / d['f']:.9f} d, "
                         f"T0 (phase 0) = HJD {d['t0']}\n# value = data minus all other (non-harmonic) fitted frequencies\n")
                w_ = csv.writer(fh)
                hdr = ['hjd', 'phase', 'value', 'error', 'file']
                if d['only'] is not None:
                    hdr.append('model')
                    mod = model_curve(d['only'], d['t'])
                w_.writerow(hdr)
                for k_ in np.argsort(d['phase']):
                    row = [f"{d['t'][k_]:.6f}", f"{d['phase'][k_]:.5f}", f"{d['value'][k_]:.6f}",
                           f"{d['err'][k_]:.6f}" if d['err'] is not None else "", pa['sets'][int(d['idx'][k_])]['name']]
                    if d['only'] is not None:
                        row.append(f"{mod[k_]:.6f}")
                    w_.writerow(row)
            pa_status_var.set(f"Phase data saved: {path}")

        btns = tk.Frame(ctrl); btns.pack(fill="x", pady=(4, 2))
        ttk.Button(btns, text="Apply / Redraw", command=redraw).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Save Figure (PNG/PDF/SVG)...", command=save_fig).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Save Phase Data (CSV)...", command=save_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=2)
        tk.Label(btns, text="Changes of text fields take effect with 'Apply / Redraw' (or Enter).",
                 fg="#666", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=10)
        for k_ in ('curve', 'errbars', 'colors', 'invert', 'grid', 'legend'):
            V[k_].trace_add("write", redraw)
        V['phase0max'].trace_add("write", on_freq_change)
        win.bind("<Return>", redraw)
        tb = NavigationToolbar2Tk(cv, win, pack_toolbar=False); tb.update()
        tb.pack(side=tk.BOTTOM, fill="x")
        cv.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True)
        win._pa_state = (fig, V, redraw, save_fig, save_data)   # (for tests)
        on_freq_change()

    def _pa_draw_residuals(ax, t, y, fit, mag, tz, cols):
        """v4.24: residuals (data - all fitted frequencies) against time, with the mean of
        every night - shows night-to-night offsets, trends and beating of unfitted modes."""
        from photometry.period_analysis import nights_of
        res = fit['residuals'] if fit else y - np.mean(y)
        for k in np.unique(pa['idx']):
            m = pa['idx'] == k
            ax.plot(t[m] - tz, res[m], 'o', ms=2, color=cols[int(k) % len(cols)])
        ni = nights_of(t)
        xs, ms_, es = [], [], []
        for k in np.unique(ni):
            m = ni == k
            if m.sum() >= 3:
                xs.append(np.mean(t[m]) - tz); ms_.append(np.mean(res[m])); es.append(np.std(res[m], ddof=1) / np.sqrt(m.sum()))
        if xs:
            ax.errorbar(xs, ms_, yerr=es, fmt='s', ms=5, color='black', capsize=2, label="night mean")
            ax.legend(fontsize=7, loc='upper right')
        ax.axhline(0, color='red', lw=0.7)
        if mag:
            ax.invert_yaxis()
        rms = float(np.std(res))
        ax.set_title(("Residuals after " + ", ".join(f"f{i + 1}" for i in range(len(fit['freqs']))) if fit
                      else "Data minus mean (no frequency fitted yet)") + f"   rms = {rms:.4g} {pa['unit']}", fontsize=8)
        ax.set_xlabel(f"HJD - {tz}", fontsize=8); ax.set_ylabel(f"residual ({pa['unit']})", fontsize=8)

    def _pa_draw_oc(ax):
        """v4.24: O-C diagram of the times of maximum."""
        oc = pa.get('oc')
        if not oc or oc.get('n', 0) == 0:
            ax.text(0.5, 0.5, "O-C: press 'Times of Maximum / O-C'", ha='center', va='center',
                    transform=ax.transAxes, color='#888')
            return
        E, o, s_ = oc['E'], oc['oc'], oc['err']
        ax.errorbar(E, o, yerr=s_, fmt='o', ms=3, color='#1a3a5f', ecolor='#999', capsize=1, label="O-C of each maximum")
        if 'coef' in oc:
            ee = np.linspace(E.min(), E.max(), 200)
            ax.plot(ee, np.polyval(oc['coef'][::-1], ee), 'r-', lw=1,
                    label="parabola (period change)" if len(oc['coef']) == 3 else "straight line (period correction)")
        ax.axhline(0, color='#888', lw=0.6)
        ax.legend(fontsize=7, loc='best')
        ax.set_xlabel(f"Cycle E   (ephemeris: T0 = HJD {oc['T0']:.5f}, P = {oc['P']:.7f} d)", fontsize=8)
        ax.set_ylabel("O - C (d)", fontsize=8)
        ax.set_title(oc.get('title', ''), fontsize=8)

    def pa_draw():
        pa_fig.clear()
        if pa['t'] is None:
            pa_canvas.draw_idle()
            return
        t, y, fit = pa['t'], pa['y'], pa['fit']
        mag = 'mag' in pa['unit'].lower()
        ax1 = pa_fig.add_subplot(3, 1, 1)
        tz = int(t.min())
        cols = ['#1f4e8c', '#c45c00', '#2e7d32', '#7b1fa2', '#00838f', '#6d4c41']
        for k in np.unique(pa['idx']):
            m = pa['idx'] == k
            ax1.errorbar(t[m] - tz, y[m], yerr=pa['e'][m] if pa['e'] is not None else None, fmt='o', ms=2,
                         color=cols[int(k) % len(cols)], ecolor='#bbbbbb', elinewidth=0.6,
                         label=pa['sets'][int(k)]['name'] if len(pa['sets']) > 1 else None)
        if fit:
            from photometry.period_analysis import model_curve
            # model drawn densely only within the observed intervals (gaps stay empty)
            tt = []
            starts = np.concatenate([[0], np.where(np.diff(t) > 0.3)[0] + 1])
            ends = np.concatenate([starts[1:], [t.size]])
            fmax = max(fit['freqs']) * max(fit['nharm'])
            for a, b in zip(starts, ends):
                n = int(min(2000, max(50, (t[b - 1] - t[a]) * fmax * 40)))
                tt.append(np.linspace(t[a], t[b - 1], n)); tt.append([np.nan])
            tt = np.concatenate(tt)
            yy = model_curve(fit, np.nan_to_num(tt, nan=t[0]))
            yy[np.isnan(tt)] = np.nan
            ax1.plot(tt - tz, yy, '-', color='red', lw=0.9, label="model")
        if mag:
            ax1.invert_yaxis()
        ax1.set_xlabel(f"HJD - {tz}", fontsize=8); ax1.set_ylabel(pa['unit'], fontsize=8)
        ax1.tick_params(labelsize=7); ax1.grid(alpha=0.3)
        if len(pa['sets']) > 1 or fit:
            ax1.legend(fontsize=7, loc='best')

        ax2 = pa_fig.add_subplot(3, 1, 2)
        if pa['spec'] is not None:
            fr = pa['freq']
            x = 1.0 / fr if pa_xper_var.get() else fr
            ax2.plot(x, pa['spec'], '-', lw=0.7, color='#1a3a5f')
            ax2.set_ylabel(pa['spec_kind'], fontsize=8)
            if pa['window'] is not None:
                # spectral window centred on the highest peak, scaled to its height
                f0 = fr[int(np.nanargmax(pa['spec']))]
                wv = pa['window'] * np.nanmax(pa['spec'])
                xw = (f0 + pa['window_freq'])
                ok = (xw >= fr[0]) & (xw <= fr[-1])
                ax2.plot((1.0 / xw[ok]) if pa_xper_var.get() else xw[ok], wv[ok], '-', lw=0.7,
                         color='orange', alpha=0.8, label="spectral window (at highest peak)")
                ax2.legend(fontsize=7, loc='upper right')
            if fit:
                for i, f in enumerate(fit['freqs']):
                    xf = 1.0 / f if pa_xper_var.get() else f
                    ax2.axvline(xf, color='red', lw=0.6, ls=':')
                    ax2.text(xf, ax2.get_ylim()[1] * 0.95, f"f{i + 1}", color='red', fontsize=7, ha='center', va='top')
            if pa['preview'] is not None:
                xf = 1.0 / pa['preview'] if pa_xper_var.get() else pa['preview']
                ax2.axvline(xf, color='green', lw=0.8)
            _fmt_period_axis(ax2)
        else:
            ax2.text(0.5, 0.5, "Press 'Compute Periodogram'", ha='center', va='center', transform=ax2.transAxes, color='#888')
        ax2.tick_params(labelsize=7); ax2.grid(alpha=0.3)

        ax3 = pa_fig.add_subplot(3, 1, 3)
        bottom = pa_bottom_var.get()
        if bottom.startswith("Residuals"):
            _pa_draw_residuals(ax3, t, y, fit, mag, tz, cols)
        elif bottom.startswith("O"):
            _pa_draw_oc(ax3)
        else:
            f_ph, others, label = None, None, ""
            sel = pa_selected_freq()
            if pa['preview'] is not None:
                f_ph, label = pa['preview'], f"preview f = {pa['preview']:.6f} c/d (P = {1 / pa['preview']:.6g} d)"
                yres = fit['residuals'] if fit else y - np.mean(y)
            elif fit:
                i = sel if sel is not None else 0
                f_ph = fit['freqs'][i]
                # data minus all OTHER frequencies; v4.23: frequencies that are harmonics of the
                # selected one (2f, 3f, ...) are kept, they belong to the shape of its light curve
                from photometry.period_analysis import model_curve
                res_ = 1.0 / max(t.max() - t.min(), 1e-6)
                keep = [j for j, fj in enumerate(fit['freqs'])
                        if j == i or (fj > f_ph and abs(fj / f_ph - round(fj / f_ph)) * f_ph < res_ and round(fj / f_ph) >= 2)]
                only = dict(fit, freqs=[fit['freqs'][j] for j in keep], components=[fit['components'][j] for j in keep],
                            nharm=[fit['nharm'][j] for j in keep], offset=0.0)
                yres = fit['residuals'] + model_curve(only, t)
                extra = [f"f{j + 1}" for j in keep if j != i]
                label = (f"f{i + 1} = {f_ph:.6f} c/d (P = {1 / f_ph:.6g} d), other frequencies subtracted"
                         + (f" (harmonic{'s' if len(extra) > 1 else ''} {', '.join(extra)} kept)" if extra else ""))
            if f_ph:
                from photometry.period_analysis import phase_fold
                t0 = fit['t0'] if fit else float(np.round(np.mean(t), 1))
                ph = phase_fold(t, f_ph, t0)
                for k in np.unique(pa['idx']):
                    m = pa['idx'] == k
                    ax3.plot(np.concatenate([ph[m], ph[m] + 1]), np.concatenate([yres[m], yres[m]]), 'o', ms=2,
                             color=cols[int(k) % len(cols)])
                curve = pa_phasecurve_var.get()
                if curve.startswith("Fit") and fit and pa['preview'] is None:
                    pp = np.linspace(0, 2, 400)
                    # model of the selected frequency incl. its own harmonics and harmonic frequencies
                    # kept above (evaluated as a function of phase)
                    mm = model_curve(only, t0 + pp / f_ph) if fit else 0 * pp
                    nh_ = sum(len(c_) for c_ in only['components'])
                    ax3.plot(pp, mm, '-', color='red', lw=1,
                             label=f"fit: sine + {nh_ - 1} harmonic(s)" if nh_ > 1 else "fit: pure sine")
                    ax3.legend(fontsize=7, loc='upper right')
                elif curve.startswith("Binned"):
                    # v4.23: model-free mean curve (median per phase bin) - shows the real shape
                    # (e.g. the sawtooth of a delta Scuti star) without assuming a sinusoid
                    nb = int(max(8, min(50, round(np.sqrt(len(ph)) * 1.5))))
                    edges = np.linspace(0, 1, nb + 1)
                    cen, med = [], []
                    for a_, b_ in zip(edges[:-1], edges[1:]):
                        m_ = (ph >= a_) & (ph < b_)
                        if m_.sum() >= 2:
                            cen.append(0.5 * (a_ + b_)); med.append(np.median(yres[m_]))
                    if cen:
                        cen = np.array(cen); med = np.array(med)
                        ax3.plot(np.concatenate([cen, cen + 1]), np.concatenate([med, med]), '-o', ms=3,
                                 color='black', lw=1.2, label=f"median of {nb} phase bins")
                        ax3.legend(fontsize=7, loc='upper right')
                if mag:
                    ax3.invert_yaxis()
                ax3.set_title(label, fontsize=8)
                ax3.set_xlabel(f"Phase (t0 = HJD {t0})", fontsize=8)
                ax3.set_ylabel(f"{pa['unit']} (residual)" if fit or pa['preview'] else pa['unit'], fontsize=8)
            else:
                ax3.text(0.5, 0.5, "Phase plot: add a frequency or click into the periodogram", ha='center',
                         va='center', transform=ax3.transAxes, color='#888')
        ax3.tick_params(labelsize=7); ax3.grid(alpha=0.3)
        pa_fig.tight_layout()
        pa_canvas.draw_idle()

    def on_pa_click(event):
        if event.inaxes is None or pa['spec'] is None or len(pa_fig.axes) < 2 or event.inaxes is not pa_fig.axes[1]:
            return
        tb = getattr(pa_canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', ''):
            return     # zoom/pan active
        x = event.xdata
        if x is None or x <= 0:
            return
        f = 1.0 / x if pa_xper_var.get() else x
        # snap to the local maximum within +/- 0.5 resolution
        fr, sp = pa['freq'], pa['spec']
        T = pa['t'].max() - pa['t'].min()
        m = np.abs(fr - f) <= 0.5 / T
        if np.any(m):
            f = float(fr[m][np.nanargmax(sp[m])])
        pa['preview'] = f
        pa_status_var.set(f"Preview: f = {f:.6f} c/d, P = {1 / f:.6g} d  -  'Add Preview Frequency' adds it to the fit.")
        if pa_bottom_var.get() != "Phase plot":
            pa_bottom_var.set("Phase plot")
        else:
            pa_draw()
    pa_canvas.mpl_connect('button_press_event', on_pa_click)

    # ---- computations (in a worker thread) --------------------------------
    def _pa_grid():
        from photometry.period_analysis import frequency_grid, data_summary
        def opt(v):
            s_ = str(v.get()).strip()
            return float(s_) if s_ else None
        pmin, pmax = opt(pa_pmin_var), opt(pa_pmax_var)
        s = data_summary(pa['t'])
        fmax = 1.0 / pmin if pmin else None
        fmin = 1.0 / pmax if pmax else (0.5 / s['span'] if s['span'] > 0 else None)
        return frequency_grid(pa['t'], fmin=fmin, fmax=fmax, oversample=max(2, int(pa_over_var.get())))

    def _pa_run(job, done, msg):
        if pa['busy']:
            return
        pa['busy'] = True
        pa_status_var.set(msg)
        for b in (pa_compute_btn, pa_window_btn, pa_prew_btn):
            b.config(state=tk.DISABLED)

        def worker():
            try:
                res = job()
                root.after(0, lambda: done(res))
            except Exception as ex:
                import traceback; traceback.print_exc()
                root.after(0, lambda err=ex: (pa_status_var.set(f"Error: {err}"),
                                              messagebox.showerror("Period Analysis", str(err))))
            finally:
                def _end():
                    pa['busy'] = False
                    for b in (pa_compute_btn, pa_window_btn, pa_prew_btn):
                        b.config(state=tk.NORMAL)
                root.after(0, _end)
        import threading as _th
        _th.Thread(target=worker, daemon=True).start()

    def _pa_spectrum(freq, resid, method, nterms):
        from photometry import period_analysis as pam
        t, e = pa['t'], pa['e']
        if method.startswith("PDM"):
            return 1.0 - pam.pdm(t, resid, freq), "1 - theta (PDM)", None
        if method.startswith("BLS"):
            p, _ = pam.bls(t, resid, freq, e, magnitudes=('mag' in pa['unit'].lower()))
            return p, "BLS power (S/N)", None
        spec = pam.amplitude_spectrum(t, resid, freq, nterms=nterms)
        p, ls = pam.lomb_scargle(t, resid, freq, e, nterms=1)
        fap = pam.false_alarm_probability(ls, float(np.nanmax(p)), freq) if nterms == 1 else np.nan
        return spec, f"Amplitude ({pa['unit']})", fap

    def on_pa_compute():
        try:
            pa_preflight()
            freq = _pa_grid()
        except ValueError as ex:
            messagebox.showerror("Period Analysis", str(ex)); return
        method, nterms = pa_method_var.get(), max(1, int(pa_terms_var.get()))
        resid = pa['fit']['residuals'] if pa['fit'] else pa['y'] - np.mean(pa['y'])

        def job():
            return freq, _pa_spectrum(freq, resid, method, nterms)

        def done(res):
            freq_, (spec, kind, fap) = res
            pa.update(freq=freq_, spec=spec, spec_kind=kind, window=None, preview=None)
            i = int(np.nanargmax(spec))
            txt = (f"{method}: {len(freq_)} frequencies {freq_[0]:.4g}-{freq_[-1]:.4g} c/d; highest peak "
                   f"f = {freq_[i]:.6f} c/d (P = {1 / freq_[i]:.6g} d)")
            if fap is not None and np.isfinite(fap):
                txt += f", false-alarm probability {fap:.2g}"
            pa_status_var.set(txt + ("  [residuals after the fitted frequencies]" if pa['fit'] else ""))
            pa_draw()
        _pa_run(job, done, f"Computing {method} periodogram...")

    def on_pa_window():
        if pa['t'] is None:
            return
        try:
            freq = _pa_grid()
        except ValueError as ex:
            messagebox.showerror("Period Analysis", str(ex)); return
        from photometry.period_analysis import spectral_window
        T = pa['t'].max() - pa['t'].min()
        span = min(5.0, freq[-1] - freq[0])
        wf = np.arange(-span, span, 1.0 / (10 * T)) if T > 0 else np.linspace(-span, span, 2000)

        def job():
            return wf, spectral_window(pa['t'], wf)

        def done(res):
            pa['window_freq'], pa['window'] = res
            if pa['spec'] is None:
                pa_status_var.set("Spectral window computed - compute a periodogram to see it overlaid.")
            else:
                side = res[0][(res[1] > 0.3) & (np.abs(res[0]) > 1.5 / T)] if T > 0 else []
                strong = sorted({round(abs(x), 2) for x in side})[:6]
                pa_status_var.set("Spectral window (orange) shifted to the highest peak. Side peaks (aliases) at "
                                  f"+/- {', '.join(str(x) for x in strong) or 'none > 30 %'} c/d from every true peak.")
            pa_draw()
        _pa_run(job, done, "Computing spectral window...")

    def pa_preflight():
        if pa['t'] is None:
            raise ValueError("Please load a light curve first.")

    def _pa_refit(freqs, nharm, then_msg):
        from photometry import period_analysis as pam
        t, y, e = pa['t'], pa['y'], pa['e']
        method, nterms = pa_method_var.get(), max(1, int(pa_terms_var.get()))
        freq = pa['freq'] if pa['freq'] is not None else _pa_grid()

        def job():
            if not freqs:
                return None, None
            fit = pam.fit_frequencies(t, y, freqs, nharm, e, refine=True)
            fit['snr'], _ = pam.snr_all(t, fit, freq)
            res_ = pam.data_summary(t)['resolution']
            fit['notes'] = [pam.identify(f, fit['freqs'][:i], res_) for i, f in enumerate(fit['freqs'])]
            for i, snr in enumerate(fit['snr']):
                if np.isfinite(snr) and snr < 4:
                    fit['notes'][i] = ("S/N < 4 (not significant); " + fit['notes'][i]).strip("; ")
            spec = _pa_spectrum(freq, fit['residuals'], method, nterms)
            return fit, spec

        def done(res):
            fit, spec = res
            pa.update(fit=fit, freqs=list(fit['freqs']) if fit else [], nharm=list(nharm),
                      preview=None, window=None, freq=freq)
            if spec is not None:
                pa['spec'], pa['spec_kind'] = spec[0], spec[1]
            elif pa['spec'] is not None and not fit:
                pa['spec'], pa['spec_kind'], _ = _pa_spectrum(freq, pa['y'] - np.mean(pa['y']), method, nterms)
            pa_refresh_table(); pa_draw()
            if fit:
                pa_status_var.set(then_msg + f"  Residual rms = {fit['rms']:.4g} {pa['unit']}. "
                                  "The periodogram now shows the residuals.")
            else:
                pa_status_var.set("No frequencies.")
        _pa_run(job, done, "Fitting frequencies...")

    def on_pa_prewhiten():
        try:
            pa_preflight()
        except ValueError as ex:
            messagebox.showerror("Period Analysis", str(ex)); return
        if pa['spec'] is None:
            messagebox.showinfo("Period Analysis", "Compute a periodogram first."); return
        from photometry.period_analysis import find_peak, data_summary
        res_ = data_summary(pa['t'])['resolution']
        f = find_peak(pa['freq'], pa['spec'], exclude=pa['freqs'], resolution=res_)
        nh = max(1, int(pa_newharm_var.get()))
        _pa_refit(pa['freqs'] + [f], pa['nharm'] + [nh], f"Added f{len(pa['freqs']) + 1} = {f:.6f} c/d (all refitted).")

    def on_pa_add_preview():
        if pa['preview'] is None:
            messagebox.showinfo("Period Analysis", "Click into the periodogram first (or use 'Add Frequency...')."); return
        nh = max(1, int(pa_newharm_var.get()))
        _pa_refit(pa['freqs'] + [pa['preview']], pa['nharm'] + [nh], f"Added f = {pa['preview']:.6f} c/d.")

    def on_pa_add_manual():
        if pa['t'] is None:
            return
        from tkinter import simpledialog
        txt = simpledialog.askstring("Add frequency", "Frequency in c/d, or period in days followed by 'd' (e.g. 0.0860169d):",
                                     parent=root)
        if not txt:
            return
        try:
            txt = txt.strip().lower()
            f = 1.0 / float(txt[:-1]) if txt.endswith('d') else float(txt)
            if f <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Add frequency", f"'{txt}' is not a valid frequency/period."); return
        nh = max(1, int(pa_newharm_var.get()))
        _pa_refit(pa['freqs'] + [f], pa['nharm'] + [nh], f"Added f = {f:.6f} c/d.")

    def on_pa_remove():
        i = pa_selected_freq()
        if i is None:
            if not pa['freqs']:
                return
            i = len(pa['freqs']) - 1
        fr = pa['freqs'][:i] + pa['freqs'][i + 1:]
        nh = pa['nharm'][:i] + pa['nharm'][i + 1:]
        if fr:
            _pa_refit(fr, nh, f"Removed f{i + 1}.")
        else:
            pa.update(fit=None, freqs=[], nharm=[])
            pa_refresh_table(); on_pa_compute()

    def on_pa_set_harm():
        i = pa_selected_freq()
        if i is None:
            messagebox.showinfo("Period Analysis", "Select a frequency in the table."); return
        nh = list(pa['nharm']); nh[i] = max(1, int(pa_newharm_var.get()))
        _pa_refit(list(pa['freqs']), nh, f"f{i + 1} now fitted with {nh[i]} harmonic(s).")

    def on_pa_reset():
        pa.update(fit=None, freqs=[], nharm=[], preview=None, oc=None)
        pa_refresh_table()
        if pa['spec'] is not None:
            on_pa_compute()
        else:
            pa_draw()

    def on_pa_save():
        if not pa['fit'] and pa['spec'] is None:
            messagebox.showinfo("Period Analysis", "Nothing to save yet."); return
        from photometry import period_analysis as pam
        base = pa['sets'][0]['name'] if pa['sets'] else "light_curve"
        outdir = os.path.join("photometry_output", "period_analysis")
        os.makedirs(outdir, exist_ok=True)
        path = _pa_fd.asksaveasfilename(title="Save period analysis", initialdir=os.path.abspath(outdir),
                                        initialfile=f"periods_{base}.csv", defaultextension=".csv",
                                        filetypes=(("CSV", "*.csv"),))
        if not path:
            return
        s = pam.data_summary(pa['t'])
        info = {'files': "; ".join(lc['path'] for lc in pa['sets']), 'quantity': pa_quant_var.get(),
                'only OK points': bool(pa_only_ok_var.get()), 'mean subtracted per file': bool(pa_submean_var.get()),
                'N': s['n'], 'time span (d)': f"{s['span']:.4f}", 'nights': s['nights'],
                'frequency resolution 1/T (c/d)': f"{s['resolution']:.5g}",
                'method': pa_method_var.get(), 'frequency range (c/d)':
                    f"{pa['freq'][0]:.5g}-{pa['freq'][-1]:.5g}" if pa['freq'] is not None else "",
                'software': f"Calibra {APP_VERSION}"}
        if pa['fit']:
            pam.save_results(path, pa['fit'], info)
        else:
            with open(path, 'w') as fh:
                for k, v in info.items():
                    fh.write(f"# {k}: {v}\n")
        png = os.path.splitext(path)[0] + ".png"
        pa_fig.savefig(png, dpi=130)
        if pa['spec'] is not None:
            with open(os.path.splitext(path)[0] + "_periodogram.csv", 'w', newline='') as fh:
                w_ = csv.writer(fh); w_.writerow(['frequency_c_per_d', 'period_d', pa['spec_kind']])
                step = max(1, len(pa['freq']) // 200000)
                for f, p in zip(pa['freq'][::step], pa['spec'][::step]):
                    w_.writerow([f"{f:.7f}", f"{1 / f:.7f}", f"{p:.6g}"])
        if pa['fit']:
            with open(os.path.splitext(path)[0] + "_residuals.csv", 'w', newline='') as fh:
                w_ = csv.writer(fh); w_.writerow(['hjd', 'value', 'model', 'residual', 'set'])
                for tt_, yy_, rr_, k_ in zip(pa['t'], pa['y'], pa['fit']['residuals'], pa['idx']):
                    w_.writerow([f"{tt_:.6f}", f"{yy_:.6f}", f"{yy_ - rr_:.6f}", f"{rr_:.6f}", pa['sets'][int(k_)]['name']])
        pa_status_var.set(f"Saved: {path} (+ .png, _periodogram.csv" + (", _residuals.csv)" if pa['fit'] else ")"))

    pa_compute_btn.config(command=on_pa_compute)
    pa_window_btn.config(command=on_pa_window)
    pa_prew_btn = ttk.Button(row_f, text="Prewhiten: Add Highest Peak", command=on_pa_prewhiten)
    pa_prew_btn.pack(side=tk.LEFT, padx=2)
    for txt, cmd in (("Add Preview Frequency", on_pa_add_preview), ("Add Frequency...", on_pa_add_manual),
                     ("Remove Selected/Last", on_pa_remove), ("Reset", on_pa_reset)):
        ttk.Button(row_f, text=txt, command=cmd).pack(side=tk.LEFT, padx=2)
    row_f2 = tk.Frame(lf_pa_fr, bg="white"); row_f2.pack(fill="x", padx=8, pady=(0, 4))
    tk.Label(row_f2, text="Harmonics for a new frequency:", bg="white").pack(side=tk.LEFT)
    ttk.Spinbox(row_f2, from_=1, to=10, textvariable=pa_newharm_var, width=3).pack(side=tk.LEFT, padx=(2, 6))
    ttk.Button(row_f2, text="Apply to Selected", command=on_pa_set_harm).pack(side=tk.LEFT, padx=2)
    ttk.Button(row_f2, text="Save Results...", command=on_pa_save).pack(side=tk.LEFT, padx=(14, 2))
    ttk.Button(row_f2, text="Open Output Folder",
               command=lambda: open_path(os.path.abspath(os.path.join("photometry_output", "period_analysis")))).pack(side=tk.LEFT, padx=2)
    pa_tree.pack(fill="x", padx=8, pady=4)
    tk.Label(lf_pa_fr, textvariable=pa_quality_var, bg="white", fg="#1a3a5f", font=("Segoe UI", 9),
             wraplength=820, justify="left").pack(anchor="w", padx=8, pady=(0, 4))
    tk.Label(lf_pa_fr, bg="white", fg="#666", font=("Segoe UI", 8), wraplength=820, justify="left",
             text="Prewhitening: the highest peak of the residual periodogram is added and all frequencies are fitted "
                  "together (frequencies, amplitudes, phases; non-linear least squares). Repeat until no peak reaches "
                  "S/N = 4 (S/N = amplitude / mean residual amplitude within +/- 1 c/d). Notes mark harmonics (2f1), "
                  "combinations (f1+f2) and possible 1 c/d or 1/yr aliases. Uncertainties after Montgomery & "
                  "O'Donoghue (1999) assume white noise and are lower limits. Select a row to see its phase plot "
                  "(harmonics 2f, 3f ... of the selected frequency stay in it). Light curves are rarely sinusoidal "
                  "(delta Scuti / RR Lyrae: sawtooth): use 3-5 harmonics for such a frequency, or show the "
                  "model-free 'Binned mean' curve.").pack(anchor="w", padx=8, pady=(0, 6))

    # ---- 4. Times of maximum and O-C (v4.24) ---------------------------------
    lf_pa_oc = tk.LabelFrame(pa_container, text="4. Times of maximum and O-C", bg="white", font=("Segoe UI", 10, "bold"), fg="#1a3a5f")
    lf_pa_oc.pack(fill="x", pady=5)
    row_o = tk.Frame(lf_pa_oc, bg="white"); row_o.pack(fill="x", padx=8, pady=(6, 2))
    pa_tm_method_var = _pa_var("pa_tmax_method", "Template (fitted shape)")
    pa_tm_minpts_var = _pa_var("pa_tmax_min_points", 8, int)
    pa_tm_window_var = _pa_var("pa_tmax_window", "0.3")
    pa_oc_quad_var = _pa_var("pa_oc_quadratic", False, bool)
    pa_eph_mode_var = _pa_var("pa_ephemeris_mode", "fit")
    pa_eph_t0_var = _pa_var("pa_ephemeris_t0", "")
    pa_eph_p_var = _pa_var("pa_ephemeris_p", "")
    tk.Label(row_o, text="Method:", bg="white").pack(side=tk.LEFT)
    ttk.Combobox(row_o, textvariable=pa_tm_method_var, width=22, state="readonly",
                 values=["Template (fitted shape)", "Parabola"]).pack(side=tk.LEFT, padx=(2, 8))
    tk.Label(row_o, text="min. points per maximum:", bg="white").pack(side=tk.LEFT)
    ttk.Spinbox(row_o, from_=4, to=100, textvariable=pa_tm_minpts_var, width=4).pack(side=tk.LEFT, padx=(2, 8))
    tk.Label(row_o, text="window (+/- phase):", bg="white").pack(side=tk.LEFT)
    ttk.Entry(row_o, textvariable=pa_tm_window_var, width=5).pack(side=tk.LEFT, padx=(2, 8))
    row_o2 = tk.Frame(lf_pa_oc, bg="white"); row_o2.pack(fill="x", padx=8, pady=2)
    tk.Label(row_o2, text="Ephemeris (C):", bg="white").pack(side=tk.LEFT)
    ttk.Radiobutton(row_o2, text="from the fit", variable=pa_eph_mode_var, value="fit").pack(side=tk.LEFT, padx=4)
    ttk.Radiobutton(row_o2, text="given:  T0 (HJD of maximum)", variable=pa_eph_mode_var, value="manual").pack(side=tk.LEFT, padx=(4, 2))
    ttk.Entry(row_o2, textvariable=pa_eph_t0_var, width=15).pack(side=tk.LEFT, padx=2)
    tk.Label(row_o2, text="P (d)", bg="white").pack(side=tk.LEFT)
    ttk.Entry(row_o2, textvariable=pa_eph_p_var, width=13).pack(side=tk.LEFT, padx=2)
    ttk.Checkbutton(row_o2, text="fit a parabola (period change)", variable=pa_oc_quad_var).pack(side=tk.LEFT, padx=8)
    row_o3 = tk.Frame(lf_pa_oc, bg="white"); row_o3.pack(fill="x", padx=8, pady=(2, 4))
    pa_oc_text_var = tk.StringVar(value="")

    def on_pa_tmax():
        fit = pa['fit']
        if not fit:
            messagebox.showinfo("Times of maximum", "Fit at least the main frequency first (section 3)."); return
        from photometry import period_analysis as pam
        i = pa_selected_freq()
        i = 0 if i is None else i
        mag = 'mag' in pa['unit'].lower()
        try:
            window = float(pa_tm_window_var.get())
            minpts = int(pa_tm_minpts_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Times of maximum", "Window and minimum number of points must be numbers."); return
        method = 'poly' if pa_tm_method_var.get().startswith("Parabola") else 'template'
        mx = pam.times_of_maximum(pa['t'], pa['y'], fit, i, pa['e'], magnitudes=mag, method=method,
                                  window=window, min_points=minpts)
        if not mx:
            messagebox.showwarning("Times of maximum", "No maximum could be determined: no cycle has enough points "
                                   "on both sides of the expected maximum (try a lower minimum number of points "
                                   "or a wider window).")
            return
        if pa_eph_mode_var.get() == "manual":
            try:
                T0 = float(pa_eph_t0_var.get()); P = float(pa_eph_p_var.get())
                if T0 < 2400000:
                    T0 += 2400000.0 if T0 > 10000 else 0.0
            except ValueError:
                messagebox.showerror("Ephemeris", "Please give T0 (HJD) and P (days) as numbers."); return
            for m in mx:          # renumber cycles for the given ephemeris
                m['E'] = int(np.round((m['tmax'] - T0) / P))
            src = "given ephemeris"
        else:
            T0, P, _ = pam.ephemeris(fit, i, mag)
            src = "ephemeris from the fit"
        oc = pam.oc_analysis(mx, T0, P, quadratic=bool(pa_oc_quad_var.get()))
        oc['maxima'] = mx
        oc['freq_index'] = i
        lines = [f"{len(mx)} times of maximum of f{i + 1} ({'template fit' if method == 'template' else 'parabola'}), "
                 f"median error {np.median([m['err'] for m in mx]) * 1440:.2f} min; O-C against the {src} "
                 f"T0 = HJD {T0:.5f}, P = {P:.8f} d."]
        if 'coef' in oc:
            lines.append(f"O-C rms = {oc['rms'] * 1440:.2f} min, reduced chi2 = {oc['chi2_red']:.2f}.  Corrected ephemeris: "
                         f"T0 = HJD {oc['T0_new']:.5f} +/- {oc['T0_new_err']:.5f}, P = {oc['P_new']:.8f} +/- {oc['P_new_err']:.8f} d")
            if 'dP_dE' in oc:
                yr = oc['dP_dt'] * 365.25
                sig = abs(oc['dP_dE'] / oc['dP_dE_err']) if oc['dP_dE_err'] > 0 else np.nan
                lines.append(f"Period change: dP/dE = {oc['dP_dE']:.3e} +/- {oc['dP_dE_err']:.1e} d/cycle  "
                             f"(dP/dt = {yr:.3e} d/yr, (1/P) dP/dt = {yr / P:.2e} /yr; {sig:.1f} sigma)")
        else:
            lines.append("Too few maxima (or only one cycle number) for a line/parabola fit.")
        oc['title'] = f"O-C of f{i + 1}: {len(mx)} maxima"
        pa['oc'] = oc
        pa_oc_text_var.set("\n".join(lines))
        pa_bottom_var.set("O-C diagram")
        pa_draw()

    def on_pa_save_maxima():
        oc = pa.get('oc')
        if not oc or not oc.get('maxima'):
            messagebox.showinfo("Times of maximum", "Compute the times of maximum first."); return
        outdir = os.path.join("photometry_output", "period_analysis")
        os.makedirs(outdir, exist_ok=True)
        base = pa['sets'][0]['name'] if pa['sets'] else "light_curve"
        path = _pa_fd.asksaveasfilename(title="Save times of maximum", initialdir=os.path.abspath(outdir),
                                        initialfile=f"maxima_{base}.csv", defaultextension=".csv",
                                        filetypes=(("CSV", "*.csv"),))
        if not path:
            return
        with open(path, 'w', newline='') as fh:
            fh.write(f"# times of maximum light (HJD) of f{oc['freq_index'] + 1}; ephemeris T0 = {oc['T0']:.6f}, P = {oc['P']:.9f}\n")
            if 'coef' in oc:
                fh.write(f"# corrected: T0 = {oc['T0_new']:.6f} +/- {oc['T0_new_err']:.6f}, P = {oc['P_new']:.9f} +/- {oc['P_new_err']:.9f}\n")
            if 'dP_dE' in oc:
                fh.write(f"# dP/dE = {oc['dP_dE']:.4e} +/- {oc['dP_dE_err']:.1e} d/cycle\n")
            w_ = csv.writer(fh)
            w_.writerow(['E', 'hjd_max', 'err_d', 'o_minus_c_d', 'n_points', 'method', 'rms_fit'])
            for m, o in zip(oc['maxima'], oc['oc']):
                w_.writerow([m['E'], f"{m['tmax']:.6f}", f"{m['err']:.6f}", f"{o:.6f}", m['n'], m['method'], f"{m['rms']:.5f}"])
        pa_status_var.set(f"Times of maximum saved: {path}")

    ttk.Button(row_o3, text="Times of Maximum / O-C", command=on_pa_tmax).pack(side=tk.LEFT, padx=2)
    ttk.Button(row_o3, text="Save Maxima (CSV)...", command=on_pa_save_maxima).pack(side=tk.LEFT, padx=2)
    tk.Label(lf_pa_oc, textvariable=pa_oc_text_var, bg="white", fg="#1a3a5f", font=("Segoe UI", 9),
             wraplength=820, justify="left").pack(anchor="w", padx=8, pady=(0, 4))
    tk.Label(lf_pa_oc, bg="white", fg="#666", font=("Segoe UI", 8), wraplength=820, justify="left",
             text="For the selected frequency (default f1) one time of maximum light is determined per observed cycle "
                  "(other, non-harmonic frequencies are subtracted first). 'Template' shifts the fitted light-curve "
                  "shape (use enough harmonics) and works for any shape; 'Parabola' fits the points around the "
                  "brightest point. O-C = observed - calculated time; a straight line gives a corrected period, "
                  "a parabola a period change. For a given ephemeris (e.g. GCVS epoch and period) choose 'given'. "
                  "See period_analysis_guide.md for details.").pack(anchor="w", padx=8, pady=(0, 6))

    tk.Label(pa_container, textvariable=pa_status_var, fg="#d32f2f", bg="white", font=("Arial", 10, "bold"),
             wraplength=820, justify="left").pack(anchor="w", pady=4)
    pa_plot_frame.pack(fill="both", expand=True, pady=4)
    pa_plotbar = tk.Frame(pa_plot_frame, bg="#e8eef5", bd=1, relief="groove")
    pa_plotbar.pack(side=tk.TOP, fill="x")
    tk.Label(pa_plotbar, text="Bottom plot:", bg="#e8eef5", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(6, 2))
    for _txt in ("Phase plot", "Residuals vs time", "O-C diagram"):
        tk.Radiobutton(pa_plotbar, text=_txt, variable=pa_bottom_var, value=_txt, bg="#e8eef5",
                       activebackground="#e8eef5").pack(side=tk.LEFT, padx=4)
    ttk.Button(pa_plotbar, text="Phase Plot Window (edit / save)...",
               command=lambda: pa_open_phase_window()).pack(side=tk.RIGHT, padx=6, pady=2)
    pa_toolbar = NavigationToolbar2Tk(pa_canvas, pa_plot_frame, pack_toolbar=False)
    pa_toolbar.update()
    pa_canvas.toolbar = pa_toolbar
    pa_toolbar.pack(side=tk.TOP, fill="x")
    pa_canvas.get_tk_widget().pack(fill="both", expand=True)

    def pa_open_files(paths):
        """Used by other tabs: load these light curves and show the tab."""
        if pa_load_paths(list(paths), replace=True):
            switch_tab("period")
    vars_dict['pa_open_files'] = pa_open_files

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
    tk.Label(info_frame, text=f"Version: {APP_VERSION} \tLatest Update: {APP_DATE}", font=("Arial", 10), bg="white").pack(fill="x")
    
    tk.Label(about_container, text="Description:", font=("Arial", 11, "bold"), fg=primary_blue, bg="white").pack(fill="x", pady=(10, 5))
    desc_text = (
        "Calibra uses star detection, sub-pixel PSF fitting, aperture photometry with a CCD/CMOS noise model\n"
        "(including averaged stacks), zero-point calibration and the determination of colour transformations\n"
        "between filters. For the analysis, Calibra can use the online catalogues ATLAS-RefCat2, APASS DR9,\n"
        "Landolt Standards Catalogue and Gaia DR3, or a user-provided catalogue, and it marks known variables\n"
        "from the AAVSO VSX.\n"
        "\n"
        "Calibra produces light curves of variable stars from a series of images (calibrated magnitudes with an\n"
        "AAVSO-formatted report, relative flux, or raw ADU), computes standard B, V and B-V by differential\n"
        "photometry with an ensemble of comparison stars, searches image sequences for unknown variables\n"
        "(Data Mining) and measures light curves of many known variables in the field at once (Field Variables;\n"
        "targets from a list or from AAVSO VSX / GCVS). Pre-processing covers bias/flat calibration, plate solving\n"
        "with ASTAP and the splitting of one-shot-colour images; an interactive FITS viewer is included.\n"
    )
    tk.Label(about_container, text=desc_text, justify=tk.LEFT, font=("Arial", 10), bg="white").pack(fill="x")

    tk.Label(about_container, text="Recent changes:", font=("Arial", 11, "bold"), fg=primary_blue, bg="white").pack(fill="x", pady=(10, 5))
    recent_text = (
        "v4.45  Local ATLAS-RefCat2 tiles read completely (first line with hex flags was taken as a header)\n"
        "v4.44  Light curves: faint-target mode - forced photometry at a fixed position, frames combined (groups / selected rows)\n"
        "v4.43  Light curves: UT date (YYYY-MM-DD) of mid-exposure added next to the UT time\n"
        "v4.42  Light curves: UT of mid-exposure (HH:MM:SS) instead of JD in table and CSV\n"
        "v4.41  HJD correct for sexagesimal site coordinates in the header (MWT frames)\n"
        "v4.40  Planner table: azimuth in degrees only\n"
        "v4.39  Planner: azimuth in degrees, RA/Dec and meridian columns, constellation sort; horizontal scrolling\n"
        "v4.38  Night plan for NINA: exposure times, durations, eclipses, NINA target set\n"
        "v4.37  Observation Planner: targets for a night (GCVS list, priority list, open clusters)\n"
        "v4.36  Instrument setups (telescope, camera, site, filters, settings) - switch in the sidebar\n"
        "v4.35  AAVSO sequences stored per variable and reused without a new query\n"
        "v4.34  FITS Viewer opens at once: catalogue data in the background, local copies, faster drawing\n"
        "v4.33  Bias/flat/dark folders: frames with another type in the header are used (warning)\n"
        "v4.32  APASS DR10 download from the AAVSO archive: button, or automatic for missing fields\n"
        "v4.31  APASS DR10 CSV files from the AAVSO download page usable as local catalogue\n"
        "v4.30  Star names looked up in the local GCVS list first (works offline), then SIMBAD / VSX\n"
        "v4.29  Master frames from a bias folder and a flat folder (every FITS file in it, names do not matter)\n"
        "v4.28  tools/catalog_overview.py: which catalogue files can be deleted (Dec / magnitude limits)\n"
        "v4.27  Master bias/flat/dark from calibration folders; automatic masters; calibration on the fly\n"
        "v4.26  Local catalogue copies (ATLAS, Gaia, APASS, VSX, GCVS list); online only when needed\n"
        "v4.25  Period Analysis: plot selector above the plot; editable phase-plot window (save PNG/PDF/SVG, CSV)\n"
        "v4.24  Period Analysis: fit quality, residuals vs time, times of maximum and O-C; guide period_analysis_guide.md\n"
        "v4.23  L (luminance) light curves; SAAF/GCVS CSV target lists; phase-plot curve selectable (fit / binned / none)\n"
        "v4.22  Plate solving: position guess (object name or RA/Dec), each solution guides the next image\n"
        "v4.21  Period Analysis tab: Lomb-Scargle / PDM / BLS, spectral window, prewhitening, phase plots\n"
        "v4.20  Several nights / pointings: field-of-view checks for all images, consistent comparison stars\n"
        "v4.19  Field Variables tab: light curves of many variables (list file or VSX/GCVS query); .fts files\n"
        "v4.18  Light Curves action bar, named setups, dated output files, automatic comparison stars from Data Mining\n"
        "v4.17  Much faster Light Curves and Data Mining; Data Mining remembers its measurements\n"
        "v4.16  Corrected noise model, weighted colour transformation, ensemble differential photometry with errors\n"
        "v4.15  Bug-fix release (filter handling, FITS viewer, calibration, file manager, error messages)\n"
        "v4.14  Light-curve modes: calibrated magnitudes, relative flux, raw ADU\n"
        "Full history: CHANGELOG.md   |   Open ideas: TODO.md\n"
    )
    tk.Label(about_container, text=recent_text, justify=tk.LEFT, font=("Arial", 9), bg="white").pack(fill="x")

    tk.Label(about_container, text="Developed by Stephan Pomp, Google DeepMind Antigravity and Claude (Anthropic)",
             font=("Arial", 9, "italic"), fg="#555", bg="white").pack(fill="x", pady=(10, 0))

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
        "- Set the Aperture Radius to approximately 1.5-2x the FWHM of your stars.\n"
        "- Use Settings > Astro & Catalog > Region Selection to focus on specific targets or avoid edges.\n"
        "- For stacked images, enter the 'Blank-sky noise check' factor from the pipeline log as Sky Noise Factor (Settings).\n"
        "- Save one Light-Curve setup per variable star (Light Curves > Save Target Setup...).\n"
        "- Several known variables in one field (e.g. long-period variables over many nights): Field Variables tab.\n"
        "- Periods (also several, e.g. on top of a main period): Period Analysis tab - check the spectral window for aliases."
    )
    tk.Label(help_frame, text=help_info, justify=tk.LEFT, font=("Arial", 9, "italic")).pack(pady=10)

    # Developer Info
    dev_info = (
        "\nDevelopers & Contact:\n"
        "Developed by Stephan Pomp, Google DeepMind Antigravity and Claude (Anthropic)\n"
        "Contact: stephan.pomp@gmail.com"
    )
    tk.Label(help_frame, text=dev_info, justify=tk.LEFT, font=("Arial", 9), fg="#555").pack(side=tk.BOTTOM, anchor="w")

    # Create Sidebar buttons
    create_sidebar_button(f"{get_icon('📂', '')}  File Manager".strip(), "files")
    create_sidebar_button(f"{get_icon('⚙', '')}  Pre-processing".strip(), "pre")
    create_sidebar_button(f"{get_icon('🔍', '')}  Analysis & Calib".strip(), "analysis")
    create_sidebar_button(f"{get_icon('📈', '')}  Light Curves".strip(), "ts")
    create_sidebar_button(f"{get_icon('🚀', '')}  Data Mining".strip(), "datamining")
    create_sidebar_button(f"{get_icon('✨', '')}  Field Variables".strip(), "fieldvars")
    create_sidebar_button(f"{get_icon('〰', '')}  Period Analysis".strip(), "period")
    create_sidebar_button(f"{get_icon('🔭', '')}  Observation Planner".strip(), "planner")
    create_sidebar_button(f"{get_icon('🔧', '')}  Settings".strip(), "settings")
    create_sidebar_button(f"{get_icon('ℹ', '')}  About Calibra".strip(), "about")
    create_sidebar_button(f"{get_icon('❓', '')}  User Help".strip(), "help")
    
    # --- v4.37: OBSERVATION PLANNER (photometry/planner_tab.py) ---
    def _plan_to_lc(t):
        from astropy.coordinates import SkyCoord as _SC
        import astropy.units as _u
        c_ = _SC(float(t['ra']) * _u.deg, float(t['dec']) * _u.deg)
        ts_target_name_var.set(t.get('name', ''))
        ts_target_ra_var.set(c_.ra.to_string(unit=_u.hourangle, sep=':', precision=1, pad=True))
        ts_target_dec_var.set(c_.dec.to_string(unit=_u.deg, sep=':', precision=0, alwayssign=True, pad=True))
        ts_target_mode_var.set("name")
        switch_tab("ts")

    def _plan_to_fv(targets):
        from photometry.local_catalogs import name_key as _nk
        have = {_nk(t.get('name', '')) for t in fv_targets}
        n = 0
        for t in targets:
            if t.get('ra') is None or _nk(t.get('name', '')) in have:
                continue
            fv_targets.append({'name': t.get('name', ''), 'ra': float(t['ra']), 'dec': float(t['dec']),
                               'type': t.get('type') or '', 'period': t.get('period'), 'max': t.get('max'),
                               'min': t.get('min'), 'source': 'planner', 'use': True})
            have.add(_nk(t.get('name', ''))); n += 1
        fv_refresh_tree()
        return n
    try:
        from photometry import planner_tab as _ptab
        _ptab.build(tab_planner, root, vars_dict, {'to_light_curves': _plan_to_lc, 'to_field_variables': _plan_to_fv})
    except Exception as _e:
        import traceback; traceback.print_exc()
        tk.Label(tab_planner, text=f"Observation planner could not be built: {_e}", fg="red").pack()

    # --- v4.36: INSTRUMENT SETUPS (sidebar: choose / manage) ---
    from photometry import instruments as _inst
    active_inst_var = tk.StringVar(value="")
    vars_dict["active_instrument"] = (active_inst_var, str)
    inst_box = tk.Frame(sidebar_frame, bg="#132b47")
    inst_box.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(4, 10))
    tk.Label(inst_box, text="Instrument setup:", fg="#c9d6e6", bg="#132b47", font=("Segoe UI", 8, "bold")).pack(anchor="w")
    inst_combo = ttk.Combobox(inst_box, state="readonly", width=20)
    inst_combo.pack(fill=tk.X, pady=2)
    inst_info_var = tk.StringVar(value="")
    tk.Label(inst_box, textvariable=inst_info_var, fg="#9fb3c8", bg="#132b47", font=("Segoe UI", 7),
             wraplength=165, justify=tk.LEFT).pack(anchor="w")
    inst_hint_btn = tk.Button(inst_box, text="", font=("Segoe UI", 7, "bold"), fg="#132b47", bg="#ffd27f",
                              relief="flat", wraplength=160, justify=tk.LEFT)
    inst_hint = {'name': None}

    def inst_current_values():
        out = {}
        for k in _inst.SETTING_KEYS:
            if k in vars_dict and isinstance(vars_dict[k], (tuple, list)) and hasattr(vars_dict[k][0], 'get'):
                try:
                    out[k] = vars_dict[k][0].get()
                except tk.TclError:
                    pass
        return out

    def inst_refresh(select=None):
        names = _inst.list_names()
        inst_combo['values'] = names
        cur = select or active_inst_var.get()
        if cur in names:
            inst_combo.set(cur)
        s = _inst.load(cur) if cur in names else None
        inst_info_var.set(_inst.summary(s) if s else "")

    def inst_autosave(*_):
        """Store the current settings in the active setup (called on switching and on exit)."""
        name = active_inst_var.get()
        if not name:
            return
        s = _inst.load(name)
        if s is None:
            return
        _inst.snapshot(s, inst_current_values())
        _inst.save(s)
    vars_dict['instrument_autosave'] = inst_autosave

    def inst_apply(name):
        s = _inst.load(name)
        if s is None:
            return 0
        vals = _inst.values_to_apply(s)
        n = 0
        for k, v in vals.items():
            if k in vars_dict and isinstance(vars_dict[k], (tuple, list)) and hasattr(vars_dict[k][0], 'set'):
                try:
                    vars_dict[k][0].set(v); n += 1
                except tk.TclError:
                    pass
        try:
            tidy_numbers()
        except Exception:
            pass
        try:
            vars_dict['cal_after_load']()
        except Exception:
            pass
        return n

    def inst_switch(name, quiet=False):
        """Activate setup 'name': current settings -> old setup, settings of 'name' -> Calibra."""
        old = active_inst_var.get()
        if name == old:
            inst_refresh(); return
        inst_autosave()
        s = _inst.load(name)
        if s is None:
            return
        if not s.get('settings'):
            # new setup without stored settings: it starts from the current settings,
            # its own site and filter names (if given) are applied
            _inst.snapshot(s, inst_current_values(), sync_descriptive=False); _inst.save(s)
            n = 0
        else:
            n = inst_apply(name)
        active_inst_var.set(name)
        if n == 0:
            inst_apply_descriptive(name)
        inst_hint_btn.pack_forget(); inst_hint['name'] = None
        inst_refresh(name)
        print(f"Instrument setup '{name}' active" + (f" ({n} settings applied)." if n else "."))
        try:
            save_session()
        except Exception:
            pass

    inst_combo.bind("<<ComboboxSelected>>", lambda e: inst_switch(inst_combo.get()))

    def inst_check_files(paths):
        """After loading frames: do they belong to another setup? (header: sensor, focal length...)"""
        if not paths:
            return
        try:
            from astropy.io import fits as _fits
            h = _fits.getheader(paths[0])
            if h.get('NAXIS', 0) < 2:
                h = _fits.getheader(paths[0], 1)
            best, sc = _inst.best_match(h)
            act = active_inst_var.get()
            act_sc = _inst.match_score(_inst.load(act), h) if act and _inst.load(act) else 0.0
            if best and best != act and sc >= 2.0 and sc > act_sc + 1.0:
                inst_hint['name'] = best
                inst_hint_btn.config(text=f"These frames look like '{best}' - click to switch")
                inst_hint_btn.pack(fill=tk.X, pady=(4, 0))
                print(f"Note: the loaded frames match the instrument setup '{best}' better than the active "
                      f"'{act}' (sensor size / focal length / camera). Switch in the sidebar if needed.")
            else:
                inst_hint_btn.pack_forget(); inst_hint['name'] = None
        except Exception as e:
            print(f"Instrument check: {e}")
    inst_hint_btn.config(command=lambda: inst_switch(inst_hint['name']) if inst_hint['name'] else None)
    vars_dict['instrument_check'] = inst_check_files

    def inst_manage():
        """Dialog: list of setups, instrument / site / filter / planning data, fill from FITS header."""
        import re
        import time
        win = tk.Toplevel(root); win.title("Instrument Setups"); win.geometry("900x640")
        left = tk.Frame(win); left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        tk.Label(left, text="Setups", font=("Arial", 10, "bold")).pack(anchor="w")
        lb = tk.Listbox(left, width=24, height=18, exportselection=False); lb.pack(fill=tk.Y, expand=True)
        right = tk.Frame(win); right.pack(side=tk.LEFT, fill="both", expand=True, padx=8, pady=8)
        cur = {'name': None}
        fvars = {}

        def row(parent, r, label, key, width=22, kind='str'):
            tk.Label(parent, text=label).grid(row=r, column=0, sticky=tk.W, padx=4, pady=1)
            v = tk.BooleanVar() if kind == 'bool' else tk.StringVar()
            if kind == 'bool':
                ttk.Checkbutton(parent, variable=v).grid(row=r, column=1, sticky=tk.W)
            else:
                ttk.Entry(parent, textvariable=v, width=width).grid(row=r, column=1, sticky=tk.W, padx=4, pady=1)
            fvars[key] = (v, kind)

        g1 = tk.LabelFrame(right, text="Instrument"); g1.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        for r, (lab, key, kind) in enumerate((
                ("Name:", "name", 'str'), ("Description:", "description", 'str'),
                ("Telescope:", "instrument.telescope", 'str'), ("Aperture (mm):", "instrument.aperture_mm", 'num'),
                ("Focal length (mm):", "instrument.focal_length_mm", 'num'), ("Reducer / corrector:", "instrument.reducer", 'str'),
                ("Camera:", "instrument.camera", 'str'), ("Pixel size (um):", "instrument.pixel_um", 'num'),
                ("Sensor width (px):", "instrument.sensor_x", 'num'), ("Sensor height (px):", "instrument.sensor_y", 'num'),
                ("Binning:", "instrument.binning", 'num'), ("Colour camera (OSC):", "instrument.osc", 'bool'),
                ("Capture gain:", "instrument.gain", 'num'), ("Capture offset:", "instrument.offset", 'num'))):
            row(g1, r, lab, key, kind=kind)
        g2 = tk.LabelFrame(right, text="Site"); g2.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        for r, (lab, key, kind) in enumerate((
                ("Site name:", "site.name", 'str'), ("Latitude (deg N):", "site.lat", 'num'),
                ("Longitude (deg E):", "site.lon", 'num'), ("Elevation (m):", "site.elevation_m", 'num'),
                ("Time zone (e.g. Europe/Stockholm):", "site.timezone", 'str'))):
            row(g2, r, lab, key, kind=kind)
        g4 = tk.LabelFrame(g2, text="Planning"); g4.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        row(g4, 0, "Exposure times (s):", "planning.exposures_s", kind='list')
        row(g4, 1, "Brightest mag:", "planning.mag_bright", width=8, kind='num')
        row(g4, 2, "Faintest mag:", "planning.mag_faint", width=8, kind='num')
        row(g4, 3, "ZP V (e-/s at V=0, empty=estimate):", "planning.zp_v_e", width=8, kind='num')
        row(g4, 4, "Sky brightness (mag/arcsec2):", "planning.sky_mag", width=8, kind='num')
        row(g4, 5, "Typical seeing (arcsec):", "planning.seeing_arcsec", width=8, kind='num')
        g3 = tk.LabelFrame(right, text="Filters (name in the FITS header / position in the filter wheel for NINA)")
        g3.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        for c, b in enumerate(_inst.BANDS):
            tk.Label(g3, text=b, font=("Arial", 9, "bold")).grid(row=0, column=c * 2, columnspan=2)
            v1, v2 = tk.StringVar(), tk.StringVar()
            ttk.Entry(g3, textvariable=v1, width=9).grid(row=1, column=c * 2, padx=2)
            ttk.Entry(g3, textvariable=v2, width=3).grid(row=1, column=c * 2 + 1, padx=(0, 6))
            fvars[f"filters.{b}.fits_name"] = (v1, 'str'); fvars[f"filters.{b}.nina_position"] = (v2, 'num')
        derived_var = tk.StringVar(value="")
        tk.Label(right, textvariable=derived_var, fg="#1a3a5f", justify=tk.LEFT, wraplength=620).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=4)
        tk.Label(right, text="The analysis settings of a setup (CCD gain, read noise, saturation, apertures, detection, "
                             "zero points, colour/extinction coefficients, calibration folders, plate-solving field ...) "
                             "are the normal Calibra settings: they are stored in the active setup automatically and "
                             "restored when you switch to it.", fg="#555", font=("Arial", 8), wraplength=620,
                 justify=tk.LEFT).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=4)

        def getp(d, key):
            for part in key.split('.'):
                d = (d or {}).get(part) if isinstance(d, dict) else None
            return d

        def setp(d, key, val):
            parts = key.split('.')
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val

        def show(name):
            s = _inst.load(name)
            if s is None:
                return
            cur['name'] = name
            for key, (v, kind) in fvars.items():
                val = getp(s, key)
                if kind == 'bool':
                    v.set(bool(val))
                elif kind == 'list':
                    v.set(", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in (val or [])))
                else:
                    v.set("" if val is None else (f"{val:g}" if isinstance(val, float) else str(val)))
            n_set = len(s.get('settings') or {})
            derived_var.set(_inst.summary(s) + f"\n{n_set} analysis settings stored"
                            + (f" (last {s['settings_saved']})" if s.get('settings_saved') else "")
                            + ("   - ACTIVE setup" if name == active_inst_var.get() else ""))

        def collect():
            s = _inst.load(cur['name']) if cur['name'] else None
            if s is None:
                return None
            for key, (v, kind) in fvars.items():
                if key == 'name':
                    continue
                raw = v.get()
                if kind == 'bool':
                    val = bool(raw)
                elif kind == 'num':
                    raw = str(raw).strip().replace(',', '.')
                    try:
                        val = float(raw) if raw else None
                        if val is not None and val.is_integer() and key.endswith(('sensor_x', 'sensor_y', 'binning', 'nina_position')):
                            val = int(val)
                    except ValueError:
                        raise ValueError(f"'{raw}' is not a number ({key})")
                elif kind == 'list':
                    val = [float(x) for x in re.split(r'[,;\s]+', str(raw).strip()) if x]
                else:
                    val = str(raw).strip()
                setp(s, key, val)
            return s

        def fill_list(select=None):
            lb.delete(0, tk.END)
            names = _inst.list_names()
            for n in names:
                lb.insert(tk.END, n + ("   (active)" if n == active_inst_var.get() else ""))
            sel = select or cur['name'] or active_inst_var.get()
            if sel in names:
                i = names.index(sel); lb.selection_set(i); lb.see(i); show(sel)

        def on_select(*_):
            i = lb.curselection()
            if i:
                show(_inst.list_names()[i[0]])
        lb.bind("<<ListboxSelect>>", on_select)

        def do_save(quiet=False):
            try:
                s = collect()
            except ValueError as e:
                messagebox.showerror("Instrument Setups", str(e), parent=win); return False
            if s is None:
                return False
            new_name = fvars['name'][0].get().strip()
            _inst.save(s)
            if new_name and new_name != s['name']:
                try:
                    _inst.rename(s['name'], new_name)
                except ValueError as e:
                    messagebox.showerror("Instrument Setups", str(e), parent=win); return False
                if active_inst_var.get() == s['name']:
                    active_inst_var.set(new_name)
                cur['name'] = new_name
            if cur['name'] == active_inst_var.get():
                # site coordinates and filter names of the active setup go to the settings at once
                inst_apply_descriptive(cur['name'])
            fill_list(cur['name']); inst_refresh()
            return True

        def do_new(copy_from=None):
            from tkinter import simpledialog
            nm = simpledialog.askstring("New setup", "Name of the new setup (e.g. 'Refractor 102 + ASI6200', 'MWT', "
                                        "'Vespera 2'):", parent=win)
            if not nm or not nm.strip():
                return
            if nm.strip() in _inst.list_names():
                messagebox.showerror("Instrument Setups", f"'{nm}' exists already.", parent=win); return
            if copy_from:
                s = _inst.load(copy_from); s['name'] = nm.strip(); s['created'] = time.strftime('%Y-%m-%d')
            else:
                s = _inst.template(nm.strip())
                _inst.snapshot(s, inst_current_values())       # starts from the current settings
            _inst.save(s)
            cur['name'] = nm.strip(); fill_list(cur['name']); inst_refresh()

        def do_delete():
            if not cur['name']:
                return
            if cur['name'] == active_inst_var.get():
                messagebox.showinfo("Instrument Setups", "The active setup cannot be deleted - switch to another one first.",
                                    parent=win); return
            if messagebox.askyesno("Instrument Setups", f"Delete the setup '{cur['name']}'?", parent=win):
                _inst.delete(cur['name']); cur['name'] = None; fill_list(); inst_refresh()

        def do_fill():
            if not cur['name']:
                return
            paths = [loaded_files[int(iid)]['path'] for iid in tree.get_children() if tree.item(iid, 'values')[0] == '[X]']
            if not paths:
                sel = tree.selection()
                paths = [loaded_files[int(sel[0])]['path']] if sel else []
            if not paths:
                messagebox.showinfo("Instrument Setups", "Check (or select) a light frame of this instrument in the "
                                    "File Manager first.", parent=win); return
            from astropy.io import fits as _fits
            h = _fits.getheader(paths[0])
            if h.get('NAXIS', 0) < 2:
                h = _fits.getheader(paths[0], 1)
            if not do_save(quiet=True):
                return
            s = _inst.load(cur['name'])
            filled = _inst.fill_from_header(s, h)
            fh = _inst.from_header(h)['filter']
            if fh:
                from photometry.filters import classify_band
                b = classify_band(fh, s['filters']['V']['fits_name'] or 'Vmag', s['filters']['B']['fits_name'] or 'Bmag',
                                  s['filters']['L']['fits_name'] or 'L')
                if b in s['filters'] and not s['filters'][b]['fits_name']:
                    s['filters'][b]['fits_name'] = fh; filled.append(f"filters.{b}")
            _inst.save(s); show(cur['name'])
            messagebox.showinfo("Instrument Setups", (f"From {os.path.basename(paths[0])}:\n" + "\n".join(filled))
                                if filled else "Nothing new found in the header (only empty fields are filled).",
                                parent=win)

        def do_nina_filters():
            """Filter names and wheel positions from an existing NINA sequence (v4.38)."""
            if not cur['name']:
                return
            from tkinter import filedialog
            from photometry import nina_export as _ne
            from photometry.filters import classify_band
            p_ = filedialog.askopenfilename(title="NINA sequence or target set (XML)", parent=win,
                                            initialdir=_ne.default_folder(os.path.dirname(os.path.abspath(__file__))),
                                            filetypes=(("NINA XML", "*.xml"), ("all files", "*.*")))
            if not p_:
                return
            try:
                fm = _ne.filters_from_nina_file(p_)
            except Exception as e:
                messagebox.showerror("Instrument Setups", f"Could not read {p_}: {e}", parent=win); return
            if not do_save(quiet=True):
                return
            s = _inst.load(cur['name'])
            got = []
            for fname, pos in fm.items():
                b = classify_band(fname, 'Vmag', 'Bmag', 'L')
                if not b:
                    b = fname.strip().upper()[:1] if fname.strip().upper()[:1] in _inst.BANDS and len(fname.strip()) == 1 else None
                if b in s['filters']:
                    s['filters'][b]['fits_name'] = fname
                    s['filters'][b]['nina_position'] = pos
                    got.append(f"{b}: {fname} = position {pos}")
            _inst.save(s); show(cur['name'])
            messagebox.showinfo("Instrument Setups", ("Filters taken over:\n" + "\n".join(got)) if got else
                                "No V/B/L/R/G/I filters recognised in this file.", parent=win)

        btns = tk.Frame(left); btns.pack(fill=tk.X, pady=4)
        for txt, cmd in (("New (current settings)", do_new), ("Duplicate", lambda: do_new(cur['name'])),
                         ("Delete", do_delete)):
            ttk.Button(btns, text=txt, command=cmd).pack(fill=tk.X, pady=1)
        bot = tk.Frame(right); bot.grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(bot, text="Fill from FITS Header (checked frame)", command=do_fill).pack(side=tk.LEFT, padx=2)
        ttk.Button(bot, text="Filters from NINA File...", command=do_nina_filters).pack(side=tk.LEFT, padx=2)
        ttk.Button(bot, text="Save", command=do_save).pack(side=tk.LEFT, padx=2)
        def do_activate():
            if do_save():
                inst_switch(cur['name'])
                fill_list(cur['name'])
        ttk.Button(bot, text="Save and Activate", command=do_activate).pack(side=tk.LEFT, padx=2)
        ttk.Button(bot, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=2)
        fill_list()
    vars_dict['instrument_manage'] = inst_manage

    def inst_apply_descriptive(name):
        """Site coordinates and filter names of the active setup -> Calibra settings."""
        s = _inst.load(name)
        if s is None:
            return
        vals = _inst.values_to_apply(dict(s, settings={}))
        for k, v in vals.items():
            if k in vars_dict:
                try:
                    vars_dict[k][0].set(v)
                except tk.TclError:
                    pass

    tk.Button(inst_box, text="Manage Setups...", font=("Segoe UI", 8), fg="white", bg="#2e5b8a", relief="flat",
              activebackground="#1f446e", activeforeground="white", command=inst_manage).pack(fill=tk.X, pady=(2, 0))

    def instruments_after_load():
        """At start: make sure there is a setup; the session's settings stay as they are."""
        names = _inst.list_names()
        if not names:
            s = _inst.template("Standard")
            s['description'] = "Created from the settings Calibra had before instrument setups existed (v4.36)."
            _inst.snapshot(s, inst_current_values())
            _inst.save(s)
            names = ["Standard"]
            print("Instrument setups: 'Standard' created from the current settings - rename it and add your "
                  "other setups in the sidebar (Manage Setups...).")
        if active_inst_var.get() not in names:
            active_inst_var.set(names[0])
        inst_refresh()
        try:
            inst_check_files([f['path'] for f in loaded_files][:1])
        except Exception:
            pass
    vars_dict['instruments_after_load'] = instruments_after_load

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
    sys.stdout.start_polling(root)
    sys.stderr.start_polling(root)

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
    try:
        vars_dict['cal_after_load']()          # v4.27: calibration settings from the session
    except Exception as _e:
        print(f"Calibration settings: {_e}")
    try:
        vars_dict['localcat_after_load']()     # v4.26: local catalogue folder from the session
    except Exception as _e:
        print(f"Local catalogues: {_e}")
    try:
        vars_dict['instruments_after_load']()  # v4.36: instrument setups
    except Exception as _e:
        print(f"Instrument setups: {_e}")
    tidy_numbers()

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
