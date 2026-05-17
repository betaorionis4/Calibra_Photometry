"""
FITS Viewer Module for StarID/Calibra Photometry Pipeline.
Provides an interactive GUI for inspecting FITS files, performing real-time 
photometry, and selecting variable/reference stars.
"""

import numpy as np
import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from scipy.spatial import cKDTree
from photometry.calibration import get_cached_catalog, save_to_cache, fetch_online_catalog, get_vsx_stars
from photometry.gui_utils import add_copy_context_menu, SelectableLabel

class FITSViewer:
    """
    An interactive FITS image viewer with star inspection, Gaussian fitting,
    and bidirectional data synchronization with the main photometry application.
    """
    def __init__(self, parent, fits_path, ref_catalog="ATLAS", default_zp=23.399, config=None, initial_stars=None, aavso_stars=None, export_callback=None, aperture_export_callback=None):
        """
        Initialize the FITS viewer.

        Args:
            parent: The parent tkinter window/frame.
            fits_path (str): Path to the FITS file to load.
            ref_catalog (str): Name of the catalog to use for cross-identification.
            default_zp (float): Default photometric zero point.
            config (dict): Configuration dictionary for aperture/fit parameters.
            initial_stars (dict): Stars to automatically mark on load.
            aavso_stars (list): List of AAVSO sequence stars for marking.
            export_callback (callable): Function to call when exporting star selections.
            aperture_export_callback (callable): Function to call when exporting aperture settings.
        """
        self.parent = parent
        self.fits_path = fits_path
        self.ref_catalog = ref_catalog
        self.default_zp = default_zp
        self.aavso_stars = aavso_stars or []
        self.base_name = os.path.splitext(os.path.basename(fits_path))[0]
        
        # Load FITS Data and WCS header
        try:
            with fits.open(fits_path) as hdul:
                self.data = hdul[0].data.astype(float)
                self.header = hdul[0].header
                self.wcs = WCS(self.header)
                self.has_wcs = self.wcs.has_celestial
        except Exception as e:
            messagebox.showerror("Error", f"Could not load FITS file: {e}")
            self.parent.destroy()
            return

        self.config = config or {}
        self.initial_stars = initial_stars
        self.export_callback = export_callback
        self.aperture_export_callback = aperture_export_callback

        # State management for marked stars
        self.variable_star = None # {'x', 'y', 'id', 'markers', ...}
        self.check_star = None
        self.on_the_spot_refs = [] # List of {'x', 'y', 'id', 'markers', ...}
        self.on_the_spot_zp = None

        # In-memory catalog cache to avoid repeated HTTP queries (Fix #4)
        self._catalog_cache = {}  # keyed on rounded (ra, dec)

        # Attempt to load results from a previous automated pipeline run
        self.results_data = []
        self.results_tree = None
        self._load_results()
        
        # UI Setup
        self.parent.title(f"FITS Viewer: {os.path.basename(fits_path)}")
        
        # Set Window Icon (Calibra Logo)
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibra_logo.png")
        if os.path.exists(logo_path):
            try:
                from PIL import Image, ImageTk
                icon_img = Image.open(logo_path)
                icon_img = icon_img.resize((32, 32), Image.Resampling.LANCZOS)
                self._icon = ImageTk.PhotoImage(icon_img) # Keep reference
                self.parent.iconphoto(True, self._icon)
            except:
                pass
        
        # --- Window Geometry & Layout Calculation ---
        # The viewer uses a fixed base height (750px) and calculates a total width 
        # sufficient to house both the main astronomical image canvas and two 
        # sidebar panels for data display and controls.
        ny, nx = self.data.shape
        base_h = 750
        panel_w = 200    # Target width for each of the two right-side detail panels
        canvas_w = 1200   # Target width for the main Matplotlib FITS canvas
        
        # Calculate final window width: Canvas + 2 * Sidebars
        win_w = canvas_w + (panel_w * 2)
        self.parent.geometry(f"{win_w}x{base_h}")
        
        # Status bar at the bottom: Provides real-time RA/Dec and pixel feedback
        # Using tk.Label instead of ttk.Label for better color/padding control
        self.status_var = tk.StringVar(value="Hover for coordinates | Scroll to zoom | Click to inspect star")
        self.status_bar = SelectableLabel(self.parent, textvariable=self.status_var, 
                                          bg="#e1e1e1", 
                                          font=("Arial", 10), padx=10)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Main Layout Architecture ---
        # We use a nested PanedWindow system to allow users to interactively resize
        # the main sections (image vs. data) and sub-sections (inspection vs. references).
        
        # Primary Horizontal Split: [Left: FITS Image] | [Right: All Star Detail Panels]
        self.main_paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left side: Matplotlib Canvas for FITS display
        # weight=4 ensures that the image section takes most of the space on window expansion
        self.canvas_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.canvas_frame, weight=8)

        # Right side: Detail and Control panels
        # The sidebar columns are nested within this horizontal paned window
        self.right_paned = ttk.PanedWindow(self.main_paned, orient=tk.HORIZONTAL)
        self.main_paned.add(self.right_paned, weight=3) 

        # 1. Column A: Variable & Inspection Panel (Vertical stack)
        self.var_inspection_paned = ttk.PanedWindow(self.right_paned, orient=tk.VERTICAL)
        self.right_paned.add(self.var_inspection_paned, weight=2)
        
        # Live star details (text box)
        self.details_frame = ttk.LabelFrame(self.var_inspection_paned, text="Live Inspection", padding=5)
        self.var_inspection_paned.add(self.details_frame, weight=1)
        
        # Radial Profile Plot
        self.profile_frame = ttk.LabelFrame(self.var_inspection_paned, text="Radial Profile", padding=2)
        self.var_inspection_paned.add(self.profile_frame, weight=2)
        
        # Variable Star info
        self.var_frame = ttk.LabelFrame(self.var_inspection_paned, text="Variable Star", padding=5)
        self.var_inspection_paned.add(self.var_frame, weight=1)

        # 2. Column B: Reference & Check Stars Panel (Vertical stack)
        self.ref_paned = ttk.PanedWindow(self.right_paned, orient=tk.VERTICAL)
        self.right_paned.add(self.ref_paned, weight=1)

        # List of Ref/Check stars
        self.ref_frame = ttk.LabelFrame(self.ref_paned, text="Ref & Check Stars", padding=10)
        self.ref_paned.add(self.ref_frame, weight=4)

        # Aperture control inputs
        self.ap_frame = ttk.LabelFrame(self.ref_paned, text="Aperture Settings", padding=5)
        self.ref_paned.add(self.ap_frame, weight=1)

        # Initialize text areas with read-only state
        self.details_text = tk.Text(self.details_frame, wrap=tk.WORD, state=tk.DISABLED, 
                                    bg="#f8f9fa", font=("Courier", 10), width=25, height=12)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        add_copy_context_menu(self.details_text)

        self.var_text = tk.Text(self.var_frame, wrap=tk.WORD, state=tk.DISABLED,
                                bg="#fff5f5", font=("Courier", 10), width=25, height=12)
        self.var_text.pack(fill=tk.BOTH, expand=True)
        add_copy_context_menu(self.var_text)

        self.ref_text = tk.Text(self.ref_frame, wrap=tk.WORD, state=tk.DISABLED,
                                bg="#f1f8f1", font=("Courier", 9), width=25)
        self.ref_text.pack(fill=tk.BOTH, expand=True)
        add_copy_context_menu(self.ref_text)

        # Button to export selected stars back to the main app
        if self.export_callback:
            self.export_btn = ttk.Button(self.ref_frame, text="Export Stars to LC Tab", command=self._on_export_click)
            self.export_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
            
        # Aperture Controls: Linked to tkinter variables
        self.ap_var = tk.DoubleVar(value=self.config.get('aperture_radius', 8.0))
        self.ann_in_var = tk.DoubleVar(value=self.config.get('annulus_inner', 15.0))
        self.ann_out_var = tk.DoubleVar(value=self.config.get('annulus_outer', 20.0))
        
        def create_ap_field(label, var, row):
            """Helper to create labeled entry fields in the aperture panel."""
            tk.Label(self.ap_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=2)
            ttk.Entry(self.ap_frame, textvariable=var, width=6).grid(row=row, column=1, sticky=tk.W, padx=2)
            
        create_ap_field("Aperture:", self.ap_var, 0)
        create_ap_field("Annulus In:", self.ann_in_var, 1)
        create_ap_field("Annulus Out:", self.ann_out_var, 2)
        
        self.export_ap_btn = ttk.Button(self.ap_frame, text="Export Aps to Settings", command=self._on_export_ap_click)
        self.export_ap_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)
        
        # Setup Main Matplotlib Figure
        self.fig = Figure(figsize=(8, 8))
        if self.has_wcs:
            self.ax = self.fig.add_subplot(111, projection=self.wcs)
        else:
            self.ax = self.fig.add_subplot(111)
            
        # Scaling for astronomical images
        zscale = ZScaleInterval(contrast=0.15)
        vmin, vmax = zscale.get_limits(self.data)
        
        # Ensure the background isn't blown out by noise
        bg_median = np.nanmedian(self.data)
        if vmin < bg_median:
            vmin = bg_median
            
        self.im = self.ax.imshow(self.data, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
        self.ax.set_title(os.path.basename(fits_path))
        
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
        
        # Add Coordinate Grid Lines if WCS is available
        if self.has_wcs:
            try:
                # Configure RA/Dec Grid Lines
                # Use slightly higher alpha and explicit color for better contrast
                self.ax.coords[0].grid(color='red', alpha=0.5, linestyle='-', linewidth=0.5)   # RA
                self.ax.coords[1].grid(color='green', alpha=0.5, linestyle='-', linewidth=0.5) # Dec
                
                # Configure Tick Labels and Ticks
                self.ax.coords[0].set_axislabel('Right Ascension', color='red', fontsize=10)
                self.ax.coords[1].set_axislabel('Declination', color='green', fontsize=10)
                
                # Ensure ticks are visible and expanded as requested
                self.ax.coords[0].set_ticks(color='red', size=8, width=1)
                self.ax.coords[1].set_ticks(color='green', size=8, width=1)
            except:
                pass
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Radial Profile Plot Setup
        self.profile_fig = Figure(figsize=(4, 4))
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, master=self.profile_frame)
        self.profile_canvas.draw()
        self.profile_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.marker = None
        self.ref_markers = []
        
        # Event bindings
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Load Initial Stars if provided (delayed to ensure UI is ready)
        if self.initial_stars:
            self.parent.after(500, self._auto_mark_initial_stars)

        # Draw AAVSO sequence markers if provided
        if self.aavso_stars:
            self.parent.after(800, self._draw_aavso_markers)
        
        # Right-click Context Menu for star role assignment
        self.context_menu = tk.Menu(self.parent, tearoff=0)
        self.context_menu.add_command(label="Mark as Variable Star (Red)", command=self._add_variable_star)
        self.context_menu.add_command(label="Mark as Check Star (Blue)", command=self._add_check_star)
        self.context_menu.add_command(label="Mark as Reference Star (Green)", command=self._add_on_the_spot_ref)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Remove Marker", command=self._remove_marker_at_click)
        self.context_menu.add_command(label="Clear ALL Markers", command=self._clear_all_markers)
        
        self._last_click_pos = (0, 0) # Track last right-click coordinate
        
        # Cleanup
        self.parent.protocol("WM_DELETE_WINDOW", self.on_close)

    def _draw_aavso_markers(self):
        """
        Draw yellow circles and AUID labels for all known AAVSO reference stars in the field.
        """
        if not self.has_wcs or not self.aavso_stars:
            return
            
        # Determine filter from header to show relevant magnitude
        header_filt = str(self.header.get('FILTER', 'V')).upper()
        mag_key = 'V_mag' if 'V' in header_filt else 'B_mag'
        
        for star in self.aavso_stars:
            try:
                # Project RA/Dec to pixel coordinates
                sky = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg)
                px, py = self.wcs.world_to_pixel(sky)
                
                # Verify if the star is within the visible image bounds
                ny, nx = self.data.shape
                if 0 <= px < nx and 0 <= py < ny:
                    # Draw a distinct yellow circle for sequence stars
                    circ = patches.Circle((px, py), radius=15, edgecolor='yellow', facecolor='none', linewidth=1.5, alpha=0.7)
                    self.ax.add_patch(circ)
                    
                    # Add text label with AUID and Magnitude
                    mag = star.get(mag_key, np.nan)
                    mag_label = f"{mag:.2f}" if not np.isnan(mag) else "na"
                    label = f"{star['auid']}\n({mag_label})"
                    self.ax.text(px + 18, py, label, color='yellow', fontsize=8, fontweight='bold', alpha=0.9,
                                 bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1))
            except Exception as e:
                print(f"Error marking AAVSO star {star.get('auid')}: {e}")
        
        self.canvas.draw_idle()

    def on_close(self):
        """
        Clean up resources and autosave current selections before closing.
        """
        try:
            # Autosave selections and aperture settings back to the main app (Fix)
            self._on_export_click(quiet=True)
            self._on_export_ap_click(quiet=True, refresh=False) # Skip refresh on close
        except:
            pass

        self.fig.clear()
        self.ref_markers = []
        self.parent.destroy()

    def _load_results(self):
        """
        Load automated photometry results from the pipeline's output folder.
        Robustly resolves the path relative to the project structure (Fix #13).
        Builds a spatial index for fast lookups (Fix #6).
        """
        # Resolve path relative to workspace root (parent of photometry/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, 'photometry_output', f'targets_auto_{self.base_name}.csv')
        
        if not os.path.exists(csv_path):
            return
        
        results = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Extract and clean refined coordinates
                        row['refined_x'] = float(row['refined_x']) if row.get('refined_x') else float(row['raw_x'])
                        row['refined_y'] = float(row['refined_y']) if row.get('refined_y') else float(row['raw_y'])
                        results.append(row)
                    except:
                        continue
            
            if results:
                self.results_data = results
                # Build cKDTree for O(log n) lookups (Fix #6)
                # Note: CSV results are 1-based, tree matches these directly
                coords = np.array([[s['refined_x'], s['refined_y']] for s in results])
                self.results_tree = cKDTree(coords)
                
        except Exception as e:
            pass
    def _update_details(self, text):
        """Update the Live Inspection text panel."""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, text)
        self.details_text.config(state=tk.DISABLED)

    def on_hover(self, event):
        """Update the status bar with mouse coordinates and WCS RA/Dec."""
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            # Matplotlib uses 0-based coords; display as 1-based for FITS standard
            status = f"X: {x+1:.1f} Y: {y+1:.1f}"
            if self.has_wcs:
                try:
                    coord = self.wcs.pixel_to_world(x, y)
                    ra_str = coord.ra.to_string(unit='hour', sep=':', precision=2)
                    dec_str = coord.dec.to_string(unit='deg', sep=':', precision=2)
                    status += f" | RA: {ra_str} Dec: {dec_str}"
                except:
                    pass
            self.status_var.set(status)

    def on_click(self, event):
        """Handle mouse click events for star selection and context menus."""
        if event.inaxes != self.ax:
            return
        
        # Right Click: Show context menu
        if event.button == 3: 
            self._last_click_pos = (event.xdata, event.ydata)
            self.context_menu.post(int(event.guiEvent.x_root), int(event.guiEvent.y_root))
            return
            
        if event.button != 1:
            return

        x_click, y_click = event.xdata, event.ydata
        
        # Search for a star in existing pipeline results using spatial index (Fix #6)
        found_in_csv = False
        if self.results_tree:
            # CSV coords are 1-based, event data is 0-based
            dist, idx = self.results_tree.query([x_click + 1, y_click + 1], distance_upper_bound=10.0)
            
            if dist < 10.0:
                nearest_star = self.results_data[idx]
                found_in_csv = True
                details = f"--- Pipeline Result ---\n\n"
                details += f"ID:     {nearest_star['id']}\n"
                details += f"X:      {nearest_star['refined_x']:.2f}\n"
                details += f"Y:      {nearest_star['refined_y']:.2f}\n\n"
                
                if nearest_star.get('ra_hms'):
                    details += f"RA: {nearest_star['ra_hms']}\n"
                    details += f"Dec:{nearest_star['dec_dms']}\n\n"
                
                details += f"SNR:    {nearest_star.get('snr', 'N/A')}\n"
                details += f"Peak:   {nearest_star.get('peak_adu', 'N/A')} ADU\n\n"
                
                # Verify identity against online catalogs if possible
                if self.has_wcs:
                    try:
                        ra_deg = float(nearest_star['ra_deg'])
                        dec_deg = float(nearest_star['dec_deg'])
                        cat_star = self._query_catalog(ra_deg, dec_deg)
                        if cat_star:
                            v_mag = cat_star.get('V_mag')
                            v_str = f"{v_mag:.3f}" if isinstance(v_mag, (int, float)) and not np.isnan(v_mag) else "N/A"
                            b_mag = cat_star.get('B_mag')
                            b_str = f"{b_mag:.3f}" if isinstance(b_mag, (int, float)) and not np.isnan(b_mag) else "N/A"
                            
                            if cat_star.get('source') == 'AAVSO':
                                details += f"--- AAVSO Identification ---\n"
                            else:
                                details += f"--- Catalog Comparison ---\n"
                                
                            details += f"ID: {cat_star.get('display_name', 'Star')}\n"
                            details += f"Catalog V: {v_str}\n"
                            details += f"Catalog B: {b_str}\n\n"
                            
                            if cat_star.get('is_variable'):
                                details += f"!!! VARIABLE STAR !!!\n"
                                details += f"Type: {cat_star.get('var_type', 'N/A')}\n\n"
                            
                            nearest_star['is_variable'] = cat_star.get('is_variable', False)
                    except:
                        pass


                mag_err = nearest_star.get('mag_calibrated_err', '')
                if mag_err:
                    details += f"Error:     ±{mag_err}\n"
                
                # Zero Point Logic: Use ensemble ZP if available, else default ZP
                active_zp = self.default_zp
                zp_label = f"ZP={self.default_zp:.3f}"

                if self.on_the_spot_zp is not None:
                    active_zp = self.on_the_spot_zp
                    zp_label = f"Ens ZP={self.on_the_spot_zp:.3f}"

                inst_f = float(nearest_star.get('net_flux', 0))
                if inst_f > 0:
                    inst_m = -2.5 * np.log10(inst_f)
                    cal_m = inst_m + active_zp
                    details += f"\nMeas Mag:  {cal_m:.3f}*\n"
                    details += f"(* using {zp_label})\n"

                details += f"\nVariable:  {nearest_star.get('is_variable', 'Unknown')}\n"
                
                self._update_details(details)
                # Draw selection crosshair
                self._draw_marker(nearest_star['refined_x'] - 1, nearest_star['refined_y'] - 1)

        # Perform real-time Gaussian fit to update the radial profile plot
        # If we found it in CSV, we suppress the basic fit's crosshair and details 
        # to preserve the rich CSV metadata.
        if found_in_csv:
            res = self._fit_at(x_click, y_click)
            if res:
                self._update_profile_plot(res)
        else:
            self._basic_fit(x_click, y_click)

    def _draw_marker(self, x, y, is_ref=False):
        """
        Draw or update the selection crosshair.
        
        Args:
            x, y (float): Coordinates in 0-based pixel space.
            is_ref (bool): If True, suppresses the red cross (used when marking roles).
        """
        # Clear existing temporary marker
        if self.marker:
            self.marker.remove()
            self.marker = None
        
        # Draw red cross for inspection unless it's a role marker
        if not is_ref:
            self.marker, = self.ax.plot(x, y, 'r+', markersize=15, mew=2)
        
        self.canvas.draw_idle()

    def _basic_fit(self, x_click, y_click, suppress_cross=False):
        """
        Perform a 2D Gaussian fit at the clicked location and display results.
        
        Args:
            x_click, y_click (float): Mouse click coordinates.
            suppress_cross (bool): Whether to skip drawing the selection cross.
        """
        res = self._fit_at(x_click, y_click)
        if not res:
            self._update_details("Fit Failed.")
            return

        # Update the radial profile plot with the new fit data
        self._update_profile_plot(res)

        # Zero Point application
        active_zp = self.default_zp
        zp_label = f"ZP={self.default_zp:.3f}"
        if self.on_the_spot_zp is not None:
            active_zp = self.on_the_spot_zp
            zp_label = f"Ens ZP={self.on_the_spot_zp:.3f}"
            
        cal_mag = res['inst_mag'] + active_zp
        
        details = f"--- Real-time Fit ---\n\n"
        details += f"X:      {res['x']:.2f}\n"
        details += f"Y:      {res['y']:.2f}\n\n"
        
        if self.has_wcs:
            coord = self.wcs.pixel_to_world(res['x'], res['y'])
            details += f"RA:  {coord.ra.to_string(unit='hour', sep=':', precision=1)}\n"
            details += f"Dec: {coord.dec.to_string(unit='degree', sep=':', precision=1, alwayssign=True)}\n\n"
            
            # Resolve against online catalogs
            cat_star = self._query_catalog(coord.ra.deg, coord.dec.deg)
            if cat_star:
                v_mag = cat_star.get('V_mag')
                v_str = f"{v_mag:.3f}" if isinstance(v_mag, (int, float)) and not np.isnan(v_mag) else "N/A"
                
                if cat_star.get('source') == 'AAVSO':
                    details += f"--- AAVSO Identification ---\n"
                else:
                    details += f"--- Catalog Comparison ---\n"
                    
                details += f"ID: {cat_star.get('display_name', 'Star')}\n"
                details += f"Catalog V: {v_str}\n\n"
                if cat_star.get('is_variable'):
                    details += f"!!! VARIABLE STAR !!!\n"
                    details += f"Type: {cat_star.get('var_type', 'N/A')}\n\n"
        
        details += f"FWHM:   {res['fwhm']:.2f} px\n"
        details += f"Inst M: {res['inst_mag']:.3f}\n"
        details += f"Cal M:  {cal_mag:.3f}*\n"
        details += f"(* using {zp_label})\n"
        
        self._update_details(details)
        if not suppress_cross:
            self._draw_marker(res['x'], res['y'])

    def _query_catalog(self, ra, dec):
        """
        Query online catalogs (ATLAS, APASS, GAIA, VSX) to identify a star.
        Prioritizes the AAVSO sequence stars if available.
        """
        # 1. Check AAVSO Sequence stars first (High Priority)
        if self.aavso_stars:
            click_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            for star in self.aavso_stars:
                star_coord = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg)
                if click_coord.separation(star_coord).arcsec < 4.0:
                    # Found a match in our AAVSO sequence cache!
                    # Treat AUID as an alias, but prefer Catalog magnitudes for the comparison
                    v_mag = star.get('cat_v') if star.get('cat_match') else star.get('V_mag')
                    b_mag = star.get('cat_b') if star.get('cat_match') else star.get('B_mag')
                    src = self.ref_catalog if star.get('cat_match') else 'AAVSO'
                    
                    return {
                        'display_name': star['auid'],
                        'id': star['auid'],
                        'V_mag': v_mag,
                        'B_mag': b_mag,
                        'ra_deg': star['ra'],
                        'dec_deg': star['dec'],
                        'source': src
                    }

        if not self.ref_catalog or not any(k in self.ref_catalog.upper() for k in ["ATLAS", "APASS", "GAIA", "LANDOLT"]):
            return None

        # Fix #4: Check in-memory cache first (keyed on 3-decimal rounded coords)
        cache_key = (round(ra, 3), round(dec, 3), self.ref_catalog.upper())
        if cache_key in self._catalog_cache:
            return self._catalog_cache[cache_key]

        try:
            # Fix #11: Route through disk cache (get_cached_catalog + save_to_cache)
            # instead of calling fetch_online_catalog directly
            radius_arcmin = 5.0 / 60.0
            stars = get_cached_catalog(ra, dec, radius_arcmin, self.ref_catalog, verbose=False)
            if not stars:
                stars = fetch_online_catalog(ra, dec, radius_arcmin=radius_arcmin, catalog_name=self.ref_catalog, verbose=False)
                if stars:
                    save_to_cache(stars, ra, dec, radius_arcmin, self.ref_catalog, verbose=False)

            if stars:
                # Find the nearest match in the returned list
                cat_coords = SkyCoord(ra=[s['ra_deg'] for s in stars]*u.deg, dec=[s['dec_deg'] for s in stars]*u.deg)
                target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                idx_cat, d2d_cat, _ = target_coord.match_to_catalog_sky(cat_coords)
                star = stars[idx_cat]

                # Cross-check with AAVSO VSX for variability (already uses disk cache)
                vsx = get_vsx_stars(ra, dec, radius_arcmin=1.0, verbose=False)
                if vsx:
                    vsx_coords = SkyCoord(ra=[s['ra_deg'] for s in vsx]*u.deg, dec=[s['dec_deg'] for s in vsx]*u.deg)
                    idx_v, d2d_v, _ = target_coord.match_to_catalog_sky(vsx_coords)
                    nearest_vsx = vsx[idx_v]
                    
                    var_type = nearest_vsx.get('Type', '')
                    star['vsx_name'] = nearest_vsx.get('id', '')
                    star['var_type'] = var_type
                    
                    # Distance check for VSX match (5 arcsec tolerance)
                    dist_v = d2d_v[0].arcsec if hasattr(d2d_v, '__len__') else d2d_v.arcsec
                    if dist_v < 5.0 and var_type and 'CST' not in str(var_type).upper():
                        star['is_variable'] = True
                    else:
                        star['is_variable'] = False
                else:
                    star['is_variable'] = False
                    star['vsx_name'] = ''
                    star['var_type'] = ''

                # Resolve display name priority
                display_name = star.get('vsx_name', '')
                if not display_name:
                    display_name = star.get('cat_id', '')
                
                # Gaia fallback if primary catalog search yields no name
                if not display_name and "GAIA" not in self.ref_catalog.upper():
                    try:
                        g_stars = get_cached_catalog(ra, dec, 15.0/60.0, "GAIA_DR3", verbose=False)
                        if not g_stars:
                            g_stars = fetch_online_catalog(ra, dec, radius_arcmin=15.0/60.0, catalog_name="GAIA_DR3", verbose=False)
                            if g_stars:
                                save_to_cache(g_stars, ra, dec, 15.0/60.0, "GAIA_DR3", verbose=False)
                        if g_stars:
                            g_coords = SkyCoord(ra=[s['ra_deg'] for s in g_stars]*u.deg, dec=[s['dec_deg'] for s in g_stars]*u.deg)
                            idx_g, d2d_g, _ = target_coord.match_to_catalog_sky(g_coords)
                            dist_g = d2d_g[0].arcsec if hasattr(d2d_g, '__len__') else d2d_g.arcsec
                            if dist_g < 5.0:
                                display_name = g_stars[idx_g].get('cat_id', '')
                    except:
                        pass
                
                star['display_name'] = display_name if display_name else "Star"
                # Fix #4: Store result in memory cache
                self._catalog_cache[cache_key] = star
                return star
        except Exception as e:
            print(f"Online query failed: {e}")
        # Cache misses too, to avoid re-querying failed lookups
        self._catalog_cache[cache_key] = None
        return None

    def _add_variable_star(self):
        """Assign the target star role to the last clicked location."""
        x, y = self._last_click_pos
        
        # Enforce single variable star
        if self.variable_star:
            for m in self.variable_star['markers']: m.remove()
            self.variable_star = None

        star_data = self._mark_star(x, y, "Variable", "red")
        if star_data:
            self.variable_star = star_data
            self._update_var_panel()
            self._update_ref_panel()
            self.canvas.draw_idle()

    def _add_check_star(self):
        """Assign the check star role. Displaces existing check stars to reference role."""
        x, y = self._last_click_pos
        old_check = self.check_star
        
        star_data = self._mark_star(x, y, "Check", "blue")
        if star_data:
            self.check_star = star_data
            if old_check:
                # Demote old check star to reference list
                if np.sqrt((old_check['x'] - star_data['x'])**2 + (old_check['y'] - star_data['y'])**2) > 5.0:
                    old_check['role'] = 'Reference'
                    for m in old_check['markers']: m.set_color('lime')
                    self.on_the_spot_refs.append(old_check)
            
            self._update_ref_panel()
            self._update_var_panel()
            self.canvas.draw_idle()

    def _add_on_the_spot_ref(self):
        """Assign a reference star role to the last clicked location."""
        x, y = self._last_click_pos
        star_data = self._mark_star(x, y, "Reference", "lime")
        if star_data:
            self.on_the_spot_refs.append(star_data)
            self._update_ref_panel()
            self._update_var_panel()
            self.canvas.draw_idle()

    def _mark_star(self, x, y, role, color, force_fixed=False, cat_star=None):
        """
        Internal helper to fit, identify, and visually mark a star.
        
        Args:
            x, y (float): Coordinates.
            role (str): Role name (Variable, Check, Reference).
            color (str): Matplotlib color for the markers.
            
        Returns:
            dict: Star data dictionary, or None if fitting fails.
        """
        # 1. Precise fit
        res = self._fit_at(x, y)
        if not res:
            messagebox.showwarning("Fit Failed", "Could not fit a star at this position.")
            return None
            
        # 2. WCS Identification (Skip if cat_star is already provided)
        if cat_star is None:
            if not self.has_wcs:
                messagebox.showwarning("WCS Required", "WCS is needed to identify stars.")
                return None
                
            coord = self.wcs.pixel_to_world(res['x'], res['y'])
            cat_star = self._query_catalog(coord.ra.deg, coord.dec.deg)
            if not cat_star:
                messagebox.showwarning("No Catalog Match", "No matching catalog star found.")
                return None

        # 3. Prevent duplicate marking
        self._remove_marker_at(res['x'], res['y'], quiet=True)

        # 4. Determine Aperture Radii (Static or Flexible based on FWHM)
        if self.config.get('use_flexible_aperture') and res.get('fwhm') and not force_fixed:
            ap = res['fwhm'] * self.config.get('aperture_fwhm_factor', 2.0)
            ann_in = ap + self.config.get('annulus_inner_gap', 2.0)
            ann_out = ann_in + self.config.get('annulus_width', 5.0)
            # Update UI controls
            self.ap_var.set(round(ap, 2))
            self.ann_in_var.set(round(ann_in, 2))
            self.ann_out_var.set(round(ann_out, 2))
        else:
            ap = self.ap_var.get()
            ann_in = self.ann_in_var.get()
            ann_out = self.ann_out_var.get()

        # 5. Draw Aperture and Annulus Rings
        m1 = patches.Circle((res['x'], res['y']), radius=ap, color=color, fill=False, lw=1.5)
        m2 = patches.Circle((res['x'], res['y']), radius=ann_in, color=color, fill=False, lw=0.8, linestyle='--')
        m3 = patches.Circle((res['x'], res['y']), radius=ann_out, color=color, fill=False, lw=0.8, linestyle='--')
        self.ax.add_patch(m1)
        self.ax.add_patch(m2)
        self.ax.add_patch(m3)
        
        # 6. Aperture Photometry (Manual Implementation)
        # We perform background-subtracted aperture photometry to ensure side panels 
        # reflect the user's chosen aperture/annulus settings.
        try:
            ap_size = int(max(ann_out * 2.5, 30))
            if ap_size % 2 == 0: ap_size += 1
            ap_cutout = Cutout2D(self.data, (res['x'], res['y']), (ap_size, ap_size), mode='partial')
            ap_data = ap_cutout.data
            
            # Distance map from centroid
            c_pos = ap_cutout.to_cutout_position((res['x'], res['y']))
            yy_ap, xx_ap = np.mgrid[:ap_size, :ap_size]
            dist_sq = (xx_ap - c_pos[0])**2 + (yy_ap - c_pos[1])**2
            
            # Background from annulus (using sigma-clipped median for robustness, matching main app)
            ann_mask = (dist_sq >= ann_in**2) & (dist_sq <= ann_out**2)
            if np.any(ann_mask):
                _, bg_local, _ = sigma_clipped_stats(ap_data[ann_mask], sigma=3.0, maxiters=5)
            else:
                bg_local = np.nanmedian(ap_data)
            
            # Net flux in aperture
            ap_mask = dist_sq <= ap**2
            net_flux = np.sum(ap_data[ap_mask] - bg_local)
            
            if net_flux > 0:
                inst_mag = -2.5 * np.log10(net_flux)
            else:
                inst_mag = res['inst_mag'] # Fallback to PSF fit if flux is non-positive
        except:
            inst_mag = res['inst_mag'] # Fallback on error
            
        # 7. Store data for export
        c_ref = SkyCoord(ra=cat_star['ra_deg']*u.deg, dec=cat_star['dec_deg']*u.deg)
        return {
            'x': res['x'], 'y': res['y'],
            'role': role,
            'id': cat_star.get('display_name', 'Star'),
            'ra_deg': cat_star['ra_deg'], # Fix #12: Store degrees for precision export
            'dec_deg': cat_star['dec_deg'],
            'ra_hms': c_ref.ra.to_string(unit='hour', sep=':', precision=1, pad=True),
            'dec_dms': c_ref.dec.to_string(unit='deg', sep=':', precision=1, alwayssign=True, pad=True),
            'cat_mag': cat_star.get('V_mag') if 'B' not in self.header.get('FILTER', 'V').upper() else cat_star.get('B_mag'),
            'cat_bv': cat_star.get('B_V') or (cat_star.get('B_mag') - cat_star.get('V_mag') if cat_star.get('B_mag') is not None and cat_star.get('V_mag') is not None else 0.5),
            'inst_mag': inst_mag,
            'is_variable': cat_star.get('is_variable', False),
            'raw_cat_data': cat_star,
            'markers': [m1, m2, m3]
        }

    def _remove_marker_at_click(self):
        """Remove the marker near the last right-click position."""
        x, y = self._last_click_pos
        self._remove_marker_at(x, y)

    def _remove_marker_at(self, x, y, quiet=False):
        """
        Scan all marked stars and remove the one nearest to (x, y).
        
        Args:
            x, y (float): Coordinates.
            quiet (bool): If True, suppresses "Not Found" messages.
        """
        removed = False
        # Variable star check
        if self.variable_star:
            dist = np.sqrt((self.variable_star['x'] - x)**2 + (self.variable_star['y'] - y)**2)
            if dist < 10.0:
                for m in self.variable_star['markers']: m.remove()
                self.variable_star = None
                removed = True
        
        # Check star check
        if not removed and self.check_star:
            dist = np.sqrt((self.check_star['x'] - x)**2 + (self.check_star['y'] - y)**2)
            if dist < 10.0:
                for m in self.check_star['markers']: m.remove()
                self.check_star = None
                removed = True

        # Reference ensemble check
        if not removed:
            for i, r in enumerate(self.on_the_spot_refs):
                dist = np.sqrt((r['x'] - x)**2 + (r['y'] - y)**2)
                if dist < 10.0:
                    for m in r['markers']: m.remove()
                    self.on_the_spot_refs.pop(i)
                    removed = True
                    break
        
        if removed:
            self._update_var_panel()
            self._update_ref_panel()
            self.canvas.draw_idle()
        elif not quiet:
            messagebox.showinfo("Not Found", "No marked star found near this position.")

    def _update_var_panel(self):
        """Refresh the Variable Star info panel."""
        text = ""
        if self.variable_star:
            s = self.variable_star
            text += f"ID:  {s['id']}\n"
            text += f"RA:  {s['ra_hms']}\n"
            text += f"Dec: {s['dec_dms']}\n"
            cat_m = s['cat_mag']
            text += f"Mag: {cat_m:.3f} (Catalog)\n" if cat_m is not None and not np.isnan(cat_m) else "Mag: N/A (Catalog)\n"
            text += f"Inst:{s['inst_mag']:.3f}\n"
            text += "=" * 25 + "\n"
        else:
            text = "No variable star marked.\n"
        
        self.var_text.config(state=tk.NORMAL)
        self.var_text.delete(1.0, tk.END)
        self.var_text.insert(tk.END, text)
        self.var_text.config(state=tk.DISABLED)

    def _update_ref_panel(self):
        """Refresh the Reference and Check stars info panel and recalculate ZP."""
        text = ""
        # 1. Check Star Info
        if self.check_star:
            s = self.check_star
            text += f"--- CHECK STAR (Blue) ---\n"
            text += f"ID:  {s['id']}\n"
            text += f"RA:  {s['ra_hms']}\n"
            text += f"Dec: {s['dec_dms']}\n"
            chk_m = s['cat_mag']
            text += f"Mag: {chk_m:.3f}\n" if chk_m is not None and not np.isnan(chk_m) else "Mag: N/A\n"
            text += "-" * 25 + "\n\n"
        
        # 2. Reference Ensemble Calculation
        if self.on_the_spot_refs:
            text += f"--- REFERENCES (Green) ---\n"
            # Calculate Zero Point from ensemble (Fix #10: use sigma-clipped stats)
            zps = [r['cat_mag'] - r['inst_mag'] for r in self.on_the_spot_refs
                   if r['cat_mag'] is not None and not np.isnan(r['cat_mag'])]
            if len(zps) >= 3:
                _, median_zp, _ = sigma_clipped_stats(zps, sigma=3.0, maxiters=5)
                self.on_the_spot_zp = median_zp
            elif zps:
                self.on_the_spot_zp = np.median(zps)
            if self.on_the_spot_zp is not None:
                text += f"Ensemble ZP: {self.on_the_spot_zp:.3f}\n"
            
            for i, r in enumerate(self.on_the_spot_refs):
                var_tag = " [VAR!]" if r['is_variable'] else ""
                text += f"#{i+1:<2} {r['id']}{var_tag}\n"
                text += f"    RA:  {r['ra_hms']}\n"
                text += f"    Dec: {r['dec_dms']}\n"
                ref_m = r['cat_mag']
                text += f"    Mag: {ref_m:.3f}\n" if ref_m is not None and not np.isnan(ref_m) else "    Mag: N/A\n"
                text += "-" * 25 + "\n"
        
        if not self.check_star and not self.on_the_spot_refs:
            text = "No reference or check stars marked."

        self.ref_text.config(state=tk.NORMAL)
        self.ref_text.delete(1.0, tk.END)
        self.ref_text.insert(tk.END, text)
        self.ref_text.config(state=tk.DISABLED)
        # Fix #8: Removed canvas.draw_idle() as text updates don't require canvas repaint

    def _clear_all_markers(self):
        """Remove all star markers and reset internal state."""
        if self.variable_star:
            for m in self.variable_star['markers']: m.remove()
            self.variable_star = None
        if self.check_star:
            for m in self.check_star['markers']: m.remove()
            self.check_star = None
        for r in self.on_the_spot_refs:
            for m in r['markers']: m.remove()
        self.on_the_spot_refs = []
        self.on_the_spot_zp = None
        
        self._update_var_panel()
        self._update_ref_panel()
        self.canvas.draw_idle()

    def _fit_at(self, x, y):
        """
        Perform a 2D Gaussian fit on a small cutout centered at (x, y).
        
        Args:
            x, y (float): Initial guess coordinates.
            
        Returns:
            dict: Fit results (centroid, fwhm, flux, etc.) or None.
        """
        size = 25
        try:
            cutout = Cutout2D(self.data, (x, y), (size, size), mode='partial')
            d_fit = cutout.data
            bg = np.nanmedian(d_fit)
            d_fit_sub = d_fit - bg
            
            # Find peak for initial guess
            y_init, x_init = np.unravel_index(np.argmax(d_fit_sub), d_fit_sub.shape)
            g_init = models.Gaussian2D(amplitude=np.max(d_fit_sub), x_mean=x_init, y_mean=y_init, x_stddev=2.0, y_stddev=2.0, theta=0)
            fitter = fitting.LevMarLSQFitter()
            yy, xx = np.mgrid[:size, :size]
            g_fit = fitter(g_init, xx, yy, d_fit_sub)
            
            # Convert back to original image coordinates
            x_orig, y_orig = cutout.to_original_position((g_fit.x_mean.value, g_fit.y_mean.value))
            # Fix #2 + #9: Use geometric mean of both stddevs for elliptical PSFs
            sigma_x = abs(g_fit.x_stddev.value)
            sigma_y = abs(g_fit.y_stddev.value)
            fwhm = 2.355 * np.sqrt(sigma_x * sigma_y)
            # Flux = Amplitude × 2π × σ_x × σ_y  (proper elliptical Gaussian integral)
            total_flux = g_fit.amplitude.value * 2 * np.pi * sigma_x * sigma_y
            if total_flux <= 0: return None
            
            return {
                'x': x_orig, 'y': y_orig, 'fwhm': fwhm, 
                'inst_mag': -2.5 * np.log10(total_flux),
                'data_sub': d_fit_sub,
                'g_fit': g_fit,
                'x_grid': xx, 'y_grid': yy,
                'fit_xc': g_fit.x_mean.value,
                'fit_yc': g_fit.y_mean.value
            }
        except:
            return None

    def _update_profile_plot(self, res):
        """
        Update the radial profile plot with the latest star fit.
        
        Args:
            res (dict): Fit result dictionary from _fit_at.
        """
        self.profile_ax.clear()
        
        data = res['data_sub']
        xx, yy = res['x_grid'], res['y_grid']
        xc, yc = res['fit_xc'], res['fit_yc']
        g_fit = res['g_fit']
        
        # Calculate radial distances from fitted centroid
        distances = np.sqrt((xx - xc)**2 + (yy - yc)**2)
        
        # Determine plot limits based on aperture settings
        ap = self.ap_var.get()
        ann_in = self.ann_in_var.get()
        rad_limit = max(ann_in, ap * 1.5, 10)
        
        # Scatter actual pixel values
        mask = distances <= rad_limit
        self.profile_ax.scatter(distances[mask], data[mask], color='royalblue', alpha=0.5, s=10)
        
        # Plot the fitted Gaussian curve
        r_fine = np.linspace(0, rad_limit, 100)
        self.profile_ax.plot(r_fine, g_fit(xc + r_fine, yc), color='darkorange', lw=2)
        
        # Draw vertical lines for aperture and annulus
        self.profile_ax.axvline(x=ap, color='red', ls='--', lw=1.5, label='Ap')
        self.profile_ax.axvline(x=ann_in, color='green', ls=':', lw=1, label='Ann')
        
        self.profile_ax.set_title(f"FWHM: {res['fwhm']:.2f}px", fontsize=9)
        self.profile_ax.set_xlim(0, rad_limit)
        self.profile_ax.grid(True, alpha=0.2)
        
        self.profile_canvas.draw_idle()

    def _show_status_message(self, msg):
        """Temporarily append a message to the status bar."""
        current = self.status_var.get()
        # Ensure we don't double-append
        if " [ " in current:
            current = current.split(" [ ")[0]
        self.status_var.set(f"{current}   [ {msg} ]")
        # Clear the message after 3 seconds
        self.parent.after(3000, lambda: self.status_var.set(current))

    def _on_export_ap_click(self, quiet=False, refresh=True):
        """Export current aperture and annulus settings back to the main app."""
        if not self.aperture_export_callback: return
        data = {
            'aperture': self.ap_var.get(),
            'annulus_in': self.ann_in_var.get(),
            'annulus_out': self.ann_out_var.get()
        }
        self.aperture_export_callback(data)
        
        # Update all markers in the viewer to use these new apertures
        if refresh:
            self._refresh_all_stars(force_fixed=True)
        
        if not quiet:
            self._show_status_message("Apertures Exported & Updated")

    def _refresh_all_stars(self, force_fixed=True):
        """Re-calculate photometry and re-draw markers for all stars."""
        # 1. Variable Star
        if self.variable_star:
            s = self.variable_star
            new_s = self._mark_star(s['x'], s['y'], s['role'], "red", force_fixed=force_fixed, cat_star=s.get('raw_cat_data'))
            if new_s: self.variable_star = new_s
            
        # 2. Check Star
        if self.check_star:
            s = self.check_star
            new_s = self._mark_star(s['x'], s['y'], s['role'], "blue", force_fixed=force_fixed, cat_star=s.get('raw_cat_data'))
            if new_s: self.check_star = new_s
            
        # 3. Reference Ensemble
        updated_refs = []
        for s in list(self.on_the_spot_refs):
            new_s = self._mark_star(s['x'], s['y'], s['role'], "lime", force_fixed=force_fixed, cat_star=s.get('raw_cat_data'))
            if new_s:
                updated_refs.append(new_s)
        self.on_the_spot_refs = updated_refs
        
        self._update_var_panel()
        self._update_ref_panel()
        self.canvas.draw_idle()

    def _auto_mark_initial_stars(self):
        """Automatically mark stars provided during initialization (e.g. from session load)."""
        if not self.has_wcs: return
        
        # 1. Target Variable
        v = self.initial_stars.get('variable')
        if v and v.get('ra') is not None:
            self._mark_by_coord(v['ra'], v['dec'], "Variable")
            
        # 2. Check Star
        c = self.initial_stars.get('check')
        if c and c.get('ra') is not None:
            self._mark_by_coord(c['ra'], c['dec'], "Check")
            
        # 3. Reference Ensemble
        for r in self.initial_stars.get('refs', []):
            if r.get('ra') is not None:
                self._mark_by_coord(r['ra'], r['dec'], "Reference")
        
        self._update_var_panel()
        self._update_ref_panel()
        self.canvas.draw_idle()

    def _mark_by_coord(self, ra, dec, role):
        """
        Mark a star based on world coordinates (RA/Dec).
        
        Args:
            ra, dec (float): Coordinates in degrees.
            role (str): Role to assign.
        """
        try:
            x, y = self.wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg'))
            # Bounds check
            if x < 0 or x >= self.data.shape[1] or y < 0 or y >= self.data.shape[0]:
                return
            
            # Simulate a click position for role handlers
            self._last_click_pos = (x, y)
            if role == "Variable": self._add_variable_star()
            elif role == "Check": self._add_check_star()
            elif role == "Reference": self._add_on_the_spot_ref()
        except Exception as e:
            print(f"Auto-mark failed for {role} at {ra}, {dec}: {e}")

    def _on_export_click(self, quiet=False):
        """Export all currently marked star coordinates and names to the main app."""
        if not self.export_callback: return
        
        data = {
            'variable': None,
            'check': None,
            'refs': []
        }
        
        if self.variable_star and self.has_wcs:
            # Fix #12: Use stored degrees directly for export to maintain precision
            ra = self.variable_star.get('ra_deg')
            dec = self.variable_star.get('dec_deg')
            if ra is None or dec is None: # Fallback if degrees missing
                coord = self.wcs.pixel_to_world(self.variable_star['x'], self.variable_star['y'])
                ra, dec = coord.ra.deg, coord.dec.deg
            data['variable'] = {
                'ra': ra, 'dec': dec, 
                'name': self.variable_star['id'],
                'mag': self.variable_star.get('cat_mag'),
                'bv': self.variable_star.get('cat_bv')
            }
            
        if self.check_star and self.has_wcs:
            ra = self.check_star.get('ra_deg')
            dec = self.check_star.get('dec_deg')
            if ra is None or dec is None:
                coord = self.wcs.pixel_to_world(self.check_star['x'], self.check_star['y'])
                ra, dec = coord.ra.deg, coord.dec.deg
            data['check'] = {
                'ra': ra, 'dec': dec, 
                'name': self.check_star['id'],
                'mag': self.check_star.get('cat_mag'),
                'bv': self.check_star.get('cat_bv')
            }
            
        for r in self.on_the_spot_refs:
            if self.has_wcs:
                ra = r.get('ra_deg')
                dec = r.get('dec_deg')
                if ra is None or dec is None:
                    coord = self.wcs.pixel_to_world(r['x'], r['y'])
                    ra, dec = coord.ra.deg, coord.dec.deg
                data['refs'].append({
                    'ra': ra, 'dec': dec, 
                    'name': r['id'],
                    'mag': r.get('cat_mag'),
                    'bv': r.get('cat_bv')
                })
            
        self.export_callback(data)
        if not quiet:
            self._show_status_message("Stars Exported")

    def on_scroll(self, event):
        """Implement zoom functionality centered on the mouse position."""
        if event.inaxes != self.ax: return
        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        if event.button == 'up':
            scale_factor = 1 / base_scale
        else:
            scale_factor = base_scale
            
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        # Calculate relative position of mouse to maintain focus
        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        
        self.ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
        self.ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])
        self.canvas.draw_idle()
