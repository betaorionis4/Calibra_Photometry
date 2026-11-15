# NOTE (v4.15): this module is not used by Calibra (superseded by photometry/fits_viewer.py)
# and is kept only for reference. It is known to be broken (fitter called without data).
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval


class FITSInspector:
    def __init__(self, fits_path):
        # 1. Load Data and WCS
        with fits.open(fits_path) as hdul:
            self.data = hdul[0].data.astype(float)
            self.header = hdul[0].header
            self.wcs = WCS(self.header)

        # 2. Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': self.wcs})
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(self.data)

        self.im = self.ax.imshow(self.data, origin='lower', cmap='Greys_r', vmin=vmin, vmax=vmax)
        self.ax.set_title(f"Inspecting: {fits_path}\nClick to Fit Star | Scroll to Zoom")

        # UI Elements
        self.coord_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes, color='cyan', fontsize=10,
                                       fontweight='bold')
        self.marker = None

        # 3. Connect Events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        plt.show()

    def on_hover(self, event):
        """Update coordinates in the corner on mouse hover."""
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            # Convert pixels to World Coordinates (RA/Dec)
            coord = self.wcs.pixel_to_world(x, y)
            ra_str = coord.ra.to_string(unit='hour', sep=':', precision=2)
            dec_str = coord.dec.to_string(unit='deg', sep=':', precision=2)

            self.coord_text.set_text(f"X: {x:.1f} Y: {y:.1f} | RA: {ra_str} Dec: {dec_str}")
            self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Fit a Gaussian PSF to the star under the mouse click."""
        if event.inaxes != self.ax or event.button != 1:
            return

        x_click, y_click = event.xdata, event.ydata
        size = 21  # Box size for fitting

        try:
            # Extract a cutout around the click
            cutout = Cutout2D(self.data, (x_click, y_click), (size, size), mode='partial')
            d_fit = cutout.data

            # Local background estimation (simple median)
            bg = np.nanmedian(d_fit)
            d_fit -= bg

            # Initial guess for Gaussian model
            # x_mean/y_mean are relative to the cutout (center is size//2)
            y_init, x_init = np.unravel_index(np.argmax(d_fit), d_fit.shape)
            g_init = models.Gaussian2D(amplitude=np.max(d_fit), x_mean=x_init, y_mean=y_init)

            # Fit the model
            fitter = fitting.LevMarLSQFitter()
            yy, xx = np.mgrid[:size, :size]
            g_fit = fitter(g_init, xx, yy)

            # Map fit coordinates back to full image pixels
            x_final = cutout.to_original_position((g_fit.x_mean.value, g_fit.y_mean.value))[0]
            y_final = cutout.to_original_position((g_fit.x_mean.value, g_fit.y_mean.value))[1]
            fwhm = g_fit.x_stddev.value * 2.355

            # Clear old markers and draw new one
            if self.marker: self.marker.remove()
            self.marker, = self.ax.plot(x_final, y_final, 'r+', markersize=15)

            print(f"\n--- Star Fitted at ({x_final:.2f}, {y_final:.2f}) ---")
            print(f"FWHM: {fwhm:.2f} px")
            print(f"Peak Flux: {g_fit.amplitude.value:.2f}")
            print(f"Total Flux (approx): {g_fit.amplitude.value * 2 * np.pi * g_fit.x_stddev.value ** 2:.2f}")

        except Exception as e:
            print(f"Fitting failed: {e}")

    def on_scroll(self, event):
        """Handle zooming with the scroll wheel."""
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

        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
        self.ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])
        self.fig.canvas.draw_idle()

# Run it
# inspector = FITSInspector('your_solved_file.fits')
