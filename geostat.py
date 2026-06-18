"""
Geostatistical Aperture Field Generator
========================================
Generates realistic fracture aperture fields with:
  - Variogram-based spatial correlation (Gaussian, exponential, spherical, Matérn)
  - Directional anisotropy (arbitrary principal axis angle)
  - Embedded high-flow channels (sinusoidal, straight, or random-walk)
  - Multi-scale superposition (nested variograms)
  - Log-normal or normal marginal distribution
  - Sequential Gaussian Simulation (SGS) stub for conditioning to data points

All generation is done via spectral/FFT methods (fast even for large grids)
with an optional turning-bands fallback for non-separable covariances.

Quick start
-----------
    from aperture_geostat import ApertureGenerator, ChannelOverlay, plot_aperture

    gen = ApertureGenerator(
        nx=200, ny=100, Lx=0.2, Ly=0.1,
        b_mean=1e-4, b_sigma=0.5,           # log-normal params
        variogram='gaussian',
        range_x=0.06, range_y=0.02,         # anisotropy: longer in x
        angle_deg=30,                        # rotate principal axis 30° CCW
    )
    b = gen.generate(seed=42)

    # Add two high-flow channels
    overlay = ChannelOverlay(nx=200, ny=100, Lx=0.2, Ly=0.1)
    overlay.add_sinusoidal(amplitude=0.015, wavelength=0.18,
                            width=0.008, b_factor=3.0, y_centre=0.03)
    overlay.add_random_walk(width=0.006, b_factor=2.5, y_centre=0.07, seed=7)
    b = overlay.apply(b)

    plot_aperture(b, Lx=0.2, Ly=0.1)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ──────────────────────────────────────────────────────────────────────────────
# Covariance / variogram models
# ──────────────────────────────────────────────────────────────────────────────

def covariance_model(h, model='gaussian', sill=1.0, rang=1.0, nugget=0.0,
                     kappa=1.5):
    """
    Isotropic covariance C(h) for reduced lag h = distance / range.

    Parameters
    ----------
    h      : array of reduced lags (dimensionless)
    model  : 'gaussian' | 'exponential' | 'spherical' | 'matern'
    sill   : variance contribution of this structure
    rang   : range (already folded into h = dist/range before calling)
    nugget : nugget effect added at h > 0
    kappa  : smoothness for Matern (0.5 = exponential, 1.5, 2.5, ∞ = Gaussian)
    """
    h = np.asarray(h, dtype=float)
    if model == 'gaussian':
        C = sill * np.exp(-np.pi * h**2)
    elif model == 'exponential':
        C = sill * np.exp(-3.0 * h)
    elif model == 'spherical':
        C = np.where(h < 1.0,
                     sill * (1.0 - 1.5 * h + 0.5 * h**3),
                     0.0)
    elif model == 'matern':
        from scipy.special import kv, gamma
        h = np.where(h == 0, 1e-15, h)
        factor = (2**(1 - kappa)) / gamma(kappa)
        x = np.sqrt(2 * kappa) * 3.0 * h
        C = sill * factor * (x**kappa) * kv(kappa, x)
    else:
        raise ValueError(f"Unknown variogram model '{model}'")

    # nugget: full sill at h=0, nugget elsewhere
    C = np.where(h == 0, sill + nugget, C + nugget * 0.0)
    return C

# ──────────────────────────────────────────────────────────────────────────────
# Spectral (FFT-based) unconditional simulation
# ──────────────────────────────────────────────────────────────────────────────

class ApertureGenerator:
    """
    Generate a spatially correlated, anisotropic aperture field via
    spectral simulation (Fourier-space colouring of white noise).

    The simulation proceeds in four steps:
      1. Build a 2-D anisotropic covariance field C(x,y) on the grid.
      2. Take its FFT to get the power spectrum S(kx,ky).
      3. Multiply the FFT of white noise by sqrt(S) and back-transform.
      4. Standardise to N(0,1) then transform to log-normal.

    Multiple nested structures (different ranges / models) are additive
    in the spectral domain.

    Parameters
    ----------
    nx, ny        : number of grid cells
    Lx, Ly        : domain size [m]
    b_mean        : target mean aperture [m]
    b_sigma       : log-normal standard deviation of ln(b)
                    (≈ coefficient of variation for small values)
    variogram     : base variogram model for the first structure
    range_x       : correlation range along the principal axis [m]
    range_y       : correlation range perpendicular to principal axis [m]
    angle_deg     : CCW rotation of the principal axis from x-axis [°]
    nugget_frac   : nugget as fraction of total variance (0–1)
    nested        : list of dicts, each with keys
                    {model, range_x, range_y, angle_deg, weight}
                    for additional nested structures (weights are relative)
    distribution  : 'lognormal' | 'normal'
    """

    def __init__(self, nx=200, ny=100, Lx=0.2, Ly=0.1,
                 b_mean=1e-4, b_sigma=0.5,
                 variogram='gaussian',
                 range_x=0.05, range_y=0.02,
                 angle_deg=0.0,
                 nugget_frac=0.05,
                 kappa=1.5,
                 nested=None,
                 distribution='lognormal'):

        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / nx, Ly / ny
        self.b_mean = b_mean
        self.b_sigma = b_sigma
        self.variogram = variogram
        self.range_x = range_x
        self.range_y = range_y
        self.angle_deg = angle_deg
        self.nugget_frac = nugget_frac
        self.kappa = kappa
        self.nested = nested or []
        self.distribution = distribution

    def _anisotropic_h(self, range_x, range_y, angle_deg):
        """
        Compute reduced anisotropic lag field h(i,j) on the grid.

        Uses a rotation matrix to align the covariance ellipse with
        the specified principal direction.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        # Cell-centre coordinates (periodic: use fftfreq convention)
        xi = np.fft.fftfreq(nx, d=1.0 / nx) * dx   # shape (nx,)
        yi = np.fft.fftfreq(ny, d=1.0 / ny) * dy   # shape (ny,)
        X, Y = np.meshgrid(xi, yi, indexing='ij')   # (nx, ny)

        # Rotate coordinates into principal-axis frame
        theta = np.radians(angle_deg)
        Xr =  X * np.cos(theta) + Y * np.sin(theta)
        Yr = -X * np.sin(theta) + Y * np.cos(theta)

        # Reduced lags
        h = np.sqrt((Xr / range_x)**2 + (Yr / range_y)**2)
        return h

    def _spectral_covariance(self, model, range_x, range_y, angle_deg,
                              nugget_frac, weight, kappa):
        """Return the covariance field C(x,y) for one structure."""
        h = self._anisotropic_h(range_x, range_y, angle_deg)
        sill = (1.0 - nugget_frac) * weight
        nug  = nugget_frac * weight
        C = covariance_model(h, model=model, sill=sill, nugget=0.0,
                             kappa=kappa)
        # Nugget: add nug only at h=0 (cell 0,0 in fftfreq convention)
        C[0, 0] += nug
        return C

    def generate(self, seed=None):
        """
        Simulate one realisation of the aperture field.
        
        Returns
        -------
        b : (nx, ny) ndarray of apertures [m]
        """
        rng = np.random.default_rng(seed)

        # ── 1. Build total covariance in spatial domain ──────────────────
        # Primary structure has weight=1; nested structures share proportionally
        total_weight = 1.0 + sum(s.get('weight', 1.0) for s in self.nested)

        C_total = self._spectral_covariance(
            model=self.variogram,
            range_x=self.range_x, range_y=self.range_y,
            angle_deg=self.angle_deg,
            nugget_frac=self.nugget_frac,
            weight=1.0 / total_weight,
            kappa=self.kappa,
        )
        for struct in self.nested:
            C_total += self._spectral_covariance(
                model=struct.get('model', 'gaussian'),
                range_x=struct.get('range_x', self.range_x),
                range_y=struct.get('range_y', self.range_y),
                angle_deg=struct.get('angle_deg', self.angle_deg),
                nugget_frac=struct.get('nugget_frac', self.nugget_frac),
                weight=struct.get('weight', 1.0) / total_weight,
                kappa=struct.get('kappa', self.kappa),
            )

        # ── 2. Power spectrum S = FFT(C) ─────────────────────────────────
        S = np.fft.fft2(C_total).real
        S = np.abs(S)                    # ensure non-negative (numerical noise)
        S_sqrt = np.sqrt(S)

        # ── 3. Colour white noise ─────────────────────────────────────────
        Z = rng.standard_normal((self.nx, self.ny))
        Z_hat = np.fft.fft2(Z)
        Y_hat = Z_hat * S_sqrt
        Y = np.fft.ifft2(Y_hat).real

        # ── 4. Standardise then transform ─────────────────────────────────
        Y = (Y - Y.mean()) / (Y.std() + 1e-15)

        if self.distribution == 'lognormal':
            # b = b_mean * exp(sigma*Y - sigma²/2)  → E[b] = b_mean
            b = self.b_mean * np.exp(
                self.b_sigma * Y - 0.5 * self.b_sigma**2
            )
        else:
            # Normal: clip to positive values
            b_std = self.b_mean * self.b_sigma
            b = self.b_mean + b_std * Y
            b = b.clip(self.b_mean * 0.01)

        return b

# ──────────────────────────────────────────────────────────────────────────────
# Channel overlays
# ──────────────────────────────────────────────────────────────────────────────

class ChannelOverlay:
    """
    Add deterministic or stochastic high-flow channels to an existing
    aperture field.

    Each channel is defined as a smooth mask that multiplies or adds
    a high-aperture zone on top of the background field.

    Parameters
    ----------
    nx, ny   : grid dimensions
    Lx, Ly   : domain size [m]
    """

    def __init__(self, nx, ny, Lx, Ly):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self._masks = []   # list of (mask, b_factor, mode)

    # ── individual channel builders ────────────────────────────────────

    def add_sinusoidal(self, amplitude=0.01, wavelength=None,
                        width=0.008, b_factor=3.0,
                        y_centre=None, mode='multiply'):
        """
        Channel whose centreline follows a sinusoid:
            y_c(x) = y_centre + amplitude * sin(2π x / wavelength)
        """
        if wavelength is None:
            wavelength = self.Lx
        if y_centre is None:
            y_centre = self.Ly / 2

        xi = (np.arange(self.nx) + 0.5) * self.dx
        yi = (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(xi, yi, indexing='ij')

        y_c = y_centre + amplitude * np.sin(2 * np.pi * X / wavelength)
        dist = np.abs(Y - y_c)
        mask = np.exp(-0.5 * (dist / (width / 2.355))**2)   # Gaussian cross-section
        self._masks.append((mask, b_factor, mode))

    def add_straight(self, y_centre=None, angle_deg=0.0, width=0.006,
                      b_factor=2.5, mode='multiply'):
        """
        Straight channel at angle_deg from the x-axis.
        """
        if y_centre is None:
            y_centre = self.Ly / 2

        xi = (np.arange(self.nx) + 0.5) * self.dx
        yi = (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(xi, yi, indexing='ij')

        theta = np.radians(angle_deg)
        # Signed perpendicular distance from the channel centreline
        xc = self.Lx / 2
        dist = (-(X - xc) * np.sin(theta) + (Y - y_centre) * np.cos(theta))
        mask = np.exp(-0.5 * (dist / (width / 2.355))**2)
        self._masks.append((mask, b_factor, mode))

    def add_random_walk(self, width=0.006, b_factor=2.5,
                         y_centre=None, roughness=0.3,
                         n_steps=None, seed=None, mode='multiply'):
        """
        Channel whose centreline follows a smoothed random walk in y.

        Parameters
        ----------
        roughness : std of each random step as fraction of Ly
        n_steps   : number of independent steps (default = nx // 5)
        """
        if y_centre is None:
            y_centre = self.Ly / 2
        if n_steps is None:
            n_steps = max(4, self.nx // 5)

        rng = np.random.default_rng(seed)
        # Coarse random walk then interpolate
        steps = rng.normal(0, roughness * self.Ly, n_steps)
        yc_coarse = np.cumsum(steps)
        # Centre around y_centre
        yc_coarse -= yc_coarse.mean() - y_centre
        # Clip to domain
        yc_coarse = yc_coarse.clip(width * 2, self.Ly - width * 2)
        # Interpolate to full nx resolution and smooth
        xi_coarse = np.linspace(0, self.nx - 1, n_steps)
        xi_full   = np.arange(self.nx)
        yc_full   = np.interp(xi_full, xi_coarse, yc_coarse)
        smooth_sigma = self.nx * 0.04
        yc_full = gaussian_filter(yc_full, sigma=smooth_sigma)

        xi = (np.arange(self.nx) + 0.5) * self.dx
        yi = (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(xi, yi, indexing='ij')

        # Map yc_full (in grid indices) to metres
        yc_m = yc_full   # already in metres after the clip/interp
        # But we built yc_coarse in metres; yc_full is also in metres
        y_c = yc_full.reshape(-1, 1)   # broadcast over ny
        dist = np.abs(Y - y_c)
        mask = np.exp(-0.5 * (dist / (width / 2.355))**2)
        self._masks.append((mask, b_factor, mode))

    def add_branching(self, trunk_y=None, branch_angle=25.0, width=0.005,
                       b_factor=2.0, mode='multiply', seed=None):
        """
        A trunk channel with two branches splitting off it.
        """
        if trunk_y is None:
            trunk_y = self.Ly / 2

        # Trunk (straight horizontal)
        xi = (np.arange(self.nx) + 0.5) * self.dx
        yi = (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(xi, yi, indexing='ij')

        mask = np.zeros((self.nx, self.ny))
        # Trunk: first half of domain
        split_x = self.Lx * 0.5
        trunk_mask = np.exp(-0.5 * ((Y - trunk_y) / (width / 2.355))**2)
        mask += np.where(X < split_x, trunk_mask, 0.0)

        # Two branches in the second half
        theta = np.radians(branch_angle)
        for sign in (-1, +1):
            # Branch centreline: y = trunk_y ± tan(theta) * (x - split_x)
            y_branch = trunk_y + sign * np.tan(theta) * (X - split_x)
            dist = np.abs(Y - y_branch)
            b_mask = np.exp(-0.5 * (dist / (width / 2.355))**2)
            mask += np.where(X >= split_x, b_mask, 0.0)

        mask = mask.clip(0, 1)
        self._masks.append((mask, b_factor, mode))

    # ── apply all channels ─────────────────────────────────────────────

    def apply(self, b):
        """
        Apply all registered channel overlays to aperture field b.

        mode='multiply' : b_new = b * (1 + (b_factor-1)*mask)
        mode='replace'  : b_new = b*(1-mask) + b_mean*b_factor*mask
        """
        b_out = b.copy()
        for mask, b_factor, mode in self._masks:
            if mode == 'multiply':
                b_out = b_out * (1.0 + (b_factor - 1.0) * mask)
            elif mode == 'replace':
                b_mean = b.mean()
                b_out = b_out * (1.0 - mask) + b_mean * b_factor * mask
        return b_out

# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_aperture(b, Lx, Ly, title='Aperture field', variogram=True,
                  save_path=None):
    """
    Plot aperture field with histogram and optional variograms.
    """
    nx, ny = b.shape
    dx, dy = Lx / nx, Ly / ny
    xc = (np.arange(nx) + 0.5) * dx * 1e3
    yc = (np.arange(ny) + 0.5) * dy * 1e3
    X, Y = np.meshgrid(xc, yc, indexing='ij')

    ncols = 1 
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── aperture map ──────────────────────────────────────────────────
    ax = axes
    im = ax.pcolormesh(X, Y, b * 1e6, cmap='viridis', shading='auto',
                       norm=LogNorm(vmin=b.min() * 1e6, vmax=b.max() * 1e6))
    plt.colorbar(im, ax=ax, label='Aperture b [μm]',fraction=0.023, pad=0.04)
    ax.set_title('Aperture (log colour)')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()
    return fig
