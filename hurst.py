import numpy as np
from numpy import pi, tan
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def generate_field(nx, ny, dx, dy, b_mean, b_std, L_c, H=0.8, angle=pi/4):
    # 1. Generate frequency grids scaled to physical dimensions (radians/unit)
    f1 = np.fft.fftfreq(nx, d=dx) * 2 * pi
    f2 = np.fft.fftfreq(ny, d=dy) * 2 * pi
    kx1, kx2 = np.meshgrid(f1, f2, indexing='ij')
    
    # 2. Set up anisotropic correlation limits
    kcx = 2 * pi / L_c
    kcy = 2 * pi / (L_c * tan(angle))
    
    # 3. Compute magnitude matrix
    k = np.sqrt(kx1**2 + kx2**2)
    k[0, 0] = 1e-12  # Prevent division by zero at DC component
    
    # 4. Apply elliptical cutoff adjustment (Fixed syntax here)
    k_ell = np.sqrt((kx1 / kcx)**2 + (kx2 / kcy)**2)
    k_c = np.sqrt(kcx**2 + kcy**2)
    k[k_ell < 1.0] = k_c
    
    # 5. Generate random field and filter it (Fixed ^ to ** and added np.real)
    # Using normal distribution (randn) matches mean/std normalization best
    random_field = np.random.randn(nx, ny) 
    fft_field = np.fft.fft2(random_field)
    
    # Apply spectral filter
    filtered_fft = fft_field * (k**(-(1 + H)))
    
    # Transform back to spatial domain and discard tiny imaginary residuals
    z = np.real(np.fft.ifft2(filtered_fft))
    
    # 6. Normalize and scale to target mean and standard deviation
    z = (z - np.mean(z)) / np.std(z)
    phys_std = b_std * b_mean
    w = z * phys_std + b_mean
    
    # 7. Apply lower threshold limit (closure)
    closure = 1e-10
    w[w < closure] = closure
    
    return w


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

if __name__ == '__main__':
    Lx, Ly  = 0.80, 0.80
    nx, ny  = 120, 60
    dx, dy  = Lx/nx, Ly/ny
    b_mean = 1.0e-4
    b_std=1
    w=generate_field(nx,ny,dx,dy,b_mean,b_std,L_c=10)
    plot_aperture(w,Lx,Ly)
