"""
2D Velocity Field in a Variable-Aperture Fracture — Finite Volume Method
=========================================================================
Governing equation (Reynolds lubrication / local cubic law):

    ∇ · (b³/(12μ) ∇p) = 0

Velocity recovery
------------------------------------------
Interior x-face velocity between cell (i-1,j) and (i,j):

    u_face[i,j] = -T_{i-½,j} · (p[i,j] − p[i-1,j]) / dx
    T_{i-½,j}   = harmonic_mean(b[i-1,j]³/(12μ), b[i,j]³/(12μ))

This is identical to the coefficient used in row i of the assembled
matrix, so ∇·u = 0 at every interior cell to solver precision.

Boundary face velocities (i=0 and i=nx) cannot be recovered from a
pressure difference because the penalty BC has fixed p[0]=p_in exactly,
making Δp=0 at the boundary face.  Instead they are recovered from the
local divergence-free condition on each boundary cell:

    (u_face[1,j] − u_face[0,j])/dx + (v_face[0,j+1] − v_face[0,j])/dy = 0
    ⟹  u_face[0,j] = u_face[1,j] + dx·(v_face[0,j+1] − v_face[0,j])/dy

and symmetrically at the outlet. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from geostat import ApertureGenerator, ChannelOverlay
from hurst import generate_field
# ---------------------------------------------------------------------------
# Finite Volume solver
# ---------------------------------------------------------------------------

def harmonic_mean(a, b):
    """Element-wise harmonic mean; returns 0 where a+b == 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((a + b) > 0, 2.0 * a * b / (a + b), 0.0)

def build_system(b, dx, dy, mu, p_in, p_out):

    nx, ny = b.shape
    N = nx * ny
    T = b**3 / (12.0 * mu)

    A = lil_matrix((N, N))
    rhs = np.zeros(N)

    def idx(i, j): return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            
            # --- X DIRECTION (Inlet / Outlet / Interior) ---
            if i == 0:
                # Left Boundary Face (Inlet): Distance is dx/2 -> factor of 2
                c_inlet = 2.0 * T[i, j] / dx**2
                A[k, k] += c_inlet
                rhs[k]  += c_inlet * p_in
            else:
                c = harmonic_mean(T[i, j], T[i-1, j]) / dx**2
                A[k, k] += c
                A[k, idx(i-1, j)] -= c

            if i == nx - 1:
                # Right Boundary Face (Outlet): Distance is dx/2 -> factor of 2
                c_outlet = 2.0 * T[i, j] / dx**2
                A[k, k] += c_outlet
                rhs[k]  += c_outlet * p_out
            else:
                c = harmonic_mean(T[i, j], T[i+1, j]) / dx**2
                A[k, k] += c
                A[k, idx(i+1, j)] -= c

            # --- Y DIRECTION (Periodic BCs) ---
            
            # South Neighbor (j - 1)
            if j > 0:
                c_south = harmonic_mean(T[i, j], T[i, j-1]) / dy**2
                A[k, k] += c_south
                A[k, idx(i, j-1)] -= c_south
            else:
                # Periodic wrap-around: South neighbor is at the top row (ny - 1)
                c_south_periodic = harmonic_mean(T[i, 0], T[i, ny-1]) / dy**2
                A[k, k] += c_south_periodic
                A[k, idx(i, ny-1)] -= c_south_periodic

            # North Neighbor (j + 1)
            if j < ny - 1:
                c_north = harmonic_mean(T[i, j], T[i, j+1]) / dy**2
                A[k, k] += c_north
                A[k, idx(i, j+1)] -= c_north
            else:
                # Periodic wrap-around: North neighbor is at the bottom row (0)
                c_north_periodic = harmonic_mean(T[i, ny-1], T[i, 0]) / dy**2
                A[k, k] += c_north_periodic
                A[k, idx(i, 0)] -= c_north_periodic

    return A, rhs, T
def recover_face_fluxes(p, T, dx, dy):
    """
    Recover conservative Darcy velocities at all cell faces.

    Interior x-faces  (i = 1 … nx-1)
    ----------------------------------
    Uses the harmonic-mean conductivity between the two sharing cells and
    a single-cell pressure difference — exactly the coefficient in the
    assembled matrix row, so ∇·u = 0 at interior cells by construction.

    Boundary x-faces  (i = 0, i = nx)
    -----------------------------------
    The penalty BC forces p[0,:] → p_in, making the pressure difference
    across the inlet face identically zero.  These fluxes are therefore
    recovered from the local divergence-free condition on the boundary
    cell:

        (u[1,j] − u[0,j])/dx + (v[0,j+1] − v[0,j])/dy = 0
        ⟹  u[0,j] = u[1,j] + dx·(v[0,j+1] − v[0,j])/dy

    and symmetrically at the outlet.

    Parameters
    ----------
    p    : (nx, ny)  solved pressure [Pa]
    T    : (nx, ny)  transmissivity b³/(12μ) [m³ Pa⁻¹ s⁻¹]
    dx,dy: cell sizes [m]
    p_in, p_out : boundary pressures (kept for reference; not used directly)

    Returns
    -------
    u_face : (nx+1, ny)  x-face volumetric flux [m²/s]
    v_face : (nx, ny+1)  y-face volumetric flux [m²/s]
    """
    nx, ny = p.shape

    # ── y-faces first (needed for boundary x-face extrapolation) ────────
    v_face = np.zeros((nx, ny + 1))
    v_face[:, 1:-1] = (-harmonic_mean(T[:, :-1], T[:, 1:])
                       * (p[:, 1:] - p[:, :-1]) / dy)
    # j=0 and j=ny walls remain 0 (no-flow natural BC)

    # ── interior x-faces ─────────────────────────────────────────────────
    u_face = np.zeros((nx + 1, ny))
    u_face[1:-1, :] = (-harmonic_mean(T[:-1, :], T[1:, :])
                       * (p[1:, :] - p[:-1, :]) / dx)

    # ── boundary x-faces: local divergence-free extrapolation ────────────
    # Inlet cell (i=0):  u[0,j] = u[1,j] + dx*(v[0,j+1]-v[0,j])/dy
    u_face[0, :]  = (u_face[1, :]
                     + dx * (v_face[0, 1:] - v_face[0, :-1]) / dy)

    # Outlet cell (i=nx-1):  u[nx,j] = u[nx-1,j] - dx*(v[nx-1,j+1]-v[nx-1,j])/dy
    u_face[-1, :] = (u_face[-2, :]
                     - dx * (v_face[-1, 1:] - v_face[-1, :-1]) / dy)

    return u_face, v_face


def face_to_cell(u_face, v_face,b):
    """
    Average adjacent face velocities to cell centres (plotting only).

        u_cell[i,j] = ½(u_face[i,j] + u_face[i+1,j])
        v_cell[i,j] = ½(v_face[i,j] + v_face[i,j+1])
    """
    # cell-centre averages of face fluxes
    u_cell = 0.5 * (u_face[:-1, :] + u_face[1:, :])   # (nx, ny)
    v_cell = 0.5 * (v_face[:, :-1] + v_face[:, 1:])   # (nx, ny
    u_darcy = u_cell / b   # [m/s]
    v_darcy = v_cell / b   # [m/s]

    return u_darcy,v_darcy


def check_mass_conservation(u_face, v_face, dx, dy, tol=1e-8):
    """
    Discrete divergence at every cell:

        div[i,j] = (u_face[i+1,j] − u_face[i,j])/dx
                 + (v_face[i,j+1] − v_face[i,j])/dy

    Returns div field, max|div|, and bool (True = conserved).
    """
    div = ((u_face[1:, :] - u_face[:-1, :]) / dx
           + (v_face[:, 1:] - v_face[:, :-1]) / dy)
    max_div = float(np.abs(div).max())
    return div, max_div, max_div < tol

def solve_pressure(b, dx, dy, p_in=1.0, p_out=0.0, mu=1e-3):
    """
    Full FVM solve: pressure → conservative face fluxes → cell velocities.

    Returns
    -------
    p      : (nx, ny)   pressure [Pa]
    u      : (nx, ny)   cell-centre x-velocity (face-averaged) [m/s]
    v      : (nx, ny)   cell-centre y-velocity (face-averaged) [m/s]
    q      : (nx, ny)   speed magnitude [m/s]
    u_face : (nx+1, ny) x-face velocity [m/s]  ← use for flux integrals
    v_face : (nx, ny+1) y-face velocity [m/s]
    """
    nx, ny = b.shape
    A, rhs, T = build_system(b,dx,dy,mu,p_in,p_out)
    p = spsolve(csr_matrix(A), rhs).reshape(nx, ny)

    u_face, v_face = recover_face_fluxes(p, T, dx, dy)
    u_cell, v_cell = face_to_cell(u_face, v_face,b)
    q = np.sqrt(u_cell**2 + v_cell**2)
    return p, u_cell, v_cell, q, u_face, v_face

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(b, q, u_face, v_face, dx, dy, p_in, p_out, Lx, mu):
    """Print flow diagnostics using face fluxes (exact, mass-conservative)."""
    nx, ny = b.shape
    Q_inlet  = float(np.sum(u_face[0,    :]) * dy)
    Q_mid    = float(np.sum(u_face[nx//2, :]) * dy)
    Q_outlet = float(np.sum(u_face[nx,   :]) * dy)

    b_harm  = nx / np.sum(1.0 / b.mean(axis=1))
    Q_cubic = b_harm**3 * (p_in - p_out) / (12.0 * mu * Lx)

    div, max_div, conserved = check_mass_conservation(u_face, v_face, dx, dy)

    print("\n─── Flow diagnostics ─────────────────────────────────────────")
    print(f"  Mean aperture                  : {b.mean()*1e6:.2f} μm")
    print(f"  Aperture std / mean            : {b.std()/b.mean():.3f}")
    print(f"  Flux Q at inlet  (face 0)      : {Q_inlet :.4e} m²/s")
    print(f"  Flux Q at mid    (face nx/2)   : {Q_mid   :.4e} m²/s")
    print(f"  Flux Q at outlet (face nx)     : {Q_outlet:.4e} m²/s")
    print(f"  Cubic-law Q (harmonic-mean b)  : {Q_cubic :.4e} m²/s")
    print(f"  Peak cell-centre speed         : {q.max()*1e9:.4f} nm/s  ({q.max()*1e6:.4e} m/s)")
    print(f"  Mean cell-centre speed         : {q.mean()*1e9:.4f} nm/s  ({q.mean()*1e6:.4e} m/s)")
    print(f"  Max |∇·u| (mass conservation)  : {max_div:.2e}",
          "✓ conserved" if conserved else "✗ CHECK SOLVER")
    print("──────────────────────────────────────────────────────────────\n")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(b, p, u, v, q, Lx, Ly,
                 title='2D fracture flow — variable aperture (FVM)',
                 save_path=None):
    """Four-panel figure: aperture, pressure, speed, streamlines."""
    nx, ny = b.shape
    xc = (np.arange(nx) + 0.5) * (Lx/nx) * 1e3   # mm
    yc = (np.arange(ny) + 0.5) * (Ly/ny) * 1e3
    X, Y = np.meshgrid(xc, yc, indexing='ij')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    im = ax.pcolormesh(X, Y, b*1e6, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='Aperture [μm]',fraction=0.023, pad=0.04)
    ax.set_title('Aperture field')

    ax = axes[0, 1]
    im = ax.pcolormesh(X, Y, p, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='Pressure [Pa]',fraction=0.023, pad=0.04)
    ax.contour(X, Y, p, levels=10, colors='k', linewidths=0.5, alpha=0.4)
    ax.set_title('Pressure field')

    ax = axes[1, 0]
    qpos = q[q > 0]
    norm = (mcolors.LogNorm(vmin=qpos.min()*1e6, vmax=q.max()*1e6)
            if qpos.size else None)
    im = ax.pcolormesh(X, Y, q*1e6, cmap='plasma', shading='auto', norm=norm)
    plt.colorbar(im, ax=ax, label='Speed [μm/s]',fraction=0.023, pad=0.04)
    ax.set_title('Speed magnitude (log scale)')

    ax = axes[1, 1]
    ax.streamplot(xc, yc, u.T, v.T, color=np.hypot(u,v).T,
                  cmap='cool', linewidth=0.9, density=1.5, arrowsize=0.8)
    skip = max(1, nx // 20)
    ax.quiver(X[::skip,::skip], Y[::skip,::skip],
              u[::skip,::skip], v[::skip,::skip],
              color='k', alpha=0.35, scale=q.max()*40)
    ax.set_title('Streamlines & velocity vectors')
    ax.set_xlim(xc[0], xc[-1]); ax.set_ylim(yc[0], yc[-1])

    for ax in axes.flat:
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved → {save_path}')
    plt.show()
    return fig

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    Lx, Ly  = 0.80, 0.80
    nx, ny  = 120, 120
    dx, dy  = Lx/nx, Ly/ny
    mu      = 1e-3
    p_in    = 1.0
    p_out   = 0.0
    b_mean =1.0e-4
    b_std=1.0
    # gen2 = ApertureGenerator(
    #     nx=nx, ny=ny, Lx=Lx, Ly=Ly,
    #     b_mean=1e-4, b_sigma=0.55,
    #     variogram='gaussian',
    #     range_x=0.06, range_y=0.018,   # 3:1 anisotropy ratio
    #     angle_deg=30.0,                 # principal axis rotated 30° CCW
    #     nugget_frac=0.03,
    # )
    # b = gen2.generate(seed=100)

    # overlay = ChannelOverlay(nx, ny, Lx, Ly)
    # # Provide a distinct seed to each call (e.g., 42 and 7)
    # # overlay.add_random_walk(width=0.002, b_factor=3.0,
    # #                      y_centre=0.04, roughness=0.1, seed=42)
    # # overlay.add_random_walk(width=0.005, b_factor=3.0,
    # #                      y_centre=0.01, roughness=0.1, seed=12)
    # # b = overlay.apply(b)
    b= generate_field(nx,ny,dx,dy,b_mean,b_std,L_c=.1,angle=7*np.pi/16)
    p, u, v, q, u_face, v_face = solve_pressure(
        b, dx, dy, p_in=p_in, p_out=p_out, mu=mu)

    # compute_diagnostics(b, q, u_face, v_face, dx, dy, p_in, p_out, Lx, mu)

    plot_results(b, p, u, v, q, Lx, Ly,
                 title='2D fracture flow — variable aperture (FVM)')
