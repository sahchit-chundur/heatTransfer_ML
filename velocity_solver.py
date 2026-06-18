"""
2D Velocity Field in a Variable-Aperture Fracture — Finite Volume Method
=========================================================================
Governing equation (Reynolds lubrication / local cubic law):

    ∇ · (b³/(12μ) ∇p) = 0

Boundary conditions
--------------------
  x-direction : Dirichlet pressure at inlet (i=0) and outlet (i=nx-1)
  y-direction : PERIODIC (north face of j=ny-1 connects to south face of j=0)

Velocity recovery
-----------------
Interior x-faces  (i = 1 … nx-1):
    u_face[i,j] = -T_{i-½,j} · (p[i,j] − p[i-1,j]) / dx

Boundary x-faces (i=0, i=nx):
    Recovered from local divergence-free condition on the boundary cell.

y-faces (j = 0 … ny, periodic):
    v_face[:,j] = -T_{j-½} · (p[:,j] − p[:,j-1]) / dy   for j=1…ny-1
    v_face[:,0] = v_face[:,ny] = periodic wrap face flux
    (The wrap face connects j=ny-1 → j=0; both aliases are set equal.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from hurst import generate_field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def harmonic_mean(a, b):
    """Element-wise harmonic mean; returns 0 where a+b == 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((a + b) > 0, 2.0 * a * b / (a + b), 0.0)


# ---------------------------------------------------------------------------
# Finite-Volume system assembly
# ---------------------------------------------------------------------------

def build_system(b, dx, dy, mu, p_in, p_out):
    nx, ny = b.shape
    N = nx * ny
    T = b**3 / (12.0 * mu)

    A = lil_matrix((N, N))
    rhs = np.zeros(N)

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)

            # ── X DIRECTION ──────────────────────────────────────────────
            if i == 0:
                # Inlet half-cell ghost: distance = dx/2
                c_inlet = 2.0 * T[i, j] / dx**2
                A[k, k] += c_inlet
                rhs[k]  += c_inlet * p_in
            else:
                c = harmonic_mean(T[i, j], T[i-1, j]) / dx**2
                A[k, k]           += c
                A[k, idx(i-1, j)] -= c

            if i == nx - 1:
                # Outlet half-cell ghost: distance = dx/2
                c_outlet = 2.0 * T[i, j] / dx**2
                A[k, k] += c_outlet
                rhs[k]  += c_outlet * p_out
            else:
                c = harmonic_mean(T[i, j], T[i+1, j]) / dx**2
                A[k, k]           += c
                A[k, idx(i+1, j)] -= c

            # ── Y DIRECTION (PERIODIC) ────────────────────────────────────
            j_s = (j - 1) % ny   # south neighbour (wraps)
            j_n = (j + 1) % ny   # north neighbour (wraps)

            c_s = harmonic_mean(T[i, j], T[i, j_s]) / dy**2
            A[k, k]           += c_s
            A[k, idx(i, j_s)] -= c_s

            c_n = harmonic_mean(T[i, j], T[i, j_n]) / dy**2
            A[k, k]           += c_n
            A[k, idx(i, j_n)] -= c_n

    return A, rhs, T


# ---------------------------------------------------------------------------
# Face-flux recovery
# ---------------------------------------------------------------------------

def recover_face_fluxes(p, T, dx, dy):
    """
    Recover conservative Darcy face fluxes consistent with the assembled
    matrix.

    Returns
    -------
    u_face : (nx+1, ny)   x-face volumetric flux [m²/s]
    v_face : (nx,  ny+1)  y-face volumetric flux [m²/s]
        v_face[:,0]  == v_face[:,ny]  (periodic wrap; both aliases stored)
    """
    nx, ny = p.shape

    # ── y-faces (periodic) ───────────────────────────────────────────────
    # Internal faces j = 1 … ny-1
    v_face = np.zeros((nx, ny + 1))
    v_face[:, 1:-1] = (-harmonic_mean(T[:, :-1], T[:, 1:])
                       * (p[:, 1:] - p[:, :-1]) / dy)

    # Periodic wrap face: connects j=ny-1 (south) → j=0 (north)
    # Flux is positive when flow goes from j=ny-1 toward j=0
    v_wrap = (-harmonic_mean(T[:, -1], T[:, 0])
              * (p[:, 0] - p[:, -1]) / dy)
    v_face[:, 0]  = v_wrap   # south face of j=0  (= north face of j=ny-1)
    v_face[:, ny] = v_wrap   # north face of j=ny-1 (alias)

    # ── interior x-faces ─────────────────────────────────────────────────
    u_face = np.zeros((nx + 1, ny))
    u_face[1:-1, :] = (-harmonic_mean(T[:-1, :], T[1:, :])
                       * (p[1:, :] - p[:-1, :]) / dx)

    # ── boundary x-faces: local divergence-free extrapolation ─────────────
    # Inlet cell (i=0):
    #   (u[1,j] - u[0,j])/dx + (v[0,j+1] - v[0,j])/dy = 0
    u_face[0, :]  = (u_face[1, :]
                     + dx * (v_face[0, 1:] - v_face[0, :-1]) / dy)

    # Outlet cell (i=nx-1):
    #   (u[nx,j] - u[nx-1,j])/dx + (v[nx-1,j+1] - v[nx-1,j])/dy = 0
    u_face[-1, :] = (u_face[-2, :]
                     - dx * (v_face[-1, 1:] - v_face[-1, :-1]) / dy)

    return u_face, v_face


# ---------------------------------------------------------------------------
# Cell-centre velocities (for plotting only)
# ---------------------------------------------------------------------------

def face_to_cell(u_face, v_face, b):
    """
    Average adjacent face fluxes to cell centres, then divide by aperture
    to get the Darcy velocity.

        u_cell[i,j] = ½(u_face[i,j] + u_face[i+1,j]) / b[i,j]
        v_cell[i,j] = ½(v_face[i,j] + v_face[i,j+1]) / b[i,j]

    NOTE: v_face has shape (nx, ny+1) with v_face[:,0]==v_face[:,ny],
    so the averaging is well-defined at every cell including j=ny-1.
    """
    u_cell = 0.5 * (u_face[:-1, :] + u_face[1:, :]) / b
    v_cell = 0.5 * (v_face[:, :-1] + v_face[:, 1:]) / b
    return u_cell, v_cell


# ---------------------------------------------------------------------------
# Mass conservation check
# ---------------------------------------------------------------------------

def check_mass_conservation(u_face, v_face, dx, dy, tol=1e-8):
    """
    Discrete divergence at every cell:

        div[i,j] = (u_face[i+1,j] − u_face[i,j])/dx
                 + (v_face[i,j+1] − v_face[i,j])/dy

    v_face[:,ny] is the periodic wrap face, identical to v_face[:,0],
    so the difference v_face[:,ny] - v_face[:,ny-1] is naturally included.
    """
    div = ((u_face[1:, :] - u_face[:-1, :]) / dx
           + (v_face[:, 1:] - v_face[:, :-1]) / dy)
    max_div = float(np.abs(div).max())
    return div, max_div, max_div < tol


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

def solve_pressure(b, dx, dy, p_in=1.0, p_out=0.0, mu=1e-3):
    """
    Full FVM solve: pressure → conservative face fluxes → cell velocities.

    Returns
    -------
    p      : (nx, ny)    pressure [Pa]
    u      : (nx, ny)    cell-centre x-velocity [m/s]
    v      : (nx, ny)    cell-centre y-velocity [m/s]
    q      : (nx, ny)    speed magnitude [m/s]
    u_face : (nx+1, ny)  x-face flux [m²/s]
    v_face : (nx, ny+1)  y-face flux [m²/s]  (v[:,0]==v[:,ny])
    """
    A, rhs, T = build_system(b, dx, dy, mu, p_in, p_out)
    p = spsolve(csr_matrix(A), rhs).reshape(b.shape)

    u_face, v_face = recover_face_fluxes(p, T, dx, dy)
    u_cell, v_cell = face_to_cell(u_face, v_face, b)
    q = np.sqrt(u_cell**2 + v_cell**2)
    return p, u_cell, v_cell, q, u_face, v_face


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(b, q, u_face, v_face, dx, dy, p_in, p_out, Lx, mu):
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
    print(f"  Peak cell-centre speed         : {q.max()*1e6:.4e} μm/s")
    print(f"  Mean cell-centre speed         : {q.mean()*1e6:.4e} μm/s")
    print(f"  Max |∇·u| (mass conservation)  : {max_div:.2e}",
          "✓ conserved" if conserved else "✗ CHECK SOLVER")
    print("──────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(b, p, u, v, q, Lx, Ly,
                 title='2D fracture flow — variable aperture (FVM)',
                 save_path=None):
    nx, ny = b.shape
    xc = (np.arange(nx) + 0.5) * (Lx/nx) * 1e3
    yc = (np.arange(ny) + 0.5) * (Ly/ny) * 1e3
    X, Y = np.meshgrid(xc, yc, indexing='ij')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    im = ax.pcolormesh(X, Y, b*1e6, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='Aperture [μm]', fraction=0.023, pad=0.04)
    ax.set_title('Aperture field')

    ax = axes[0, 1]
    im = ax.pcolormesh(X, Y, p, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='Pressure [Pa]', fraction=0.023, pad=0.04)
    ax.contour(X, Y, p, levels=10, colors='k', linewidths=0.5, alpha=0.4)
    ax.set_title('Pressure field')

    ax = axes[1, 0]
    qpos = q[q > 0]
    norm = (mcolors.LogNorm(vmin=qpos.min()*1e6, vmax=q.max()*1e6)
            if qpos.size else None)
    im = ax.pcolormesh(X, Y, q*1e6, cmap='plasma', shading='auto', norm=norm)
    plt.colorbar(im, ax=ax, label='Speed [μm/s]', fraction=0.023, pad=0.04)
    ax.set_title('Speed magnitude (log scale)')

    ax = axes[1, 1]
    ax.streamplot(xc, yc, u.T, v.T, color=np.hypot(u, v).T,
                  cmap='cool', linewidth=0.9, density=1.5, arrowsize=0.8)
    skip = max(1, nx // 20)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
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
    Lx, Ly = 0.80, 0.80
    nx, ny = 120, 120
    dx, dy = Lx/nx, Ly/ny
    mu     = 1e-3
    p_in   = 1.0
    p_out  = 0.0
    b_mean = 1.0e-4
    b_std  = 1.0

    b = generate_field(nx, ny, dx, dy, b_mean, b_std, L_c=0.1, angle=7*np.pi/16)
    p, u, v, q, u_face, v_face = solve_pressure(b, dx, dy, p_in=p_in, p_out=p_out, mu=mu)

    compute_diagnostics(b, q, u_face, v_face, dx, dy, p_in, p_out, Lx, mu)
    plot_results(b, p, u, v, q, Lx, Ly,
                 title='2D fracture flow — variable aperture (FVM)')