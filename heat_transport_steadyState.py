"""
Steady-State Heat Transport in a Variable-Aperture Fracture
============================================================
Solves the coupled fluid/rock temperature field by assembling a single
sparse linear system and calling a direct solver (SuperLU via splu).

State vector layout  (length = N_total = N * (1 + 2*n_rock)):
  [0      :   N]            fluid layer
  [N      : N+N*n_rock]     rock side 0  (layer 0 = fracture face … n_rock-1 = far field)
  [N+N*n_rock : N_total]    rock side 1

Boundary conditions
-------------------
  x (flow direction):
    Inlet  (i=0)     : Dirichlet T = T_in  (fluid + rock layers)
    Outlet (i=nx-1)  : zero-gradient (advection-only outflow)
  y (transverse)     : PERIODIC for both fluid and rock
  z (rock thickness) : outermost rock layer Dirichlet T = T_init

Physics
-------
  Fluid : upwind advection + aperture-weighted diffusion + wall exchange
  Rock  : in-plane diffusion (periodic y) + through-thickness diffusion
          + wall exchange at layer 0
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from velocity_solver import solve_pressure, plot_results
from hurst import generate_field

# ── Physical constants ────────────────────────────────────────────────────────
KAPPA   = 1.4e-7      # fluid  thermal diffusivity  [m²/s]
RHO_C   = 4.182e6     # fluid  volumetric heat cap. [J/m³/K]
KAPPA_S = 1.5e-6      # rock   thermal diffusivity  [m²/s]
RHO_C_S = 2.16e6      # rock   volumetric heat cap. [J/m³/K]
N_ROCK  = 5           # rock sublayers per side (default)
K_FLUID   = 0.6      # W/m/K
NU         = 3.771    # Nusselt — laminar slot flow (isothermal)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_side_slice(T_vec, N, nx, ny, n_rock, slice_j):
    """
    Assemble a cross-section (x vs z) through the full rock+fluid stack
    at a fixed y-index (slice_j).

    Returns a 2-D array of shape (1 + 2*n_rock, nx) ordered:
      row 0          : bottom rock outer layer (side 0, layer n_rock-1)
      …
      row n_rock-1   : bottom rock inner layer (side 0, layer 0)
      row n_rock     : fluid
      row n_rock+1   : top rock inner layer    (side 1, layer 0)
      …
      row 2*n_rock   : top rock outer layer    (side 1, layer n_rock-1)
    """
    def rock_off(side, layer):
        return N * (1 + side * n_rock + layer)

    rows = []
    # Bottom rock (side 0): outer → inner
    for layer in range(n_rock - 1, -1, -1):
        off = rock_off(0, layer)
        rows.append(T_vec[off:off + N].reshape(nx, ny)[:, slice_j])
    # Fluid
    rows.append(T_vec[:N].reshape(nx, ny)[:, slice_j])
    # Top rock (side 1): inner → outer
    for layer in range(n_rock):
        off = rock_off(1, layer)
        rows.append(T_vec[off:off + N].reshape(nx, ny)[:, slice_j])

    return np.array(rows)   # (1 + 2*n_rock, nx)


# ---------------------------------------------------------------------------
# Steady-state solver
# ---------------------------------------------------------------------------

def solve_heat(b, u_face, v_face, dx, dy, T_init_1, T_init_2, T_in, H_s, n_rock=N_ROCK):
    """
    Assemble and solve the steady-state heat-transport system.

    Parameters
    ----------
    b        : (nx, ny)   aperture field [m]
    u_face   : (nx+1, ny) x-face Darcy flux [m²/s]  (from velocity_solver)
    v_face   : (nx, ny+1) y-face Darcy flux [m²/s]  (v[:,0]==v[:,ny])
    dx, dy   : cell sizes [m]
    T_init   : far-field (initial) rock temperature [K or °C]
    T_in     : inlet fluid temperature [K or °C]
    H_s      : total rock half-thickness per side [m]
    n_rock   : number of rock sublayers per side

    Returns
    -------
    T_fluid : (nx, ny)  fluid temperature
    T_inner : (nx, ny)  rock temperature at fracture face (layer 0)
    T_rock  : (nx, ny)  mean active-rock temperature
    side_T  : (1+2*n_rock, nx)  cross-section slice at mid-y
    """
    nx, ny     = b.shape
    N          = nx * ny
    N_total    = N * (1 + 2 * n_rock)
    H_s_layer  = H_s / n_rock       # thickness of one rock sublayer

    

    KK = np.arange(N).reshape(nx, ny)   # flat cell indices

    rows_l, cols_l, vals_l = [], [], []

    def _add(r, c, v):
        r = np.asarray(r).ravel()
        c = np.asarray(c).ravel()
        v = np.asarray(v).ravel()
        if v.size == 1:
            v = np.full(r.shape, v[0])
        rows_l.append(r); cols_l.append(c); vals_l.append(v)

    def rock_off(side, layer):
        """Global index offset for rock block (side, layer)."""
        return N * (1 + side * n_rock + layer)

    # ── Aperture harmonic means at faces ──────────────────────────────────
    b_xf = 2 * b[:-1, :] * b[1:, :] / (b[:-1, :] + b[1:, :])   # (nx-1, ny)

    # y-faces: interior + periodic wrap
    b_yf_int = 2 * b[:, :-1] * b[:, 1:] / (b[:, :-1] + b[:, 1:])  # (nx, ny-1)
    b_yf_per = 2 * b[:, -1]  * b[:, 0]  / (b[:, -1]  + b[:, 0] )  # (nx,)
    # Full array with wrap face appended: shape (nx, ny)
    b_yf = np.concatenate([b_yf_int, b_yf_per[:, None]], axis=1)

    # =========================================================
    # 1. FLUID BLOCK
    # =========================================================

    # ── Interior x-faces (i = 1 … nx-1) ──────────────────────────────────
    # Upwind advection + diffusion assembled as a flux difference.
    # a_x > 0 ⟹ flow left→right; upwind = left cell.
    a_x = u_face[1:-1, :] / dx          # (nx-1, ny)  advective coef
    d_x = b_xf * KAPPA / dx**2          # (nx-1, ny)  diffusive coef
    k_L = KK[:-1, :]                    # left  cell indices
    k_R = KK[1:,  :]                    # right cell indices

    # Contribution of face (i-½) to left cell (outgoing) and right cell (incoming):
    _add(k_L, k_L,  np.maximum(a_x, 0) + d_x)   # left  diag  +
    _add(k_L, k_R,  np.minimum(a_x, 0) - d_x)   # left  off   -
    _add(k_R, k_R, -np.minimum(a_x, 0) + d_x)   # right diag  +
    _add(k_R, k_L, -np.maximum(a_x, 0) - d_x)   # right off   -

    # ── y-faces: ALL ny faces using the periodic wrap ─────────────────────
    # v_face has shape (nx, ny+1) with v_face[:,0]==v_face[:,ny].
    # Face j connects cell j-1 (south) to cell j (north); face 0 is the
    # periodic wrap connecting cell ny-1 (south) to cell 0 (north).
    #
    # We loop over all ny faces (j = 0 … ny-1) where face j connects:
    #   south cell : KK[:, j-1 mod ny]
    #   north cell : KK[:, j]
    # and uses flux v_face[:, j] and aperture b_yf[:, j].

    for jf in range(ny):
        j_s = (jf - 1) % ny           # south cell index
        j_n = jf                       # north cell index (face jf = south face of j_n)
        # For jf == 0: j_s = ny-1, j_n = 0  ← periodic wrap face

        a_y_jf = v_face[:, jf] / dy   # (nx,)  advective coef at face jf
        d_y_jf = b_yf[:, jf] * KAPPA / dy**2  # (nx,)

        k_S = KK[:, j_s]  # (nx,)
        k_N = KK[:, j_n]  # (nx,)

        _add(k_S, k_S,  np.maximum(a_y_jf, 0) + d_y_jf)
        _add(k_S, k_N,  np.minimum(a_y_jf, 0) - d_y_jf)
        _add(k_N, k_N, -np.minimum(a_y_jf, 0) + d_y_jf)
        _add(k_N, k_S, -np.maximum(a_y_jf, 0) - d_y_jf)

    # ── Outlet BC (i = nx-1): zero-gradient outflow ───────────────────────
    # Remove the downstream diffusion term; keep only the outgoing advection.
    _add(KK[-1, :], KK[-1, :], np.abs(u_face[-1, :]) / dx)

    # ── Fluid–rock wall exchange ──────────────────────────────────────────
    h_local = (NU * K_FLUID) / (2.0 * b)   # W/m²/K  (nx, ny)
    hf_coef = h_local / RHO_C              # m/s     (nx, ny)
    k_all   = KK.ravel()                   # (N,)

    # Fluid loses heat to both rock walls
    _add(k_all, k_all, 2.0 * hf_coef.ravel())
    for side in range(2):
        _add(k_all, rock_off(side, 0) + k_all, -hf_coef.ravel())

    # =========================================================
    # 2. ROCK BLOCKS  (side = 0 bottom, side = 1 top)
    # =========================================================
    # Through-thickness diffusion coefficient between adjacent rock layers:
    #   d_z = κ_s / H_s_layer 
    # The steady FVM equation for rock layer (side, l) per unit area:
    #   (rock gains from l-1 or fluid) + (rock loses to l+1 or outer BC)
    # We use a diffusion coefficient scaled by the layer thickness so that
    # the *residual* has units of [K/s], consistent with the fluid block.
    d_z  = KAPPA_S / H_s_layer         # [m/s]  through-thickness
    d_xr = KAPPA_S* H_s_layer / dx**2              # [m/s]  in-plane x
    d_yr = KAPPA_S * H_s_layer / dy**2                # [m/s]  in-plane y
    hs_coef = h_local / RHO_C_S           # [m/s] → exchange with fluid

    for side in range(2):
        for layer in range(n_rock - 1):   # layer n_rock-1 is Dirichlet → skip
            off     = rock_off(side, layer)
            ks      = off + k_all          # global row indices for this layer

            # ── In-plane x-diffusion (Neumann at x-boundaries) ───────────
            ks_xL = off + KK[:-1, :].ravel()
            ks_xR = off + KK[1:,  :].ravel()
            _add(ks_xL, ks_xL,  d_xr);  _add(ks_xL, ks_xR, -d_xr)
            _add(ks_xR, ks_xR,  d_xr);  _add(ks_xR, ks_xL, -d_xr)

            # ── In-plane y-diffusion (periodic) ───────────────────────────
            for jf in range(ny):
                j_s = (jf - 1) % ny
                j_n = jf
                ks_S = off + KK[:, j_s].ravel()
                ks_N = off + KK[:, j_n].ravel()
                _add(ks_S, ks_S,  d_yr);  _add(ks_S, ks_N, -d_yr)
                _add(ks_N, ks_N,  d_yr);  _add(ks_N, ks_S, -d_yr)

            # ── Through-thickness coupling ────────────────────────────────
            # Inward face (toward fluid):
            if layer == 0:
                # Layer 0 is adjacent to the fluid wall
                # Exchange coefficient: h / (ρc_rock)
                _add(ks, ks,     hs_coef.ravel())
                _add(ks, k_all, -hs_coef.ravel())
                # Also coupled through-thickness to layer 1 (outward)
                off_out = rock_off(side, 1)
                _add(ks, ks,               d_z)
                _add(ks, off_out + k_all, -d_z)
            else:
                # Inward face to layer - 1
                off_in = rock_off(side, layer - 1)
                _add(ks, ks,              d_z)
                _add(ks, off_in + k_all, -d_z)
                # Outward face to layer + 1  (or Dirichlet at n_rock-1)
                off_out = rock_off(side, layer + 1)
                _add(ks, ks,               d_z)
                _add(ks, off_out + k_all, -d_z)

    # =========================================================
    # 3. ASSEMBLE COO, MASK DIRICHLET ROWS, ENFORCE BCs
    # =========================================================
    all_r = np.concatenate(rows_l)
    all_c = np.concatenate(cols_l)
    all_v = np.concatenate(vals_l)

    # Dirichlet rows: fluid inlet (i=0, all j) + outer rock layers
    inlet_rows = KK[0, :].ravel()                    # (ny,)
    outer_rows_1 = np.arange(rock_off(0, n_rock - 1),
                  rock_off(0, n_rock - 1) + N)
    outer_rows_2 = np.arange(rock_off(1, n_rock - 1),
                  rock_off(1, n_rock - 1) + N) 
    dirichlet_rows = np.concatenate([inlet_rows, outer_rows_1,outer_rows_2])

    mask  = ~np.isin(all_r, dirichlet_rows)
    A_coo = coo_matrix((all_v[mask], (all_r[mask], all_c[mask])),
                       shape=(N_total, N_total))
    A_lil = A_coo.tolil()
    rhs   = np.zeros(N_total)

    # Fluid inlet: T = T_in  for all j at i=0
    for row in inlet_rows:
        A_lil[row, :] = 0
        A_lil[row, row] = 1.0
        rhs[row] = T_in

    # Outer rock layers: T = T_init
    for row in outer_rows_1:
        A_lil[row, :] = 0
        A_lil[row, row] = 1.0
        rhs[row] = T_init_1
    for row in outer_rows_2:
        A_lil[row, :] = 0
        A_lil[row, row] = 1.0
        rhs[row] = T_init_2
    # =========================================================
    # 4. SOLVE
    # =========================================================
    A     = A_lil.tocsc()
    A_lu  = splu(A)
    T_vec = A_lu.solve(rhs)

    # =========================================================
    # 5. EXTRACT FIELDS
    # =========================================================
    n_active = n_rock - 1

    def mean_active_rock(T):
        T_sum = np.zeros(N)
        for side in range(2):
            for layer in range(n_active):
                off = rock_off(side, layer)
                T_sum += T[off:off + N]
        return (T_sum / (2 * n_active)).reshape(nx, ny)

    T_fluid = T_vec[:N].reshape(nx, ny).copy()
    T_inner = T_vec[rock_off(0, 0):rock_off(0, 0) + N].reshape(nx, ny).copy()
    T_rock  = mean_active_rock(T_vec).copy()

    slice_j = ny // 2
    side_T  = _make_side_slice(T_vec, N, nx, ny, n_rock, slice_j)

    return T_fluid, T_inner, T_rock, side_T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_steady_state_views(T_fluid, T_inner, T_rock, side_T,
                            dx, dy, H_s, n_rock):
    """Four-panel steady-state thermal dashboard."""
    nx, ny = T_fluid.shape
    x_max  = nx * dx
    y_max  = ny * dy

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()
    cmap = 'jet'

    im0 = axs[0].imshow(T_fluid.T, extent=[0, x_max, 0, y_max],
                        origin='lower', cmap=cmap, aspect='auto')
    axs[0].set_title("Fluid Temperature $T_{fluid}$")
    axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]")
    fig.colorbar(im0, ax=axs[0], label="Temperature [°C]")

    im1 = axs[1].imshow(T_inner.T, extent=[0, x_max, 0, y_max],
                        origin='lower', cmap=cmap, aspect='auto')
    axs[1].set_title("Innermost Rock Layer $T_{inner}$  ($l=0$)")
    axs[1].set_xlabel("x [m]"); axs[1].set_ylabel("y [m]")
    fig.colorbar(im1, ax=axs[1], label="Temperature [°C]")

    im2 = axs[2].imshow(T_rock.T, extent=[0, x_max, 0, y_max],
                        origin='lower', cmap=cmap, aspect='auto')
    axs[2].set_title("Mean Active Rock Temperature $T_{rock}$")
    axs[2].set_xlabel("x [m]"); axs[2].set_ylabel("y [m]")
    fig.colorbar(im2, ax=axs[2], label="Temperature [°C]")

    im3 = axs[3].imshow(side_T, extent=[0, x_max, -H_s, H_s],
                        origin='lower', aspect='auto', cmap=cmap)
    axs[3].set_title(f"Side Slice at $y = {y_max/2:.3f}$ m")
    axs[3].set_xlabel("x [m]"); axs[3].set_ylabel("z thickness [m]")
    axs[3].axhline(0, color='white', linestyle='--', alpha=0.6,
                   label='Fluid interface')
    axs[3].legend(loc='upper right')
    fig.colorbar(im3, ax=axs[3], label="Temperature [°C]")

    plt.tight_layout()
    plt.show()
    return fig
# ---------------------------------------------------------------------------
# H_eff calculation
# ---------------------------------------------------------------------------

"NOTE: IN ORDER TO USE THIS VALUE IN SANDBOX, ONLY DIVIDE BY Lx!!!!!!"
def calculate_h_eff(T_fluid,u_face,T_in, T_init_1,T_init_2,Lx,Ly,dy):
    T_delta = T_fluid[-1,:]-T_in            #delta_T for outgoing fluid
    u_out = u_face[-1,:]                    # Darcy fluxes out
    E_out = np.dot(T_delta,u_out)*RHO_C*dy  # Energy out
    Q=E_out/Lx/Ly/2                         # Heat flux (Energy/interfacial surface area)
    h_eff = np.abs(Q/(T_in-0.5*T_init_1-0.5*T_init_2))
    return h_eff

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    Lx, Ly = 0.80, 0.80
    nx, ny = 120, 120
    dx, dy = Lx / nx, Ly / ny
    mu     = 1e-3
    p_in   = 50_000.0
    p_out  = 0.0
    H_s    = 0.05
    n_rock = 5
    T_init_1 = 100.0
    T_init_2 =100.0
    T_in   = 50.0
    b_mean = 1.0e-4
    b_std  = 1.0

    b = generate_field(nx, ny, dx, dy, b_mean, b_std, L_c=0.1, angle=7*np.pi/16)

    p, u_cell, v_cell, q, u_face, v_face = solve_pressure(
        b, dx, dy, p_in=p_in, p_out=p_out, mu=mu)

    plot_results(b, p, u_cell, v_cell, q, Lx, Ly,
                 title='2D fracture flow — variable aperture (FVM)')

    T_fluid, T_inner, T_rock, side_T = solve_heat(
        b, u_face, v_face, dx, dy, T_init_1, T_init_2, T_in, H_s, n_rock=n_rock)

    plot_steady_state_views(T_fluid, T_inner, T_rock, side_T,
                            dx, dy, H_s, n_rock)
    h_local = (NU * K_FLUID) / (2.0 * b)
    h_spatial=np.average(h_local)
    h_eff = calculate_h_eff(T_fluid, u_face,T_in,T_init_1,T_init_2,Lx,Ly,dy)
    print(f'Spatial Mean H_eff: {h_spatial}')
    print(f'Calculated H_eff: {h_eff}')