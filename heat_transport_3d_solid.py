"""
Transient Heat Transport in a Variable-Aperture Fracture — FVM
===============================================================
Process:
1.) Generate aperture field from geostat.py using FFT spectral techniques, using multiscale covariance, anisotropy, and channel overlays derived from inputs.
2.) Calculate steady state velocity field from Reynold's Lubrication Equation using velocity_solver.py.
3.) Construct matrix (_build_transport_matrix) and solve transient advection-diffusion equation (solve_heat) for temperature evolution of fluid and solid
4.) Energy Conservation check
5.) Parameter sweep (Parallelized)
6.) Global Sobol' Sensitivity Analysis (Parallelized)

Boundary conditions
-------------------
  Rock lateral faces (x=0, x=Lx, y=0, y=Ly) : Neumann zero-flux
  Rock outer face (l = N_ROCK-1)             : Dirichlet T_init
  Fluid inlet (x=0)                          : Dirichlet T_in (ramped)
  All outflow / wall faces                   : zero-gradient
"""
import os
# Block multi-threading inside linear algebra backends before they load
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import scipy
from typing import Any
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, diags, block_diag
from scipy.sparse.linalg import  splu
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from velocity_solver import solve_pressure, plot_results
from geostat import ApertureGenerator, ChannelOverlay
from hurst import generate_field
from auxillary import check_energy_conservation, _plot_energy_conservation, animate_heat_three_views, compute_interface_h_eff, stored_energy, matrix_heating_power, _make_side_slice, ramp_T_in
from scipy.stats import qmc
import concurrent.futures

# ── physical constants ────────────────────────────────────────────────────────
KAPPA   = 1.4e-7      # fluid  thermal diffusivity  [m²/s]
RHO_C   = 4.182e6     # fluid  ρc                   [J/m³/K]
KAPPA_S = 1.5e-6      # granite thermal diffusivity [m²/s]
RHO_C_S = 2.16e6      # granite ρc                  [J/m³/K]
N_ROCK  = 5          # rock sublayers per side (default)

# ─────────────────────────────────────────────────────────────────────────────
# Matrix assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_transport_matrix(b, u_face, v_face, dx, dy, H_s, n_rock=N_ROCK):
    """
    Assemble (1+2*n_rock)*N × (1+2*n_rock)*N operator L via COO format.

    All loops over grid cells are replaced by NumPy array operations.
    The only remaining Python loops iterate over 2*(n_rock-1) rock layers.
    """
    nx, ny    = b.shape
    N         = nx * ny
    N_total   = N * (1 + 2 * n_rock)
    H_s_layer = H_s / n_rock
    k_fluid   = 0.6    # W/m/K
    Nu        = 4.117  # laminar slot flow

    KK = np.arange(N).reshape(nx, ny)   # (nx, ny) flat cell indices

    rows_l, cols_l, vals_l = [], [], []

    def _add(r, c, v):
        rows_l.append(np.asarray(r).ravel())
        cols_l.append(np.asarray(c).ravel())
        vals_l.append(np.broadcast_to(np.asarray(v),
                                       np.asarray(r).shape).ravel().copy())

    def rock_off(side, layer):
        return N * (1 + side * n_rock + layer)

    # harmonic-mean apertures at faces
    b_xf = 2*(b[:-1,:]*b[1:,:]) / (b[:-1,:]+b[1:,:])   # (nx-1, ny)
    b_yf = 2*(b[:,:-1]*b[:,1:]) / (b[:,:-1]+b[:,1:])    # (nx, ny-1)

    # ── fluid block ───────────────────────────────────────────────────────

    # Interior x-faces  (nx-1, ny)
    a_x = u_face[1:-1, :] / dx
    d_x = b_xf * KAPPA / dx**2
    k_L = KK[:-1, :];  k_R = KK[1:, :]

    _add(k_L, k_L,  np.maximum(a_x, 0) + d_x)
    _add(k_L, k_R,  np.minimum(a_x, 0) - d_x)
    _add(k_R, k_R, -np.minimum(a_x, 0) + d_x)
    _add(k_R, k_L, -np.maximum(a_x, 0) - d_x)

    # Interior y-faces  (nx, ny-1)
    a_y = v_face[:, 1:-1] / dy
    d_y = b_yf * KAPPA / dy**2
    k_B = KK[:, :-1];  k_T = KK[:, 1:]

    _add(k_B, k_B,  np.maximum(a_y, 0) + d_y)
    _add(k_B, k_T,  np.minimum(a_y, 0) - d_y)
    _add(k_T, k_T, -np.minimum(a_y, 0) + d_y)
    _add(k_T, k_B, -np.maximum(a_y, 0) - d_y)

    # Outlet BC (i = nx-1): zero-gradient
    _add(KK[-1, :], KK[-1, :], np.abs(u_face[-1, :]) / dx)

    # Fluid-rock exchange — local Nusselt coefficient, varies per cell
    h_local = (Nu * k_fluid) / (2.0 * b)   # (nx, ny)  [W/m²/K]
    hf_coef = h_local / RHO_C              # (nx, ny)  [m/s]
    k_all   = KK.ravel()                   # (N,)

    _add(k_all, k_all, 2.0 * hf_coef.ravel())   # loses heat to both sides
    for side in range(2):
        _add(k_all, rock_off(side, 0) + k_all, -hf_coef.ravel())

    # ── rock blocks  (active layers l = 0 … n_rock-2 only) ───────────────
    d_z  = KAPPA_S / H_s_layer            # through-thickness  [m/s]
    d_xr = H_s_layer * KAPPA_S / dx**2   # in-plane x         [m/s]
    d_yr = H_s_layer * KAPPA_S / dy**2   # in-plane y         [m/s]

    # Innermost-layer rock exchange coefficient — also varies per cell
    hs_coef = h_local / RHO_C_S           # (nx, ny)  [m/s]

    for side in range(2):
        for layer in range(n_rock - 1):   # outer Dirichlet layer skipped
            off = rock_off(side, layer)
            ks  = off + k_all              # all cell indices for this layer

            # ── in-plane x-diffusion (Neumann on lateral faces) ──────────
            ks_xL = off + KK[:-1, :].ravel()
            ks_xR = off + KK[1:,  :].ravel()
            _add(ks_xL, ks_xL,  d_xr);  _add(ks_xL, ks_xR, -d_xr)
            _add(ks_xR, ks_xR,  d_xr);  _add(ks_xR, ks_xL, -d_xr)

            # ── in-plane y-diffusion ──────────────────────────────────────
            ks_yB = off + KK[:, :-1].ravel()
            ks_yT = off + KK[:, 1: ].ravel()
            _add(ks_yB, ks_yB,  d_yr);  _add(ks_yB, ks_yT, -d_yr)
            _add(ks_yT, ks_yT,  d_yr);  _add(ks_yT, ks_yB, -d_yr)

            # ── through-thickness: inward (toward fluid) ──────────────────
            if layer == 0:
                # innermost — local Nusselt coupling, varies per cell
                _add(ks, ks,     hs_coef.ravel())
                _add(ks, k_all, -hs_coef.ravel())
            else:
                off_in = rock_off(side, layer - 1)
                _add(ks, ks,              d_z)
                _add(ks, off_in + k_all, -d_z)

            # ── through-thickness: outward (toward next / Dirichlet layer) ─
            off_out = rock_off(side, layer + 1)
            _add(ks, ks,               d_z)
            _add(ks, off_out + k_all, -d_z)

    # ── assemble COO → CSR ────────────────────────────────────────────────
    all_r = np.concatenate(rows_l)
    all_c = np.concatenate(cols_l)
    all_v = np.concatenate(vals_l)

    # Zero out Dirichlet rows by masking them from the COO triplets
    inlet_rows = np.arange(ny)
    outer_rows = np.concatenate([
        np.arange(rock_off(s, n_rock - 1), rock_off(s, n_rock - 1) + N)
        for s in range(2)
    ])
    dirichlet_rows = np.concatenate([inlet_rows, outer_rows])

    mask = ~np.isin(all_r, dirichlet_rows)
    L = coo_matrix((all_v[mask], (all_r[mask], all_c[mask])),
                   shape=(N_total, N_total)).tocsr()
    L.sum_duplicates()
    L.eliminate_zeros()

    return L, np.zeros(N_total)

def solve_heat(b, u_face, v_face, dx, dy, dt, interval, T_init, T_in, steps,
               H_s, n_rock=N_ROCK, theta=1.0):
    """
    Fully-implicit (theta=1) or Crank-Nicolson integration of the coupled
    fluid + multi-layer granite system.

    theta defaults to 1.0 to suppress stiffness oscillations arising from the
    fast fluid-rock exchange eigenvalue  λ·dt >> 1.

    State-vector length : (1 + 2*n_rock) * N

    Returns
    -------
    times, fields, rock_fields, E_arr, dE_arr, dF_arr
        rock_fields : mean temperature across all ACTIVE rock layers
                      (l = 0 … n_rock-2, both sides)
    """
    nx, ny    = b.shape
    N         = nx * ny
    N_total   = N * (1 + 2 * n_rock)
    H_s_layer = H_s / n_rock
    n_active  = n_rock - 1       # active (non-Dirichlet) layers per side

    T_vec = np.full(N_total, float(T_init))

    L, f_bc = _build_transport_matrix(b, u_face, v_face, dx, dy, H_s, n_rock)

    bv  = b.ravel()
    B_f = diags(bv                     / dt)
    B_s = diags(np.full(N, H_s_layer)  / dt)
    B   = block_diag([B_f] + [B_s] * (2 * n_rock), format='csr')

    A_lil = (B + theta       * L).tolil()
    M_lil = (B - (1 - theta) * L).tolil()

    # ── fluid inlet Dirichlet (rows 0 … ny-1) ────────────────────────────
    for j in range(ny):
        A_lil[j, :] = 0;  A_lil[j, j] = 1.0
        M_lil[j, :] = 0

    # ── outer rock Dirichlet (l = n_rock-1 on both sides) ────────────────
    # Collect all outer-layer row indices once; reuse in time loop.
    outer_rows = []
    for side in range(2):
        offset = N * (1 + side * n_rock + (n_rock - 1))
        for cell in range(N):
            k = offset + cell
            outer_rows.append(k)
            A_lil[k, :] = 0;  A_lil[k, k] = 1.0
            M_lil[k, :] = 0
    outer_rows = np.asarray(outer_rows, dtype=int)

    A = csr_matrix(A_lil)
    M = csr_matrix(M_lil)
    del A_lil, M_lil    # free memory
    A_lu = splu(A.tocsc())
    f_rhs = f_bc * T_in

    # helper: mean of active rock layers → (nx, ny) for snapshots
    def mean_active_rock(T_vec):
        T_sum = np.zeros(N)
        for side in range(2):
            for layer in range(n_active):
                off = N * (1 + side * n_rock + layer)
                T_sum += T_vec[off : off + N]
        return (T_sum / (2 * n_active)).reshape(nx, ny)

    steps_per_snap = max(1, int(round(interval / dt)))
    E0     = stored_energy(T_vec, b, dx, dy, H_s, n_rock)
    cum_dE = 0.0
    cum_dF = 0.0
    slice_j=int(ny/2)
    times       = [0.0]
    fields      = [T_vec[:N].reshape(nx, ny).copy()]
    rock_fields = [mean_active_rock(T_vec).copy()]
    innermost_fields = [T_vec[N:2*N].reshape(nx, ny).copy()]       # bottom l=0
    side_T           = [_make_side_slice(T_vec, N, nx, ny, n_rock, slice_j)]
    E_arr  = [E0]
    dE_arr = [0.0]
    dF_arr = [0.0]

    for step in range(steps):
        T_old  = T_vec.copy()
        T_in_t = ramp_T_in((step + 1) * dt, T_init, T_in, dt * steps * 0.2)

        rhs              = M @ T_old + f_rhs
        rhs[:ny]         = T_in_t    # fluid inlet Dirichlet
        rhs[outer_rows]  = T_init    # outer rock Dirichlet

        T_vec = A_lu.solve(rhs)

        dE = (stored_energy(T_vec, b, dx, dy, H_s, n_rock)
              - stored_energy(T_old, b, dx, dy, H_s, n_rock))
        dF = dt * matrix_heating_power(
                T_vec, T_old, L, f_bc, T_in_t,
                b, dx, dy, H_s, n_rock, theta=theta)
        cum_dE += dE
        cum_dF += dF

        if (step + 1) % steps_per_snap == 0:
            t_now   = (step + 1) * dt
            T_fluid = T_vec[:N].reshape(nx, ny)
            T_rock  = mean_active_rock(T_vec)
            T_inner = T_vec[N:2*N].reshape(nx, ny)
            rel     = abs(cum_dE - cum_dF) / max(abs(cum_dE), 1e-30)

            times.append(t_now)
            fields.append(T_fluid.copy())
            rock_fields.append(T_rock.copy())
            innermost_fields.append(T_inner.copy())
            side_T.append(_make_side_slice(T_vec, N, nx, ny, n_rock, slice_j))
            E_arr.append(E0 + cum_dE)
            dE_arr.append(cum_dE)
            dF_arr.append(cum_dF)

    return (times, fields, innermost_fields, rock_fields, side_T,
            np.array(E_arr), np.array(dE_arr), np.array(dF_arr))

def _worker_ensemble_run(task_payload):
    """
    Standalone worker for the ensemble sweep. 
    Runs a single simulation for a specific (sigma, correl, seed) combination.
    """
    sigma, correl, seed, config = task_payload


    



    # Unpack static configuration
    nx, ny = config['nx'], config['ny']
    Lx, Ly = config['Lx'], config['Ly']
    dx, dy = config['dx'], config['dy']
    p_in, p_out, mu = config['p_in'], config['p_out'], config['mu']
    H_s, n_rock = config['H_s'], config['n_rock']
    dt, steps = config['dt'], config['steps']
    T_init, T_in, theta = config['T_init'], config['T_in'], config['theta']
    snap_interval = dt

    # 1. Generate Aperture Field
    gen = ApertureGenerator(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        b_mean=1e-4, b_sigma=sigma,
        variogram='exponential',
        range_x=2*correl, range_y=correl,
        angle_deg=0.0, nugget_frac=0.05,
    )
    b = gen.generate(seed=seed)
    b = np.clip(b, 0.05 * b.mean(), None)

    # 2. Flow Field Calculations
    p, u_cell, v_cell, q, u_face, v_face = solve_pressure(
        b, dx, dy, p_in=p_in, p_out=p_out, mu=mu
    )

    Q_flow = np.sum(u_face[-1, :] * b[-1, :] * dy)
    pore_volume = np.mean(b) * Lx * Ly
    # Dynamically calculate how many steps are needed to reach a target PVI (e.g., 1.5)
    # target_pvi = 0.05
    # total_simulation_time = (target_pvi * pore_volume) / Q_flow
    # dynamic_steps = int(np.ceil(total_simulation_time / dt))
    # 3. Thermal Transport Simulation
    (times, fields, innermost_fields, rock_fields, side_T,
     E_arr, dE_arr, dF_arr) = solve_heat(
        b, u_face, v_face, dx, dy,
        dt=dt, interval=snap_interval,
        T_init=T_init, T_in=T_in,
        steps=steps,
        H_s=H_s, n_rock=n_rock, theta=theta,
    )

    # 4. Calculate PVI and h_eff
    times_arr = np.asarray(times)
    h_eff, _, _ = compute_interface_h_eff(
        times_arr, fields, innermost_fields, b, q, Lx, Ly, plot=False
    )

    n_pts = min(len(times_arr), len(h_eff))
    times_arr = times_arr[:n_pts]
    h_eff = np.asarray(h_eff)[:n_pts]

    pvi = (Q_flow * times_arr) / pore_volume

    # Return the identifiers along with the data arrays so the master function 
    # knows which subplot this data belongs to.
    return (sigma, correl, seed, pvi, h_eff)

def run_ensemble_parameter_sweep(nx, ny, Lx, Ly, dx, dy,
                                 p_in, p_out, mu, H_s, n_rock,
                                 dt, steps, T_init, T_in, theta):
    """
    Runs a parallelized parameter sweep over sigma and correlation lengths using 
    an ensemble of multiple random seeds. Plots h_eff vs PVI.
    """
    b_sigmas = [0.25, 0.5, 1.5, 4.0]          
    correls = [0.005, 0.020,0.040]        
    seeds = [42, 99, 123, 555,678]      
 
    # Package static arguments into a single config dictionary
    config = {
        'nx': nx, 'ny': ny, 'Lx': Lx, 'Ly': Ly, 'dx': dx, 'dy': dy,
        'p_in': p_in, 'p_out': p_out, 'mu': mu, 'H_s': H_s, 'n_rock': n_rock,
        'dt': dt, 'steps': steps, 'T_init': T_init, 'T_in': T_in, 'theta': theta
    }

    # Generate a flat list of all simulation tasks
    all_tasks = []
    for sigma in b_sigmas:
        for correl in correls:
            for seed in seeds:
                all_tasks.append((sigma, correl, seed, config))

    total_runs = len(all_tasks)
    print(f"Beginning Parallel Ensemble Evaluation across {total_runs} total model runs...")

    # Dictionary to group results back together by their (sigma, correl) parameters
    # Format: {(sigma, correl): {'all_pvi': [], 'all_h_eff': []}}
    grouped_results = {(s, c): {'all_pvi': [], 'all_h_eff': []} for s in b_sigmas for c in correls}

    # Execute tasks in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(_worker_ensemble_run, all_tasks)
        
        for i, res in enumerate(results_iterator):
            ret_sigma, ret_correl, ret_seed, ret_pvi, ret_h_eff = res
            
            # Store the resulting arrays in the correct group
            grouped_results[(ret_sigma, ret_correl)]['all_pvi'].append(ret_pvi)
            grouped_results[(ret_sigma, ret_correl)]['all_h_eff'].append(ret_h_eff)
            
            print(f"  Ensemble Progress: {i + 1}/{total_runs} complete.", flush=True)
    # ---------------------------------------------------------
    # Plotting Sequence
    # ---------------------------------------------------------
    fig, axes = plt.subplots(len(b_sigmas), len(correls), figsize=(12, 8), sharex=True, sharey=True)
    fig.patch.set_facecolor('#1a1a2e')
 
    for row_idx, sigma in enumerate(b_sigmas):
        for col_idx, correl in enumerate(correls):
            ax = axes[row_idx, col_idx] if len(b_sigmas) > 1 else axes[col_idx]
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.15)
 
            # Retrieve the grouped data for this specific subplot
            all_pvi = grouped_results[(sigma, correl)]['all_pvi']
            all_h_eff = grouped_results[(sigma, correl)]['all_h_eff']
 
            # Interpolate onto a uniform PVI grid to compute ensemble averages safely
            max_pvi_limit = min(pvi_arr[-1] for pvi_arr in all_pvi)
            pvi_uniform   = np.linspace(0.01, max_pvi_limit, steps)
 
            h_eff_interp_list = []
            for i in range(len(all_pvi)):
                xp = all_pvi[i]
                fp = all_h_eff[i]
 
                # Filter out NaNs / Infs
                valid_mask = np.isfinite(xp) & np.isfinite(fp)
                xp_clean   = xp[valid_mask]
                fp_clean   = fp[valid_mask]
 
                # Ensure strictly monotonically increasing x for np.interp
                _, unique_indices = np.unique(xp_clean, return_index=True)
                unique_indices = np.sort(unique_indices)
 
                xp_final = xp_clean[unique_indices]
                fp_final = fp_clean[unique_indices]
 
                h_interp = np.interp(pvi_uniform, xp_final, fp_final)
                h_eff_interp_list.append(h_interp)
 
            h_eff_interp = np.array(h_eff_interp_list)
 
            # Compute statistical distributions
            h_mean = np.mean(h_eff_interp, axis=0)
            h_std  = np.std(h_eff_interp, axis=0)
            total=h_mean.mean()
            # Plot ensemble metrics
            ax.plot(pvi_uniform, h_mean, color='#4fc3f7', lw=2.5, label='Ensemble Mean')
            ax.fill_between(pvi_uniform, h_mean - h_std, h_mean + h_std,
                            color='#4fc3f7', alpha=0.15, label='$\\pm$1 Std. Dev.')
 
            for spine in ax.spines.values():
                spine.set_edgecolor('#ffffff33')
            ax.set_title(f'$\\sigma_b$={sigma} | $\\lambda_y$={correl*1e3:.1f}mm: Average = {total:.0f}', color='white', fontsize=10)
            ax.set_yscale('log')
 
    # Matrix-wide axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Pore Volumes Injected (PVI) [—]', color='white')
    for ax in axes[:, 0]:
        ax.set_ylabel('Effective $h$ [W/m²K]', color='white')
 
    axes[0, 0].legend(facecolor='#2a2a4e', edgecolor='#ffffff11', labelcolor='white', loc='best', fontsize=8)
    plt.suptitle('Ensemble Averaged $h_{eff}$ Response vs Dimensionless Volumetric Throughput',
                 color='white', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Paramters 
    Lx, Ly = 0.80, 0.80
    nx, ny  = 120, 120
    dx, dy  = Lx / nx, Ly / ny
    mu      = 1e-3
    p_in    = 50000.0
    p_out   = 0.0
    H_s     = 0.05
    n_rock  = 5
    dt      = 10
    steps   = 100
    T_init  = 100.0
    T_in    = 50.0
    theta   = 1
    b_mean =1.0e-4
    b_std=1.0
    """
    Parameter Sweeps + Sensitivity Analysis
    WARNING: Will run a lot of simulations so be careful. Parallelized.
    """

    # # Protocol 1: Execute Ensemble Sweeps normalized by PVI
    # print("Running Ensemble Parameter Analysis...")
    # run_ensemble_parameter_sweep(
    #     nx, ny, Lx, Ly, dx, dy,            # FIX 1: pass dx, dy
    #     p_in, p_out, mu, H_s, n_rock, dt, steps, T_init, T_in, theta
    # )
 
    # Protocol 2: Execute Global Sensitivity Matrix
    # print("\nInitializing Sobol Experimental Design...")
    # s1, st = compute_sobol_indices(
    #     nx, ny, Lx, Ly, dx, dy,            # FIX 1: pass dx, dy
    #     p_in, p_out, mu, H_s, n_rock, dt, steps, T_init, T_in, theta
    # )



    """
    Uncomment this block to run a single case
    """
    # gen = ApertureGenerator(
    # nx=nx, ny=ny, Lx=Lx, Ly=Ly,
    # b_mean=1e-4, b_sigma=.6,
    # variogram='exponential',
    # range_x=0.02, range_y=0.01,
    # angle_deg=30.0, nugget_frac=0.0,
    # )
    # b   = gen.generate(seed=101)
    # b = np.clip(b, 0.05 * b.mean(), None)

    b= generate_field(nx,ny,dx,dy,b_mean,b_std,L_c=.1,angle=7*np.pi/16)
    p, u_cell, v_cell, q, u_face, v_face = solve_pressure(
        b, dx, dy, p_in=p_in, p_out=p_out, mu=mu)
    plot_results(b, p, u_cell, v_cell, q, Lx, Ly,
                 title='2D fracture flow — variable aperture (FVM)')

    (times, fields, innermost_fields, rock_fields, side_T,
    E_arr, dE_arr, dF_arr) = solve_heat(
    b, u_face, v_face, dx, dy,
    T_init=100.0, T_in=50.0, dt=10, interval=1, steps=100,
    H_s=H_s, n_rock=n_rock, theta=0.5,
    )

  


    ani = animate_heat_three_views(
        times, fields, innermost_fields, side_T,
        Lx, Ly, H_s, n_rock,
        interval=150, fps=10
    )
    # # report = check_energy_conservation(
    # #     times, b, H_s=H_s,
    # #     E_arr=E_arr, dE_arr=dE_arr, dF_arr=dF_arr,
    # #     n_rock=n_rock, theta=1.0,
    # #     verbose=True, plot=True,
    # # )


    # h_eff, Q_xfer, dT = compute_interface_h_eff(
    #     times, fields, innermost_fields, b, q, Lx, Ly
    # )