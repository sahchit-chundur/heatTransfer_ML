import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# ── physical constants ────────────────────────────────────────────────────────
KAPPA   = 1.4e-7      # fluid  thermal diffusivity  [m²/s]
RHO_C   = 4.182e6     # fluid  ρc                   [J/m³/K]
KAPPA_S = 1.5e-6      # granite thermal diffusivity [m²/s]
RHO_C_S = 2.16e6      # granite ρc                  [J/m³/K]
N_ROCK  = 5    

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def animate_heat_three_views(times, fields, innermost_fields, side_T,
                              Lx, Ly, H_s, n_rock,
                              slice_j=None, cmap='inferno',
                              interval=150, save_path=None, fps=10):
    """
    Three-panel animation:
      Left   : fluid temperature plan view   (x-y)
      Centre : innermost rock layer (l=0)    plan view   (x-y)
      Right  : side-view x-z slice at y = slice_j

    The side view shows the full cross-section from the top outer rock
    boundary (Dirichlet T_init) through the fracture to the bottom outer
    boundary.  Because the physical aperture b << H_s, the fluid strip is
    drawn with the same visual height as one rock sublayer so it remains
    visible.

    Parameters
    ----------
    fields           : list of (nx, ny)           fluid snapshots
    innermost_fields : list of (nx, ny)           rock l=0 snapshots
    side_T           : list of (nx, 1+2*n_rock)   x-z cross-section data
                         col 0           : fluid at y=slice_j
                         cols 1..n_rock  : bottom rock l=0..n_rock-1
                         cols n_rock+1.. : top rock    l=0..n_rock-1
    Lx, Ly           : domain dimensions [m]
    H_s              : total rock thickness per side [m]
    n_rock           : sublayers per side
    slice_j          : y-index of the side-view slice (default ny//2)
    """
    nx, ny    = fields[0].shape
    H_s_layer = H_s / n_rock

    if slice_j is None:
        slice_j = ny // 2

    # ── plan-view coordinates [mm] ────────────────────────────────────────
    xc = (np.arange(nx) + 0.5) * Lx / nx * 1e3
    yc = (np.arange(ny) + 0.5) * Ly / ny * 1e3
    X, Y = np.meshgrid(xc, yc, indexing='ij')

    # ── side-view coordinates ─────────────────────────────────────────────
    # Visual layer height = H_s_layer [mm] for all layers including fluid.
    # z = 0 at the top edge of the fluid strip (fluid/top-rock interface).
    # z increases upward into the top rock, decreases downward into bottom rock.
    #
    # Display order top → bottom:
    #   top rock l = n_rock-1 (outermost, Dirichlet)
    #   ...
    #   top rock l = 0 (innermost)
    #   fluid  (one visual layer thick)
    #   bot rock l = 0 (innermost)
    #   ...
    #   bot rock l = n_rock-1 (outermost, Dirichlet)
    h_vis   = H_s_layer * 1e3                      # [mm] visual height per row
    n_total = 1 + 2 * n_rock
    z_top   =  n_rock * h_vis                       # top outer edge [mm]
    z_bot   = -(n_rock + 1) * h_vis                 # bottom outer edge [mm]
    z_edges = np.linspace(z_top, z_bot, n_total + 1)

    # x edges for pcolormesh
    xe = np.concatenate([
        [xc[0]  - (xc[1]  - xc[0])  / 2],
        (xc[:-1] + xc[1:]) / 2,
        [xc[-1] + (xc[-1] - xc[-2]) / 2],
    ])   # (nx+1,)

    Xsv, Zsv = np.meshgrid(xe, z_edges)   # (n_total+1, nx+1)

    def _reorder_sv(sv):
        """
        Reorder side_T to top-to-bottom display order.
        Input  sv : (nx, 1+2*n_rock)
        Output    : (n_total, nx)  — top outer at row 0
        """
        fluid = sv[:, 0:1]                   # (nx, 1)
        bot   = sv[:, 1:n_rock + 1]          # (nx, n_rock)  l=0..n_rock-1
        top   = sv[:, n_rock + 1:]           # (nx, n_rock)  l=0..n_rock-1
        # Reverse top so outermost (Dirichlet) is at top of plot
        return np.hstack([top[:, ::-1], fluid, bot]).T   # (n_total, nx)

    # ── colour limits ─────────────────────────────────────────────────────
    all_f  = np.concatenate([f.ravel() for f in fields])
    all_ir = np.concatenate([f.ravel() for f in innermost_fields])
    all_sv = np.concatenate([s.ravel() for s in side_T])
    T_min  = min(all_f.min(),  all_ir.min(),  all_sv.min())
    T_max  = max(all_f.max(),  all_ir.max(),  all_sv.max())

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax_f, ax_r, ax_s) = plt.subplots(1, 3, figsize=(21, 5))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in (ax_f, ax_r, ax_s):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.set_xlabel('x [mm]', color='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#ffffff33')

    ax_f.set_ylabel('y [mm]', color='white');  ax_f.set_aspect('equal')
    ax_r.set_ylabel('y [mm]', color='white');  ax_r.set_aspect('equal')
    ax_s.set_ylabel('z [mm]  (visual scale — each row = one sublayer)',
                    color='white')

    # Plan-view meshes
    im_f = ax_f.pcolormesh(X, Y, fields[0], cmap=cmap, shading='auto',
                           vmin=T_min, vmax=T_max)
    im_r = ax_r.pcolormesh(X, Y, innermost_fields[0], cmap=cmap, shading='auto',
                           vmin=T_min, vmax=T_max)
    y_slice_mm = yc[slice_j]
    line_f = ax_f.axhline(y_slice_mm, color='white', lw=1.2, ls='--', alpha=0.8)
    line_r = ax_r.axhline(y_slice_mm, color='white', lw=1.2, ls='--', alpha=0.8)
    # Side-view mesh
    im_s = ax_s.pcolormesh(Xsv, Zsv, _reorder_sv(side_T[0]),
                           cmap=cmap, shading='auto',
                           vmin=T_min, vmax=T_max)

    # Mark fluid / rock interfaces
    ax_s.axhline(0.0,    color='white', lw=1.2, ls='--', alpha=0.8)
    ax_s.axhline(-h_vis, color='white', lw=1.2, ls='--', alpha=0.8)

    # Region labels (placed at right margin)
    x_lbl = xc[-1] * 0.97
    ax_s.text(x_lbl,  h_vis * 0.5,   'Top rock', color='white',
              fontsize=7, ha='right', va='center')
    ax_s.text(x_lbl, -h_vis * 0.5,   'Fluid',    color='white',
              fontsize=7, ha='right', va='center')
    ax_s.text(x_lbl, -h_vis * 1.5,   'Bot rock', color='white',
              fontsize=7, ha='right', va='center')

    # Colorbars
    for im, ax, lbl in ((im_f, ax_f, 'Fluid T [°C]'),
                        (im_r, ax_r, 'Rock l=0 T [°C]'),
                        (im_s, ax_s, 'T [°C]')):
        cb = fig.colorbar(im, ax=ax, label=lbl, fraction=0.03)
        cb.ax.yaxis.label.set_color('white')
        cb.ax.tick_params(colors='white')

    y_slice_mm = yc[slice_j]
    ttl_f = ax_f.set_title('Fluid', color='white', fontsize=10)
    ttl_r = ax_r.set_title('Rock — innermost layer (l=0)', color='white', fontsize=10)
    ttl_s = ax_s.set_title(f'Side view   y = {y_slice_mm:.1f} mm',
                            color='white', fontsize=10)
    fig.suptitle(f't = {times[0]:.1f} s', color='white', fontsize=11)
    plt.tight_layout()

    def update(frame):
        im_f.set_array(fields[frame].ravel())
        im_r.set_array(innermost_fields[frame].ravel())
        im_s.set_array(_reorder_sv(side_T[frame]).ravel())

        Tf = fields[frame]
        Tr = innermost_fields[frame]
        ttl_f.set_text(f'Fluid  ⟨T⟩={Tf.mean():.1f}°C  '
                       f'T_out={Tf[-1, :].mean():.1f}°C')
        ttl_r.set_text(f'Rock l=0  ⟨T⟩={Tr.mean():.1f}°C  '
                       f'T_out={Tr[-1, :].mean():.1f}°C')
        fig.suptitle(f't = {times[frame]:.1f} s', color='white', fontsize=11)
        return [im_f, im_r, im_s, ttl_f, ttl_r,line_f,line_r]

    ani = animation.FuncAnimation(fig, update, frames=len(fields),
                                  interval=interval, blit=True, repeat=True)
    if save_path is not None:
        ext    = save_path.rsplit('.', 1)[-1].lower()
        writer = (animation.FFMpegWriter(fps=fps, bitrate=1800)
                  if ext == 'mp4' else animation.PillowWriter(fps=fps))
        ani.save(save_path, writer=writer, dpi=150)
    plt.show()
    return ani
# ─────────────────────────────────────────────────────────────────────────────
# Energy conservation check
# ─────────────────────────────────────────────────────────────────────────────

def check_energy_conservation(
        times, b, H_s,
        fields=None, rock_fields=None,
        u_face=None, v_face=None,
        dx=None, dy=None, T_in=None,
        *, E_arr=None, dE_arr=None, dF_arr=None,
        n_rock=N_ROCK, theta=1.0,
        rho_c=RHO_C, rho_c_s=RHO_C_S,
        verbose=True, plot=True):

    times  = np.asarray(times, dtype=float)
    nsnaps = len(times)

    if E_arr is not None and dE_arr is not None and dF_arr is not None:
        E_stored  = np.asarray(E_arr,  dtype=float)
        dE_stored = np.asarray(dE_arr, dtype=float)
        dE_flux   = np.asarray(dF_arr, dtype=float)
        E_fluid = E_rock = Q_series = None

    # else:
    #     if fields is None or u_face is None or v_face is None:
    #         raise ValueError(
    #             'Provide (E_arr, dE_arr, dF_arr) or '
    #             '(fields, u_face, v_face, dx, dy, T_in).')
    #     if len(fields) != nsnaps:
    #         raise ValueError('len(times) must equal len(fields).')

    #     L, f_coef = _build_transport_matrix(
    #                     b, u_face, v_face, dx, dy, H_s, n_rock)
    #     E_stored = np.zeros(nsnaps)
    #     E_fluid  = np.zeros(nsnaps)
    #     E_rock   = np.zeros(nsnaps)
    #     dE_flux  = np.zeros(nsnaps)
    #     Q_series = np.zeros(nsnaps)

    #     for n, T_vec in enumerate(fields):
    #         T_vec = np.asarray(T_vec, dtype=float).ravel()
    #         E_stored[n] = stored_energy(T_vec, b, dx, dy, H_s, n_rock,
    #                                     rho_c, rho_c_s)
    #         ef, er = stored_energy_split(T_vec, b, dx, dy, H_s, n_rock,
    #                                      rho_c, rho_c_s)
    #         E_fluid[n] = ef;  E_rock[n] = er

    #     dE_stored = E_stored - E_stored[0]
    #     for n in range(1, nsnaps):
    #         dt_n = float(times[n] - times[n-1])
    #         Q_series[n] = matrix_heating_power(
    #             fields[n], fields[n-1], L, f_coef, T_in,
    #             b, dx, dy, H_s, n_rock, theta=theta, rho_c=rho_c,
    #             rho_c_s=rho_c_s)
    #         dE_flux[n] = dE_flux[n-1] + dt_n * Q_series[n]

    abs_error   = np.abs(dE_stored - dE_flux)
    scale       = max(float(np.max(np.abs(dE_stored))), 1.0)
    rel_error   = abs_error / scale
    max_rel_err = float(rel_error[1:].max()) if nsnaps > 1 else 0.0

    if verbose:
        print('\n' + '='*65)
        print('  ENERGY CONSERVATION CHECK')
        print('='*65)
        print(f'  Rock sublayers per side : {n_rock}  '
              f'(active: {n_rock-1}, Dirichlet outer: 1)')
        print(f'  Initial stored energy  E₀ = {E_stored[0]:.4e} J')
        print(f'  Final   stored energy  E  = {E_stored[-1]:.4e} J')
        print(f'  Total energy change    ΔE = {dE_stored[-1]:.4e} J')
        print(f'  Cumulative heating     ∫Q = {dE_flux[-1]:.4e} J')
        print(f'  Peak relative error       = {max_rel_err:.4e}')
        print('-'*65)
        stride = max(1, nsnaps // 10)
        print(f"  {'Time [s]':>10}  {'E [J]':>14}  {'ΔE [J]':>14}  "
              f"{'∫Q [J]':>14}  {'Rel err':>9}")
        print('-'*65)
        for n in range(0, nsnaps, stride):
            print(f'  {times[n]:>10.1f}  {E_stored[n]:>14.4e}  '
                  f'{dE_stored[n]:>14.4e}  {dE_flux[n]:>14.4e}  '
                  f'{rel_error[n]:>9.2e}')
        print('='*65 + '\n')

    if plot:
        _plot_energy_conservation(times, E_stored, dE_stored, dE_flux,
                                  rel_error, E_fluid, E_rock)

    return dict(times=times, E_stored=E_stored, dE_stored=dE_stored,
                dE_flux=dE_flux, abs_error=abs_error, rel_error=rel_error,
                max_rel_error=max_rel_err, Q_series=Q_series,
                E_fluid=E_fluid, E_rock=E_rock)


def _plot_energy_conservation(times, E_stored, dE_stored, dE_flux, rel_error,
                              E_fluid=None, E_rock=None, save_path=None):
    has_split = (E_fluid is not None and E_rock is not None)
    ncols = 4 if has_split else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    fig.patch.set_facecolor('#1a1a2e')

    def _style(ax):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#ffffff33')

    for ax in axes:
        _style(ax)

    axes[0].plot(times, E_stored,             'r-',  lw=2,   label='Stored E(t)')
    axes[0].plot(times, E_stored[0]+dE_flux,  'c--', lw=1.5, label='E₀ + ∫Q dt')
    axes[0].set_xlabel('Time [s]'); axes[0].set_ylabel('Thermal energy [J]')
    axes[0].set_title('Stored vs integrated heating')
    axes[0].legend(fontsize=8, facecolor='#2a2a4e', labelcolor='white')
    axes[0].grid(True, alpha=0.15)

    axes[1].plot(times, dE_stored, 'r-',  lw=2,   label='ΔE stored')
    axes[1].plot(times, dE_flux,   'c--', lw=1.5, label='∫Q dt')
    axes[1].set_xlabel('Time [s]'); axes[1].set_ylabel('Cumulative energy [J]')
    axes[1].set_title('Cumulative balance')
    axes[1].legend(fontsize=8, facecolor='#2a2a4e', labelcolor='white')
    axes[1].grid(True, alpha=0.15)

    if len(times) > 1:
        axes[2].semilogy(times[1:], rel_error[1:], 'orange', lw=2)
    axes[2].axhline(1e-3, color='#ffffff44', ls='--', lw=1, label='0.1 %')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('|ΔE − ∫Q| / max|ΔE|')
    axes[2].set_title('Relative conservation error')
    axes[2].legend(fontsize=8, facecolor='#2a2a4e', labelcolor='white')
    axes[2].grid(True, alpha=0.15, which='both')

    if has_split:
        axes[3].plot(times, E_fluid,  color='#4fc3f7', lw=2, label='Fluid')
        axes[3].plot(times, E_rock,   color='#ef9a9a', lw=2, label='Rock (all layers)')
        axes[3].plot(times, E_stored, 'w--', lw=1, alpha=0.5, label='Total')
        axes[3].set_xlabel('Time [s]'); axes[3].set_ylabel('Thermal energy [J]')
        axes[3].set_title('Fluid / rock partition')
        axes[3].legend(fontsize=8, facecolor='#2a2a4e', labelcolor='white')
        axes[3].grid(True, alpha=0.15)

    plt.suptitle('Heat transport — energy conservation',
                 color='white', fontsize=11, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Effective heat-transfer coefficient
# ─────────────────────────────────────────────────────────────────────────────

def compute_interface_h_eff(times, fields, innermost_fields, b, q, Lx, Ly, 
                            k_fluid=0.6, Nu=4.117, plot=True):
    """
    Calculates the macroscopic effective heat transfer coefficient (h_eff) 
    at the fluid-rock interface by integrating the local thermal fluxes.
    
    Parameters
    ----------
    times, fields, innermost_fields : arrays returned by solve_heat
    b : 2D array of apertures
    Lx, Ly : domain dimensions
    """
    times = np.asarray(times)
    nsnaps = len(times)
    nx, ny = b.shape
    dx, dy = Lx / nx, Ly / ny
    
    # Total surface area of the fracture walls (top + bottom)
    A_contact = 2.0 * Lx * Ly  
    
    # Calculate the static local heat transfer coefficient matrix [W/m^2/K]
    # h_local = (Nu * k_f) / D_h  --> where D_h = 2*b
    h_local = (Nu * k_fluid) / (2.0 * b)
    
    h_eff   = np.full(nsnaps, np.nan)
    Q_total = np.zeros(nsnaps)
    dT_mean = np.zeros(nsnaps)
    dT_flowweighted = np.zeros(nsnaps)  # ← add this
    for n in range(nsnaps):
        T_f    = fields[n]
        T_wall = innermost_fields[n] # l=0 rock layer (skin temperature)
        
        # Local temperature difference driving heat into the rock
        dT_local = T_f - T_wall
        dT_flowweighted[n] = np.sum(q * dT_local * dx * dy) / np.sum(q * dx * dy)
        # Total heat flux [W] across both top and bottom interfaces
        # Q = sum( 2_sides * h_local * A_cell * dT_local )
        Q_total[n] = np.sum(2.0 * h_local * dT_local * dx * dy)
        
        # Global mean temperature difference
        dT_mean[n] = np.mean(T_f) - np.mean(T_wall)
        
    # Calculate global h_eff [W/m^2/K], avoiding division by zero at t=0
    valid = np.abs(dT_flowweighted) > 1e-6
    h_eff[valid] = Q_total[valid] / (A_contact * dT_flowweighted[valid])
    total = h_eff[valid].mean()
    if plot:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax1.set_facecolor('#1a1a2e')
        ax1.tick_params(colors='white')
        for sp in ax1.spines.values():
            sp.set_edgecolor('#ffffff33')
            
        ax1.plot(times[valid], h_eff[valid], color='#4fc3f7', lw=2, label='Global $h_{eff}$')
        
        # Plot the theoretical bounds based on the geometry
        h_min, h_max = h_local.min(), h_local.max()
        ax1.axhline(np.mean(h_local), color='#ff8a65', ls='--', lw=1.5, 
                    label=f'Spatial Mean of $h_{{local}}$ ({np.mean(h_local):.0f})')
        
        ax1.set_xlabel('Time [s]', color='white')
        ax1.set_ylabel('Effective $h$ [W/m²K]', color='#4fc3f7')
        ax1.set_title(f'Interface Heat Transfer Coefficient over Time: Average ={total}', color='white')
        ax1.legend(fontsize=9, facecolor='#2a2a4e', labelcolor='white', loc='best')
        ax1.grid(True, alpha=0.15)
        
        plt.tight_layout()
        plt.show()

    return h_eff[valid], Q_total, dT_flowweighted

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def stored_energy(T_vec, b, dx, dy, H_s, n_rock=N_ROCK,
                  rho_c=RHO_C, rho_c_s=RHO_C_S):
    """
    Total thermal energy [J] in the domain.

    Fluid  : inlet cells (i=0) excluded — Dirichlet-prescribed.
    Rock   : all layers included (outer Dirichlet cells add only a constant
             offset that cancels in ΔE, so inclusion is harmless).
    """
    nx, ny    = b.shape
    N         = b.size
    H_s_layer = H_s / n_rock

    # fluid (skip inlet strip)
    bv_int  = b[1:, :].ravel()
    T_f_int = T_vec[ny:N]
    E_f     = rho_c * np.dot(bv_int, T_f_int) * dx * dy

    # rock: sum over all sides and all layers
    E_rock = 0.0
    for side in range(2):
        for layer in range(n_rock):
            offset = N * (1 + side * n_rock + layer)
            T_l    = T_vec[offset : offset + N]
            E_rock += rho_c_s * H_s_layer * np.sum(T_l) * dx * dy

    return float(E_f + E_rock)

def matrix_heating_power(T_vec, T_prev_vec, L, f_coef, T_in, b, dx, dy, H_s,
                          n_rock=N_ROCK, theta=1.0,
                          rho_c=RHO_C, rho_c_s=RHO_C_S):
    """
    Net power [W] entering the domain, consistent with the CN/implicit
    discretisation.
    """
    N   = b.size
    vol = dx * dy

    rho_c_vec = np.concatenate([
        np.full(N,            rho_c),    # fluid
        np.full(N * 2*n_rock, rho_c_s), # all rock layers (both sides)
    ])

    T_flat      = np.asarray(T_vec,      dtype=float).ravel()
    T_prev_flat = np.asarray(T_prev_vec, dtype=float).ravel()

    residual = (theta       * (-L @ T_flat)
                + (1-theta) * (-L @ T_prev_flat)
                + f_coef    * T_in)

    return float(vol * np.dot(rho_c_vec, residual))

def _make_side_slice(T_vec, N, nx, ny, n_rock, j):
    """
    Extract a 2D x-z cross-section at y = j through every layer.

    Returns array of shape (nx, 1 + 2*n_rock):
        col 0              : fluid
        cols 1 .. n_rock   : bottom rock  l = 0 (innermost) .. n_rock-1
        cols n_rock+1 ..   : top    rock  l = 0 (innermost) .. n_rock-1
    """
    cols = [T_vec[:N].reshape(nx, ny)[:, j]]
    for side in range(2):
        for layer in range(n_rock):
            off = N * (1 + side * n_rock + layer)
            cols.append(T_vec[off:off + N].reshape(nx, ny)[:, j])
    return np.column_stack(cols)   # (nx, 1+2*n_rock)

def ramp_T_in(t, T_init, T_in, ramp_time):
    """Cosine ramp from T_init to T_in over ramp_time seconds."""
    if t >= ramp_time:
        return float(T_in)
    alpha = 0.5 * (1.0 - np.cos(np.pi * t / ramp_time))
    return float(T_init + (T_in - T_init) * alpha)