import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, diags, block_diag
from scipy.sparse.linalg import  splu
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from velocity_solver import solve_pressure, plot_results
from hurst import generate_field
from auxillary import animate_heat_three_views, compute_interface_h_eff, stored_energy, matrix_heating_power, _make_side_slice, ramp_T_in
