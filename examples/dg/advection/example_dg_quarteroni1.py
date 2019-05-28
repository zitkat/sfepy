from examples.dg.example_dg_common import *

example_name = "adv_2D_tens"
dim = int(example_name[example_name.index("D") - 1])

filename_mesh = "../mesh/tens_2D_mesh20.vtk"

approx_order = 1
t0 = 0.
t1 = .2
CFL = .4

# get_common(approx_order, CFL, t0, t1, None, get_ic)
angle = 0.0 # - nm.pi / 5
rotm = nm.array([[nm.cos(angle), -nm.sin(angle)],
                 [nm.sin(angle), nm.cos(angle)]])
velo = nm.sum(rotm.T * nm.array([1., 1.]), axis=-1)[:, None]
materials = {
    'a': ({'val': [velo], '.flux': 0.0},),
}

regions = {
    'Omega'     : 'all',
    'left' : ('vertices in x == 0', 'edge'),
    'right': ('vertices in x == 1', 'edge'),
    'top' : ('vertices in y == 1', 'edge'),
    'bottom': ('vertices in y == 0', 'edge')
}

fields = {
    'density': ('real', 'scalar', 'Omega', str(approx_order) + 'd', 'DG', 'legendre')  #
}

variables = {
    'u': ('unknown field', 'density', 0, 1),
    'v': ('test field', 'density', 'u'),
}

@local_register_function
def get_ic(x, ic=None):
    return gsmooth(x[..., 0:1]) * gsmooth(x[..., 1:])

ics = {
    'ic': ('Omega', {'u.0': 'get_ic'}),
}

integrals = {
    'i': 2 * approx_order,
}

equations = {
    'Advection': """
                   dw_volume_dot.i.Omega(v, u)
                   + dw_s_dot_mgrad_s.i.Omega(a.val, u, v)
                   - dw_dg_advect_laxfrie_flux.i.Omega(a.flux, a.val, v, u) = 0
                  """
}

solvers = {
    "tss": ('ts.tvd_runge_kutta_3',
            {"t0"     : t0,
             "t1"     : t1,
             'limiter': IdentityLimiter,
             'verbose': True}),
    'nls': ('nls.newton', {}),
    'ls' : ('ls.scipy_direct', {})
}

options = {
    'ts'              : 'tss',
    'nls'             : 'newton',
    'ls'              : 'ls',
    'save_times'      : 100,
    'active_only'     : False,
    'output_format'   : 'msh',
    'pre_process_hook': get_cfl_setup(CFL)
}
