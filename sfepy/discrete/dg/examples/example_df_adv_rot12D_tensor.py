import numpy as nm
from  numpy.linalg import norm
import matplotlib.pyplot as plt


# sfepy imports
from sfepy.discrete.fem import Mesh
from sfepy.discrete.fem.meshio import UserMeshIO
from sfepy.base.base import Struct
from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete.conditions import InitialCondition, EssentialBC, Conditions
from sfepy.terms.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.solvers.ts_solvers import SimpleTimeSteppingSolver
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.discrete.fem.meshio import VTKMeshIO
from sfepy.base.conf import ProblemConf
from sfepy.terms.terms_dot import ScalarDotMGradScalarTerm, DotProductVolumeTerm


# local import
from dg_terms import AdvFluxDGTerm, ScalarDotMGradScalarDGTerm
# from dg_equation import Equation
from dg_tssolver import EulerStepSolver, DGTimeSteppingSolver, RK3StepSolver
from dg_field import DGField

from my_utils.inits_consts import left_par_q, gsmooth, const_u, ghump, superic

mesh = gen_block_mesh((1., 1.), (100, 2), (.5, 0.5))
pi = nm.pi
coors = mesh.coors
angle = pi/5
rotm = nm.array([[nm.cos(angle),  -nm.sin(angle)],
                 [nm.sin(angle),  nm.cos(angle)]]).T
rotcoors = nm.sum(coors[..., None] * rotm[None, :, :], axis=-2)


rotmesh = gen_block_mesh((1., 1.), (100, 2), (.5, 0.5), coors=rotcoors)
outfile = "output/mesh/tensor_rot12D_mesh.vtk"
meshio = VTKMeshIO(outfile)
meshio.write(outfile, rotmesh)

mesh = rotmesh

#vvvvvvvvvvvvvvvv#
approx_order = 2
CFL = 1.
#^^^^^^^^^^^^^^^^#

velo = nm.array([[-1., 0.]]).T
max_velo = nm.max(nm.abs(velo))

t0 = 0
t1 = 2

dx = nm.min(mesh.cmesh.get_volumes(2))
dt = dx / norm(velo) * CFL/(2*approx_order + 1)
# time_steps_N = int((tf - t0) / dt) * 2
tn = int(nm.ceil((t1 - t0) / dt))
dtdx = dt / dx
print("Space divided into {0} cells, {1} steps, step size is {2}".format(mesh.n_el, len(mesh.coors), dx))
print("Time divided into {0} nodes, {1} steps, step size is {2}".format(tn - 1, tn, dt))
print("Courant number c = max(abs(u)) * dt/dx = {0}".format(max_velo * dtdx))


integral = Integral('i', order=5)

domain = FEDomain('domain_tensor_12D', mesh)
omega = domain.create_region('Omega', 'all')
field = DGField('dgfu', nm.float64, 'scalar', omega,
                approx_order=approx_order)

u = FieldVariable('u', 'unknown', field, history=1)
v = FieldVariable('v', 'test', field, primary_var_name='u')

MassT = DotProductVolumeTerm("adv_vol(v, u)", "v, u", integral, omega, u=u, v=v)

a = Material('a', val=[velo])
StiffT = ScalarDotMGradScalarDGTerm("adv_stiff(a.val, u, v)", "a.val, u, v", integral, omega,
                                    u=u, v=v, a=a, mode="grad_virtual")

FluxT = AdvFluxDGTerm(integral, omega, u=u, v=v, a=a)

eq = Equation('balance', MassT + StiffT - FluxT)
eqs = Equations([eq])


def ic_wrap(x, ic=None):
    x = nm.sum(x[..., None] * rotm.T[None, :, :], axis=-2)
    return gsmooth(x[..., 0:1])

# X = nm.arange(0, 1, 0.005)
# Y = nm.arange(0, 1, 0.005)
# X, Y = nm.meshgrid(X, Y)
# coors = nm.dstack((X[:, :, None], Y[:, :, None]))
#
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# Z = ic_wrap(coors)
# surf = ax.plot_surface(X, Y, Z[:, :, 0], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False, alpha=.6)
#
# fig = plt.figure()
# ax = fig.gca()
# ax.contour(X, Y, Z[..., 0])
# ax.scatter(rotcoors[:, 0], rotcoors[:, 1])
# plt.show()


ic_fun = Function('ic_fun', ic_wrap)
ics = InitialCondition('ic', omega, {'u.0': ic_fun})

pb = Problem('advection', equations=eqs, conf=Struct(options={"save_times": 100}, ics={},
                                   ebcs={}, epbcs={}, lcbcs={}, materials={}))
pb.setup_output(output_dir="./output/adv_tens_rot12D", output_format="msh")
# pb.set_bcs(ebcs=Conditions([left_fix_u, right_fix_u]))
pb.set_ics(Conditions([ics]))

state0 = pb.get_initial_state()
pb.save_state("output/state0_tensor_rot12D.msh", state=state0)

ls = ScipyDirect({})
nls_status = IndexedStruct()
nls = RK3StepSolver({}, lin_solver=ls, status=nls_status)

tss = DGTimeSteppingSolver({'t0': t0, 't1': t1, 'n_step': tn},
                                nls=nls, context=pb, verbose=True)

#---------
#| Solve |
#---------
pb.set_solver(tss)
pb.solve()