import numpy as nm
from numpy import dot
import matplotlib.pyplot as plt
from numpy import newaxis as nax
import numpy.linalg as nla



class TSSolver:

    def __init__(self, eq, ic, bc, limiter, basis):
        self.equation = eq
        self.mesh = eq.mesh
        self.basis = basis
        self.limiter = limiter
        self.initial_cond = self.sampleIC(self.mesh, ic, self.intGauss2, self.basis)
        self.boundary_cond = bc

    def initialize(self, t0, tend, tsteps):
        dt = float(tend - t0) / tsteps
        dx = nm.max(self.mesh.coors[1:] - self.mesh.coors[:-1])
        dtdx = dt / dx
        maxa = abs(nm.max(self.equation.terms[1].a(self.mesh.coors)))
        print("Space divided into {0} cells, {1} steps, step size is {2}".format(self.mesh.n_el, len(self.mesh.coors),
                                                                                 dx))
        print("Time divided into {0} nodes, {1} steps, step size is {2}".format(tsteps - 1, tsteps, dt))
        print("Courant number c = max(abs(u)) * dt/dx = {0}".format(maxa * dtdx))
        A = nm.zeros((2, self.mesh.n_el, self.mesh.n_el), dtype=nm.float64)
        b = nm.zeros((2, self.mesh.n_el, 1), dtype=nm.float64)
        u = nm.zeros((2, self.mesh.n_el + 2, tsteps, 1), dtype=nm.float64)

        # bc
        u[:, 0, 0] = self.boundary_cond["left"]
        u[:, -1, 0] = self.boundary_cond["right"]
        # ic
        u[:, 1:-1, 0] = self.initial_cond
        return A, b, dt, u

    # Move to problem class?
    def sampleIC(self, mesh, ic, quad, basis):
        sic = nm.zeros((2, self.mesh.n_el, 1), dtype=nm.float64)

        c = (mesh.coors[1:] + mesh.coors[:-1])/2  # center
        s = (mesh.coors[1:] - mesh.coors[:-1])/2  # scale
        sic[0, :] = quad(lambda t: ic(c + t*s))/2
        sic[1, :] = 3*quad(lambda t: t*ic(c + t*s))/2
        return sic

    # Will be taken care of in Integral class
    @staticmethod
    def intGauss2(f):

        x_1 = - nm.sqrt(1./3.)
        x_2 = nm.sqrt(1./3.)

        return f(x_1) + f(x_2)

    @staticmethod
    def intGauss3(f):
        x_0 = 0
        x_1 = - nm.sqrt(3./5.)
        x_2 = nm.sqrt(3./5.)

        w_0 = 8./9.
        w_1 = 5. / 9.

        return w_0 * f(x_0) + w_1 * f(x_1) + w_1 * f(x_2)

    # Separate class for limiters?
    @staticmethod
    def moment_limiter(u):
        """
        Krivodonova(2007): Limiters for high-order discontinuous Galerkin methods

        :param u: solution at time step n
        :return: limited solution
        """

        def minmod(a, b, c):
            seq = (nm.sign(a) == nm.sign(b)) & (nm.sign(b) == nm.sign(c))

            res = nm.zeros(nm.shape(a))
            res[seq] = nm.sign(a[seq]) * nm.minimum.reduce([nm.abs(b[seq]),
                                                            nm.abs(a[seq]),
                                                            nm.abs(c[seq])])

            return res

        idx = nm.arange(nm.shape(u[0, 1:-1])[0])
        nu = nm.copy(u)
        for l in range(1, 0, -1):
            tilu = minmod(nu[l, 1:-1][idx],
                          nu[l-1, 2:][idx] - nu[l-1, 1:-1][idx],
                          nu[l-1, 1:-1][idx] - nu[l-1, :-2][idx])
            idx = tilu != nu
            nu[l, 1:-1][idx] = tilu[idx]
        return nu

    def solve(self, t0, tend, tsteps=10):
        raise NotImplemented


class RK3Solver(TSSolver):
    """
    Runge-Kutta of order 3, with limiter
    """

    def solve(self, t0, tend, tsteps=10):

        A, b, dt, u = self.initialize(t0, tend, tsteps)

        # setup RK3 specific arrays
        u1 = nm.zeros((2, self.mesh.n_el + 2, 1), dtype=nm.float64)
        u2 = nm.zeros((2, self.mesh.n_el + 2, 1), dtype=nm.float64)

        for it in range(1, tsteps):
            # ----1st stage----
            # bcs
            u1[:, 0] = self.boundary_cond["left"]
            u1[:, -1] = self.boundary_cond["right"]

            # get RHS
            A[:] = 0
            b[:] = 0
            self.equation.evaluate(dw_mode="matrix", asm_obj=A, diff_var="u")
            self.equation.evaluate(dw_mode="vector", asm_obj=b, diff_var=None, u=u[:, :, it-1])

            # get update u1
            # maybe use: for more general cases
            #                                     dot(nm.linalg.inv(A[0]), b[0])
            #                                     dot(nm.linalg.inv(A[1]), b[1])
            u1[0, 1:-1] = u[0, 1:-1, it-1] + dt * b[0] / nm.diag(A[0])[:, nax]
            u1[1, 1:-1] = u[1, 1:-1, it-1] + dt * b[1] / nm.diag(A[1])[:, nax]

            # limit
            u1 = self.limiter(u1)

            # ----2nd stage----
            # bcs
            u2[:, 0] = self.boundary_cond["left"]
            u2[:, -1] = self.boundary_cond["right"]

            # get RHS
            A[:] = 0
            b[:] = 0
            self.equation.evaluate(dw_mode="matrix", asm_obj=A, diff_var="u")
            self.equation.evaluate(dw_mode="vector", asm_obj=b, diff_var=None, u=u1[:, :])

            # get update u2
            u2[0, 1:-1] = (3 * u[0, 1:-1, it - 1] + u1[0, 1:-1]
                           + dt * b[0] / nm.diag(A[0])[:, nax]) / 4
            u2[1, 1:-1] = (3 * u[1, 1:-1, it - 1] + u1[1, 1:-1]
                           + dt * b[1] / nm.diag(A[1])[:, nax]) / 4

            # limit
            u2 = self.limiter(u2)

            # ----3rd stage-----
            # get RHS
            A[:] = 0
            b[:] = 0
            self.equation.evaluate(dw_mode="matrix", asm_obj=A, diff_var="u")
            self.equation.evaluate(dw_mode="vector", asm_obj=b, diff_var=None, u=u2[:, :])

            # get update u3
            u[0, 1:-1, it] = (u[0, 1:-1, it - 1] + 2 * u2[0, 1:-1]
                              + 2*dt * b[0] / nm.diag(A[0])[:, nax]) / 3
            u[1, 1:-1, it] = (u[1, 1:-1, it - 1] + 2 * u2[1, 1:-1]
                              + 2*dt * b[1] / nm.diag(A[1])[:, nax]) / 3

            # limit
            u[:, :, it] = self.limiter(u[:, :, it])

        return u, dt

class EUSolver(TSSolver):
    """
    Euler method with limiter
    """

    def solve(self, t0, tend, tsteps=10):
        """

        :param t0:
        :param tend:
        :param tsteps:
        :return:
        """
        A, b, dt, u = self.initialize(t0, tend, tsteps)
        self.equation.terms[1].get_state_variables()[0].setup_dof_info()
        di = self.equation.terms[1].get_state_variables()[0].di
        self.equation.terms[1].get_state_variables()[0].setup_initial_conditions(self.ics)

        for it in range(1, tsteps):
            A[:] = 0
            b[:] = 0

            self.equation.evaluate(dw_mode="matrix", asm_obj=A, diff_var="u")
            self.equation.evaluate(dw_mode="vector", asm_obj=b, diff_var="u")

            u[0, 1:-1, it] = u[0, 1:-1, it - 1] + dt * b[0] / nm.diag(A[0])[:, nax]
            u[1, 1:-1, it] = u[1, 1:-1, it - 1] + dt * b[1] / nm.diag(A[1])[:, nax]

            u[:, :, it] = self.limiter(u[:, :, it])

        return u, dt