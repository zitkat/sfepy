import numpy as nm


class DGTerm:

    def __init__(self, mesh):
        self.mesh = mesh

    def evaluate(self, mode="weak", diff_var=None,
                 standalone=True, ret_status=False, **kwargs):
        if diff_var == self.diff_var:
            u = kwargs.pop("u", None)
            fargs = self.get_fargs(None, u, mode=mode, standalone=False)
            out = [None, None]
            status = self.function(out, *fargs)
            return out[0], out[1]
        else:
            return None, None

    @staticmethod
    def assemble_to(asm_obj, val, iels, mode="vector"):
        if (asm_obj is not None) and (iels is not None):
            if mode == "vector":
                if (len(iels) == 2) and (nm.shape(val)[0] == len(iels[0])):
                    for ii in iels[0]:
                        asm_obj[ii][iels[1]] = (asm_obj[ii][iels[1]].T + val[ii]).T
                else:
                    asm_obj[iels] = asm_obj[iels] + val

            elif mode == "matrix":
                if (len(iels) == 3) and (nm.shape(val)[0] == len(iels[0])):
                    for ii in iels[0]:
                        asm_obj[ii][iels[1], iels[2]] = asm_obj[ii][iels[1], iels[2]] + val[ii]
                else:
                    asm_obj[iels] = asm_obj[iels] + val
            else:
                raise ValueError("Unknown assembly mode '%s'" % mode)


class AdvIntDGTerm(DGTerm):

    def __init__(self, mesh):
        DGTerm.__init__(self, mesh)
        self.vvar = "v"
        self.diff_var = "u"

    def get_fargs(self, *args, **kwargs):

        val = nm.vstack(((self.mesh.coors[1:] - self.mesh.coors[:-1]).T,
                         (self.mesh.coors[1:] - self.mesh.coors[:-1]).T/3))
        # integral over element with constant test
        # function is just volume of the element

        fargs = (val,)
        return fargs

    def function(self, out, vals):
        out[:] = vals
        status = None
        return status

    def evaluate(self, mode="weak", diff_var="u",
                 standalone=True, ret_status=False, **kwargs):
        if diff_var == self.diff_var:
            fargs = self.get_fargs()
            out = nm.zeros((2, self.mesh.n_el))
            self.function(out, *fargs)
            iels = ([0, 1], nm.arange(len(self.mesh.coors) - 1), nm.arange(len(self.mesh.coors) - 1))
            # values go on to the diagonal, in sfepy this is assured
            # by mesh connectivity induced by basis
            return out, iels
        else:
            return None, None


class AdvFluxDGTerm(DGTerm):
    """
    So far Lax-Friedrichs flux for a*u,
    a can be variable sample over domain
    """

    def __init__(self, mesh, a):
        DGTerm.__init__(self, mesh)
        self.a = a
        self.vvar = None
        self.diff_var = None

    def get_fargs(self, test, state, mode="weak",
                  standalone=True, ret_status=False, **kwargs):

        a = self.a
        fargs = (state, a)
        return fargs

    def function(self, out, u, a):
        # for Legendre basis integral of higher order
        # functions of the basis is zero,
        # hence we calculate integral
        #
        # int_{j-1/2}^{j+1/2} f(u)dx
        #
        # only from the zero order function, over [-1, 1] - hence the 2
        intg = a * u[0, 1:-1].T * 2

        #  the Lax-Friedrichs flux is
        #       F(a, b) = 1/2(f(a) + f(b)) + max(f'(w)) / 2 * (a - b)
        # in our case a and b are values to the left and right of the element boundary
        # for Legendre basis these are:
        # u_left = U_0 + U_1 + U_2 + ...
        # u_right = U_0 - U_1 + U_2 + ... = sum_0^{order} (-1)^p * U_p

        # left flux is calculated in j_-1/2  where U(j-1) and U(j) meet
        # right flux is calculated in j_+1/2 where U(j) and U(j+1) meet

        fl = a * (u[0, :-2] + u[1, :-2] +
                 (u[0, 1:-1] - u[1, 1:-1])).T / 2 + \
             nm.abs(a) * (u[0, :-2] + u[1, :-2] -
                         (u[0, 1:-1] - u[1, 1:-1])).T / 2

        fp = a * (u[0, 1:-1] + u[1, 1:-1] +
                 (u[0, 2:] - u[1, 2:])).T / 2 + \
             nm.abs(a) * (u[0, 1:-1] + u[1, 1:-1] -
                         (u[0, 2:] - u[1, 2:])).T / 2

        val = nm.vstack((fl - fp, - fl - fp + intg))

        # placement is simple, but getting the values requires looping over neighbours
        iels = ([0, 1], nm.arange(len(self.mesh.coors) - 1))  # just fill the vector

        vals = nm.vstack((fl - fp, - fl - fp + intg))
        out[:] = vals, iels
        status = None
        return status

    def get_stab_cond(self, ic=None):
        return abs(nm.max(self.a))

class HypfFluxDGTerm(DGTerm):
    """
    So far Lax-Friedrichs flux term for general f(u)
    """

    def __init__(self, mesh, f, df):
        DGTerm.__init__(self, mesh)
        self.f = f
        self.df = df
        self.vvar = None
        self.diff_var = None

    def get_fargs(self, test, state, mode="weak",
                  standalone=True, ret_status=False, **kwargs):
        f = self.f
        df = self.df
        fargs = (state, f, df)
        return fargs

    def function(self, out, u, f, df):
        # for Legendre basis integral of higher order
        # functions of the basis is zero,
        # hence we calculate integral
        #
        # int_{j-1/2}^{j+1/2} f(u)dx
        #
        # only from the zero order function, over [-1, 1] - hence the 2
        # TODO use reconstructed solution for integral
        intg = f(u[0, 1:-1].T) * 2

        #  the Lax-Friedrichs flux is
        #       F(a, b) = 1/2(f(a) + f(b)) + max(f'(w)) / 2 * (a - b)
        # in our case a and b are values to the left and right of the element boundary
        # for Legendre basis these are:
        # u_left = U_0 + U_1 + U_2 + ...
        # u_right = U_0 - U_1 + U_2 + ... = sum_0^{order} (-1)^p * U_p

        # left flux is calculated in j_-1/2  where U(j-1) and U(j) meet
        # right flux is calculated in j_+1/2 where U(j) and U(j+1) meet
        # TODO use reconstructed solution for flux
        fl = (f(u[0, :-2]) + f(u[1, :-2]) +
             (f(u[0, 1:-1]) - f(u[1, 1:-1]))).T / 2 + \
             nm.abs(nm.max(df(u[0, 1:-1]))) * (u[0, :-2] + u[1, :-2] -
                              (u[0, 1:-1] - u[1, 1:-1])).T / 2

        fp = (f(u[0, 1:-1]) + f(u[1, 1:-1]) +
             (f(u[0, 2:]) - f(u[1, 2:]))).T / 2 + \
             nm.abs(nm.max(df(u[0, 1:-1]))) * (u[0, 1:-1] + u[1, 1:-1] -
                              (u[0, 2:] - u[1, 2:])).T / 2

        val = nm.vstack((fl - fp, - fl - fp + intg))

        # placement is simple, but getting the values requires looping over neighbours
        iels = ([0, 1], nm.arange(len(self.mesh.coors) - 1))  # just fill the vector
        out[:] = val, iels
        status = None
        return status

    def get_stab_cond(self, ic=nm.linspace(0, 1)):
        return nm.max(nm.abs(self.df(ic)))


class DiffFluxDGTerm(DGTerm):

    def __init__(self, mesh, coef):
        DGTerm.__init__(self, mesh)
        self.coef = coef

    def get_fargs(self, test, state, mode="weak",
                  standalone=True, ret_status=False, **kwargs):
        return state, self.coef

    def function(self, out, u, coef):
        raise NotImplemented
