import numpy as nm

# sfepy imports
from sfepy.terms.terms import Term, terms
from sfepy.base.base import (get_default, output, assert_,
                             Struct, basestr, IndexedStruct)
from sfepy.terms.terms_dot import ScalarDotMGradScalarTerm

from sfepy.discrete.dg.dg_field import get_unraveler, get_raveler


class AdvectDGFluxTerm(Term):
    r"""
    Lax-Friedrichs flux term for advection of scalar quantity :math:`p` with the advection velocity
    :math:`\ul{a}` given as a material parameter (a known function of space and time).

    Part of the discretization of
        math::

    Is residual only! Use with state variable history [-1].

    :Definition:

    .. math::

        \int_{\partial{T_K}} \vec{n} \cdot f^{*} (p_{in}, p_{out})q

        where
            f^{*}(p_{in}, p_{out}) =  \vec{a}  \frac{p_{in} + p_{out}}{2}  + (1 - \alpha) \vec{n}
            C \frac{ p_{in} - p_{out}}{2},

        $\alpha \in [0, 1]$, $\alpha = 0$ for upwind scheme, $\alpha = 1$ for central scheme,  and

            C = \max_{u \in [?, ?]} \abs{n_x \pdiff{a_1}{u} + n_y \pdiff{a_2}{u}}

        the $p_{in}$ resp. $p_{out}$ is solution on the boundary of the element provided
        by element itself resp. its neighbor and a is advection velocity.

    :Arguments 1:
        - material : :math:`\ul{a}`
        - virtual  : :math:`q`
        - state    : :math:`p`

    :Arguments 3:
        - material    : :math:`\ul{a}`
        - virtual     : :math:`q`
        - state       : :math:`p`
        - opt_material : :math: `\alpha`
    """

    alf = 0
    name = "dw_dg_advect_laxfrie_flux"
    modes = ("weak",)
    arg_types = ('opt_material', 'material_advelo', 'virtual', 'state')
    arg_shapes = [{'opt_material'   : '.: 1',
                   'material_advelo': 'D, 1',
                   'virtual'        : (1, 'state'),
                   'state'          : 1
                   },
                  {'opt_material': None}]
    integration = 'volume'
    symbolic = {'expression': 'div(a*u)*w',
                'map'       : {'u': 'state', 'a': 'material', 'v': 'virtual'}
                }

    def get_fargs(self, alpha, advelo, test, state,
                  mode=None, term_mode=None, diff_var=None, **kwargs):

        if alpha is not None:
            self.alf = alpha  # extract alpha value regardless of shape

        if diff_var is not None:
            output("Diff var is not None in residual only term {} ! Skipping.".format(self.name))
            return None, None, None, 0
        else:
            field = state.field
            region = field.region

            if "DG" not in field.family_name:
                raise ValueError("Used DG term with non DG field {} of family {}".format(field.name, field.family_name))

            fargs = (state, field, region, advelo[:, 0, :, 0])
            return fargs

    # noinspection PyUnreachableCode
    def function(self, out, state, field, region, advelo):

        if state is None:
            out[:] = 0
            return None

        fc_n = field.get_cell_normals_per_facet(region)
        facet_base_vals = field.get_facet_base(base_only=True)
        in_fc_v, out_fc_v, weights = field.get_both_facet_state_vals(state, region)
        # get sane facet base shape
        fc_b = facet_base_vals[:, 0, :, 0, :].T  # (n_el_nod, n_el_facet, n_qp)

        # get maximal wave speeds at facets
        C = nm.abs(nm.einsum("ifk,ik->if", fc_n, advelo))

        fc_v_avg = (in_fc_v + out_fc_v)/2
        fc_v_jmp = in_fc_v - out_fc_v

        central = nm.einsum("ik,ifq->ifkq", advelo, fc_v_avg)
        upwind = (1 - self.alf) / 2. * nm.einsum("if,ifk,ifq->ifkq", C, fc_n, fc_v_jmp)

        cell_fluxes = nm.einsum("ifk,ifkq,dfq,ifq->id", fc_n, central + upwind, fc_b, weights)


        out[:] = 0.0
        n_el_nod = field.n_el_nod
        for i in range(n_el_nod):
            out[:, :, i, 0] = cell_fluxes[:, i, None]

        status = None
        return status


class DiffusionDGFluxTerm(Term):
    name = "dw_dg_diffusion_flux"
    modes = ("weak",)
    arg_types = ('material_diff_tensor', 'virtual', 'state')
    arg_shapes = [{'material_diff_tensor': '1, 1',
                   'virtual'             : (1, 'state'),
                   'state'               : 1
                   }]
    integration = 'volume'
    symbolic = {'expression': 'div(D*grad(u))',
                'map'       : {'u': 'state', 'a': 'material', 'v': 'virtual'}
                }

    def get_fargs(self, diff_tensor, test, state,
                  mode=None, term_mode=None, diff_var=None, **kwargs):
        if diff_var is not None:
            # TODO will this term be residual only?
            output("Diff var is not None in residual only term {} ! Skipping.".format(self.name))
            return None, None, None, 0
        else:
            field = state.field
            region = field.region

            if "DG" not in field.family_name:
                raise ValueError("Used DG term with non DG field {} of family {}".format(field.name, field.family_name))

            fargs = (state, field, region, diff_tensor[:, 0, :, :])
            return fargs

    def function(self, out, state, field, region, D):

        if state is None:
            out[:] = 0.0
            return None

        fc_n = field.get_cell_normals_per_facet(region)
        inner_facet_base, outer_facet_base, _ = field.get_both_facet_base_vals(state, region,
                                                                               derivative=False
                                                                               )
        inner_facet_state_d, outer_facet_state_d, _ = field.get_both_facet_state_vals(state, region,
                                                                                      derivative=True
                                                                                      )
        inner_facet_base_d, outer_facet_base_d, _ = field.get_both_facet_base_vals(state, region,
                                                                                   derivative=True)
        inner_facet_state, outer_facet_state, weights = field.get_both_facet_state_vals(state, region,
                                                                                        derivative=False
                                                                                        )
        avgDdState = (nm.einsum("ikl,ifkq->ifkq", D, inner_facet_state_d) +
                      nm.einsum("ikl,ifkq->ifkq", D, outer_facet_state_d)) / 2.
        jmpBase = inner_facet_base  # - outer_facet_base

        avgDdbase = (nm.einsum("ikl,idfkq->idfkq", D, inner_facet_base_d)) / 2.
        # nm.einsum("ikl,idfkq->idfkq", D, outer_facet_base_d)) / 2.
        jmpState = inner_facet_state - outer_facet_state

        int1 = nm.einsum("ifkq , ifk, idfq, ifq -> id", avgDdState, fc_n, jmpBase, weights)

        int2 = nm.einsum("idfkq, ifk, ifq , ifq -> id", avgDdbase, fc_n, jmpState, weights)

        # nonsymetric diffusion form - opposite signs
        # cell_fluxes = int1 - int2
        # symetric diffusion form
        cell_fluxes = int1 + int2
        # incomplete
        # cell_fluxes = int1

        out[:] = 0.0
        n_el_nod = field.n_el_nod
        for i in range(n_el_nod):
            out[:, :, i, 0] = cell_fluxes[:, i, None]

        status = None
        return status


class LeftDiffusionDGFluxTerm(DiffusionDGFluxTerm):
    name = "dw_dg_left_diffusion_flux"

    def function(self, out, state, field, region, D):

        if state is None:
            out[:] = 0.0
            return None

        fc_n = field.get_cell_normals_per_facet(region)
        inner_facet_base, outer_facet_base, _ = field.get_both_facet_base_vals(state, region,
                                                                               derivative=False
                                                                               )
        inner_facet_state_d, outer_facet_state_d, weights = field.get_both_facet_state_vals(state, region,
                                                                                            derivative=True
                                                                                            )

        avgDdState = (nm.einsum("ikl,ifkq->ifkq", D, inner_facet_state_d) +
                      nm.einsum("ikl,ifkq->ifkq", D, outer_facet_state_d)) / 2.
        jmpBase = inner_facet_base  # - outer_facet_base

        int1 = nm.einsum("ifkq , ifk, idfq, ifq -> id", avgDdState, fc_n, jmpBase, weights)

        out[:] = 0.0
        n_el_nod = field.n_el_nod
        for i in range(n_el_nod):
            out[:, :, i, 0] = int1[:, i, None]

        status = None
        return status


class RightDiffusionDGFluxTerm(DiffusionDGFluxTerm):
    name = "dw_dg_right_diffusion_flux"

    def function(self, out, state, field, region, D):

        if state is None:
            out[:] = 0.0
            return None

        fc_n = field.get_cell_normals_per_facet(region)
        inner_facet_base_d, outer_facet_base_d, _ = field.get_both_facet_base_vals(state, region,
                                                                                   derivative=True)
        inner_facet_state, outer_facet_state, weights = field.get_both_facet_state_vals(state, region,
                                                                                        derivative=False
                                                                                        )

        avgDdbase = (nm.einsum("ikl,idfkq->idfkq", D, inner_facet_base_d)) / 2.
        # nm.einsum("ikl,idfkq->idfkq", D, outer_facet_base_d)) / 2.
        jmpState = inner_facet_state - outer_facet_state

        int2 = nm.einsum("idfkq, ifk, ifq , ifq -> id", avgDdbase, fc_n, jmpState, weights)

        out[:] = 0.0
        n_el_nod = field.n_el_nod
        for i in range(n_el_nod):
            out[:, :, i, 0] = int2[:, i, None]

        status = None
        return status


class DiffusionInteriorPenaltyTerm(Term):
    name = "dw_dg_interior_penal"
    modes = ("weak",)
    arg_types = ('material_Cw', 'virtual', 'state')
    arg_shapes = [{'material_Cw': '.: 1',
                   'virtual'    : (1, 'state'),
                   'state'      : 1
                   }]

    def get_fargs(self, Cw, test, state, mode=None, term_mode=None, diff_var=None, **kwargs):

        if diff_var is not None:
            # TODO will this term be residual only?
            output("Diff var is not None in residual only term {} ! Skipping.".format(self.name))
            return None, None, None, 0
        else:
            field = state.field
            region = field.region

            if "DG" not in field.family_name:
                raise ValueError("Used DG term with non DG field {} of family {}".format(field.name, field.family_name))

            fargs = (state, field, region, Cw)
            return fargs

    def function(self, out, state, field, region, Cw):

        if state is None:
            out[:] = 0.0
            return None

        inner_facet_state, outer_facet_state, weights = field.get_both_facet_state_vals(state, region,
                                                                                        derivative=False
                                                                                        )
        inner_facet_base, outer_facet_base, _ = field.get_both_facet_base_vals(state, region,
                                                                               derivative=False)
        facet_vols = nm.sum(weights, axis=-1)

        jmp_state = inner_facet_state - outer_facet_state
        jmp_base = inner_facet_base  # - outer_facet_base
        sigma = Cw / facet_vols

        n_el_nod = nm.shape(inner_facet_base)[1]
        cell_penalty = nm.einsum("if,ifq,idfq,ifq->id", sigma, jmp_state, jmp_base, weights)

        out[:] = 0.0
        for i in range(n_el_nod):
            out[:, :, i, 0] = cell_penalty[:, i, None]

        status = None
        return status


class NonlinearHyperDGFluxTerm(Term):
    alf = 0
    name = "dw_dg_nonlinear_laxfrie_flux"
    modes = ("weak",)
    arg_types = ('opt_material', 'material_fun', 'material_fun_d', 'virtual', 'state')
    arg_shapes = [{'opt_material'  : '.: 1',
                   'material_fun'  : '.: 1',
                   'material_fun_d': '.: 1',
                   'virtual'       : (1, 'state'),
                   'state'         : 1
                   },
                  {'opt_material': None}]
    integration = 'volume'
    symbolic = {'expression': 'div(f(u))*w',
                'map'       : {'u': 'state', 'v': 'virtual', 'f': 'function'}
                }

    def get_fargs(self, alpha, fun, dfun, test, state,
                  mode=None, term_mode=None, diff_var=None, **kwargs):

        if alpha is not None:
            self.alf = nm.max(alpha)  # extract alpha value regardless of shape

        self.fun = fun
        self.dfun = dfun

        if diff_var is not None:
            output("Diff var is not None in residual only term {} ! Skipping.".format(self.name))
            return None, None, None, 0, 0
        else:
            field = state.field
            region = field.region

            if "DG" not in field.family_name:
                raise ValueError(
                        "Used DG term with non DG field {} of family {}!".format(field.name, field.family_name))

            fargs = (state, field, region, fun, dfun)
            return fargs

    # noinspection PyUnreachableCode
    def function(self, out, state, field, region, f, df):
        if state is None:
            out[:] = 0.0
            return None

        fc_n = field.get_cell_normals_per_facet(region)
        facet_base_vals = field.get_facet_base(base_only=True)
        in_fc_v, out_fc_v, weights = field.get_both_facet_state_vals(state, region)

        fc_b = facet_base_vals[:, 0, :, 0, :].T  # (n_el_nod, n_el_facet, n_qp)

        n_cell = field.n_cell
        n_el_nod = field.n_el_nod
        n_el_facets = field.n_el_facets

        # get maximal wave speeds at facets
        df_in = df(in_fc_v)
        df_out = df(out_fc_v)
        fc_n__dot__df_in = nm.einsum("ifk,ifqk->ifq", fc_n, df_in)
        fc_n__dot__df_out = nm.einsum("ifk,ifqk->ifq", fc_n, df_out)
        dfdn = nm.stack((fc_n__dot__df_in, fc_n__dot__df_out), axis=-1)
        C = nm.amax(nm.abs(dfdn), axis=(-2, -1))

        fc_f_avg = (f(in_fc_v) + f(out_fc_v)) / 2.
        fc_v_jmp = in_fc_v - out_fc_v

        central = fc_f_avg
        upwind = (1 - self.alf) / 2. * nm.einsum("if,ifk,ifq->ifqk", C, fc_n, fc_v_jmp)

        cell_fluxes = nm.einsum("ifk,ifqk,dfq,ifq->id", fc_n, central + upwind, fc_b, weights)

        out[:] = 0.0
        for i in range(n_el_nod):
            out[:, :, i, 0] = cell_fluxes[:, i, None]

        status = None
        return status


from sfepy.linalg import dot_sequences


class NonlinScalarDotGradTerm(Term):
    r"""
    Volume dot product of a scalar gradient dotted with a material vector with
    a scalar.

    :Definition:

    .. math::
        \int_{\Omega} q \ul{y} \cdot \nabla p \mbox{ , }
        \int_{\Omega} p \ul{y} \cdot \nabla q

    :Arguments 1:
        - material : :math:`\ul{y}`
        - virtual  : :math:`q`
        - state    : :math:`p`

    :Arguments 2:
        - material : :math:`\ul{y}`
        - state    : :math:`p`
        - virtual  : :math:`q`
    """
    name = 'dw_ns_dot_grad_s'
    arg_types = (('material_fun', 'material_fun_d', 'virtual', 'state'),
                 ('material_fun', 'material_fun_d', 'state', 'virtual'))
    arg_shapes = [{'material_fun'        : '.: 1',
                   'material_fun_d'      : '.: 1',
                   'virtual/grad_state'  : (1, None),
                   'state/grad_state'    : 1,
                   'virtual/grad_virtual': (1, None),
                   'state/grad_virtual'  : 1}]
    modes = ('grad_state', 'grad_virtual')

    @staticmethod
    def function(out, out_qp, geo, fmode):
        status = geo.integrate(out, out_qp)
        return status

    def get_fargs(self, fun, dfun, var1, var2,
                  mode=None, term_mode=None, diff_var=None, **kwargs):
        vg1, _ = self.get_mapping(var1)
        vg2, _ = self.get_mapping(var2)

        if diff_var is None:
            if self.mode == 'grad_state':
                # TODO check correct shapes for integration
                geo = vg1
                bf_t = vg1.bf.transpose((0, 1, 3, 2))
                val_qp = dfun(self.get(var2, 'val')[..., 0])
                val_grad_qp = self.get(var2, 'grad')
                val = dot_sequences(val_qp, val_grad_qp, 'ATB')
                out_qp = dot_sequences(bf_t, val_grad_qp, 'ATB')

            else:
                geo = vg2
                val_qp = fun(self.get(var1, 'val'))[..., 0, :].swapaxes(-2, -1)
                out_qp = dot_sequences(vg2.bfg, val_qp, 'ATB')

            fmode = 0

        else:
            # TODO what in matrix mode?
            if self.mode == 'grad_state':
                geo = vg1
                bf_t = vg1.bf.transpose((0, 1, 3, 2))
                out_qp = dot_sequences(bf_t, vg2.bfg, 'ATB')

            else:
                geo = vg2
                out_qp = dot_sequences(vg2.bfg, vg1.bf, 'ATB')

            fmode = 1

        return out_qp, geo, fmode
