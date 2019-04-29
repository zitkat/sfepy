import numpy as nm


# sfepy imports
from sfepy.terms.terms import Term, terms
from sfepy.base.base import (get_default, output, assert_,
                             Struct, basestr, IndexedStruct)

from sfepy.discrete.dg.dg_field import get_unraveler, get_raveler


def unravel_sol(state):
    """
    Unpacks solution from flat vector to
    (n_cell, n_el_nod, n_comp)


    :param state:
    """
    u = state.data[0]
    n_cell = state.field.n_cell
    n_el_nod = state.field.poly_space.n_nod
    unravel = get_unraveler(n_el_nod, n_cell)
    return unravel(u)

class AdvectDGFluxTerm(Term):
    r"""
    Lax-Friedrichs flux term for advection of scalar quantity :math:`p` with the advection velocity
    :math:`\ul{a}` given as a material parameter (a known function of space and time).

    Part of the discretization of
        math::



    Is residual only! Use with state variable history [-1].

    :Definition:

    .. math::

        \int_{\partial{T_K}} \vec{n} \cdot f^{*} (p_{in}, p_{out})\cdot q

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
    arg_shapes = [{'opt_material': '.: 1',
                  'material_advelo': 'D, 1',
                  'virtual': (1, 'state'),
                  'state': 1
                  },
                {'opt_material': None}]
    integration = 'volume'
    # symbolic = {'expression' : 'div(a*u)',
    #             'map': {'u': 'state', 'a': 'material'}
    # }

    def get_fargs(self, alpha, advelo, test, state,
                  mode=None, term_mode=None, diff_var=None, **kwargs):

        if alpha is not None:
            # FIXME this is only hotfix to get scalar!
            self.alf = nm.max(alpha) # extract alpha value regardless of shape

        if diff_var is not None:
            output("Diff var is not None in residual only term {} ! Skipping.".format(self.name))
            return None, None, None, None, None, advelo[:, 0, :, 0], 0
        else:
            field = state.field
            region = field.region

            if not "DG" in field.family_name:
                raise ValueError("Used DG term with non DG field {} of family {}".format(field.name, field.family_name))

            shapes = (field.n_cell, field.n_el_nod)
            cell_normals = field.get_cell_normals_per_facet(region)
            facet_base_vals = field.get_facet_base(base_only=True)
            inner_facet_qp_vals, outer_facet_qp_vals, weights = field.get_both_facet_qp_vals(state, region)

            self.dofs = unravel_sol(state)

            fargs = (shapes, inner_facet_qp_vals, outer_facet_qp_vals,
                     facet_base_vals, weights, cell_normals, advelo[:, 0, :, 0])
            return fargs

    # noinspection PyUnreachableCode
    def function(self, out, shapes, in_fc_v, out_fc_v, fc_b, whs, fc_n, advelo):
        """

        :param out:
        :param shapes: (n_cell, n_el_nod)
        :param in_fc_v: inner values for facets per cell, shape = (n_cell, n_el_faces, 1)
        :param out_fc_v: outer values for facets per cell, shape = (n_cell, n_el_faces, 1)
        :param fc_b: values of basis in facets qps, shape = (1, n_el_facet, n_qp)
        :param whs: weights of the qps on facets, shape = (n_cell, n_el_facet, n_qp
        :param fc_n: facet normals, shape = (n_cell, n_el_facets)
        :param advelo: advection velocity, shape = (n_cell, 1)
        :return:
        """
        if shapes is None:
            out[:] = 0.0
            return None

        n_cell = shapes[0]
        n_el_nod = shapes[1]
        n_el_facets = fc_n.shape[-2]
        #  Calculate integrals over facets representing Lax-Friedrichs fluxes
        C = nm.abs(nm.sum(fc_n * advelo[:, None, :], axis=-1))[:, None]
        facet_fluxes = nm.zeros((n_cell, n_el_facets, n_el_nod))
        for facet_n in range(n_el_facets):
            for mode_n in range(n_el_nod):
                fc_v_p = in_fc_v[:, facet_n, :] + out_fc_v[:, facet_n, :]
                fc_v_m = in_fc_v[:, facet_n, :] - out_fc_v[:, facet_n, :]
                central = advelo[:, None, :] * fc_v_p[:, :, None] / 2.
                upwind = ((1 - self.alf)/2. * C[:, :, facet_n] * fc_n[:, facet_n])[..., None, :] * fc_v_m[:, :, None]

                facet_fluxes[:, facet_n, mode_n] = nm.sum(fc_n[:, facet_n] *
                                                     nm.sum((central + upwind) *
                                                            (fc_b[None, :, 0, facet_n, 0, mode_n] *
                                                             whs[:, facet_n, :])[..., None], axis=1),
                                                     axis=1)

        cell_fluxes = nm.sum(facet_fluxes, axis=1)

        # 1D plots
        if False:
            dofs = self.dofs
            # TODO remove 1D plots
            from my_utils.visualizer import plot_1D_legendre_dofs, reconstruct_legendre_dofs
            import matplotlib.pyplot as plt
            x = self.region.domain.mesh.coors
            plot_1D_legendre_dofs(x, (dofs,))
            ww, xx = reconstruct_legendre_dofs(x, 1, dofs.T[0, ..., None, None])
            plt.plot(xx, ww[:, 0], label="recon")

            # plt.figure("Vals outer")
            # plt.plot(x[:-1], out_fc_v[:, 0], marker=".", label="left outer", color="b",  ls="")
            # plt.plot(x[1:], out_fc_v[:, 1], marker=".",  label="right outer", color="r",  ls="")
            # plt.plot(xx, ww[:, 0], label="recon")
            # plt.legend()
            #
            # # Plot mesh
            # XN = x[-1]
            # X1 = x[0]
            # Xvol = XN - X1
            # X = (x[1:] + x[:-1]) / 2
            # plt.vlines(x[:, 0], ymin=0, ymax=.5, colors="grey")
            # plt.vlines((X1, XN), ymin=0, ymax=.5, colors="k")
            # plt.vlines(X, ymin=0, ymax=.3, colors="grey", linestyles="--")
            #
            # plt.figure("Vals inner")
            # plt.plot(xx, ww[:, 0], label="recon")
            # plt.plot(x[:-1], in_fc_v[:, 0], marker=".", label="left in", color="b", ls="")
            # plt.plot(x[1:], in_fc_v[:, 1], marker=".", label="right in", color="r", ls="")
            # plt.legend()
            #
            # # Plot mesh
            # XN = x[-1]
            # X1 = x[0]
            # Xvol = XN - X1
            # X = (x[1:] + x[:-1]) / 2
            # plt.vlines(x[:, 0], ymin=0, ymax=.5, colors="grey")
            # plt.vlines((X1, XN), ymin=0, ymax=.5, colors="k")
            # plt.vlines(X, ymin=0, ymax=.3, colors="grey", linestyles="--")
            # plt.legend()

            for mode_n in range(n_el_nod):
                plt.figure("Flux {}".format(mode_n))
                fig = plt.gcf()
                fig.clear()
                plt.plot(xx, ww[:, 0], label="recon")
                plt.plot(xx, ww[:, 0], label="recon")
                plt.plot(x[:-1], facet_fluxes[:, 0, mode_n], marker=">", label="flux {} left".format(mode_n), color="b", ls="")
                plt.plot(x[1:], facet_fluxes[:, 1,  mode_n], marker=">", label="flux {} right".format(mode_n), color="r", ls="")


                # Plot mesh
                XN = x[-1]
                X1 = x[0]
                Xvol = XN - X1
                X = (x[1:] + x[:-1]) / 2
                plt.vlines(x[:, 0], ymin=0, ymax=.5, colors="grey")
                plt.vlines((X1, XN), ymin=0, ymax=.5, colors="k")
                plt.vlines(X, ymin=0, ymax=.3, colors="grey", linestyles="--")
                plt.legend()

            # plt.show()

            for mode_n in range(n_el_nod):
                plt.figure("Flux {}".format(mode_n))
                fig = plt.gcf()
                fig.clear()
                plt.plot(xx, ww[:, 0], label="recon")
                plt.plot(xx, ww[:, 0], label="recon")
                plt.plot(X, cell_fluxes[:, mode_n], marker="D", label="cell flux {}".format(mode_n), color="r", ls="")


                # Plot mesh
                XN = x[-1]
                X1 = x[0]
                Xvol = XN - X1
                X = (x[1:] + x[:-1]) / 2
                plt.vlines(x[:, 0], ymin=0, ymax=.5, colors="grey")
                plt.vlines((X1, XN), ymin=0, ymax=.5, colors="k")
                plt.vlines(X, ymin=0, ymax=.3, colors="grey", linestyles="--")
                plt.legend()
                plt.close('all')


        # 2D plots
        if False:
            # TODO remove 2D plots
            import matplotlib.pyplot as plt
            import sfepy.postprocess.plot_cmesh as pc

            cmesh = self.region.domain.mesh.cmesh

            def plot_facet_normals(ax, cmesh, normals, scale, col='m'):
                dim = cmesh.dim
                ax = pc._get_axes(ax, dim)

                edim = dim - 1
                coors = cmesh.get_centroids(edim)
                coors = pc._to2d(coors)

                cmesh.setup_connectivity(dim, edim)
                c2f = cmesh.get_conn(dim, edim)
                for ic, o1 in enumerate(c2f.offsets[:-1]):
                    o2 = c2f.offsets[ic + 1]
                    for ifal, ifa in enumerate(c2f.indices[o1:o2]):
                        # print(ic, ifal, ifa)
                        cc = nm.array([coors[ifa], coors[ifa] + scale * normals[ic, ifal]])
                        # print(cc)
                        ax.plot(*cc.T, color=col)
                        # ax.plot([cc[1, 0]], [cc[1, 1]], color=col, marker=">")


                return ax


            ax = pc.plot_cmesh(
                None, cmesh,
                wireframe_opts={'color': 'k', 'linewidth': 2},
                entities_opts=[
                    {'color': 'k', 'label_global': 12, 'label_local': 8, 'size': 20},  # vertices
                    {'color': 'b', 'label_global': 6, 'label_local': 8, 'size': 10},  # faces
                    {'color': 'r', 'label_global': 12, 'size': 1},  # cells
                ])
            # for i in range(n_el_nod):
            #     ax = plot_facet_normals(ax, cmesh, facet_fluxes[:, :, i, None] * fc_n, 1, col='r')

            ax = plot_facet_normals(ax, cmesh, fc_n, .01, col='m')
            plt.show()

        out[:] = 0.0
        for i in range(n_el_nod):
            out[:, :, i, 0] = cell_fluxes[:, i, None]

        status = None
        return status
