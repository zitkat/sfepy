import numpy as nm
from sfepy.terms.extmods import terms
from sfepy.terms.cache import DataCache

class StateInVolumeQPDataCache( DataCache ):
    name = 'state_in_volume_qp'
    arg_types = ('state', 'get_vector')

    def __init__( self, name, arg_names, history_sizes = None ):
        DataCache.__init__( self, name, arg_names, ['state'], history_sizes,
                            terms.dq_state_in_qp )
        
    def init_data(self, key, ckey, term, **kwargs):
        state, aux = self.get_args( **kwargs )

        n_el, n_qp = state.get_data_shape(ckey[-1], term.integral)[:2]
        shape = (n_el, n_qp, state.n_components, 1)

#        print self.name, key, ckey, shape
        DataCache.init_data(self, key, ckey, shape, dtype=state.dtype)

    def update(self, key, term, ih, **kwargs):
        state, get_vector = self.get_args( **kwargs )
        ap, vg = term.get_approximation(state)
        ckey = self.get_key(term)

        if ih == 0:
            bf = ap.get_base('v', 0, term.integral)
            vec = get_vector( state )
            self.function_complex(self.data[key][ckey][ih], vec,
                                  0, bf, ap.econn)
        else:
            print 'history update!'
            print kwargs['history']
            raise NotImplementedError

class StateInSurfaceQPDataCache( DataCache ):
    name = 'state_in_surface_qp'
    arg_types = ('state',)
    region_matters = True
    
    def __init__( self, name, arg_names, history_sizes = None ):
        DataCache.__init__( self, name, arg_names, ['state'], history_sizes,
                            terms.dq_state_in_qp )
        
    def init_data(self, key, ckey, term, **kwargs):
        state, = self.get_args( **kwargs )

        region_name, ig = term.get_current_group()[1:]
        n_fa, n_qp = state.get_data_shape(ig, term.integral, 'surface',
                                           region_name)[:2]
        shape = (n_fa, n_qp, state.n_components, 1)
        
        DataCache.init_data(self, key, ckey, shape, dtype=state.dtype)

    def update(self, key, term, ih, **kwargs):
        ckey = self.get_key(term)
        state, = self.get_args( **kwargs )

        ap, sg = term.get_approximation(state)
        sd = ap.surface_data[term.region.name]

        bf = ap.get_base(sd.face_type, 0, term.integral)
        econn = sd.get_connectivity(ap.is_surface)

        self.function_complex(self.data[key][ckey][ih], state(), 0, bf, econn)

class CauchyStrainDataCache( DataCache ):
    name = 'cauchy_strain'
    arg_types = ('state', 'get_vector')

    def __init__( self, name, arg_names, history_sizes = None ):
        DataCache.__init__( self, name, arg_names, ['strain'],
                            history_sizes, terms.dq_cauchy_strain )
        
    def init_data(self, key, ckey, term, **kwargs):
        state, aux = self.get_args( **kwargs )

        n_el, n_qp, dim = state.get_data_shape(ckey[-1], term.integral)[:3]
        sym = dim * (dim + 1) / 2
        shape = (n_el, n_qp, sym, 1)

#        print self.name, key, ckey, shape
        DataCache.init_data( self, key, ckey, shape )

    def update(self, key, term, ih, **kwargs):
        ckey = self.get_key(term)
        if ih > 0:
            print 'history update!'
            print kwargs['history']
            raise NotImplementedError
        state, get_vector = self.get_args( **kwargs )

        ap, vg = term.get_approximation(state)
        vec = get_vector( state )
        self.function( self.data['strain'][ckey][ih], vec, 0, vg, ap.econn )
        is_finite = nm.isfinite( self.data[key][ckey][ih] )
        if not nm.alltrue( is_finite ):
            ii = nm.where( is_finite == False )
            print ii
            print self.data[key][ckey][ih][ii]
            print 'infinite strains in', ckey
            raise ValueError
        self.valid['strain'][ckey] = True
                
class GradScalarDataCache( DataCache ):
    name = 'grad_scalar'
    arg_types = ('state',)

    def __init__( self, name, arg_names, history_sizes = None ):
        DataCache.__init__( self, name, arg_names, ['grad'], history_sizes,
                            terms.dq_grad )
        
    def init_data(self, key, ckey, term, **kwargs):
        state, = self.get_args( **kwargs )

        n_el, n_qp, dim = state.get_data_shape(ckey[-1], term.integral)[:3]
        shape = (n_el, n_qp, dim, 1)

#        print self.name, key, ckey, shape
        DataCache.init_data(self, key, ckey, shape, dtype=state.dtype)

    def update(self, key, term, ih, **kwargs):
        state, = self.get_args( **kwargs )
        ap, vg = term.get_approximation(state)
        ckey = self.get_key(term)

        self.function_complex(self.data[key][ckey][ih], state(),
                              0, vg, ap.econn)

class GradVectorDataCache( GradScalarDataCache ):
    name = 'grad_vector'
    arg_types = ('state',)

    def init_data(self, key, ckey, term, **kwargs):
        state, = self.get_args( **kwargs )

        n_el, n_qp, dim = state.get_data_shape(ckey[-1], term.integral)[:3]
        shape = (n_el, n_qp, dim, dim)

#        print self.name, key, ckey, shape
        DataCache.init_data(self, key, ckey, shape, dtype=state.dtype)

class DivVectorDataCache( DataCache ):
    name = 'div_vector'
    arg_types = ('state',)

    def __init__( self, name, arg_names, history_sizes = None ):
        DataCache.__init__( self, name, arg_names, ['div'], history_sizes,
                            terms.dq_div_vector )
        
    def init_data(self, key, ckey, term, **kwargs):
        state, = self.get_args( **kwargs )

        n_el, n_qp = state.get_data_shape(ckey[-1], term.integral)[:2]
        shape = (n_el, n_qp, 1, 1)

#        print self.name, key, ig, shape
        DataCache.init_data(self, key, ckey, shape, dtype=state.dtype)

    def update(self, key, term, ih, **kwargs):
        state, = self.get_args( **kwargs )
        ap, vg = term.get_approximation(state)
        ckey = self.get_key(term)

        self.function_complex(self.data[key][ckey][ih], state(),
                              0, vg, ap.econn)
