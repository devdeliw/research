import gdspy
import ruamel.yaml as yaml 

from qnldraw import Params, Chip
from qnldraw.junction import JunctionArray, JunctionLead, simulate_evaporation
from qnldraw.shapes import Rectangle


class SingleQubit: 
    def __init__(self, chip, params): 
        self.chip = chip 
        self.params = params 
        self.layers = params['global.layers']

    def antenna(self): 
        # -- capacitive pads \& antennas to chip
        antenna_params = self.params['antenna']

        pad_gap = antenna_params['pad_gap']
        pad_length, pad_width = antenna_params['pad_dim'] 
        antenna_length, antenna_width = antenna_params['antenna_dim'] 

        pads = [
            Rectangle(pad_length, pad_width).translate(
                -(pad_gap/2 + pad_length/2 + antenna_length), 0), 
            Rectangle(pad_length, pad_width).translate(
                (pad_gap/2 + pad_length/2 + antenna_length), 0), 
        ]

        antennas = [ 
            Rectangle(antenna_length, antenna_width).translate(
                -(pad_gap/2 + antenna_length/2), 0), 
            Rectangle(antenna_length, antenna_width).translate(
                (pad_gap/2 + antenna_length/2), 0), 
        ]

        # -- adding components to chip 
        layer = self.layers['antenna']
        self.chip.add_component(antennas, 'antennas', layers=layer) 
        self.chip.add_component(pads, 'pads', layers=layer) 

        self.pad_gap = pad_gap

        return 

    def junction(self): 
        junction_params = self.params['junction']

        # -- junction leads
        lead_params = junction_params['leads']
        junction_lead = JunctionLead(**lead_params)

        junction_leads = [
            junction_lead.place( 
                location = [
                    -( 
                        self.pad_gap/2 
                        + lead_params['total_length']
                        - lead_params['inner.length']
                        - lead_params['taper_length'] 
                    ), 
                    0,
                ]
            ), 
            junction_lead.place(
                  location = [
                      (
                          self.pad_gap/2
                          + lead_params['total_length']
                          - lead_params['inner.length']
                          - lead_params['taper_length']
                      ),
                      0,
                  ],
                  mirror = 'y',
              ),
        ]

        # -- junction connector to antenna
        connector_params = junction_params['connector'] 

        lx, ly = connector_params['connector_dim']
        connector_gap = connector_params['connector_gap']
        
        junction_connectors = [
            Rectangle(lx, ly).translate(-(connector_gap/2 + lx/2), 0),
            Rectangle(lx, ly).translate((connector_gap/2 + lx/2), 0), 
        ]

        # -- junction fingers 
        finger_params = junction_params['fingers'] 

        # -- central finger 
        central_params = finger_params['central'] 

        lx, ly = central_params['length'], central_params['width']
        undercut_lx = central_params['undercut_length']
        undercut_extension = central_params['undercut_extension']
        
        junction_finger = Rectangle(lx, ly).translate(lx/2, 0)
        junction_undercut = Rectangle(
            undercut_lx, ly+undercut_extension).translate(-undercut_lx/2, 0)

        # -- thin cut leads
        cut_params = junction_params['cut_parameter']

        llx, lly = cut_params['L_length'], cut_params['L_height']
        loffset = cut_params['L_offset']

        rlx, rly = cut_params['R_length'], cut_params['R_height']
        roffset = cut_params['R_offset']

        leads = [
            Rectangle(llx, lly).translate(-undercut_lx - llx/2, loffset), 
            Rectangle(rlx, rly).translate(rlx/2, -roffset), 
        ]

        # -- antisymmetric fingers 
        antisym_params = finger_params['antisymmetric']

        gap = antisym_params['gap']
        offset = antisym_params['offset']
        lx, ly = antisym_params['length'], antisym_params['width']
        undercut_lx = antisym_params['undercut_length']

        antisym_fingers = [
            Rectangle(lx, ly).translate(-gap/2, offset),
            Rectangle(lx, ly).translate(gap/2, -ly+offset),
        ]

        antisym_undercuts = [ 
            Rectangle(undercut_lx, ly).translate(-gap/2 - lx + (undercut_lx - ly)/2, offset), 
            Rectangle(undercut_lx, ly).translate(gap/2 + lx + (undercut_lx - lx)/2, offset-ly), 
        ]

        # -- adding all components to chip 

        layer = self.layers['lead']
        undercut_layer = self.layers['undercut']

        ## junction leads 
        self.chip.add_component(junction_leads, 'junction_leads', layers=layer) 

        ## junction antenna connectors 
        self.chip.add_component(junction_connectors, 'junction_connectors', layers=layer)

        ## central junction finger \& undercut
        self.chip.add_component(junction_finger, 'junction_finger', layers=layer) 
        self.chip.add_component(junction_undercut, 'junction_undercut', layers=undercut_layer)

        ## thin cut parameter leads 
        self.chip.add_component(leads, 'cut parameter leads', layers=layer) 

        ## asymmetric junction fingers \& undercut 
        self.chip.add_component(antisym_fingers, 'antisym fingers', layers=layer) 
        self.chip.add_component(antisym_undercuts, 'antisym undercuts', layers=undercut_layer) 

        return 

    def array(self): 
        array_params = self.params['array']
        antisym_params = self.params['junction.fingers.antisymmetric']

        gap = antisym_params['gap']
        connector_params = self.params['junction.connector']
        depth = ( 
            antisym_params['width']
            + antisym_params['offset'] 
            + connector_params['connector_dim'][1]/2 
        )

        # -- junction array, the # of junctions is defined in params
        array = JunctionArray(**array_params) 
 
        left_array = array.place(location=[-gap/2, -depth], node='wire1') 
        right_array = array.place(location=[gap/2, -depth], node='wire1')

        bottom_connector = Rectangle(
              right_array.node("wire2")[0] - left_array.node("wire2")[0], 
              array_params["overlap"][1],
          ).translate(
              0,
              -(
                  left_array.node("wire1")[1]
                  - left_array.node("wire2")[1]
                  + depth
                  - array_params["overlap"][1] / 2
              ),
          )

        array_components = [left_array, right_array, bottom_connector] 

        # -- adding arrays and connector to chip 
        layer = self.layers['array']
        self.chip.add_component(array_components, 'junction arrays', layers=layer) 

        return 

    def evap_simulation(self): 
        cells = self.chip.render(name='fluxonium', draw_border=False)
        polys = cells[0].get_polygons(by_spec=True) 

        highdose_layers = self.layers['lead'] + [self.layers['array'][0]]
        lowdose_layers = self.layers['undercut'] + [self.layers['array'][1]]


        highdose = gdspy.PolygonSet(
            polys[(highdose_layers[0], 0)] 
            + polys[(highdose_layers[1], 0)], )
        lowdose = gdspy.PolygonSet(
            polys[(lowdose_layers[0], 0)] 
            + polys[(lowdose_layers[1], 0)], )

        evap_params = self.params['evaporation']
        evap_sim = simulate_evaporation(lowdose, highdose, **evap_params) 

        for i, (layer, evap) in enumerate(zip(self.layers['evap'], evap_sim)): 
            self.chip.add_component(evap, f'evap_{i}', layers=layer)

        return 


    def build(self): 
        self.antenna()
        self.junction()
        self.array() 
        self.evap_simulation()

        return self.chip 


if __name__ == '__main__': 
    with open('single_qubit.yaml') as f: 
        params = yaml.load(f, Loader=yaml.SafeLoader)
        params = Params(params) 

        chip = Chip() 

        class_ = SingleQubit(chip, params)
        chip = class_.build() 
        cells = chip.render('single_qubit', include_refs=True)

        lib = gdspy.GdsLibrary() 
        lib.unit = 1.0e-6 
        lib.precision = 1e-9
        lib.write_gds(
                '/Users/devaldeliwala/research/AQT/fluxonium_fishbone/gds/single_qubit.gds', 
                cells=[cells[0].flatten()],
        )






        
        

