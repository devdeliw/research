import gdspy
import ruamel.yaml as yaml
import qnldraw as qd
import numpy as np 
import os 

from qnldraw import Params, Chip
from qnldraw.junction import JunctionArray, JunctionLead
from qnldraw.junction import simulate_evaporation
from qnldraw.shapes import Rectangle


class FluxoniumQubit: 
    def __init__(self, chip, params):
        self.chip = chip
        self.params = params 
        self.layers = params['global.layers'] 

        # -- Storing essential spacings
        # -- Used in multiple methods 
        self.antenna_gap = self.params['antenna.antenna_gap']
        self.connector_gap = self.params['dolan.connector.connector_gap']

    def param_check(self): 
        self.central_finger_params = self.params['dolan.finger.central']
        self.array_params = self.params['array']
        self.dolan_params = self.params['dolan.junction_lead']
        self.array_pad_params = self.params['dolan.finger.array_pad'] 

        min_connector_gap = (
            self.central_finger_params['central_dim'][0] 
            + self.central_finger_params['central_undercut_dim'][0] 
            + 0.5
        )

        if self.connector_gap < min_connector_gap: 
            self.params['dolan.connector'].update({'connector_gap': min_connector_gap})
            self.connector_gap = self.params['dolan.connector.connector_gap']

        min_connector_gap = (
            self.array_params['overlap'][0]
            + 2*(self.array_params['undercut'])
        )

        if self.connector_gap < min_connector_gap: 
            self.params['dolan.connector'].update({'connector_gap': min_connector_gap})
            self.connector_gap = self.params['dolan.connector.connector_gap']

        min_antenna_gap = 2*(
            self.connector_gap 
            + self.array_pad_params['gap_extra'] 
            + self.array_pad_params['l_extra_offset'] 
            + self.dolan_params['extension'] 
            + self.dolan_params['taper_length']
            + self.dolan_params['inner.length']
        )

        if self.antenna_gap < min_antenna_gap: 
            self.params['antenna'].update({'antenna_gap': min_antenna_gap})
            self.antenna_gap = self.params['antenna.antenna_gap'] 

        return 

    def lowdose(self, objects, exclusion, offset, layer): 
        lead_lowdose = qd.offset(
            objects, 
            offset, 
            join_first=True, 
            join='round', 
            tolerance=1, 
        )
        lead_lowdose = qd.boolean(
            lead_lowdose, exclusion,  'not', layer=layer
        )
        self.chip.add_component(
            lead_lowdose, 'lowdose', layer=layer
        )

        return 

    def antenna(self): 
        # -- Capacitive Pads \& Antennas
        antenna_params = self.params['antenna'] 
        antenna_layers = self.layers['antenna']

        antenna_gap = antenna_params['antenna_gap']
        pad_length, pad_width = antenna_params['pad_dim'] 
        antenna_length, antenna_width = antenna_params['antenna_dim']

        undercut_extension = antenna_params['undercut_extension']

        pads = [
            Rectangle(pad_length, pad_width).translate(
                -(antenna_gap/2 + antenna_length + pad_length/2), 0), 
            Rectangle(pad_length, pad_width).translate(
                +(antenna_gap/2 + antenna_length + pad_length/2), 0), 
        ]

        antennas = [ 
            Rectangle(antenna_length, antenna_width).translate(
                -(antenna_gap/2 + antenna_length/2), 0), 
            Rectangle(antenna_length, antenna_width).translate(
                +(antenna_gap/2 + antenna_length/2), 0), 
        ]

        undercuts = [
            Rectangle(pad_length+2*undercut_extension, pad_width).translate(
                -(antenna_gap/2 + antenna_length + pad_length/2), 0), 
            Rectangle(pad_length+2*undercut_extension, pad_width).translate(
                +(antenna_gap/2 + antenna_length + pad_length/2), 0), 
        ]

        undercuts = [
                qd.boolean(undercuts[0], pads[0], 'not'),
                qd.boolean(undercuts[1], pads[1], 'not'), 
        ]

        main_layer = antenna_layers['main']
        undercut_layer = antenna_layers['undercut']

        # -- Adding to chip 
        self.chip.add_component(pads, 'pads', layers=main_layer) 
        self.chip.add_component(antennas, 'antennas', layers=main_layer)
        self.chip.add_component(undercuts, 'undercuts', layers=undercut_layer)

        return self.chip

    def dolan(self, antisymmetric=False): 
        dolan_params = self.params['dolan'] 
        dolan_layers = self.layers['dolan']

        # -- JunctionLeads 
        junction_lead_params = dolan_params['junction_lead']
        junction_lead = JunctionLead(**junction_lead_params) 

        junction_leads = [
            junction_lead.place(
                location = [
                    -(
                        self.antenna_gap/2 
                        + junction_lead_params['total_length']
                        - junction_lead_params['inner.length'] 
                        - junction_lead_params['taper_length']
                    ), 
                    0, 
                ],
            ), 
            junction_lead.place(
                location = [
                    +(
                        self.antenna_gap/2
                        + junction_lead_params['total_length']
                        - junction_lead_params['inner.length']
                        - junction_lead_params['taper_length']
                    ),
                    0,
                ],
                mirror = 'y',
            ),
        ]
      
        # -- Junction-Antenna connector leads 
        connector_params = dolan_params['connector'] 

        connector_gap = connector_params['connector_gap'] 
        lx = abs(junction_leads[1].node('lead_0')[0]-connector_gap/2)
        ly = connector_params['connector_width']

        junction_connectors = [
                Rectangle(lx, ly).translate(-(connector_gap/2 + lx/2), 0), 
                Rectangle(lx, ly).translate((connector_gap/2 + lx/2), 0), 
        ]

        # -- Junction Fingers 
        finger_params = dolan_params['finger']

        ## -- Central Finger 
        central_params = finger_params['central'] 

        lx, ly = central_params['central_dim']
        undercut_lx, undercut_ly = central_params['central_undercut_dim']

        central_finger = Rectangle(lx, ly).translate(lx/2, 0)
        central_undercut = Rectangle(undercut_lx, undercut_ly).translate(-undercut_lx/2, 0)

        x_min, x_max = central_undercut.get_bounding_box()[0][0], central_finger.get_bounding_box()[1][0]
        x_offset = -(x_min + x_max)/2 

        central_finger.translate(x_offset, 0)
        central_undercut.translate(x_offset, 0)

        ## -- Thin Cut Leads conneciting central finger to array pads 
        cut_params = dolan_params['cut_parameter']

        llx = abs(central_undercut.get_bounding_box()[0][0] - junction_connectors[0].get_bounding_box()[1][0])
        lly = cut_params['L_width']
        l_offset = cut_params['L_y_offset']

        rlx = abs(central_undercut.get_bounding_box()[1][0] - junction_connectors[1].get_bounding_box()[0][0])
        rly = cut_params['R_width']
        r_offset = cut_params['R_y_offset']


        cut_leads = [
            Rectangle(llx, lly).translate(central_undercut.get_bounding_box()[0][0] - llx/2, l_offset), 
            Rectangle(rlx, rly).translate(central_undercut.get_bounding_box()[1][0] + rlx/2, -r_offset), 
        ]

        # -- Symmetric or Antisymmetric array pads 
        array_pad_params = finger_params['array_pad']

        array_pad_gap_extra = array_pad_params['gap_extra']
        array_pad_offs = array_pad_params['offset']
        left_pad_extra_offset = array_pad_params['l_extra_offset']

        lx, ly = array_pad_params['array_pad_dim']
        undercut_lx = array_pad_params['undercut_length']

        if not antisymmetric: 
            l_offs = -ly+array_pad_offs 
            r_offs = -ly+array_pad_offs 
        else: 
            l_offs =  array_pad_offs 
            r_offs = -ly+array_pad_offs

        array_pad_undercuts = [
                Rectangle(undercut_lx, ly).translate(
                    -connector_gap/2-array_pad_gap_extra-left_pad_extra_offset, l_offs),
                Rectangle(undercut_lx, ly).translate(
                    +connector_gap/2+array_pad_gap_extra, r_offs), 
        ]
        array_pad_fingers = [
                Rectangle(lx, ly).translate(
                    -connector_gap/2 - array_pad_gap_extra+(undercut_lx+lx)/2-left_pad_extra_offset, l_offs), 
                Rectangle(lx, ly).translate(
                    +connector_gap/2 + array_pad_gap_extra+(undercut_lx+lx)/2, r_offs), 
        ]

        # -- Adding all components to chip 
        main_layer = dolan_layers['main']
        undercut_layer = dolan_layers['undercut']

        ## -- JunctionLeads 
        self.chip.add_component(junction_leads, 'junction leads', layers=main_layer) 

        ## -- Junction-Antenna Connectors 
        self.chip.add_component(junction_connectors, 'junction connectors', layers=main_layer) 

        ## -- Central Junction Finger \& Undercut 
        self.chip.add_component(central_finger, 'central finger', layers=main_layer) 
        self.chip.add_component(central_undercut, 'junction undercut', layers=undercut_layer)

        ## -- Thin Cut Parameter Leads 
        self.chip.add_component(cut_leads, 'cut parameter leads', layers=main_layer)

        ## Sym or Asym Array Pads \& Undercut 
        self.chip.add_component(array_pad_fingers, 'array pad fingers', layers=main_layer) 
        self.chip.add_component(array_pad_undercuts, 'array pad undercuts', layers=undercut_layer)

        self.array_pad_undercuts = array_pad_undercuts


        return self.chip 

    def array(self): 
        array_params = self.params['array']
        array_layers = self.layers['array']
        array_pad_params = self.params['dolan.finger.array_pad']
        connector_params = self.params['dolan.connector']
        gap = connector_params['connector_gap'] + array_pad_params['gap_extra']


        array_pad_width = array_pad_params['array_pad_dim'][1]
        connector_width = connector_params['connector_width']
        array_pad_offset = array_pad_params['offset']

        y_pos = self.array_pad_undercuts[1].get_bounding_box()[0][1]
        x_pos = self.array_pad_undercuts[1].get_bounding_box()[0][0] + (
            self.array_pad_undercuts[1].get_bounding_box()[1][0]
            - self.array_pad_undercuts[1].get_bounding_box()[0][0]
        )/2 


                    

        # -- JunctionArray; The # of array elements defined in params 
        array = JunctionArray(**array_params) 

        left_array = array.place(location=[-x_pos, y_pos], node='wire1') 
        right_array = array.place(location=[x_pos, y_pos], node='wire1') 

        bottom_connection = Rectangle( 
                right_array.node('wire2')[0] - left_array.node('wire2')[0], 
                self.params['array_bottom_connection'], 
        ).translate(
                0, 
                left_array.node('wire2')[1] - self.params['array_bottom_connection']/2 + array_params['overlap'][1]
        )
        array_components = [left_array, right_array, bottom_connection] 

        main_layer = array_layers['main']
        # -- Adding JunctionArray \& Bottom Connection to chip 
        self.chip.add_component(array_components, 'junction arrays', layers=main_layer)

        return self.chip 

    def evap(self):
        evap_params = self.params['evap']

        # -- Assigning Layers to HD or LD
        highdose_layer = self.layers['dolan.main'] + [self.layers['array.main'][0]]
        lowdose_layer = self.layers['dolan.undercut'] + [self.layers['array.main'][1]]

        cells = self.chip.render(name='fluxonium', draw_border=False) 
        polys = cells[0].get_polygons(by_spec=True)

        highdose_components = gdspy.PolygonSet(
                polys[(highdose_layer[0], 0)]
                + polys[(highdose_layer[1], 0)], 
        )
        lowdose_components = gdspy.PolygonSet(
                polys[(lowdose_layer[0], 0)]
                + polys[(lowdose_layer[1], 0)], 
        )

        evap_sim = simulate_evaporation(
            lowdose_components, 
            highdose_components, 
            **evap_params
        ) 

        # -- Adding to chip
        for idx, (layer, evap) in enumerate(zip(self.layers['evap'], evap_sim)): 
            self.chip.add_component(evap, f'evap_{idx}', layers=layer) 

        return self.chip 

    def square_mask(self, size = [520, 520]):
        # -- Square Mask 
        x_dim, y_dim = size 
        square = Rectangle(x_dim, y_dim)

        mask_layer = self.layers['mask.main']
        self.chip.add_component(square, 'square mask', layers=mask_layer) 

        return self.chip 

    def build(self, antenna=True, dolan=True, array=True, square_mask=True, evap=True): 
        self.param_check()
        if antenna: 
            self.antenna() 
        if dolan: 
            self.dolan()
        if array:
            self.array() 
        if square_mask: 
            self.square_mask() 
        if evap: 
            self.evap() 

        return self.chip


class FluxoniumQubitArray:
    def __init__(self, chip, params): 
        self.chip_array = chip
        self.params = params 
        self.layers = params['global.layers']

        single_qubit_chip = Chip()
        self.qubit = FluxoniumQubit(single_qubit_chip, params).build()

    def build(self, nx, ny, dx, dy):
        # -- Building array of Fluxonium qubits 
        cells = self.qubit.render(name='fluxonium', draw_border=False) 
        polys = cells[0].get_polygons(by_spec=True)

        # -- All layers of individual qubit
        antenna_layers = self.layers['antenna']
        dolan_layers = self.layers['dolan']
        array_layers = self.layers['array']
        evap_layers = self.layers['evap']
        mask_layer = self.layers['mask']

        layers = [ 
                antenna_layers['main'][0],
                antenna_layers['undercut'][0], 
                dolan_layers['main'][0],
                dolan_layers['undercut'][0], 
                array_layers['main'][0], 
                array_layers['main'][1], 
                mask_layer['main'][0], 
                evap_layers[0], 
                evap_layers[1],
        ]

        # -- Converting SingleQubit chip into constituent components 
        # -- Assigning layers to each component 
        # -- Adding back components to universal chip 
        self.count = 0
        self.empty_layer = []
        def poly_to_component(layer, x, y):
            ansatz_component = qd.components.Component()
            poly_cell = gdspy.Cell(f'init_{self.count}')
            try: 
                poly_set = gdspy.PolygonSet(polys[(layer, 0)])

                poly_cell.add(poly_set)
                component = qd.components.Component.fromcell(poly_cell)
                ansatz_component.add(component).assign_layers(layer)
                
                self.count+=1
                return ansatz_component.place((x, y))
            except: 
                if layer not in self.empty_layer:
                    print(f'Layer {layer} not found in SingleQubit Chip')

                self.empty_layer.append(layer)
                self.count+=1
                return 

        # -- Evenly spaced x, y positions for array 
        xs = dx*(np.arange(nx) - 0.5*(nx-1))
        ys = dy*(np.arange(ny) - 0.5*(ny-1))

        # -- Adding each component of each array to universal chip 
        for y in ys: 
            for x in xs: 
                for layer in layers: 
                    component = poly_to_component(layer, x, y)
                    if component: 
                        self.chip_array.add_component(component)

        return self.chip_array
          



if __name__ == '__main__': 
    with open('params/fluxonium_qubit.yaml') as f: 
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params = Params(params) 

    chip = Chip() 

    class_ = FluxoniumQubitArray(chip, params)
    chip = class_.build(1, 1, 2000, 2000)

    cells = chip.render('single_qubit', include_refs=True)

    lib = gdspy.GdsLibrary() 
    lib.unit = 1.0e-6 
    lib.precision = 1e-9

    file_dir = 'gds/' 
    file_name = 'fluxonium_array'
    
    if not os.path.exists(file_dir): 
        os.makedirs(file_dir)

    lib.write_gds(
        f'{file_dir}{file_name}.gds', 
        cells=[cells[0].flatten()], 
    )








        

