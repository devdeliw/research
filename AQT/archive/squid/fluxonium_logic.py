import gdspy
import ruamel.yaml as yaml
import qnldraw as qd
import numpy as np 
import os 
from qnldraw import Params, Chip
from qnldraw.junction import JunctionArray, JunctionLead
from qnldraw.junction import simulate_evaporation
from qnldraw.shapes import Rectangle

class Fluxonium: 
    def __init__(self, chip, params): 
        self.chip = chip 
        self.params = params 
        self.layers = params['global.layers']

        # check if antenna params & connector params exist 
        # if so, store universally for use in multiple methods 
        if self.params['antenna.antenna_gap']: 
            self.antenna_gap = self.params['antenna.antenna_gap']
        if self.params['dolan.connector.connector_gap']: 
            self.connector_gap = self.params['dolan.connector.connector_gap']

    def param_check(self): 
        # only runs if all topology components are being generated
        # logic to ensure all the parameters in self.params "make sense" 
        self.array_params = self.params['array'] 
        self.dolan_params = self.params['dolan']

        central_finger_params = self.dolan_params['finger.central']
        junction_lead_params = self.dolan_params['junction_lead']
        array_pad_params = self.dolan_params['finger.array_pad'] 

        ## connector gap must be large enough to ensure junction 
        ## arrays don't collide and must be larger than the 
        ## central finger and central finger undercut 
        min_connector_gap = max(
                (
                    central_finger_params['central_dim'][0] 
                    + central_finger_params['central_undercut_dim'][0]
                    + 1 # arbitrary num based on aesthetic
                ), 
                (
                    self.array_params['overlap'][0]
                    + 2*self.array_params['undercut']
                ),
                2*(
                    central_finger_params['central_dim'][0]
                    + self.params['global.lowdose.offset']
                    + 1 # arbitrary num based on aesthetic
                ),
                2*(
                    central_finger_params['central_undercut_dim'][0]
                    + self.params['global.lowdose.offset'] 
                    + 1 # arbitrary num based on aesthetic
                ), 
        )

        ## update self.connector_gap  
        if self.connector_gap < min_connector_gap: 
            self.params['dolan.connector'].update({'connector_gap': min_connector_gap}) 
            self.connector_gap = self.params['dolan.connector.connector_gap'] 

        ## antenna gap must be larger than the connector gap
        ## and the taper + inner length of the dolan junction leads 
        ## and the gap between the junction array pads 
        min_antenna_gap = 2*(
            self.connector_gap
            + array_pad_params['gap_extra'] 
            + array_pad_params['l_extra_offset'] 
            + junction_lead_params['extension']
            + junction_lead_params['taper_length'] 
            + junction_lead_params['inner.length'] 
        )

        if self.antenna_gap < min_antenna_gap: 
            self.params['antenna'].update({'antenna_gap': min_antenna_gap})
            self.antenna_gap = self.params['antenna.antenna_gap'] 

        return

    def lowdose(self, objects, exclusion, offset, layer):
        # method for adding lowdose undercuts 
        # around 'objects' parameters while 
        # maintaining no overlap with 'exclusion' 
        lead_lowdose = qd.offset(
            objects, 
            offset, 
            join_first=True, 
            join='round', 
            tolerance=1, 
        )
        lead_lowdose = qd.boolean(
            lead_lowdose, exclusion, 'not'
        )
        self.chip.add_component(
            lead_lowdose, 'lowdose', layers=layer
        )

        return

    def antenna(self): 
        # capactive pads & antennae 
        antenna_params = self.params['antenna'] 
        antenna_layers = self.layers['antenna']

        antenna_gap = antenna_params['antenna_gap']
        pad_length, pad_width = antenna_params['pad_dim'] 
        antenna_length, antenna_width = antenna_params['antenna_dim']

        # outer capacitive pads 
        self.pads = [
            Rectangle(pad_length, pad_width).translate(
                -(antenna_gap/2 + antenna_length + pad_length/2), 0), 
            Rectangle(pad_length, pad_width).translate(
                +(antenna_gap/2 + antenna_length + pad_length/2), 0), 
        ]

        # antennae connecting pads to dolan junction leads 
        self.antennas = [ 
            Rectangle(antenna_length, antenna_width).translate(
                -(antenna_gap/2 + antenna_length/2), 0), 
            Rectangle(antenna_length, antenna_width).translate(
                +(antenna_gap/2 + antenna_length/2), 0), 
        ]

        main_layer = antenna_layers['main']

        # -- Adding to chip 
        self.chip.add_component(self.pads, 'pads', layers=main_layer) 
        self.chip.add_component(self.antennas, 'antennas', layers=main_layer)

        return 

    def dolan(self, antisymmetric=False): 
        # all dolan bridge components 
        dolan_params = self.params['dolan'] 
        dolan_layers = self.layers['dolan']

        # dolan junction leads 
        junction_lead_params = dolan_params['junction_lead']
        junction_lead = JunctionLead(**junction_lead_params) 

        self.junction_leads = [
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

        ## inner junction - junctionlead connector leads 
        connector_params = dolan_params['connector']
        connector_gap = connector_params['connector_gap'] 

        ## dimension of connector leads 
        connector_lx = np.abs(self.junction_leads[1].node('lead_0')[0] - connector_gap/2)
        connector_ly = connector_params['connector_width']

        self.junction_connectors = [
            Rectangle(connector_lx, connector_ly).translate(
                -(connector_gap/2 + connector_lx/2), 0), 
            Rectangle(connector_lx, connector_ly).translate(
                +(connector_gap/2 + connector_lx/2), 0), 
        ]

        ### junction fingers 
        finger_params = dolan_params['finger']
        ### central finger 
        central_params = finger_params['central']

        lx, ly = central_params['central_dim']
        undercut_lx, undercut_ly = central_params['central_undercut_dim'] 

        self.central_finger = Rectangle(lx, ly).translate(lx/2, 0) 
        self.central_undercut = Rectangle(undercut_lx, undercut_ly).translate(-undercut_lx/2, 0)

        ### centering the central finger-central undercut combo around origin 
        x_min = self.central_undercut.get_bounding_box()[0][0]
        x_max = self.central_finger.get_bounding_box()[1][0]
        x_offset = (x_max - x_min) / 2 

        #self.central_finger.translate(np.abs(x_offset), 0)
        #self.central_undercut.translate(np.abs(x_offset), 0)

        ### thin cut leads connecting central finger(s) to connectors
        cut_params = dolan_params['cut_parameter']

        ### left and right thin cut lead lengths 
        llx = np.abs(
            self.central_undercut.get_bounding_box()[0][0]
            - self.junction_connectors[0].get_bounding_box()[1][0]
        )
        rlx = np.abs(
            self.central_undercut.get_bounding_box()[1][0] 
            - self.junction_connectors[1].get_bounding_box()[0][0]
        )

        ### left and right thin cut lead widths 
        lly = cut_params['L_width'] 
        rly = cut_params['R_width'] 

        ### left and right thin cut lead vertical offset from origin
        l_offset = cut_params['L_y_offset']
        r_offset = cut_params['R_y_offset']

        self.cut_leads = [ 
            Rectangle(llx, lly).translate(
                self.central_undercut.get_bounding_box()[0][0] - llx/2, 
                +l_offset
            ),
            Rectangle(rlx, rly).translate(
                self.central_undercut.get_bounding_box()[1][0] + rlx/2, 
                -r_offset
            )
        ]

        #### junction array pads w/ antisymmetric capability 
        array_pad_params = finger_params['array_pad'] 

        array_pad_offs = array_pad_params['offset']
        array_pad_gap_extra = array_pad_params['gap_extra'] 
        left_pad_extra_offs = array_pad_params['l_extra_offset']

        lx, ly = array_pad_params['array_pad_dim']
        undercut_lx = array_pad_params['undercut_length'] 

        #### antisymmetric logic 
        if not antisymmetric: 
            #### both array pads at same y 
            l_offs = -ly + array_pad_offs 
            r_offs = -ly + array_pad_offs 
        else: 
            #### left array pad above origin 
            l_offs = array_pad_offs 
            r_offs = -ly + array_pad_offs

        self.array_pad_undercuts = [
                Rectangle(undercut_lx, ly).translate(
                    -connector_gap/2-array_pad_gap_extra-left_pad_extra_offs, l_offs),
                Rectangle(undercut_lx, ly).translate(
                    +connector_gap/2+array_pad_gap_extra, r_offs), 
        ]
        self.array_pad_fingers = [
                Rectangle(lx, ly).translate(
                    -connector_gap/2 - array_pad_gap_extra+(undercut_lx+lx)/2-left_pad_extra_offs
                    , l_offs
                ), 
                Rectangle(lx, ly).translate(
                    +connector_gap/2 + array_pad_gap_extra+(undercut_lx+lx)/2
                    , r_offs
                ), 
        ]

        ##### adding all dolan components to chip 
        main_layer = dolan_layers['main']
        undercut_layer = dolan_layers['undercut'] 

        self.chip.add_component(self.junction_leads, 'junction leads', layers=main_layer) 

        #####  dolan bridge - dolan junction lead connectors 
        self.chip.add_component(self.junction_connectors, 'junction connectors', layers=main_layer) 

        #####  central junction finger & undercut 
        self.chip.add_component(self.central_finger, 'central finger', layers=main_layer) 
        self.chip.add_component(self.central_undercut, 'junction undercut', layers=undercut_layer)

        #####  thin cut parameter leads 
        self.chip.add_component(self.cut_leads, 'cut parameter leads', layers=main_layer)

        ##### array pads and undercut 
        self.chip.add_component(self.array_pad_fingers, 'array pad fingers', layers=main_layer) 
        self.chip.add_component(self.array_pad_undercuts, 'array pad undercuts', layers=undercut_layer)

        ###### adding an undercut around the antenna and all of dolan 
        ###### while maintaining an exclusion with thin cut leads 
        ###### and array pads 
        objects = [] 
        exclusion = []
        if hasattr(self, 'pads'):
            objects.append([self.pads[0], self.pads[1]])
            objects.append([self.antennas[0], self.antennas[1]])
        if hasattr(self, 'junction_leads'): 
            objects.append([self.junction_leads[0], self.junction_leads[1]])
            objects.append([self.junction_connectors[0], self.junction_connectors[1]])
            exclusion.append(
                [
                    self.cut_leads[0], 
                    self.cut_leads[1], 
                    self.array_pad_fingers[0], 
                    self.array_pad_fingers[1], 
                    self.array_pad_undercuts[0], 
                    self.array_pad_undercuts[1],
                ]
            )

        objects = np.array(objects).flatten()
        exclusion = np.array(exclusion).flatten()

        lowdose_layer = self.layers['antenna.undercut']
        lowdose_offs = self.params['global.lowdose.offset']
        self.lowdose(objects, np.concatenate((objects, exclusion)).flatten(), offset=lowdose_offs, layer=lowdose_layer)

        return 

    def array(self): 
        array_params = self.params['array']
        array_layers = self.layers['array']

        if hasattr(self, 'array_pad_undercuts'): 
            arr_pad_bbox = self.array_pad_undercuts[1].get_bounding_box()

            # determining absolute (x,y) positions to place junction 
            # arrays directly underneath array pads 
            x_pos = arr_pad_bbox[0][0] + (arr_pad_bbox[1][0] - arr_pad_bbox[0][0])/2
            y_pos = arr_pad_bbox[0][1]
        else: 
            x_pos = array_params['overlap'][0] + array_params['undercut'] + 1
            y_pos = 0
                    
        # junction array 
        array = JunctionArray(**array_params) 
        left_array = array.place(location=[-x_pos, y_pos], node='wire1') 
        right_array = array.place(location=[x_pos, y_pos], node='wire1') 

        # bottom connection between left and right junction arrays
        bottom_connection = Rectangle( 
                right_array.node('wire2')[0] - left_array.node('wire2')[0], 
                self.params['array_bottom_connection'], 
        ).translate(
                0, 
                left_array.node('wire2')[1] - self.params['array_bottom_connection']/2 + array_params['overlap'][1]
        )
        
        # all junction array components 
        self.array_components = [left_array, right_array, bottom_connection] 

        main_layer = array_layers['main']
        # adding junction array and bottom connection to chip 
        self.chip.add_component(self.array_components, 'junction arrays', layers=main_layer)

        return self.chip 

    def evap(self): 
        evap_params = self.params['evap']

        # assigning layers to HD or LD 
        highdose_layer = self.layers['dolan.main'] + [self.layers['array.main'][0]]
        lowdose_layer = self.layers['dolan.undercut'] + [self.layers['array.main'][1]]

        cells = self.chip.render(name='fluxonium', draw_border=False) 
        polys = cells[0].get_polygons(by_spec=True)

        highdose_components = []
        lowdose_components = []
        try: 
            highdose_components.extend(polys[(highdose_layer[0], 0)]) 
        except (NameError, KeyError) as e:
            pass
        try: 
            highdose_components.extend(polys[(highdose_layer[1], 0)])
        except (NameError, KeyError) as e:
            pass
        try:
            lowdose_components.extend(polys[(lowdose_layer[0], 0)])
        except (NameError, KeyError) as e:
            pass
        try:
            lowdose_components.extend(polys[(lowdose_layer[1], 0)])
        except (NameError, KeyError) as e:
            pass

        highdose_components = gdspy.PolygonSet(highdose_components)
        lowdose_components = gdspy.PolygonSet(lowdose_components)

        # perform evaporation simulation 
        evap_sim = simulate_evaporation(
            lowdose_components, 
            highdose_components, 
            **evap_params
        ) 

        # adding to chip
        for idx, (layer, evap) in enumerate(zip(self.layers['evap'], evap_sim)): 
            self.chip.add_component(evap, f'evap_{idx}', layers=layer) 

        return self.chip

    def square_mask(self, size = [520, 520]):
        # square mask surrounding fluxonium qubit 
        x_dim, y_dim = size 
        square = Rectangle(x_dim, y_dim)

        mask_layer = self.layers['mask.main']
        self.chip.add_component(square, 'square mask', layers=mask_layer) 

        return self.chip 

    def build(self, antenna=False, dolan=False, array=False, mask=True, evap=False): 

        if all([antenna, dolan, array]): 
            self.param_check() 
        if antenna: 
            self.antenna() 
        if dolan: 
            self.dolan()
        if array:
            self.array() 
        if mask: 
            self.square_mask() 
        if evap: 
            self.evap() 

        return self.chip 


class FluxoniumArray:
    def __init__(self, chip, params, 
                 antenna=True, 
                 dolan=True, 
                 array=True, 
                 mask=True, 
                 evap=True): 

        self.chip_array = chip
        self.params = params 
        self.layers = params['global.layers']

        single_qubit_chip = Chip()
        self.qubit = Fluxonium(single_qubit_chip, params).build(
            antenna=antenna, 
            dolan=dolan, 
            array=array, 
            mask=mask, 
            evap=evap
        )

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
                    print(f'Layer {layer} not found in chip; likely excluded during build')

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

    class_ = FluxoniumArray(chip, params, 
                            antenna=True, 
                            dolan=True, 
                            array=True, 
                            mask=True, 
                            evap=True
    )

    chip = class_.build(nx=1, ny=1, dx=2000, dy=2000)

    cells = chip.render('single_qubit', include_refs=True)

    lib = gdspy.GdsLibrary() 
    lib.unit = 1.0e-6 
    lib.precision = 1e-9

    file_dir = 'gds/' 
    file_name = 'fluxonium_final'
    
    if not os.path.exists(file_dir): 
        os.makedirs(file_dir)

    lib.write_gds(
        f'{file_dir}{file_name}.gds', 
        cells=[cells[0].flatten()], 
    )



























