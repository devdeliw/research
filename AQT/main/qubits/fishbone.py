import qnldraw as qd 
import ruamel.yaml as yaml 
import numpy as np 
import gdspy 
import os 

from qnldraw import components, Params, Chip 
from qnldraw.shapes import Rectangle
from qnldraw.junction import JunctionArray, JunctionLead, simulate_evaporation 

class CapacitorAntennae(components.Component):
    """
    Class for generating component for outer capacitor pads 
    and inductive leads. 

    """
    __name__ = 'CAPACITOR' 
    __draw__ = True 

    def draw(
            self, 
            lx, 
            ly, 
            lead_lx, 
            lead_ly, 
            gap, 
    ): 

        """
        Args: 
            lx, ly (float): dimensions of outer antenna pads.
            lead_lx, lead_ly (float): dimensions of inductive leads. 
            gap (float): gap between inductive leads for qubit.

        Returns: 
            capacitor_antennae joint component object. 

        """

        nodes = {} 

        # Capacitor Pads 
        left_pad = Rectangle(lx, ly).translate(
                -(gap/2 + lead_lx + lx/2), 0
        )
        right_pad = Rectangle(lx, ly).translate(
                +(gap/2 + lead_lx + lx/2), 0
        )

        # Connector Leads 
        left_lead = Rectangle(lead_lx, lead_ly).translate(
                -(gap/2 + lead_lx/2), 0
        )
        right_lead = Rectangle(lead_lx, lead_ly).translate(
                +(gap/2 + lead_lx/2), 0
        )

        # Establishing capacitor nodes 
        nodes['llead_end'] = (-gap/2, 0) 
        nodes['rlead_end'] = (+gap/2, 0)

        nodes['lpad_end'] = (-gap/2-lead_lx-lx, 0) 
        nodes['rpad_end'] = (+gap/2+lead_lx+lx, 0) 

        # Adding to component
        capacitor_antennae = [
            left_pad, right_pad, left_lead, right_lead
        ]
        self.add(capacitor_antennae) 

        return nodes 

class Connectors(components.Component): 
    """ 
    Class for generating inner connecting leads between the 
    JunctionLead objects and the central capactive finger. 

    """
    __name__ = 'CONNECTOR'
    __draw__ = True 

    def draw(
            self, 
            lx, 
            ly, 
            gap,
    ):
        """ 
        Args: 
            lx, ly (float): dimensions of connector leads. 
            gap (float): gap between connector leads. 

        Returns: 
            joint connector lead component object. 
        """
        nodes = {}

        # Left and right leads connecting JunctionLeads 
        # To inner capacitive fingers 
        l_connector = Rectangle(lx, ly).translate(-(gap/2 + lx/2), 0) 
        r_connector = Rectangle(lx, ly).translate(+(gap/2 + lx/2), 0) 

        # Establishing connector nodes 
        nodes['llead_end'] = (-gap/2, 0) 
        nodes['rlead_end'] = (+gap/2, 0)

        # Adding to component
        connectors = [
                l_connector, r_connector, 
        ]
        self.add(connectors) 

        return nodes 



class Render:
    """
    Class for rendering entire `fishbone`-architecture 
    fluxonium qubit. 

    """

    def __init__(self, chip, params): 
        """ 
        Args: 
            chip: qnldraw.Chip component
            params: Parameter dictionary object.

        """
        self.chip = chip 
        self.params = params 
        self.layers = params['layers']
        return

    def build_antenna(self): 
        # Antenna parameters
        antenna_params = self.params['antenna']
        pad_lx, pad_ly = antenna_params['pad_dim']
        lead_lx, lead_ly = antenna_params['lead_dim']
        gap = antenna_params['gap']

        # Build antenna component object
        antennae = CapacitorAntennae(
                pad_lx, 
                pad_ly, 
                lead_lx, 
                lead_ly,
                gap, 
        ).place((0,0))

        # Add component to chip 
        self.chip.add_component(
            antennae, 
            cid='capacitor', 
            layers=self.layers['antenna'], 
        )

        self.antennae = antennae
        return 

    def build_junction_leads(self): 
        # JunctionLead parameters 
        lead_params = self.params['junction_lead']
        junction_lead = JunctionLead(**lead_params) 

        # Normally, total_length or outer_length can be used
        # But, here I require an outer_length parameter 
        outer_length = lead_params['outer.length'] 

        # Establish placement location of JunctionLeads 
        # Relative to antennae lead nodes 
        llead_loc = np.subtract(
            self.antennae.node('llead_end'), [outer_length, 0]
        )
        rlead_loc = np.add(
            self.antennae.node('rlead_end'), [outer_length, 0]
        )

        # Place JunctionLead objects at correct location
        left_lead = junction_lead.place(location=llead_loc) 
        right_lead = junction_lead.place(location=rlead_loc, mirror='y')

        junction_leads = [left_lead, right_lead]
        # Add components to chip
        self.chip.add_component(
            junction_leads,
            cid='junction_leads', 
            layers=self.layers['leads']
        )

        self.junction_leads = junction_leads
        return 

    def build_connectors(self): 
        # Connector params
        connector_params = self.params['connector'] 
        gap = connector_params['gap'] 

        # Calculated connector dimensions
        lx = abs(self.junction_leads[0].node('lead_0')[0] + gap/2) 
        ly = connector_params['width'] 

        # Generate connector component
        connectors = Connectors(lx, ly, gap).place((0, 0))

        # Add component to chip
        self.chip.add_component(
                connectors, 
                cid='connectors', 
                layers=self.layers['leads']
        )

        self.connectors = connectors
        return 

    def build_fingers(self, antisymmetric=False): 
        """
        Method for building the finger-undercut components. 

        Includes: 
         - central finger-undercut component  
         - array pad finger-undercut component

        The central finger-undercut component lies at the center of the qubit. 
        The array pad finger-undercut component connects to the JunctionArray. 

        """
        # Finger params
        finger_params = self.params['fingers']

        # Finger undercut layer
        def central_finger(): 
            # Central params
            central_params = finger_params['central_fingers'] 
            lx, ly = central_params['finger_dim']
            undercut_lx, undercut_ly = central_params['undercut_dim']

            # Generate central finger component
            central_fingers = Rectangle(lx, ly)
            central_fingers.translate(lx/2, 0)

            # Add component to chip
            chip.add_component(
                central_fingers, 
                cid='capacitive_finger', 
                layers=self.layers['leads'] 
            )

            # ---- undercut ---- 

            # Generate central undercut component
            central_undercut = Rectangle(undercut_lx, undercut_ly)
            central_undercut.translate(-undercut_lx/2, 0) 

            # Add component to chip
            self.chip.add_component(
                central_undercut, 
                cid='capacitive_finger_undercut', 
                layers=self.layers['lowdose'], 
            )

            # `Node` for central undercut; will be used for inductive lead positioning
            self.central_undercut_edge = (-undercut_lx, 0) 
            self.central_undercut = central_undercut
            return 

        def array_pad(): 
            # Array pad params 
            array_params = finger_params['array_pads']
            pad_length = array_params['length']
            pad_width = array_params['width'] 
            lowdose_length = array_params['lowdose_length']

            # Distance between end of connector lead and 
            # closest end of array pad 
            x_offset = array_params['x_offset'] 

            # Distance that the array-pad protrudes off 
            # the connector leads 
            y_offset = array_params['y_offset'] 

            # Generate the right pad component
            right_pad = Rectangle(pad_length, pad_width)
            left_pad = Rectangle(pad_length, pad_width) 

            # ---- right pad ---- 
            # Similar to a node; will be used for positioning undercuts 
            right_pad_edge = (
                self.connectors.node('rlead_end')[0] + x_offset + lowdose_length,
                -(pad_width-y_offset),
            )
            right_pad.translate(
                right_pad_edge[0] + pad_length/2, 
                right_pad_edge[1]
            )
        
            # ---- left pad ---- 

            # If antisymmetric coupling, position the left 
            # array_pad further up the connector leads
            if antisymmetric: 
                self.antisymmetric = True 
                left_pad_edge = (
                    self.connectors.node('llead_end')[0] - x_offset, 
                    +y_offset,
                )
                left_pad.translate(
                    left_pad_edge[0] - pad_length/2, 
                    left_pad_edge[1],
                )
            else: 
                self.antisymmetric = False 
                left_pad_edge = (
                    self.connectors.node('llead_end')[0] - x_offset, 
                    -(pad_width-y_offset),
                )
                left_pad.translate(
                    left_pad_edge[0] - pad_length/2,
                    left_pad_edge[1],
                )


            array_pad_fingers = [left_pad, right_pad]
            # Add components to chip 
            chip.add_component(
                array_pad_fingers, 
                cid='array_pads', 
                layers=self.layers['leads']
            )

            # ---- undercut ---- 
            l_undercut = Rectangle(lowdose_length, pad_width)
            r_undercut = Rectangle(lowdose_length, pad_width) 

            # Positioning
            l_undercut_loc = (
                left_pad_edge[0] - pad_length - lowdose_length/2, 
                left_pad_edge[1]
            )
            r_undercut_loc = (
                right_pad_edge[0] - lowdose_length/2, 
                right_pad_edge[1]
            )
            l_undercut.translate(l_undercut_loc[0], l_undercut_loc[1])
            r_undercut.translate(r_undercut_loc[0], r_undercut_loc[1]) 

            array_pad_undercuts = [l_undercut, r_undercut] 
            # Add components to chip 
            self.chip.add_component(
                array_pad_undercuts, 
                cid='array_pad_undercuts', 
                layers=self.layers['lowdose']
            )

            # `Nodes` will also be used for inductive lead positioning
            self.right_pad_edge = right_pad_edge 
            self.left_pad_edge = left_pad_edge
            # y-value of base of array_pads; will be used for JunctionArrays
            self.right_pad_base = (
                right_pad_edge[0] - lowdose_length/2, 
                right_pad_edge[1] - pad_width/2,
            )
            self.left_pad_base = (
                -self.right_pad_base[0], 
                +self.right_pad_base[1],
            )

            self.array_pad_fingers = array_pad_fingers 
            self.array_pad_undercuts = array_pad_undercuts 

            return 

        central_finger() 
        array_pad()
        return 

    def build_inductive_leads(self): 
        # Inductive lead params 
        cut_params = self.params['inductive_leads'] 
        lly = cut_params['l_width'] 
        rly = cut_params['r_width'] 
        ly_offset = cut_params['ly_offset'] 
        ry_offset = cut_params['ry_offset']

        # Calculating lengths of leads 
        llx = abs(self.left_pad_edge[0] - self.central_undercut_edge[0]) 
        rlx = abs(self.right_pad_edge[0] - self.central_undercut_edge[1])

        # Generating lead components
        left_lead = Rectangle(llx, lly) 
        right_lead = Rectangle(rlx, rly) 

        # Positioning 
        # Left y_offset is dependent on antisymmetricity
        if self.antisymmetric: 
            ly_offset = +ly_offset 
        else: 
            ly_offset = -ly_offset
            
        left_lead = Rectangle(llx, lly).translate(
                self.central_undercut_edge[0]-llx/2, ly_offset,
        )
        right_lead = Rectangle(rlx, rly).translate(
                self.central_undercut_edge[1]+rlx/2, -ry_offset,
        )

        inductive_leads = [left_lead, right_lead] 
        # Adding components to chip 
        self.chip.add_component(
                inductive_leads, 
                cid='inductive_leads', 
                layers=self.layers['leads']
        )

        self.inductive_leads = inductive_leads
        return 

    def build_junction_array(self): 
        array_params = self.params['array'] 

        # Generating JunctionArray component
        array = JunctionArray(**array_params)

        # Positioning
        left_array = array.place(self.left_pad_base, node='wire1') 
        right_array = array.place(self.right_pad_base, node='wire1') 

        # ---- bottom array connection ---- 
        bottom_connection_width = self.params['bottom_connection_width']
        bottom_connection = Rectangle(
                right_array.node('wire2')[0] - left_array.node('wire2')[0],
                bottom_connection_width
        )
        bottom_connection.translate(
                0, 
                left_array.node('wire2')[1] - bottom_connection_width/2 + array_params['overlap'][1]
        )

        junction_array = [left_array, right_array, bottom_connection] 
        # Adding components to chip
        self.chip.add_component(
                junction_array, 
                cid='junction_array',
                layers=self.layers['array']
        )
        
        self.junction_array = junction_array
        return

    def lowdose(self): 
        """
        Adds lowdose around the antenna, inductive leads, and JunctionLeads. 

        """ 
        objects = [
                self.antennae,
                self.junction_leads[0],
                self.junction_leads[1], 
                self.connectors, 
        ]

        exclusion = [
                self.antennae, 
                self.junction_leads[0], 
                self.junction_leads[1], 
                self.connectors, 
                self.inductive_leads[0], 
                self.inductive_leads[1], 
                self.array_pad_fingers[0], 
                self.array_pad_fingers[1], 
                self.array_pad_undercuts[0], 
                self.array_pad_undercuts[1], 
        ]
       
        lowdose = qd.offset(
                objects,
                self.params['lowdose_offset'], 
                join_first=True, 
                join='round', 
                tolerance=1, 
        )
        lowdose = qd.boolean(
                lowdose,
                exclusion, 
                'not', 
        )

        # Adding component to chip 
        self.chip.add_component(
                lowdose, 
                cid='lowdose',
                layers=self.layers['lowdose'] 
        )
        return 

    def evap(self): 
        evap_params = self.params['evap']

        # Assigning layers to HD or LD 
        highdose_layer = [self.layers['leads']] + [self.layers['array'][0]] 
        lowdose_layer = [self.layers['lowdose']] + [self.layers['array'][1]]

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

        # Perform evaporation simulation 
        evap_sim = simulate_evaporation(
            lowdose_components, 
            highdose_components, 
            **evap_params
        ) 

        # Adding to chip
        for idx, (layer, evap) in enumerate(zip(self.layers['evap'], evap_sim)): 
            self.chip.add_component(evap, f'evap_{idx}', layers=layer) 

        return self.chip

    def render(
        self, 
        antenna=True, 
        junction=True,
        fingers=True, 
        connectors=True,
        leads=True,
        array=True,
        lowdose=False,
        simulate_evap=True
    ): 
        """ 
        Renders the qubit. 

        """

        def logic(): 
            """ 
            Ensures qubit can be rendered if certain components are 
            not generated. For components whose nodes are used to
            position other components, these values will be defaulted. 
    
            """
            if not antenna: 
                self.antennae = CapacitorAntennae(100, 50, 300, 10, 60).place((0, 0))
            if not connectors: 
                self.connectors = Connectors(7.85, 1, 13.3).place((0, 0))
            if not fingers: 
                self.left_pad_edge = (-7.4, 0.1)
                self.right_pad_edge = (8.1, -0.9)
                self.left_pad_base = (-7.75, -1.4)
                self.right_pad_base = (7.75, -1.4)
                self.central_undercut_edge = (-0.4, 0)
                self.antisymmetric=False
            return 

        logic()
        if antenna: 
            self.build_antenna()
        if junction: 
            self.build_junction_leads() 
        if connectors: 
            self.build_connectors()
        if fingers: 
            self.build_fingers(antisymmetric=True)
        if leads: 
            self.build_inductive_leads()
        if array:
            self.build_junction_array()
        if lowdose: 
            self.lowdose()
        if simulate_evap: 
            self.evap()
        return 










if  __name__ == '__main__':

    param_dir = './parameters'
    outfile = 'fishbone.gds'
    gds_dir = './pattern_files/'

    with open(f'{param_dir}/fishbone_params.yaml') as f: 
        params = yaml.load(f, Loader=yaml.SafeLoader) 
    params = Params(params)

    lib = gdspy.GdsLibrary()
    lib.unit = 1.0e-6           # μm
    lib.precision = 1.0e-9      # nm

    chip = Chip() 
    Render(chip, params).render()

    cells = chip.render('fishbone_qubit', include_refs=True)
    lib.write_gds(os.path.join(gds_dir, outfile), cells=[cells[0].flatten()])





