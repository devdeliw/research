import qnldraw as qd 
from qnldraw import components, Chip
from qnldraw.shapes import Rectangle
from qnldraw.junction import JunctionLead, Bandage, simulate_evaporation
from qnldraw import Params 

import os
import gdspy 
import numpy as np 
import ruamel.yaml as yaml 

class Bandages(components.Component):
    __name__ = "BANDAGES" 
    __draw__ = True 

    def draw(self, lx, ly, gap): 
        nodes = {}

        # Generating bandage component 
        bandage = Bandage(lx, ly) 

        # Positioning
        l_bandage = bandage.place((-gap/2-lx/2, 0)) 
        r_bandage = bandage.place((+gap/2+lx/2, 0)) 

        # Establishing important bandage nodes 
        nodes['l_edge'] = (-gap/2, 0)
        nodes['r_edge'] = (+gap/2, 0) 

        # Adding to component
        bandages = [l_bandage, r_bandage]
        self.add(bandages)

        return nodes 

class Render: 
    """ 
    Class for rendering FD4 qubit architecture. 

    """
    def __init__(self, chip, params): 
        self.chip = chip 
        self.params = params 
        self.hdlayer = params['layers.highdose']
        self.ldlayer = params['layers.lowdose'] 

    def build_bandages(self): 
        # Bandage params 
        bandage_params = params['bandages'] 
        lx, ly = bandage_params['lx'], bandage_params['ly'] 
        gap = bandage_params['gap']
        layer = self.params['layers.bandage']

        # Generating bandage component
        bandages = Bandages(lx, ly, gap)
        
        # Adding components to chip
        self.chip.add_component(
                bandages, 
                cid='bandages', 
                layers=layer, 
        )

        self.bandages = bandages 
        return 

    def build_junction_leads(self, add_to_chip=True):
        # JunctionLead params 
        junction_lead_params = self.params['junction_leads']
        bandage_overlap = junction_lead_params['bandage_overlap']

        # Generating junction lead component
        junction_lead = JunctionLead(**junction_lead_params)

        # Positioning relative to bandages
        l_lead_loc = (
                self.bandages.node('l_edge')[0] - bandage_overlap, 
                self.bandages.node('l_edge')[1]
        )
        r_lead_loc = (
                self.bandages.node('r_edge')[0] + bandage_overlap, 
                self.bandages.node('r_edge')[1]
        )

        # Placement
        l_lead = junction_lead.place(l_lead_loc)
        r_lead = junction_lead.place(r_lead_loc, mirror='y') 

        junction_leads = [l_lead, r_lead] 
        # Adding components to chip
        if add_to_chip:
            self.chip.add_component(
                    junction_leads, 
                    cid='junction_leads', 
                    layers=self.hdlayer, 
            )

        self.junction_leads = junction_leads 
        return 

    def build_inductive_leads(self): 
        # Inductive lead params
        lead_params = self.params['inductive_leads']
        lead_width = lead_params['width']
        gap = lead_params['gap']

        # Calculate length of inductive lead relative to junction leads
        lead_length = abs(self.junction_leads[1].node('lead_0')[0] - gap/2) 

        # Positioning 
        l_lead = Rectangle(lead_length, lead_width).translate(
                self.junction_leads[0].node('lead_0')[0] + lead_length/2, 0
        )
        r_lead = Rectangle(lead_length, lead_width).translate(
                self.junction_leads[1].node('lead_0')[0] - lead_length/2, 0
        )

        leads = [l_lead, r_lead] 
        # Adding components to chip
        self.chip.add_component(
                leads, 
                cid='inductive_leads', 
                layers=self.hdlayer, 
        )

        # Component nodes for positioning dolan junction leads 
        self.l_lead_edge = (-gap/2, lead_width/2)
        self.r_lead_edge = (+gap/2, lead_width/2)

        self.inductive_leads = leads 
        return 

    def build_outer_junction(self, add_to_chip=True): 
        # Outer junction params 
        outer_params = self.params['outer_junction'] 

        # Outer junction component
        outer_lead = JunctionLead(**outer_params)

        # Origin positions for outer junction leads
        left_lead_location = (
                self.l_lead_edge[0] + outer_params['inner.width']/2, 
                self.l_lead_edge[1] - (outer_params['total_length']+outer_params['extension']),
        )

        right_lead_location = (
                self.r_lead_edge[0] - outer_params['inner.width']/2, 
                self.r_lead_edge[1] - (outer_params['total_length']+outer_params['extension']), 
        )

        # Positioning
        l_lead = outer_lead.place(left_lead_location, rotation=90)
        r_lead = outer_lead.place(right_lead_location, rotation=90)

        leads = [l_lead, r_lead]
        # Adding components to chip 
        if add_to_chip:
            self.chip.add_component(
                    leads,
                    cid='outer_junction_leads', 
                    layers=self.hdlayer, 
            )

        self.outer_junction_leads = leads 
        return

    def build_dolan_leads(self): 
        # Dolan lead params 
        dolan_lead_params = self.params['antisym_leads'] 
        l_lx = dolan_lead_params['left_lx']
        r_lx = dolan_lead_params['right_lx']
        width = dolan_lead_params['width']

        # Calculating positions 
        left_location = (
                self.outer_junction_leads[0].node('lead_2')[0] + l_lx/2, 
                self.outer_junction_leads[0].node('lead_2')[1] - width/2,
        )

        # Needed for right location calculation
        junction_inner_width = self.params['outer_junction.inner.width'] 

        right_location = (
                self.outer_junction_leads[1].node('lead_0')[0] - r_lx/2 - junction_inner_width/2, 
                self.outer_junction_leads[1].node('lead_0')[1] - width/2, 
        )

        # Generating components 
        l_lead = Rectangle(l_lx, width).translate(
                left_location[0], left_location[1],
        )
        r_lead = Rectangle(r_lx, width).translate(
                right_location[0], right_location[1],
        )

        leads = [l_lead, r_lead]
        # Adding components to chip 
        self.chip.add_component(
                leads, 
                cid='dolan_leads', 
                layers=self.hdlayer, 
        )

        # Nodes for positioning inner dolan junction afterward
        self.right_edge_base = (
            right_location[0] - r_lx/2, right_location[1] - width/2,
        )
        self.left_edge_base = (
            left_location[0] + l_lx/2, left_location[1] + width/2, 
        )

        self.antisym_dolan_leads = leads
        return 

    def build_inner_dolan(self, add_to_chip=True): 
        # Inner dolan JunctionLead params 
        dolan_params = self.params['inner_junction'] 

        junction_lead = JunctionLead(**dolan_params) 

        # Calculating junction lead origin locations 
        left_lead_location = (
                self.left_edge_base[0] - dolan_params['outer.width']/2, 
                self.left_edge_base[1]
        )
        right_lead_location = (
                self.right_edge_base[0] + dolan_params['outer.width']/2, 
                self.right_edge_base[1]
        )

        # Positioning 
        l_lead = junction_lead.place(left_lead_location, rotation=90)
        r_lead = junction_lead.place(right_lead_location, rotation=-90) 

        leads = [l_lead, r_lead] 
        # Adding components to chip 
        if add_to_chip:
            self.chip.add_component(
                    leads, 
                    cid='inner_dolan_leads',
                    layers=self.hdlayer, 
            )

        self.inner_dolan_leads = leads
        return

    def build_central_lead(self): 
        # Central lead params 
        lead_params = self.params['central_lead']
        lx, ly = lead_params['lx'], lead_params['ly']

        # Calculating lead location 
        location = (
                self.inner_dolan_leads[0].node('lead_1')[0] + lx/2, 
                self.inner_dolan_leads[0].node('lead_1')[1] + ly/2, 
        )

        # Calculating width of undercut
        undercut_ly = abs(
            self.inner_dolan_leads[1].node('lead_0')[1] - (location[1] + ly/2)
        )

        undercut_location = (
                location[0], 
                location[1] + ly/2 + undercut_ly/2, 
        )

        # Generating components and positioning
        lead = Rectangle(lx, ly).translate(
                location[0], location[1]
        ) 
        undercut_lead = Rectangle(lx, undercut_ly).translate(
                undercut_location[0], undercut_location[1]
        )

        # Adding highdose lead 
        self.chip.add_component(
                lead, 
                cid='central_lead', 
                layers=self.hdlayer, 
        )

        # Adding lowdose lead 
        self.chip.add_component(
                undercut_lead, 
                cid='central_lead_undercut', 
                layers=self.ldlayer, 
        )

        self.central_lead = lead 
        self.central_undercut_lead = undercut_lead
        return

    def build_array(self): 
        # Array params 
        array_params = self.params['array'] 
        outer_junction_params = self.params['outer_junction']

        number = array_params['number']
        length = outer_junction_params['outer.width']
        width = array_params['width'] 
        gap = array_params['gap'] 

        # Establishing upper origins for junction array before looping
        left_top_origin = [
            length/2 - self.outer_junction_leads[0].node("origin")[0],
            width - self.outer_junction_leads[0].node("origin")[1],
        ]
        right_top_origin = [
            length/2 - self.outer_junction_leads[1].node("origin")[0],
            width - self.outer_junction_leads[1].node("origin")[1],
        ]


        left_lowdose_origin = [
            left_top_origin[0],
            left_top_origin[1] + gap
        ] 
        right_lowdose_origin = [
            right_top_origin[0], 
            right_top_origin[1] + gap
        ]

        # Building array
        cid = 0
        for _ in range(number // 2): 
            left_rect = Rectangle(length, width, left_top_origin)
            right_rect = Rectangle(length, width, right_top_origin)

            left_lowdose_rect = Rectangle(length, gap, left_lowdose_origin)
            right_lowdose_rect = Rectangle(length, gap, right_lowdose_origin) 

            # Adding highdose to component
            self.chip.add_component(
                    [left_rect, right_rect], 
                cid=f'array_{cid}', 
                layers=self.hdlayer, 
            )

            # Adding lowdose to component except for lowest array rung
            if cid != number//2 - 1: 
                self.chip.add_component(
                        [left_lowdose_rect, right_lowdose_rect], 
                        cid=f'array_ld_{cid}', 
                        layers=self.ldlayer, 
                )

                # Updating origins, moving one down the array
                left_top_origin = np.subtract(
                    left_top_origin, 
                    [0, -(width+gap)]
                )
                right_top_origin = np.subtract(
                    right_top_origin, 
                    [0, -(width+gap)]
                )
                left_lowdose_origin = np.subtract(
                    left_lowdose_origin, 
                    [0, -(width+gap)]
                )
                right_lowdose_origin = np.subtract(
                    right_lowdose_origin, 
                    [0, -(width+gap)],
                )
            cid+=1

        # ---- bottom connection ---- 
        bottom_params = self.params['bottom_junction']

        # Lower junction leads 
        left_origin = (
                bottom_params['outer.width']/2 - left_top_origin[0], 
                -left_top_origin[1], 
        )
        right_origin = (
                bottom_params['outer.width']/2 - right_top_origin[0], 
                -right_top_origin[1], 
        )

        # Generating component 
        bottom_junctions = JunctionLead(**bottom_params)
        left_junction = bottom_junctions.place(left_origin, rotation=-90)
        right_junction = bottom_junctions.place(right_origin, rotation=-90) 

        # Connecting wire lead 
        bottom_origin = (
                bottom_params['inner.width']/2 
                - left_junction.node('origin')[0], 
                + bottom_params['total_length']
                + bottom_params['inner.width']
                - left_junction.node('origin')[1], 
        )

        length = abs(
                left_junction.node('origin')[0] - right_junction.node('origin')[0]
        )
        length += bottom_params['inner.width']
        width = bottom_params['inner.width']

        # Bottom connector lead component
        bottom_lead = Rectangle(length, width, bottom_origin)

        # Adding bottom junctions to component 
        self.chip.add_component(
                [left_junction, right_junction], 
                cid='bottom_junctions',
                layers=self.hdlayer, 
        )

        # Adding connecting lead to component 
        self.chip.add_component(
                bottom_lead, 
                cid='bottom_lead', 
                layers=self.hdlayer, 
        ) 

        self.left_bottom_junction = left_junction
        self.right_bottom_junction = right_junction 
        self.bottom_lead = bottom_lead
        return 

    def evap(self): 
        """
        Method for performing evaporation simulation. 

        """
        # Extract all polygons currently in chip
        cell = self.chip.render('junction_mask', include_refs=False)[0]
        all_objects = cell.get_polygons(by_spec=True) 

        # Filter highdose and lowdose components
        highdose = gdspy.PolygonSet(all_objects[(self.hdlayer, 0)])
        lowdose = gdspy.PolygonSet(all_objects[(self.ldlayer, 0)])

        # Perform evap sim
        evaporated = simulate_evaporation(
            lowdose, highdose, **self.params['evaporation']
        )

        # Add evaporation layers to chip
        for idx, (layer, evap) in enumerate(
            zip(self.params['layers.evap'], evaporated)
        ):
            self.chip.add_component(evap, cid=f"evap_{idx}", layers=layer)
        return


    def render(
        self,
        bandages=True, 
        junction_leads=True, 
        inductive_leads=True, 
        outer_junction_leads=True, 
        dolan_junction_leads=True, 
        inner_dolan_leads=True, 
        central_lead=True, 
        junction_array=True, 
        simulate_evap=True
    ): 


        def logic():
            """
            Method to ensure gds file still gets rendered. 
            Many components depend on the positions of others, 
            so in the case an important component is not generated, 
            set its default position. 

            """
            if not bandages:
                self.bandages = Bandages(10, 15, 95)
            if not junction_leads: 
                # still requires the `junction_lead` parameters
                # to exist in the .yaml file 
                self.build_junction_leads(add_to_chip=False)
            if not inductive_leads: 
                self.l_lead_edge = (-6.35, 0.5)
                self.r_lead_edge = (+6.35, 0.5)
            if not outer_junction_leads: 
                # still requires the 'outer_junction` parameters 
                # to exist in the .yaml file
                self.build_outer_junction(add_to_chip=False)
            if not dolan_junction_leads: 
                self.left_edge_base = (0.3, -4.92) 
                self.right_edge_base = (-0.3, -0.5)
            if not inner_dolan_leads: 
                # still requires the 'inner_junction' parameters 
                # to exist in the .yaml file
                self.build_inner_dolan(add_to_chip=False)
            return

        logic()
        if bandages: 
            self.build_bandages()
        if junction_leads:
            self.build_junction_leads()
        if inductive_leads:
            self.build_inductive_leads()
        if outer_junction_leads:
            self.build_outer_junction() 
        if dolan_junction_leads:
            self.build_dolan_leads()
        if inner_dolan_leads:
            self.build_inner_dolan()
        if central_lead:
            self.build_central_lead()
        if junction_array:
            self.build_array()
        if simulate_evap: # do only if everything else is also rendered
            self.evap()

        return 



if __name__ == '__main__': 
    param_dir = './parameters'
    outfile = 'fd4.gds'
    gds_dir = './pattern_files/'

    with open(f'{param_dir}/fd4_params.yaml') as f: 
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params = Params(params)

    lib = gdspy.GdsLibrary() 
    lib.unit = 1.0e-6           # μm
    lib.precision = 1.0e-9      # nm

    chip = Chip() 
    Render(chip, params).render() 

    cells = chip.render('fd4_qubit', include_refs=True)
    lib.write_gds(os.path.join(gds_dir, outfile), cells=[cells[0].flatten()])



            





        





