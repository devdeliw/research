import os

import gdspy
import numpy as np
from numpy.core.multiarray import inner
import qnldraw as qd
import qnldraw.junction as qj
import qnldraw.shapes as shapes
import ruamel.yaml as yaml
from qnldraw import Chip, Params
from qnldraw.junction import JunctionArray, JunctionLead, simulate_evaporation
from qnldraw.shapes import Rectangle


class FD4:
    def __init__(self, chip, params):
        self.chip = chip
        self.params = params
        self.layers = params["global.layers"]

        # -- Spacing between inner ends of JunctionLeads
        self.inner_spacing = params["global.inner_spacing"]

        # -- Spacing between outer ends of JunctionLeads
        self.outer_spacing = params["global.outer_spacing"]

    def lowdose(self, objects, exclusion, offset):
        # -- Method for lowdose offset
        lead_lowdose = qd.offset(
            objects,
            offset,
            join_first=True,
            join="round",
            tolerance=1,
        )
        # -- Boolean Operation \& Adding to chip
        lead_lowdose = qd.boolean(
            lead_lowdose, exclusion, "not", layer=self.layers["lowdose"]
        )
        self.chip.add_component(
            lead_lowdose, cid="lowdose", layers=self.layers["lowdose"]
        )

    def bandages(self):
        bandage_params = self.params["bandage"]
        # -- How much Bandage(s) overlaps junction leads
        bandage_lx = bandage_params["lx"]
        bandage_overlap = bandage_params["bandage_overlap"]

        bandage = qj.Bandage(**bandage_params)

        # -- Left \& Right Bandages
        self.bandages = [
            bandage.place(
                (
                    -(self.outer_spacing / 2 - bandage_overlap + bandage_lx / 2),
                    0,
                )
            ),
            bandage.place(
                (
                    +(self.outer_spacing / 2 - bandage_overlap + bandage_lx / 2),
                    0,
                )
            ),
        ]

        # -- Adding to chip
        bandage_layers = self.layers["bandage"]
        self.chip.add_component(self.bandages, "bandages", layers=bandage_layers)

        return self.chip

    def outer_junction_leads(self):
        # -- Outer Junction Leads
        junction_params = self.params["outer_junction_lead"]

        junction = JunctionLead(**junction_params)

        # -- Left \& Right JunctionLeads
        self.outer_junctions = [
            junction.place((-self.outer_spacing / 2, 0)),
            junction.place((+self.outer_spacing / 2, 0), mirror="y"),
        ]

        # -- Adding to chip
        junction_layer = self.layers["highdose"]
        self.chip.add_component(
            self.outer_junctions, "junction leads", layers=junction_layer
        )

        return self.chip

    def connector(self):
        # -- Connectors to inner junction
        connector_params = self.params["connector"]
        junction_params = self.params["outer_junction_lead"]
        dolan_params = self.params["dolan"]

        junction_length = junction_params["total_length"] + junction_params["extension"]

        connector_width = connector_params["width"]
        junction_lead_gap = self.outer_spacing - 2 * junction_length
        self.right_lx = junction_lead_gap / 2 + dolan_params["finger_offset"]
        self.left_lx = junction_lead_gap / 2 - self.inner_spacing - connector_width

        # -- Left \& Right Connector Leads
        left_origin_x = self.outer_junctions[0].node("lead_0")[0]
        right_origin_x = self.outer_junctions[1].node("lead_0")[0]
        self.connectors = [
            Rectangle(self.left_lx, connector_width).translate(
                left_origin_x + self.left_lx / 2, 0
            ),
            Rectangle(self.right_lx, connector_width).translate(
                right_origin_x - self.right_lx / 2, 0
            ),
        ]

        # -- Adding to chip
        connector_layer = self.layers["highdose"]
        self.chip.add_component(
            self.connectors, "connector leads", layers=connector_layer
        )

        return self.chip

    def array_junction_leads(self):
        # -- JunctionLeads connecting to JunctionArray
        junction_params = self.params["array_junction_lead"]

        connector_params = self.params["connector"]
        connector_width = connector_params["width"]

        junction_total_length = (
            junction_params["total_length"] + junction_params["extension"]
        )
        junction_inner_width = junction_params["inner.width"]

        junction = JunctionLead(**junction_params)

        left_junction_origin = np.add(
            self.outer_junctions[0].node("lead_0"),
            [
                self.left_lx + junction_inner_width / 2,
                -(junction_total_length - connector_width / 2),
            ],
        )
        right_junction_origin = [-left_junction_origin[0], left_junction_origin[1]]

        # -- Left \& Right Array-JunctionLeads
        self.array_junction_leads = [
            junction.place(left_junction_origin, rotation=90),
            junction.place(right_junction_origin, rotation=90),
        ]

        # -- Adding to chip
        array_junction_lead_layer = self.layers["highdose"]
        self.chip.add_component(
            self.array_junction_leads,
            "array junction leads",
            layers=array_junction_lead_layer,
        )

        return self.chip

    def dolan(self):
        dolan_params = self.params["dolan"]
        junction_params = self.params["array_junction_lead"]
        connector_params = self.params["connector"]
        connector_width = connector_params["width"]

        dolan_leads = JunctionLead(**dolan_params)

        # -- Calculating origins of Dolan JunctionLeads
        right_connector_bb = self.connectors[1].get_bounding_box()[0]
        right_origin = [
            right_connector_bb[0] + dolan_params["outer.width"] / 2,
            right_connector_bb[1],
        ]
        left_origin = [
            -right_origin[0],
            (
                right_origin[1]
                - dolan_params["total_length"] * 2
                - dolan_params["junction_gap"]
            ),
        ]

        # -- Left \& Right Dolan JunctionLeads
        self.dolan_leads = [
            dolan_leads.place(left_origin, rotation=90),
            dolan_leads.place(right_origin, rotation=-90),
        ]

        # -- Central Dolan Finger in Junction
        central_finger_lx = dolan_params["finger_length"]
        central_finger_ly = dolan_params["finger_width"]

        bottom_left_corner = [
            self.dolan_leads[0].node("inner_lead")[0] - dolan_params["inner.width"] / 2,
            self.dolan_leads[0].node("inner_lead")[1],
        ]
        central_finger_origin = [
            bottom_left_corner[0] + central_finger_lx / 2,
            bottom_left_corner[1] + central_finger_ly / 2,
        ]

        self.central_finger = Rectangle(central_finger_lx, central_finger_ly).translate(
            central_finger_origin[0], central_finger_origin[1]
        )
        
        central_finger_undercut_ly = abs(
            self.dolan_leads[0].node('lead_0')[1]
            + central_finger_ly 
            -self.dolan_leads[1].node('lead_0')[1]
        )

        self.central_finger_undercut = Rectangle(
            central_finger_lx, 
            central_finger_undercut_ly).translate(
                central_finger_origin[0], 
                central_finger_origin[1]+central_finger_ly/2+central_finger_undercut_ly/2
        )

        # -- Left Connection to Inner Dolan JunctionLeads
        right_bound = (
            self.dolan_leads[0].node("origin")[0] + dolan_params["outer.width"] / 2
        )
        left_bound = (
            self.array_junction_leads[0].node("origin")[0]
            - junction_params["inner.width"] / 2
        )

        bottom_connection_length = abs(right_bound - left_bound)

        bottom_connection_origin = [
            right_bound - bottom_connection_length / 2,
            self.dolan_leads[0].node("origin")[1] - connector_width / 2,
        ]
        self.bottom_wire = Rectangle(
            bottom_connection_length, connector_width
        ).translate(bottom_connection_origin[0], bottom_connection_origin[1])

        # -- Adding to chip
        dolan_lead_layer = self.layers['highdose']
        dolan_undercut_layer = self.layers['lowdose']
        self.chip.add_component(
            self.dolan_leads, "dolan junction leads", layers=dolan_lead_layer
        )
        self.chip.add_component(
            self.central_finger, "central finger", layers=dolan_lead_layer
        )
        self.chip.add_component(
            self.central_finger_undercut, 'central finger undercut', layers=dolan_undercut_layer
        )
        self.chip.add_component(
            self.bottom_wire, "bottom connection", layers=dolan_lead_layer
        )

        ##  -- Adding lowdose to Outer Junction Leads, Connectors
        objects = [
            self.connectors[0],
            self.connectors[1],
            self.outer_junctions[0],
            self.outer_junctions[1],
        ]
        exclusion = [
            self.array_junction_leads[0],
            self.array_junction_leads[1],
            self.dolan_leads[1],
        ]
        self.lowdose(
            objects,
            exclusion=objects + exclusion,
            offset=self.params["outer_junction_lead.undercut"],
        )

        return self.chip

    def junction_array(self):
        array_params = self.params["array"]
        junction_params = self.params["array_junction_lead"]

        number = array_params["number"]
        width = array_params["width"]
        gap = array_params["gap"]
        length = junction_params["outer.width"]

        # -- Determining Origin to Start JunctionArray
        left_lead_origin = self.array_junction_leads[0].node("origin")
        right_lead_origin = self.array_junction_leads[1].node("origin")

        left_top_origin = [
            length / 2 - left_lead_origin[0],
            width - left_lead_origin[1],
        ]
        right_top_origin = [
            length / 2 - right_lead_origin[0],
            width - right_lead_origin[1],
        ]

        left_low_dose_origin = [left_top_origin[0], left_top_origin[1] + gap]
        right_low_dose_origin = [right_top_origin[0], right_top_origin[1] + gap]

        # -- Building Josephson Array
        cid = 0
        for _ in range(number // 2):
            left_rect = shapes.Rectangle(length, width, left_top_origin)
            right_rect = shapes.Rectangle(length, width, right_top_origin)

            left_lowdose_rect = shapes.Rectangle(length, gap, left_low_dose_origin)
            right_lowdose_rect = shapes.Rectangle(length, gap, right_low_dose_origin)

            self.chip.add_component(
                [left_rect, right_rect],
                cid=f"array_{cid}",
                layers=self.layers["highdose"],
            )

            if cid != number // 2 - 1:
                self.chip.add_component(
                    [left_lowdose_rect, right_lowdose_rect],
                    cid=f"array_ld_{cid}",
                    layers=self.layers["lowdose"],
                )

                left_top_origin = np.subtract(left_top_origin, [0, -(width + gap)])
                right_top_origin = np.subtract(right_top_origin, [0, -(width + gap)])

                left_low_dose_origin = np.subtract(
                    left_low_dose_origin, [0, -(width + gap)]
                )
                right_low_dose_origin = np.subtract(
                    right_low_dose_origin, [0, -(width + gap)]
                )

            cid += 1
        self.left_origin = left_top_origin
        self.right_origin = right_top_origin

        return self.chip

    def bottom_connection(self):
        bottom_params = self.params["bottom_junction"]

        # -- Lower JunctionLeads
        left_origin = [
            bottom_params["outer.width"] / 2 - self.left_origin[0],
            -self.left_origin[1],
        ]
        right_origin = [
            bottom_params["outer.width"] / 2 - self.right_origin[0],
            -self.right_origin[1],
        ]

        bottom_junctions = qj.JunctionLead(**bottom_params)
        self.left_bottom_junction = bottom_junctions.place(left_origin, rotation=-90)
        self.right_bottom_junction = bottom_junctions.place(right_origin, rotation=-90)

        # -- Connecting Bottom Lead
        bottom_origin = [
            bottom_params["inner.width"] / 2
            - self.left_bottom_junction.node("origin")[0],
            +bottom_params["total_length"]
            + bottom_params["inner.width"]
            - self.left_bottom_junction.node("origin")[1],
        ]

        length = abs(
            self.left_bottom_junction.node("origin")[0]
            - self.right_bottom_junction.node("origin")[0]
        )
        length += bottom_params["inner.width"]
        width = bottom_params["inner.width"]

        self.bottom_wire = shapes.Rectangle(length, width, bottom_origin)

        # -- Adding to chip
        bottom_connection_layer = self.layers["highdose"]
        self.chip.add_component(
            [self.left_bottom_junction, self.right_bottom_junction],
            cid="bottom_junctions",
            layers=bottom_connection_layer,
        )
        self.chip.add_component(
            self.bottom_wire, cid="bottom_wire", layers=bottom_connection_layer
        )
        return self.chip

    def evap(self):
        cell = self.chip.render("junction_mask", include_refs=False)[0]
        all_objects = cell.get_polygons(by_spec=True)

        highdose = gdspy.PolygonSet(all_objects[(self.layers["highdose"], 0)])
        lowdose = gdspy.PolygonSet(all_objects[(self.layers["lowdose"], 0)])

        evaporated = qj.simulate_evaporation(lowdose, highdose, **self.params["evap"])

        for idx, (layer, evap) in enumerate(zip(self.layers["evap"], evaporated)):
            self.chip.add_component(evap, cid=f"evap_{idx}", layers=layer)

        return self.chip

    def build(
        self, bandages=True, junction_leads=True, dolan=True, array=True, evap=True
    ):
        # Add Bandages to chip
        if bandages:
            self.bandages()

        # Add outer junction leads and connectors to chip
        if junction_leads:
            self.outer_junction_leads()
            self.connector()
            self.array_junction_leads()

        # Add dolan bridge & inner josephson junction to chip
        if dolan:
            self.dolan()

        # Add josephson array to chip
        if array:
            self.junction_array()
            self.bottom_connection()

        # Evaporation simulation
        if evap:
            self.evap()

        return self.chip


if __name__ == "__main__":
    with open("params/fd4_qubit.yaml") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params = Params(params)

    chip = Chip()
    class_ = FD4(chip, params)
    chip = class_.build()

    cells = chip.render("FD4", include_refs=True)
    lib = gdspy.GdsLibrary()
    lib.unit = 1.0e-6
    lib.precision = 1e-9

    file_dir = "gds/"
    file_name = "FD4"

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    lib.write_gds(f"{file_dir}/{file_name}.gds", cells=[cells[0].flatten()])
