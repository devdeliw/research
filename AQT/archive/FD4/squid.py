 from collections.abc import Iterable
from itertools import product

import gdspy
import numpy as np
import qnldraw as qd
import qnldraw.junction as qj
import qnldraw.shapes as shapes
import ruamel.yaml as yaml
from qnldraw import Angle, Params, components, paths
from qnldraw.paths import Trace


class SQUID_Dolan:
    """
    Class to construct the FD4 Fluxonium Dolan Junction.

    Builds the FD4 Fluxonium Dolan junction mask using QNLDraw.
    Includes bandages, leads, traces, and evaporation simulation.

    parameters:
        chip (qnldraw.Chip): The chip object where the junction components will be added.
        params (qnldraw.Params): A qnl.Params object containing the necessary design parameters

    defs:
        bandages():
            creates the bandage structures on the ends of the chip.

        junction_lead_bandage():
            constructs the junction leads, attaches bandages to them, and adds these components to the chip.

        define_layers():
            defines the high-dose and low-dose gds layers for lithography.

        lowdose(objects, exclusion):
            generates low-dose lead regions by offsetting the given objects and excluding specified areas.

        traces():
            creates the wire traces that connect various parts of the junction and adds them to the chip.

        dolan():
            constructs the Dolan bridges as part of the junction design.

        junction_array():
            builds an array of junctions if required by the design parameters, adding them to the chip.

        bottom_connection():
            creates the bottom trace connection of the junction.

        simulate_evaporation():
            simulates the evaporation process to model the final physical structure of the junction,
            adding the resulting components to the chip.

        run():
            executes the full sequence of methods to build the junction and returns the updated chip object.
    """

    def __init__(self, chip, params):
        self.chip = chip
        self.params = params

    # -- bandages
    def bandages(self):
        bandage = qj.Bandage(**self.params["bandages"])
        return bandage

    # -- with junction leads
    def junction_lead_bandage(self):
        pad_spacing = self.params["pad_spacing"]
        lead_params = self.params["leads"]

        lead_v_separation = np.sqrt(self.params["squid_loop_area"])
        lead_h_separation = np.sqrt(self.params["squid_loop_area"])

        lead_params.update(
            {"extension": (lead_params["inner.width"] - self.params["wire.width"]) / 2}
        )

        leads = qj.JunctionLead(**lead_params)
        dist = pad_spacing / 2 + lead_params["pad_overlap"]

        self.left_lead = leads.place((-dist, 0))
        self.right_lead = leads.place((dist, 0), mirror="y")

        bandage = self.bandages()
        bandage_param = self.params["bandages"]
        self.left_bandage = bandage.attach(
            self.left_lead, "origin", "right", offset=(bandage_param["lead_overlap"], 0)
        )
        self.right_bandage = bandage.attach(
            self.right_lead,
            "origin",
            "left",
            offset=(-bandage_param["lead_overlap"], 0),
        )

        self.chip.add_component(
            [self.left_lead, self.right_lead], "leads", layers=self.hdlayer
        )
        self.chip.add_component(
            [self.left_bandage, self.right_bandage],
            "bandages",
            layers=self.params["bandages"]["layer"],
        )
        return

    # low and high dose layers for simulation
    def define_layers(self):
        self.hdlayer = self.params["layers"]["highdose"]
        self.ldlayer = self.params["layers"]["lowdose"]
        return

    # -- method for lowdose leakage
    def lowdose(self, objects, exclusion):
        lead_lowdose = qd.offset(
            objects,
            self.params["leads.undercut"],
            join_first=True,
            join="round",
            tolerance=1,
        )
        lead_lowdose = qd.boolean(lead_lowdose, exclusion, "not", layer=self.ldlayer)
        self.chip.add_component(lead_lowdose, cid="lowdose", layers=self.ldlayer)
        return

    def traces(self):
        # -- wires from junction leads
        pad_spacing = self.params["pad_spacing"]
        lead_params = self.params["leads"]

        junction_lead_tot = (
            pad_spacing + 2 * lead_params["pad_overlap"] + 2 * self.params["wire.width"]
        )
        junction_lead_ind = (
            lead_params["total_length"]
            + (lead_params["inner.width"] + params["wire.width"]) / 2
        )

        junction_gap = abs(junction_lead_tot - 2 * junction_lead_ind)

        left_trace_length = junction_gap / 2 - (
            lead_params["outer.width"] + params["wire.undercut"] + 1
        )
        right_trace_length = junction_gap / 2
        trace_width = self.params["wire.width"]

        self.left_trace = shapes.Rectangle(
            left_trace_length,
            trace_width,
            origin=-np.subtract(self.left_lead.node("lead_0"), [0, trace_width / 2]),
        )
        self.right_trace = shapes.Rectangle(
            right_trace_length + 0.13,
            trace_width,
            origin=-np.subtract(
                self.right_lead.node("lead_0"),
                [right_trace_length + 0.13, trace_width / 2],
            ),
        )
        outer_lead_params = self.params["outer_junction"]
        outer_lead_length = outer_lead_params["total_length"]
        outer_lead_width = outer_lead_params["inner.width"]
        extension = outer_lead_params["extension"]

        # -- outer vertical leads
        outer_leads = qj.JunctionLead(**outer_lead_params)
        self.outer_left_lead = outer_leads.place(
            np.add(
                self.left_lead.node("lead_0"),
                [
                    left_trace_length + outer_lead_width / 2,
                    -(extension + outer_lead_length) + outer_lead_width / 2,
                ],
            ),
            rotation=90,
        )
        self.outer_right_lead = outer_leads.place(
            [
                abs(self.outer_left_lead.node("origin")[0]),
                self.outer_left_lead.node("origin")[1],
            ],
            rotation=90,
        )
        # -- left wire to inner junction
        taper_length = outer_lead_params["taper_length"]
        inner_length = outer_lead_params["inner.length"]

        self.left_inner_trace = shapes.Rectangle(
            lead_params["outer.width"] + self.params["wire.undercut"] + 1.3,
            outer_lead_width,
            origin=-np.add(
                self.outer_left_lead.node("origin"),
                [-outer_lead_width / 2, taper_length],
            ),
        )

        self.chip.add_component(
            [self.left_trace, self.right_trace], cid="traces", layers=self.hdlayer
        )
        self.chip.add_component(
            [self.outer_left_lead, self.outer_right_lead],
            cid="outer_junctions",
            layers=self.hdlayer,
        )
        self.chip.add_component(
            self.left_inner_trace, cid="left_inner_trace", layers=self.hdlayer
        )
        return

    def dolan(self):
        dolan_params = self.params["junction"]
        lead_params = self.params["leads"]

        # -- lower and upper dolan junctions
        dolan_junction_leads = qj.JunctionLead(**dolan_params)

        # -- lower
        lower_origin = [
            self.left_inner_trace.get_bounding_box()[1][0]
            - dolan_params["outer.width"] / 2,
            self.left_inner_trace.get_bounding_box()[1][1],
        ]
        self.lower_dolan_junction = dolan_junction_leads.place(
            lower_origin, rotation=90
        )

        # -- upper
        upper_origin = [
            self.right_trace.get_bounding_box()[0][0] + dolan_params["outer.width"] / 2,
            self.right_trace.get_bounding_box()[0][1],
        ]
        self.upper_dolan_junction = dolan_junction_leads.place(
            upper_origin, rotation=-90
        )

        # -- the lower horizontal trace in between the JunctionLeads
        middle_origin = [
            dolan_params["inner.width"] / 2
            - self.lower_dolan_junction.node("lead_0")[0],
            -self.lower_dolan_junction.node("lead_0")[1],
        ]
        self.middle_trace = shapes.Rectangle(
            dolan_params["middle_length"],
            dolan_params["middle_width"],
            origin=middle_origin,
        )
        # -- the lowdose layer in between the dolan junction leads
        lowdose_origin = [
            middle_origin[0],
            middle_origin[1] - dolan_params["middle_width"],
        ]
        height = abs(
            self.upper_dolan_junction.node("lead_0")[1]
            + (middle_origin[1] - dolan_params["middle_width"])
        )
        self.middle_lowdose_layer = shapes.Rectangle(
            dolan_params["middle_length"], height, origin=lowdose_origin
        )

        # -- adding lowdose around junctionLeads
        objects = [self.left_trace, self.right_trace, self.left_lead, self.right_lead]
        exclusion = [
            self.outer_left_lead,
            self.outer_right_lead,
            self.upper_dolan_junction,
        ]
        self.lowdose(objects=objects, exclusion=objects + exclusion)

        self.chip.add_component(
            [self.lower_dolan_junction, self.upper_dolan_junction],
            cid="lower_dolan_junction",
            layers=self.hdlayer,
        )
        self.chip.add_component(
            self.middle_trace, cid="middle_trace", layers=self.hdlayer
        )
        self.chip.add_component(
            self.middle_lowdose_layer, cid="middle_ld_trace", layers=self.ldlayer
        )
        return

    def junction_array(self):
        array_params = self.params["array"]
        outer_params = self.params["outer_junction"]

        number = array_params["number"]
        width = array_params["width"]
        length = outer_params["outer.width"]
        gap = array_params["gap"]

        left_top_origin = [
            outer_params["outer.width"] / 2 - self.outer_left_lead.node("origin")[0],
            width - self.outer_left_lead.node("origin")[1],
        ]
        right_top_origin = [
            outer_params["outer.width"] / 2 - self.outer_right_lead.node("origin")[0],
            width - self.outer_right_lead.node("origin")[1],
        ]
        left_low_dose_origin = [left_top_origin[0], left_top_origin[1] + gap]
        right_low_dose_origin = [right_top_origin[0], right_top_origin[1] + gap]

        # -- building the josephson array
        cid = 0
        for _ in range(number // 2):
            left_rect = shapes.Rectangle(length, width, left_top_origin)
            right_rect = shapes.Rectangle(length, width, right_top_origin)

            left_lowdose_rect = shapes.Rectangle(length, gap, left_low_dose_origin)
            right_lowdose_rect = shapes.Rectangle(length, gap, right_low_dose_origin)

            self.chip.add_component(
                [left_rect, right_rect], cid=f"array_{cid}", layers=self.hdlayer
            )

            if cid != number // 2 - 1:
                self.chip.add_component(
                    [left_lowdose_rect, right_lowdose_rect],
                    cid=f"array_ld_{cid}",
                    layers=self.ldlayer,
                )

            if cid != number // 2 - 1:
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
        return

    def bottom_connection(self):
        bottom_params = self.params["bottom_junction"]

        # -- lower junction leads
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

        # -- connecting wire trace
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

        self.chip.add_component(
            [self.left_bottom_junction, self.right_bottom_junction],
            cid="bottom_junctions",
            layers=self.hdlayer,
        )
        self.chip.add_component(
            self.bottom_wire, cid="bottom_wire", layers=self.hdlayer
        )
        return

    def simulate_evaporation(self):
        cell = self.chip.render("junction_mask", include_refs=False)[0]
        all_objects = cell.get_polygons(by_spec=True)

        highdose = gdspy.PolygonSet(all_objects[(self.hdlayer, 0)])
        lowdose = gdspy.PolygonSet(all_objects[(self.ldlayer, 0)])

        evaporated = qj.simulate_evaporation(
            lowdose, highdose, **self.params["evaporation"]
        )

        for idx, (layer, evap) in enumerate(
            zip(self.params["evaporation.layers"], evaporated)
        ):
            self.chip.add_component(evap, cid=f"evap_{idx}", layers=layer)

        return

    def run(self):
        self.define_layers()
        self.junction_lead_bandage()
        self.traces()
        self.dolan()
        self.junction_array()
        self.bottom_connection()
        self.simulate_evaporation()
        return self.chip


if __name__ == "__main__":
    with open("FD4/squid.yaml") as f:
        params = yaml.load(f, Loader=yaml.Loader)
    params = Params(params)

    chip = qd.Chip()
    class_ = SQUID_Dolan(chip, params)
    chip = class_.run()

    cells = chip.render("chip_with_leads_and_bandages", include_refs=True)
    lib = gdspy.GdsLibrary()
    lib.unit = 1.0e-6
    lib.precision = 1e-9
    lib.write_gds("FD4/pattern_files/squid_dolan.gds", cells=[cells[0].flatten()])
