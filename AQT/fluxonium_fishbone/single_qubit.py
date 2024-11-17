import os

import gdspy
import matplotlib.pyplot as plt
import numpy as np
import qnldraw as qd
import qnldraw.junction as qj
import ruamel.yaml as yaml
from qnldraw import Chip, Params, components, shapes
from qnldraw.components import Component
from qnldraw.junction import JunctionArray, JunctionLead
from qnldraw.paths import Trace
from qnldraw.shapes import Rectangle


class Single:
    def __init__(self, chip, params):
        self.chip = chip
        self.params = params
        self.main = params["global"]

    def antenna(self):
        # -- adding capactive pads and antennas to chip
        antenna_params = self.params["antenna"]

        pad_length, pad_width = antenna_params["pad_dim"]
        antenna_length, antenna_width = antenna_params["antenna_dim"]
        pad_gap = antenna_params["pad_gap"]

        antennas = [
            Rectangle(antenna_length, antenna_width).translate(
                -(pad_gap / 2 + antenna_length / 2), 0
            ),
            Rectangle(antenna_length, antenna_width).translate(
                (pad_gap / 2 + antenna_length / 2), 0
            ),
        ]

        pads = [
            Rectangle(pad_length, pad_width).translate(
                -(pad_gap / 2 + antenna_length + pad_length / 2), 0
            ),
            Rectangle(pad_length, pad_width).translate(
                (pad_gap / 2 + antenna_length + pad_length / 2), 0
            ),
        ]

        self.pad_gap = pad_gap
        self.chip.add_component(
            antennas, "antenna_leads", layers=antenna_params["layer"]
        )
        self.chip.add_component(pads, "pads", layers=antenna_params["layer"])

        return

    def junction(self):
        junction_params = self.params["junction"]

        lead_params = junction_params["leads"]

        # -- junction leads
        junction_lead = JunctionLead(**lead_params)
        left_junction_lead = junction_lead.place(
            location=[
                -(
                    self.pad_gap / 2
                    + lead_params["total_length"]
                    - lead_params["inner.length"]
                    - lead_params["taper_length"]
                ),
                0,
            ]
        )
        right_junction_lead = junction_lead.place(
            location=[
                (
                    self.pad_gap / 2
                    + lead_params["total_length"]
                    - lead_params["inner.length"]
                    - lead_params["taper_length"]
                ),
                0,
            ],
            mirror="y",
        )

        # -- junction connector to antenna
        connector_params = junction_params["connector"]

        lx, ly = connector_params["connector_dim"]
        left_junction_connector = Rectangle(lx, ly).translate(
            -(connector_params["connector_gap"] / 2 + lx / 2),
            0,
        )
        right_junction_connector = Rectangle(lx, ly).translate(
            (connector_params["connector_gap"] / 2 + lx / 2),
            0,
        )

        finger_params = junction_params["fingers"]

        # -- central junction finger
        central_params = finger_params["central"]
        junction_finger = Rectangle(
            central_params["length"], central_params["width"]
        ).translate((central_params["length"] / 2), 0)
        junction_undercut = Rectangle(
            central_params["undercut_length"],
            central_params["width"] + central_params["undercut_extension"],
        ).translate(-central_params["undercut_length"] / 2, 0)

        # -- antisymmetric junction fingers
        antisym_params = finger_params["antisymmetric"]

        left_antisym_finger = Rectangle(
            antisym_params["length"], antisym_params["width"]
        ).translate(-antisym_params["gap"] / 2, antisym_params["offset"])
        right_antisym_finger = Rectangle(
            antisym_params["length"], antisym_params["width"]
        ).translate(
            antisym_params["gap"] / 2,
            -(antisym_params["width"] - antisym_params["offset"]),
        )

        left_antisym_finger_undercut = Rectangle(
            antisym_params["undercut_length"], antisym_params["width"]
        ).translate(
            (
                -antisym_params["gap"] / 2
                - antisym_params["length"]
                + (antisym_params["undercut_length"] - antisym_params["width"]) / 2
            ),
            antisym_params["offset"],
        )
        right_antisym_finger_undercut = Rectangle(
            antisym_params["undercut_length"], antisym_params["width"]
        ).translate(
            (
                antisym_params["gap"] / 2
                + antisym_params["length"]
                + (antisym_params["undercut_length"] - antisym_params["length"]) / 2
            ),
            -(antisym_params["width"] - antisym_params["offset"]),
        )

        # -- cut parameters
        cut_params = junction_params["cut_parameter"]
        left_lead = Rectangle(cut_params["L_length"], cut_params["L_height"]).translate(
            -central_params["undercut_length"] - cut_params["L_length"] / 2,
            cut_params["L_offset"],
        )
        right_lead = Rectangle(
            cut_params["R_length"], cut_params["R_height"]
        ).translate(cut_params["R_length"] / 2, -cut_params["R_offset"])

        # -- junction leads
        self.chip.add_component(
            [left_junction_lead, right_junction_lead],
            "lead",
            layers=lead_params["layer"],
        )
        # -- junction antenna connectors
        self.chip.add_component(
            [left_junction_connector, right_junction_connector],
            "connector",
            layers=lead_params["layer"],
        )
        # -- central junction finger & undercut
        self.chip.add_component(junction_finger, "finger", layers=lead_params["layer"])
        self.chip.add_component(
            junction_undercut, "finger undercut", layers=self.main["layers.undercut"]
        )
        # -- anyisymmetric junction fingers & undercut
        self.chip.add_component(
            [left_antisym_finger, right_antisym_finger],
            "antisym finger",
            layers=lead_params["layer"],
        )
        self.chip.add_component(
            [left_antisym_finger_undercut, right_antisym_finger_undercut],
            "antisym undercut",
            layers=self.main["layers.undercut"],
        )

        # -- cut parameter leads
        self.chip.add_component(
            [left_lead, right_lead], "cut param leads", layers=lead_params["layer"]
        )

        return

    def array(self):
        array_params = self.params["array"]
        antisym_params = self.params["junction.fingers.antisymmetric"]
        gap = antisym_params["gap"]
        connector_params = self.params["junction.connector"]
        depth = (
            antisym_params["width"]
            + antisym_params["offset"]
            + connector_params["connector_dim"][1] / 2
        )
        array = JunctionArray(**array_params)

        left_array = array.place(location=[-gap / 2, -depth], node="wire1")
        right_array = array.place(location=[gap / 2, -depth], node="wire1")

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

        self.chip.add_component(
            [left_array, right_array, bottom_connector],
            "junction arrays",
            layers=[0, 1, 2],
        )

        return chip


if __name__ == "__main__":
    with open("single_qubit.yaml") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
        params = Params(params)

        chip = qd.Chip()
        class_ = Single(chip, params)
        class_.antenna()
        class_.junction()
        chip = class_.array()

        cells = chip.render("temp", include_refs=True)
        lib = gdspy.GdsLibrary()
        lib.unit = 1.0e-6
        lib.precision = 1e-9
        lib.write_gds("gds/single_qubit.gds", cells=[cells[0].flatten()])
