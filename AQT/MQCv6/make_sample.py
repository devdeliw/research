import os
import re
import gdspy
import numpy as np
import ruamel.yaml as yaml

import qnldraw as qd
import qnldraw.library as qlib

from qnldraw import components, paths, shapes, Chip, Params, Angle
from qnldraw.paths import CPW


class ReadoutBus(components.Component): 
    __name__ = 'BUS'
    __draw__ = True

    def draw(self, cpw, length, length_reduction, coupler): 
        nodes = {}

        cpw_trace = CPW(**cpw, direction = '+y')
        cpw_trace.segment(length, '+y').open(cpw_trace.total_width())

        idc = qlib.InlineIDC(
            cpw=cpw,
            name='IDC',
            **coupler
        ).place(cpw_trace.initial_position(), node='cpw0', rotation=-90)

        nodes['top'] = cpw_trace.current_position() - (0, cpw_trace.total_width())
        nodes['bottom'] = cpw_trace.initial_position()

        nodes['qb_top'] = nodes['top'] - (0, length_reduction)
        nodes['qb_bottom'] = nodes['bottom'] + (0, length_reduction)

        nodes['center'] = (nodes['top'] + nodes['bottom']) / 2

        self.add([cpw_trace, idc])
        self.add_cutout(cpw_trace.cutout)

        return nodes


def draw_multiplexed_readout(chip, mpr_params):
    bus = ReadoutBus(cpw=mpr_params['cpw'], **mpr_params['purcell_filter'])
    bus = bus.place((0, mpr_params['bus_y_center']), node='center')

    chip.add_component(bus, cid='BUS', layers=mpr_params['layer'])

def draw_qubits(chip, qubit_params):

    Q = qlib.FloatingPads(
        pads={'x': 135, 'y': 545}, 
        spacing=65, 
        cutout={'x': 535, 'y': 745}
    )

    Q0 = Q.place()
    Q1 = Q.place((0, 2000))

    chip.add_component(Q0, layers=3)
    chip.add_component(Q1, layers=5) 

    return



if __name__ == '__main__':
    param_dir = './parameters'
    outfile = 'mask6.gds'
    gds_dir = './pattern_files/'

    with open(os.path.join(param_dir, 'mqcv6.yaml'), 'r') as f:
        sample_params = Params(yaml.load(f, yaml.Loader))

    with open(os.path.join(param_dir, 'multiplexed_readout.yaml'), 'r') as f:
        mpr_params = Params(yaml.load(f, yaml.Loader))

    lib = gdspy.GdsLibrary()
    lib.unit = 1.0e-6           # um
    lib.precision = 1.0e-9      # nm

    chip = Chip()
    draw_qubits(chip, sample_params)
    draw_multiplexed_readout(chip, mpr_params)

    ## This will be explained later
    intersections = [
        ([cid for cid in chip.components if re.match(r'^Q\d+', cid)] +
        [cid for cid in chip.components if re.match(r'^COUPLING', cid)])
    ]

    mask = chip.render('MASK', intersections=intersections, draw_border=True)
    lib.write_gds(os.path.join(gds_dir, outfile), cells=mask)












