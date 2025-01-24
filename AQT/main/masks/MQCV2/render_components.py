import os 
import gdspy
import numpy as np 
import ruamel.yaml as yaml

import qnldraw as qd 
import qnldraw.library as qlib 

from qnldraw import components, paths, shapes, Angle, Chip, Params
from qnldraw.paths import CPW 


class FloatingPads(components.Component): 
    __name__ = 'FloatingPads'
    __draw__ = True 

    def draw(self, pads, cutout, gap, **kwargs):
        if isinstance(pads, list): 
            pad1, pad2 = pads
        else: 
            pad1, pad2 = pads, pads 

        q1x, q2x = pad1['x'], pad2['x'] 
        qy = max(pad1['y'], pad2['y'])

        # Empty capacitor pad gaps 
        qpads = [
            shapes.Rectangle(q1x, qy, 'right').translate(-gap/2, 0), 
            shapes.Rectangle(q2x, qy, 'left').translate(+gap/2, 0),
        ]

        # Encompassing cutout rectangle 
        cx, cy = cutout['x'], cutout['y'] 

        # Offset for asymmetry 
        try: 
            offset = cutout['offset']
        except KeyError: 
            offset = np.zeros(2)

        cutout = shapes.Rectangle(cx, cy).translate(*offset)

        self.add(qd.boolean(cutout, qpads, 'not'))
        self.add_cutout(cutout) 

        # Adding placement nodes 
        nodes = {
            'pad1': np.array((-gap/2 - q1x, 0)),
            'pad2': np.array((+gap/2 + q2x, 0)), 
            'pad1_top': np.array((-gap/2 - q1x/2, qy/2)), 
            'pad2_top': np.array((+gap/2 + q2x/2, qy/2)),
            'pad1_bottom': np.array((-gap/2 - q1x/2, -qy/2)), 
            'pad2_bottom': np.array((+gap/2 + q2x/2, -qy/2)), 
            'cutout_left': np.array((-cx/2 + offset[0], 0)), 
            'cutout_right': np.array((cx/2 + offset[0], 0)), 
            'cutout_top': np.add(offset, np.array((0, cy/2))), 
            'cutout_bottom': np.add(offset, np.array((0, -cy/2))), 
        }

        return nodes 


class InlineIDC(components.Component): 
    __name__ = 'IDC'
    __draw__ = True 

    def draw(self, cpw, taper, fingers, **kwargs):
        nodes = {}
        
        # Meandering central fingers of IDC
        # ---------------------------------
        n_fingers = fingers['num'] 
        finger_w = fingers['width'] 
        finger_l = fingers['length']
        spacing = fingers['horizontal_gap']
        gap = fingers['vertical_gap'] 
        distance = 2*spacing + 2*finger_w 

        nl = int(np.ceil(n_fingers/2))
        nr = n_fingers - nl 

        left_fingers = gdspy.Path(
            width=finger_w, 
            initial_point=(-finger_l/2, 0), 
            number_of_paths=nl, 
            distance=distance,
        ).segment(finger_l-gap, '+x') 

        offset = (nl - nr - 1) * distance/2
        right_fingers = gdspy.Path(
            width=finger_w, 
            initial_point=(+finger_l/2, offset), 
            number_of_paths=nr, 
            distance=distance, 
        ).segment(finger_l-gap, '-x')

        fingers = qd.boolean(left_fingers, right_fingers, 'or')

        # Outer rectangle for finger boolean cutoff 
        # -----------------------------------------
        box = gdspy.Rectangle(*fingers.get_bounding_box())
        (_, y0), (_, y1) = fingers.get_bounding_box()

        # Final Fingers
        fingers = qd.boolean(box, fingers, 'not')

        # Adding nodes for taper positioning 
        nodes['cpw0'] = np.array((-taper['length'] - finger_l/2, 0))
        nodes['cpw1'] = np.array((+taper['length'] + finger_l/2, 0))

        # Tapers and Encompassing Segments 
        # -------------------------------- 

        p = paths.CPW(**cpw, start=nodes['cpw0'])
        p.taper(taper['length'], width=y1-y0, gap=taper['gap'])
        p.segment(finger_l, '+x')
        p.taper(taper['length'], **cpw) 

        self.add(fingers.translate(0, -(y0+y1)/2))
        self.add(p)

        return nodes 


class ReadoutBus(components.Component): 
    __name__ = 'ReadoutBus'
    __draw__ = True 

    def draw(self, cpw, length, length_reduction, coupler): 
        nodes={}
        
        cpw_trace = paths.CPW(**cpw, direction='+y')
        cpw_trace.segment(length, '+y').open(cpw_trace.total_width())

        idc = InlineIDC(
            cpw=cpw,
            **coupler, 
        ).place(cpw_trace.initial_position(), node='cpw0', rotation=-90)

        # Placement nodes 
        nodes['top'] = cpw_trace.current_position() - (0, cpw_trace.total_width())
        nodes['bottom'] = cpw_trace.initial_position()
        nodes['qb_top'] = nodes['top'] - (0, length_reduction)
        nodes['qb_bottom'] = nodes['bottom'] + (0, length_reduction)
        nodes['center'] = (nodes['top'] + nodes['bottom']) / 2

        self.add(cpw_trace)
        self.add(idc)

        return nodes 


class CouplingResonator(components.Component): 
    __name__ = 'CouplingResonator'
    __draw__ = True 

    def draw(self, cpw, span, Ltotal, Lneck, radius, coupler, cutout):
        nodes = {}

        # Initializing the Coplanar Waveguide
        start = np.array(((cpw['width'] + cpw['gap'])/2, 0))
        left_cpw = CPW(**cpw, start=-1*start, direction='+y') 
        right_cpw = CPW(**cpw, start=1*start, direction='+y')

        hlength = span/2 - radius - start[0] 
        left_cpw.segment(L_neck).turn(radius, 'l').segment(hlength)
        right_cpw.segment(L_neck).turn(radius, 'r').segment(hlength)

        # Rectangular cutout 
        cutout_rect = shapes.Rectangle(cutout['x'], cutout['y'], origin='top')

        self.add(left_cpw)
        self.add(right_cpw)
        self.add(cutout_rect) 

        return nodes 


            






        










if __name__ == "__main__": 
    param_dir = './parameters/' 
    outfile = 'test.gds'
    gds_dir = './pattern_files/' 

    with open(os.path.join(param_dir, 'component_params.yaml'), 'r') as f: 
        readout_params = Params(yaml.load(f, yaml.SafeLoader))

    lib = gdspy.GdsLibrary() 
    lib.unit = 1.0e-6
    lib.precision = 1.0e-9 

    chip = Chip()

    read = CouplingResonator


    read = read.place()
    chip.add_component(read, layers=1)

    mask = chip.render('test', draw_border=True) 
    lib.write_gds(os.path.join(gds_dir, outfile), cells=mask)
