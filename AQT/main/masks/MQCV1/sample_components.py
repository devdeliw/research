import os
import gdspy
import numpy as np

import qnldraw as qd 
import qnldraw.library as qlib
from qnldraw import components, shapes, Angle, paths, Chip
from qnldraw.paths import CPW 



class FloatingPads(components.Component): 
    """ 
    Builds Capacitor Pads for QNL's standard Floating Qubits. 
        
    """
    __name__ = 'FloatingPads' 
    __draw__ = True 

    def draw(self, pads, spacing, cutout): 

        if isinstance(pads, list): 
            pad1, pad2 = pads 
        else: 
            pad1, pad2 = pads, pads 

        del self.params['pads'] 
        self.params['pad1'] = pad1 
        self.params['pad2'] = pad2 

        q1x, q1y = pad1['x'], pad1['y'] 
        q2x, q2y = pad2['x'], pad2['y'] 

        qy = max(q1y, q2y) 

        qpads = [ 
            shapes.Rectangle(q1x, q1y, 'right').translate(-spacing/2, 0), 
            shapes.Rectangle(q2x, q2y, 'left').translate(+spacing/2, 0),
        ]

        cx, cy = cutout['x'], cutout['y'] 

        try: 
            offset = np.array(cutout['offset']) 
        except KeyError: 
            offset = np.zeros(2)
        
        cutout = shapes.Rectangle(cx, cy).translate(*offset)

        self.add(qd.boolean(cutout, qpads, 'not')) 
        self.add_cutout(cutout)

        nodes = {
            'pad1': np.array((-spacing/2 - q1x, 0)), 
            'pad2': np.array((spacing/2 + q2x, 0)), 
            'pad1_top': np.array((-spacing/2 - q1x / 2, qy / 2)),
            'pad2_top': np.array((spacing/2 + q2x / 2, qy / 2)),
            'pad1_bottom': np.array((-spacing/2 - q1x / 2, -qy / 2)),
            'pad2_bottom': np.array((spacing/2 + q2x / 2, -qy / 2)),
            'cutout_left': np.array((-cx/2 + offset[0], 0)),
            'cutout_right': np.array((cx/2 + offset[0], 0)),
            'cutout_top': offset + (0, cy/2),
            'cutout_bottom': offset + (0, -cy/2)
        }
        return nodes

class InlineIDC(components.Component): 
    __name__ = 'IDC' 
    __draw__ = True 

    def draw(self, cpw, taper, fingers):
        nodes = {}

        if isinstance(cpw, list):
            cpw1, cpw2 = cpw
        else:
            cpw1 = cpw
            cpw2 = cpw

        num_fingers, finger_width, finger_length = fingers['num'], fingers['width'], fingers['length']
        finger_spacing, gap = fingers['horizontal_gap'], fingers['vertical_gap']

        n_l = int(np.ceil(num_fingers / 2))
        n_r = num_fingers - n_l

        distance = (finger_spacing + finger_width)

        left_fingers = gdspy.Path(
            finger_width, 
            initial_point=(-finger_length/2, 0), 
            distance=2*distance, 
            number_of_paths=n_l
        ).segment(finger_length - gap)

        offset = (n_l - n_r - 1) * distance
        right_fingers = gdspy.Path(
            finger_width,
            initial_point=(finger_length/2, offset),
            distance=2*distance,
            number_of_paths=n_r
        ).segment(finger_length - gap, '-x')

        fingers = gdspy.boolean(left_fingers, right_fingers, 'or')
        (_, y0), (_, y1) = fingers.get_bounding_box()
        box = gdspy.Rectangle(*fingers.get_bounding_box())

        nodes['cpw0'] = np.array((-taper['length'] - finger_length / 2, 0))
        nodes['cpw1'] = np.array((taper['length'] + finger_length / 2, 0))

        p = paths.CPW(**cpw1, start=nodes['cpw0'])
        p.taper(taper['length'], width=y1 - y0, gap=taper['gap'])
        p.segment(finger_length)
        p.taper(taper['length'], **cpw2)

        self.add(gdspy.boolean(box, fingers, 'not').translate(0, -(y0 + y1)/2))
        self.add(p)
        self.add_cutout(p.cutout)
        return nodes

class ReadoutBus(components.Component): 
    __name__ = 'BUS'
    __draw__ = True 

    def draw(self, cpw, length, length_reduction, coupler): 
        nodes = {} 

        # Initialize Coplanar waveguide (CPW) 
        cpw_trace = CPW(**cpw, direction='+y')
        cpw_trace.segment(length, '+y')

        # InlineIDC component 
        idc = qlib.InlineIDC(
            cpw=cpw, 
            name='IDC', 
            **coupler, 
        ).place(cpw_trace.initial_position(), node='cpw0', rotation=-90) 

        # Adding nodes 
        nodes['top'] = cpw_trace.current_position() - (0, cpw_trace.total_width())
        nodes['bottom'] = cpw_trace.initial_position()

        nodes['qb_top'] = nodes['top'] - (0, length_reduction)
        nodes['qb_bottom'] = nodes['bottom'] + (0, length_reduction)

        nodes['center'] = (nodes['top'] + nodes['bottom']) / 2

        self.add([cpw_trace, idc])
        self.add_cutout(cpw_trace.cutout)

        return nodes

class CouplingResonator(components.Component):

    __name__ = 'COUPLINGRESONATOR'
    __draw__ = True

    def draw(self, cpw, span, Ltotal, Lneck, radius, coupler, cutout, **kwargs):
        nodes = {}

        ## Make left and right CPW traces
        start = np.array(((cpw['width'] + cpw['gap']) / 2, 0))
        left_cpw = CPW(**cpw, start=-1*start, direction='+y')
        right_cpw = CPW(**cpw, start=1*start, direction='+y')

        ## Draw the CPW sections
        hlength = span / 2 - radius - start[0]
        left_cpw.segment(Lneck).turn(radius, 'l').segment(hlength)
        right_cpw.segment(Lneck).turn(radius, 'r').segment(hlength)

        ## Make cutout rectangle for CPS trace
        cutout_rect = shapes.Rectangle(cutout['x'], cutout['y'], origin='top')

        ## Compute length of CPS trace, the first straight segment is 105 um
        Lmeander = Ltotal - left_cpw.length - radius * np.pi/2 - 105

        cps_trace = CPW(width=cpw['gap'], gap=cpw['width'], direction='-y')
        cps_trace.turn(radius, 'l').segment(105).meander(
            num_segments=int(np.ceil(Lmeander / (cutout['x'] / 1.8 + np.pi*radius))),
            length=Lmeander + cpw['width']/2,
            radius=radius,
            turn=-1,
            length_type='total',
            extra_turns='start'
        )

        ## Make the connection between the two
        cps_trace.open(cpw['width'], direction=cps_trace.get_direction() + 180)

        cps = qd.boolean(cutout_rect, cps_trace, 'not')

        ## Attach the coupling pads at the end
        left_coupler = coupler.place(left_cpw.current_position(), 'cpw', rotation=90)
        right_coupler = coupler.place(right_cpw.current_position(), 'cpw', rotation=-90)

        ## Add nodes
        nodes['left'] = left_coupler.node('pad')
        nodes['right'] = right_coupler.node('pad')

        self.add([left_coupler, right_coupler, left_cpw, right_cpw, cps])
        self.add_cutout([left_cpw.cutout, right_cpw.cutout, cutout_rect])

        return nodes

class HorizontalCouplingResonator(components.Component):

    __name__ = 'COUPLINGRESONATOR'
    __draw__ = True

    def draw(self, cpw, lx, ly1, ly2, radius, meander, coupler, **kwargs):
        nodes = {}

        cpw_trace = CPW(**cpw, start=(-lx/2, ly1), direction='-y')

        y = ly1
        if not isinstance(meander['lengths'], list):
            meander['lengths'] = [meander['lengths']]*len(meander['ys'])

        for y_m, Lm in zip(meander['ys'], meander['lengths']):
            cpw_trace.segment(y - (y_m +2*radius)) \
                .turn(radius, 'l') \
                .meander(2, Lm, radius, turn=-1) \
                .turn(radius, 'l')

            _, y = cpw_trace.current_position()

        cpw_trace.segment(y - radius) \
            .turn(radius, 'l') \
            .segment(lx - 2*radius) \
            .turn(radius, 'l') \
            .segment(ly2 - radius)

        left_coupler = coupler.place(cpw_trace.initial_position(), 'cpw')
        right_coupler = coupler.place(cpw_trace.current_position(), 'cpw')

        self.add([left_coupler, right_coupler, cpw_trace])
        self.add_cutout(cpw_trace.cutout)

        nodes['left'] = left_coupler.node('pad')
        nodes['right'] = right_coupler.node('pad')

        return nodes

class ReadoutResonator(components.Component):

    __name__ = 'READOUTRESONATOR'
    __draw__ = True

    def draw(self, cpw, L, lx, l_bus_coupler, meander_gap, cpad, radius, **kwargs):
        nodes = {}
        cpad = cpad.place(
            (cpad.get_params('gap'), 0), 'pad', rotation=90
        )

        cpw_trace = CPW(**cpw, start=cpad.node('cpw'))

        cpw_trace.turn(radius, 'l').segment(10).turn(radius, 'r')

        l = lx - cpw_trace.current_position()[0] - radius - cpw_trace.total_width() / 2

        cpw_trace.segment(l) \
            .turn(radius, 'r') \
            .segment(l_bus_coupler) \
            .turn(radius, 'r') \
            .segment(630)

        cpad_contribution = cpad.get_params('cpw_length') + cpad.get_params('pad.length') / 2
        meander_length = L - cpw_trace.length - cpad_contribution

        cpw_trace.meander(
            num_segments=2,
            length=meander_length,
            radius=radius,
            turn=1,
            length_type='total',
            extra_turns='start'
        )

        self.add([cpad, cpw_trace])
        self.add_cutout(cpw_trace.cutout)

        return nodes

class ControlLine(components.Component):

    __name__ = 'CONTROLLINE'
    __draw__ = True

    def draw(self, cpw, launch, start, end, intersection, direction):
        nodes = {}

        ## Place the launch component
        launch = launch.place(start, 'cutout', rotation=direction)

        ## Create CPW
        direction = Angle(direction)
        cpw_trace = CPW(**cpw, start=launch.node('cpw'))
        cpw_trace.segment(0, direction=direction)

        ## Intersection is relative distance from cutout
        length = intersection - np.abs(launch.node('cutout') - launch.node('cpw'))[0]
        shift = (end - start)[1]
        shift *= -1 if direction > 90 else 1

        cpw_trace.shift(shift, radius=180, shift_position=length)

        length = np.abs(cpw_trace.current_position() - end)[0] - cpw['gap']

        cpw_trace.segment(length).open(cpw['gap'])

        nodes['end'] = cpw_trace.current_position()
        self.add([launch, cpw_trace])
        self.add_cutout(cpw_trace.cutout)

        return nodes

class ReadoutLine(components.Component):

    __name__ = 'READOUTLINE'
    __draw__ = True

    def draw(self, cpw, radius, start, launch, launch_position, shift_x_position, hero_bond, **kwargs):
        nodes = {}

        # Compute hero bond positions
        x_left = hero_bond['x'] - hero_bond['gap'] / 2
        x_right = hero_bond['x'] + hero_bond['gap'] / 2
        _, y = start - hero_bond['y_offset']

        hero_launch_left = launch.place((x_left, y), 'cutout', rotation=180)
        hero_launch_right = launch.place((x_right, y), 'cutout')

        # CPW between readout bus and left hero launch
        cpw_trace_1 = CPW(**cpw, start=start)
        cpw_trace_1.segment(start[1] - y - radius, direction='-y')
        cpw_trace_1.turn(radius, 'l') \

        length, _ = hero_launch_left.node('cpw') - cpw_trace_1.current_position()
        cpw_trace_1.segment(length)

        # CPW between right hero launch and readout launch
        start = hero_launch_right.node('cpw')
        cpw_trace_2 = CPW(**cpw, start=hero_launch_right.node('cpw'))

        _, shift_y = launch_position - start
        shift_x, _ = shift_x_position - (start - hero_launch_right.node('cutout'))

        cpw_trace_2.shift(shift_y, radius, shift_position=shift_x)

        final_launch = launch.place(launch_position, 'cutout', rotation=180)
        length, _ = final_launch.node('cpw') - cpw_trace_2.current_position()
        cpw_trace_2.segment(length)

        self.add([cpw_trace_1, cpw_trace_2, hero_launch_left, hero_launch_right, final_launch])
        self.add_cutout([cpw_trace_1.cutout, cpw_trace_2.cutout])

        return nodes





if __name__ == '__main__': 
    param_dir = './parameters/' 
    outfile = 'test.gds'
    gds_dir = './pattern_files/' 


    lib = gdspy.GdsLibrary() 
    lib.unit = 1.0e-6
    lib.precision = 1.0e-9 

    chip = Chip() 
    Q = FloatingPads(
        {'x': 135, 'y': 545}, 
        spacing=65, 
        cutout={'x': 535, 'y': 745}, 
    )

    Q0 = Q.place() 
    chip.add_component(Q0, layers=1)

    mask = chip.render('MASK', draw_border=True) 
    lib.write_gds(os.path.join(gds_dir, outfile), cells=mask)



