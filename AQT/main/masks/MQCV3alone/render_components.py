import os 
import gdspy 
import numpy as np 
import ruamel.yaml as yaml 
import qnldraw as qd 
import qnldraw.library as qlib 

from qnldraw import shapes, paths, Chip, Params, components
from qnldraw.paths import CPW


class ReadoutBus(components.Component): 
    __name__ = 'READOUTBUS'
    __draw__ = True 

    """ 
    Builds the central readout bus and inline IDC component. 

    Args: 
        cpw (Params): Coplanar WaveGuide parameters.
        coupler (Params): Purcell filter parameters. 

    Returns: 
        dict: The locations of the nodes of the component. 
    """

    def draw(self, cpw, coupler, **kwargs): 
        # initialize coplanar waveguide path 
        cpw_path = CPW(**cpw, direction='+y')

        length = coupler['length']
        cpw_path.segment(length, '+y')

        idc_params = coupler['coupler']
        idc = qlib.InlineIDC(cpw=cpw, **idc_params)
        idc = idc.place(
                location=cpw_path.initial_position(), 
                node='cpw0', 
                rotation=-90, 
        )

        nodes = { 
            'top': cpw_path.current_position(),
            'bottom': cpw_path.initial_position(),
            'center': (cpw_path.current_position() + cpw_path.initial_position())/2
        }

        # Adding readout bus to component
        self.add(qd.boolean(idc, cpw_path, 'or'))
        return nodes 

class CouplingResonator(components.Component): 
    __name__ = 'COUPLINGRESONATOR'
    __draw__ = True

    """
    Builds a CouplingResonator component that connects two 
    FloatingPad qubits. 

    Args: 
        cpw (Params): The Coplanar Waveguide parameters. 
        Lneck (float): The length of the CPW neck of the coupling resonator. 
        radius (float): the radius of the meandering turns of the resonator. 
        span (float): Length of horizontal segments. 
        turns (int): Number of meandering turns (# of horizontal segments).
                     *not including the first half segment. 
        claw (Params): The connecting claw parameters.
        cutout (Params): Dimension parameters for the surrounding rectangle. 
    Returns: 
        dict: The locations of the nodes of the component. 
    """

    def draw(self, cpw, Lneck, Lsegment, span, cutout, turns, radius, **kwargs): 
        nodes = {} 

        # initialize coplanar waveguide path 
        start = np.array(((cpw['width'] + cpw['gap'])/2, 0))
        l_cpw = CPW(**cpw, direction='+y', start=-1*start)
        r_cpw = CPW(**cpw, direction='+y', start=+1*start)

        l_cpw.segment(Lneck)
        r_cpw.segment(Lneck) 

        # left and right cpws connecting claws to resonator
        hlength = span / 2 - radius - start[0] 
        l_cpw.turn(radius, 'l').segment(hlength)
        r_cpw.turn(radius, 'r').segment(hlength)

        # claw connector objects
        claw_connector = qd.boolean(l_cpw, r_cpw, 'or')

        # initialize resonator trace gap
        first_segment_length = Lsegment/2 - radius 
        cps_trace = CPW(
            gap=cpw['width'], 
            width=cpw['gap'], 
            direction='-y'
        ) 
        cps_trace.turn(radius, 'l').segment(first_segment_length)
        cps_trace.meander(
            num_segments=turns, 
            length=Lsegment, 
            radius=radius, 
            turn=-1,
            extra_turns='start', 
        )

        # rectangular surrounding cutout 
        cutout = shapes.Rectangle(cutout['x'], cutout['y'], origin='top')

        # make meandering cps_trace a gap 
        coupler = qd.boolean(cutout, cps_trace, 'not')

        # final coupling resonator component 
        coupling_resonator = qd.boolean(coupler, claw_connector, 'or')


        self.add(coupling_resonator)


        return nodes



        



        return 

def readout_bus(chip, readout_bus_params):
    bus = ReadoutBus(
            cpw=readout_bus_params['cpw'],
            coupler=readout_bus_params['purcell_filter'],
    )

    chip.add_component(bus, layers=readout_bus_params['layer'])
    return

def coupling_resonator(chip, resonator_params): 
    resonator = CouplingResonator(
        cpw=resonator_params['cpw'],
        **resonator_params['vertical_couplers']
    )

    chip.add_component(resonator, layers=resonator_params['layer'])
    return 

if __name__ == '__main__': 
    param_dir = './parameters/'
    out_dir = './pattern_files/'
    filename = 'components.gds'
    
    lib = gdspy.GdsLibrary() 
    lib.unit = 1.0e-6
    lib.precision = 1.0e-9 

    with open(os.path.join(param_dir, 'readoutbus.yaml'), 'r') as f: 
        readout_bus_params = Params(yaml.load(f, yaml.SafeLoader))
    with open(os.path.join(param_dir, 'coupling_resonator.yaml'), 'r') as f: 
        coupling_params = Params(yaml.load(f, yaml.SafeLoader))

    chip = Chip() 
    readout_bus(chip, readout_bus_params)
    #coupling_resonator(chip, coupling_params)


    
    mask = chip.render('mask_components')
    lib.write_gds(os.path.join(out_dir, filename), cells=mask) 





