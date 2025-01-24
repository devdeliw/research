import os 
import gdspy 
import ruamel.yaml as yaml 
import numpy as np 
import qnldraw as qd 
import qnldraw.library as qlib 

from qnldraw import Chip, Params, shapes, paths












if __name__ == '__main__': 
    param_dir = './parameters/' 
    out_dir = './pattern_files/'
    filename = 'mqcv3.gds'

    lib = gdspy.GdsLibrary() 
    lib.unit = 1.0e-6
    lib.precision = 1.0e-9 

    chip = Chip() 

    mask = chip.render('mask3', draw_border=True)
    lib.write_gds(os.path.join(out_dir, filename), cells=mask) 
