import gdspy
import numpy as np 

""" renders my devdeliw logo """

lib = gdspy.GdsLibrary() 
cell = lib.new_cell('logo') 

def render_moon(): 
    moon = gdspy.FlexPath(
        [ 
            (-3.28, 10.12), 
        ], 0.1
    )
    moon.smooth(
        [ 
            (-4.65, -0.5), 
            (6.38, 0.46),
        ],
        relative=False
    )
    moon2 = gdspy.FlexPath(
        [
            (6.38, 0.46)
        ], 0.1
    )
    moon2.smooth(
        [
            (-2, 1), 
            (-3.28, 10.12),
        ], 
        relative=False
    )
    cell.add(moon)
    cell.add(moon2)

    return

def render_devdeliw(): 
    devdeliw = gdspy.Text('DEVDELIW', 2.5, (10, -10))
    devdeliw.translate(-18, 3.37)

    cell.add(devdeliw)

    return 

def render_cat():
    cat = gdspy.FlexPath(
        [(-0.7, 0.1), (0, 0)], 0.1
    )
    cat.smooth(
        [ 
            (-0.05, 2),
            (1.3, 5), 
        ]

    )
    cat.segment((1.5, 5.6))
    cat.smooth(
        [
            (1.3, 6.2), 
            (1.35, 6.9),
        ],
        relative=False
    )
    cat.segment((1.0, 7.4))
    cat.segment((1.6, 7.25))
    cat.smooth(
        [
            (2.3, 7.4),
            (2.8, 7.35),
        ], 
        relative=False
    )
    cat.segment((2.9, 7.7))
    cat.smooth(
        [
            (3, 7.5),
            (3.17, 6.59),
            (3.13, 5.6),
            (3.25, 4.48),
            (3.15, 2.27),
        ], 
        relative=False
    )
    cat.segment((3.15, 2.27)) 
    cat.smooth(
        [
            (3.36, 1.08), 
            (3.13, -0.28),
        ],
        relative=False
    )
    cat.segment((3.13, -0.28)) 
    cat.smooth(
        [
            (3.32, -0.52), 
            (3.28, -0.6), 
        ],
        relative=False
    )

    tail = gdspy.FlexPath(
        [(-3.8, -1.3)], 0.1
    )

    tail.smooth(
        [
           (-3.65, -1.82), 
            (-2.89, -2.4),
            (-2.35, -2.45),
            (-2.18, -2.37),
            (-2.22, -2.33),
            (-2.29, -2.29), 
            (-2.614, -2.12),
            (-3.157, -1.87),
            (-3.31, -1.7),
            (-3.25, -1.62),
        ],
        relative=False
    )

    cell.add(cat)
    cell.add(tail)

    return 



def render(): 
    render_cat()
    render_moon()
    render_devdeliw()
    


    return 



if __name__ == '__main__': 
    render()










# export to gds
lib.write_gds('logo.gds')

