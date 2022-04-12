""" Designer specified class definition for emitter coils used in the EM tracking system design"""

import numpy as np


class CoilModel:

    # The init function should initialise and prepare all data necessary for 'coil_field' to operate
    def __init__(self, module_config):
        self.x_points, self.y_points, self.z_points = coil_definition(module_config)

    def model_init(self):
        pass

    # MUST BE DEFINED. A User defined function which returns the magnetic field intensity H at a point in space P.
    # This function must be defined as accepting a cartesian coordinate (x,y,z),
    # and returning a three magnetic intensity values Hx, Hy and Hz
    def coil_field_total(self, px, py, pz):
        I = 1
        numPoints = self.x_points.shape[1]

        # matrix conversions

        x_points = np.matrix(self.x_points)
        y_points = np.matrix(self.y_points)
        z_points = np.matrix(self.z_points)

        ax = x_points[:, 1:numPoints] - x_points[:, 0:(numPoints - 1)]
        ay = y_points[:, 1:numPoints] - y_points[:, 0:(numPoints - 1)]
        az = z_points[:, 1:numPoints] - z_points[:, 0:(numPoints - 1)]

        bx = x_points[:, 1:numPoints] - px
        by = y_points[:, 1:numPoints] - py
        bz = z_points[:, 1:numPoints] - pz

        cx = x_points[:, 0:(numPoints - 1)] - px
        cy = y_points[:, 0:(numPoints - 1)] - py
        cz = z_points[:, 0:(numPoints - 1)] - pz

        c_mag = np.sqrt(np.square(cx) + np.square(cy) + np.square(cz))
        b_mag = np.sqrt(np.square(bx) + np.square(by) + np.square(bz))

        a_dot_b = np.multiply(ax, bx) + np.multiply(ay, by) + np.multiply(az, bz)
        a_dot_c = np.multiply(ax, cx) + np.multiply(ay, cy) + np.multiply(az, cz)

        c_cross_a_x = np.multiply(az, cy) - np.multiply(ay, cz)
        c_cross_a_y = np.multiply(ax, cz) - np.multiply(az, cx)
        c_cross_a_z = np.multiply(ay, cx) - np.multiply(ax, cy)

        c_cross_a_mag_squared = np.square(c_cross_a_x) + np.square(c_cross_a_y) + np.square(c_cross_a_z)

        scalar = np.divide((np.divide(a_dot_c, c_mag) - np.divide(a_dot_b, b_mag)), c_cross_a_mag_squared)

        hx_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_x, scalar)
        hy_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_y, scalar)
        hz_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_z, scalar)

        hx = np.sum(hx_dum, axis=1)
        hy = np.sum(hy_dum, axis=1)
        hz = np.sum(hz_dum, axis=1)

        return hx, hy, hz

    def coil_field_single(self, px, py, pz, coilindex=0):
        I = 1
        numPoints = self.x_points.shape[1]

        # matrix conversions

        x_points = np.matrix(self.x_points[coilindex, :])
        y_points = np.matrix(self.y_points[coilindex, :])
        z_points = np.matrix(self.z_points[coilindex, :])

        ax = x_points[:, 1:numPoints] - x_points[:, 0:(numPoints - 1)]
        ay = y_points[:, 1:numPoints] - y_points[:, 0:(numPoints - 1)]
        az = z_points[:, 1:numPoints] - z_points[:, 0:(numPoints - 1)]

        bx = x_points[:, 1:numPoints] - px
        by = y_points[:, 1:numPoints] - py
        bz = z_points[:, 1:numPoints] - pz

        cx = x_points[:, 0:(numPoints - 1)] - px
        cy = y_points[:, 0:(numPoints - 1)] - py
        cz = z_points[:, 0:(numPoints - 1)] - pz

        c_mag = np.sqrt(np.square(cx) + np.square(cy) + np.square(cz))
        b_mag = np.sqrt(np.square(bx) + np.square(by) + np.square(bz))

        a_dot_b = np.multiply(ax, bx) + np.multiply(ay, by) + np.multiply(az, bz)
        a_dot_c = np.multiply(ax, cx) + np.multiply(ay, cy) + np.multiply(az, cz)

        c_cross_a_x = np.multiply(az, cy) - np.multiply(ay, cz)
        c_cross_a_y = np.multiply(ax, cz) - np.multiply(az, cx)
        c_cross_a_z = np.multiply(ay, cx) - np.multiply(ax, cy)

        c_cross_a_mag_squared = np.square(c_cross_a_x) + np.square(c_cross_a_y) + np.square(c_cross_a_z)

        scalar = np.divide((np.divide(a_dot_c, c_mag) - np.divide(a_dot_b, b_mag)), c_cross_a_mag_squared)

        hx_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_x, scalar)
        hy_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_y, scalar)
        hz_dum = (I / (4 * np.pi)) * np.multiply(c_cross_a_z, scalar)

        hx = np.sum(hx_dum, axis=1)
        hy = np.sum(hy_dum, axis=1)
        hz = np.sum(hz_dum, axis=1)

        return hx, hy, hz


# Function used to define the dimensions of each coil in the square coil model
def coil_definition(model_config):
    if str(model_config['model_name']).upper() == 'SQUARE_MODEL':
        model_type = model_config['model_name']
        coil_turns = model_config['num_turns']
        turn_length = model_config['trace_length']
        trace_width = model_config['trace_width']
        trace_spacing = model_config['trace_spacing']
        trace_thickness = model_config['trace_thickness']

        x_centres = np.array(model_config['centers_x'])
        x_centres = x_centres * 1.0e-3
        y_centres = np.array(model_config['centers_y'])
        y_centres = y_centres * 1.0e-3

        num_coils = x_centres.shape[0]

        [x_points_a, y_points_a, z_points_a] = _spiralCoilDimensionCalc(coil_turns,
                                                                       turn_length,
                                                                       trace_width,
                                                                       trace_spacing,
                                                                       trace_thickness, np.pi / 4)
        [x_points_v, y_points_v, z_points_v] = _spiralCoilDimensionCalc(coil_turns,
                                                                       turn_length,
                                                                       trace_width,
                                                                       trace_spacing,
                                                                       trace_thickness, np.pi / 2)

        x_points_1 = x_points_v + x_centres[0]
        x_points_2 = x_points_a + x_centres[1]
        x_points_3 = x_points_v + x_centres[2]
        x_points_4 = x_points_a + x_centres[3]
        x_points_5 = x_points_a + x_centres[4]
        x_points_6 = x_points_v + x_centres[5]
        x_points_7 = x_points_a + x_centres[6]
        x_points_8 = x_points_v + x_centres[7]

        y_points_1 = y_points_v + y_centres[0]
        y_points_2 = y_points_a + y_centres[1]
        y_points_3 = y_points_v + y_centres[2]
        y_points_4 = y_points_a + y_centres[3]
        y_points_5 = y_points_a + y_centres[4]
        y_points_6 = y_points_v + y_centres[5]
        y_points_7 = y_points_a + y_centres[6]
        y_points_8 = y_points_v + y_centres[7]

        z_points_1 = z_points_v
        z_points_2 = z_points_a
        z_points_3 = z_points_v
        z_points_4 = z_points_a
        z_points_5 = z_points_a
        z_points_6 = z_points_v
        z_points_7 = z_points_a
        z_points_8 = z_points_v

        x_points = np.array(
            [x_points_1, x_points_2, x_points_3, x_points_4, x_points_5, x_points_6, x_points_7, x_points_8])
        y_points = np.array(
            [y_points_1, y_points_2, y_points_3, y_points_4, y_points_5, y_points_6, y_points_7, y_points_8])
        z_points = np.array(
            [z_points_1, z_points_2, z_points_3, z_points_4, z_points_5, z_points_6, z_points_7, z_points_8])

        return x_points, y_points, z_points
    elif str(model_config['model_name']).upper() == 'SQUARE_MODEL_FAST':

        x_centres = np.array(model_config['centers_x'])
        x_centres = x_centres * 1.0e-3
        y_centres = np.array(model_config['centers_y'])
        y_centres = y_centres * 1.0e-3

        num_coils = x_centres.shape[0]


        x_points_a = np.array([-61.5e-3/np.sqrt(2), 0, 61.5e-3 / np.sqrt(2),0, - 61.5e-3 / np.sqrt(2)])
        y_points_a = np.array([0, - 61.5e-3 / np.sqrt(2), 0, 61.5e-3 / np.sqrt(2), 0])
        z_points_a = np.zeros([1, 5]) -1.6e-3 / 2

        x_points_v = [-61.5e-3 / 2, 61.5e-3 / 2, 61.5e-3 / 2, - 61.5e-3 / 2, - 61.5e-3 / 2]
        y_points_v = [-61.5e-3 / 2, - 61.5e-3 / 2, 61.5e-3 / 2, 61.5e-3 / 2, - 61.5e-3 / 2]
        z_points_v = np.zeros([1, 5]) - 1.6e-3 / 2

        x_points_1 = x_points_v + x_centres[0]
        x_points_2 = x_points_a + x_centres[1]
        x_points_3 = x_points_v + x_centres[2]
        x_points_4 = x_points_a + x_centres[3]
        x_points_5 = x_points_a + x_centres[4]
        x_points_6 = x_points_v + x_centres[5]
        x_points_7 = x_points_a + x_centres[6]
        x_points_8 = x_points_v + x_centres[7]

        y_points_1 = y_points_v + y_centres[0]
        y_points_2 = y_points_a + y_centres[1]
        y_points_3 = y_points_v + y_centres[2]
        y_points_4 = y_points_a + y_centres[3]
        y_points_5 = y_points_a + y_centres[4]
        y_points_6 = y_points_v + y_centres[5]
        y_points_7 = y_points_a + y_centres[6]
        y_points_8 = y_points_v + y_centres[7]

        z_points_1 = z_points_v
        z_points_2 = z_points_a
        z_points_3 = z_points_v
        z_points_4 = z_points_a
        z_points_5 = z_points_a
        z_points_6 = z_points_v
        z_points_7 = z_points_a
        z_points_8 = z_points_v

        x_points = np.array(
            [x_points_1, x_points_2, x_points_3, x_points_4, x_points_5, x_points_6, x_points_7, x_points_8])
        y_points = np.array(
            [y_points_1, y_points_2, y_points_3, y_points_4, y_points_5, y_points_6, y_points_7, y_points_8])
        z_points = np.array(
            [z_points_1, z_points_2, z_points_3, z_points_4, z_points_5, z_points_6, z_points_7, z_points_8])

        return x_points, y_points, z_points
    elif str(model_config['model_name']).upper() == 'SQUARE_MODEL_GAP':
        model_type = model_config['model_name']
        coil_turns = model_config['num_turns']
        turn_length = model_config['trace_length']
        trace_width = model_config['trace_width']
        trace_spacing = model_config['trace_spacing']
        trace_thickness = model_config['trace_thickness']

        x_centres = np.array(model_config['centers_x'])
        x_centres = x_centres * 1.0e-3
        y_centres = np.array(model_config['centers_y'])
        y_centres = y_centres * 1.0e-3

        num_coils = x_centres.shape[0]


        [x_points_v, y_points_v, z_points_v] = _spiralCoilDimensionCalc(coil_turns,
                                                                       turn_length,
                                                                       trace_width,
                                                                       trace_spacing,
                                                                       trace_thickness, np.pi / 2)

        x_points_1 = x_points_v + x_centres[0]
        x_points_2 = x_points_v + x_centres[1]
        x_points_3 = x_points_v + x_centres[2]
        x_points_4 = x_points_v + x_centres[3]
        x_points_5 = x_points_v + x_centres[4]
        x_points_6 = x_points_v + x_centres[5]
        x_points_7 = x_points_v + x_centres[6]
        x_points_8 = x_points_v + x_centres[7]

        y_points_1 = y_points_v + y_centres[0]
        y_points_2 = y_points_v + y_centres[1]
        y_points_3 = y_points_v + y_centres[2]
        y_points_4 = y_points_v + y_centres[3]
        y_points_5 = y_points_v + y_centres[4]
        y_points_6 = y_points_v + y_centres[5]
        y_points_7 = y_points_v + y_centres[6]
        y_points_8 = y_points_v + y_centres[7]

        z_points_1 = z_points_v
        z_points_2 = z_points_v
        z_points_3 = z_points_v
        z_points_4 = z_points_v
        z_points_5 = z_points_v
        z_points_6 = z_points_v
        z_points_7 = z_points_v
        z_points_8 = z_points_v

        x_points = np.array(
            [x_points_1, x_points_2, x_points_3, x_points_4, x_points_5, x_points_6, x_points_7, x_points_8])
        y_points = np.array(
            [y_points_1, y_points_2, y_points_3, y_points_4, y_points_5, y_points_6, y_points_7, y_points_8])
        z_points = np.array(
            [z_points_1, z_points_2, z_points_3, z_points_4, z_points_5, z_points_6, z_points_7, z_points_8])

        return x_points, y_points, z_points
    elif str(model_config['model_name']).upper() == 'RECTANGLE_6LAYER_MODEL_GAP':

        model_type = model_config['model_name']
        coil_turns = int(model_config['num_turns'] / 3)  # Divide total number of turns by 3 for generation
        turn_length = model_config['trace_length']
        trace_width = model_config['trace_width']
        trace_spacing = model_config['trace_spacing']
        thickness = model_config['trace_thickness']

        x_centres = np.array(model_config['centers_x'])
        x_centres = x_centres * 1.0e-3
        y_centres = np.array(model_config['centers_y'])
        y_centres = y_centres * 1.0e-3

        num_coils = x_centres.shape[0]


        # HARD CODED RECTANGLES FOR 6-LAYER FIELD GENERATOR
        turn_length1 = 97.9e-3
        turn_length2 = 215.3e-3
        nturns_rect = 47    # 141 / 3 is the number of turns on a double layer board



        # Generate rectangles using _spiralRectCoilDimensionCalc()
        [x_points_v_rect, y_points_v_rect, z_points_v_rect] = _spiralRectCoilDimensionCalc(nturns_rect,
                                                                        turn_length1,
                                                                        turn_length2,
                                                                        trace_width,
                                                                        trace_spacing,
                                                                        thickness, 0)
        # Generate square coils using _spiralCoilDimensionCalc()
        [x_points_v, y_points_v, z_points_v] = _spiralCoilDimensionCalc(coil_turns,
                                                                            turn_length,
                                                                            trace_width,
                                                                            trace_spacing,
                                                                            thickness, np.pi / 2)

        x_points_1 = x_points_v + x_centres[0]
        x_points_2 = x_points_v + x_centres[1]
        x_points_3 = x_points_v + x_centres[2]
        x_points_4 = x_points_v_rect + x_centres[3]
        x_points_5 = x_points_v_rect + x_centres[4]
        x_points_6 = x_points_v + x_centres[5]
        x_points_7 = x_points_v + x_centres[6]
        x_points_8 = x_points_v + x_centres[7]

        y_points_1 = y_points_v + y_centres[0]
        y_points_2 = y_points_v + y_centres[1]
        y_points_3 = y_points_v + y_centres[2]
        y_points_4 = y_points_v_rect + y_centres[3]
        y_points_5 = y_points_v_rect + y_centres[4]
        y_points_6 = y_points_v + y_centres[5]
        y_points_7 = y_points_v + y_centres[6]
        y_points_8 = y_points_v + y_centres[7]

        z_points_1 = z_points_v
        z_points_2 = z_points_v
        z_points_3 = z_points_v
        z_points_4 = z_points_v_rect
        z_points_5 = z_points_v_rect
        z_points_6 = z_points_v
        z_points_7 = z_points_v
        z_points_8 = z_points_v

        # Disgusting magnetics hack to create a null field to in order to keep filament arrays the same size for rectangles and squares
        turn_difference = z_points_1.shape[0] - z_points_4.shape[0] # Calculate difference in number of turns
        half_difference = int(turn_difference/2)
        delta = 0.0000000001

        # Last filament vertex is the first vertex since closed coil
        lastx4 = x_points_4[0]
        lastx5 = x_points_5[0]
        lasty4 = y_points_4[0]
        lasty5 = y_points_5[0]
        lastz4 = z_points_4[0]
        lastz5 = z_points_5[0]

        # Create micro nudges
        xnudge4 = np.tile(np.array([lastx4+delta, lastx4]), half_difference)
        xnudge5 = np.tile(np.array([lastx5+delta, lastx5]), half_difference)
        ynudge4 = np.tile(np.array([lasty4+delta, lasty4]), half_difference)
        ynudge5 = np.tile(np.array([lasty5+delta, lasty5]), half_difference)
        znudge4 = np.tile(np.array([lastz4, lastz4]), half_difference)
        znudge5 = np.tile(np.array([lastz5, lastz5]), half_difference)

        x_points_4 = np.concatenate((x_points_4, xnudge4))
        x_points_5 = np.concatenate((x_points_5, xnudge5))
        y_points_4 = np.concatenate((y_points_4, ynudge4))
        y_points_5 = np.concatenate((y_points_5, ynudge5))
        z_points_4 = np.concatenate((z_points_4, znudge4))
        z_points_5 = np.concatenate((z_points_5, znudge5))


        x_points = np.array(
            [x_points_1, x_points_2, x_points_3, x_points_4, x_points_5, x_points_6, x_points_7, x_points_8])
        y_points = np.array(
            [y_points_1, y_points_2, y_points_3, y_points_4, y_points_5, y_points_6, y_points_7, y_points_8])
        z_points = np.array(
            [z_points_1, z_points_2, z_points_3, z_points_4, z_points_5, z_points_6, z_points_7, z_points_8])

        #Replicate the dual layer pattern above and below to create a total of 6 layers
        x_points = np.concatenate((x_points,x_points,x_points), axis=1)
        y_points = np.concatenate((y_points, y_points, y_points), axis=1)

        z_points_lower = z_points - 2 * thickness
        z_points_upper = z_points + 2 * thickness
        z_points = np.concatenate((z_points_lower, z_points, z_points_upper), axis=1)

        return x_points, y_points, z_points

    elif str(model_config['model_name']).upper() == 'SQUARE_6LAYER_42CM':

        model_type = model_config['model_name']
        coil_turns = int(171 / 3)  # Divide total number of turns by 3 for generation
        turn_length = 114.75e-3
        trace_width = 1.1e-3
        trace_spacing = 0.25e-3
        pcb_thickness = 2.0e-3

        x_centres = np.array([-0.145, 0, 0.145, -.1063, 0.1063, -0.145, 0, 0.145])
        y_centres = np.array([0.145, .1063, 0.145, 0, 0, -0.145, -0.1063, -0.145])

        num_coils = x_centres.shape[0]

        # HARD CODED SQAURES FOR 6-LAYER FIELD GENERATOR
        turn_length = 114.75e-3
        nturns_rect = 171  # 141 / 3 is the number of turns on a double layer board

        [x_points_a, y_points_a, z_points_a] = _spiralCoilDimensionCalc(coil_turns,
                                                                        turn_length,
                                                                        trace_width,
                                                                        trace_spacing,
                                                                        pcb_thickness, np.pi / 4)
        [x_points_v, y_points_v, z_points_v] = _spiralCoilDimensionCalc(coil_turns,
                                                                        turn_length,
                                                                        trace_width,
                                                                        trace_spacing,
                                                                        pcb_thickness, np.pi / 2)

        x_points_1 = x_points_v + x_centres[0]
        x_points_2 = x_points_a + x_centres[1]
        x_points_3 = x_points_v + x_centres[2]
        x_points_4 = x_points_a + x_centres[3]
        x_points_5 = x_points_a + x_centres[4]
        x_points_6 = x_points_v + x_centres[5]
        x_points_7 = x_points_a + x_centres[6]
        x_points_8 = x_points_v + x_centres[7]

        y_points_1 = y_points_v + y_centres[0]
        y_points_2 = y_points_a + y_centres[1]
        y_points_3 = y_points_v + y_centres[2]
        y_points_4 = y_points_a + y_centres[3]
        y_points_5 = y_points_a + y_centres[4]
        y_points_6 = y_points_v + y_centres[5]
        y_points_7 = y_points_a + y_centres[6]
        y_points_8 = y_points_v + y_centres[7]

        z_points_1 = z_points_v
        z_points_2 = z_points_a
        z_points_3 = z_points_v
        z_points_4 = z_points_a
        z_points_5 = z_points_a
        z_points_6 = z_points_v
        z_points_7 = z_points_a
        z_points_8 = z_points_v

        x_points = np.array(
            [x_points_1, x_points_2, x_points_3, x_points_4, x_points_5, x_points_6, x_points_7, x_points_8])
        y_points = np.array(
            [y_points_1, y_points_2, y_points_3, y_points_4, y_points_5, y_points_6, y_points_7, y_points_8])
        z_points = np.array(
            [z_points_1, z_points_2, z_points_3, z_points_4, z_points_5, z_points_6, z_points_7, z_points_8])

        # Replicate the dual layer pattern above and below to create a total of 6 layers
        x_points = np.concatenate((x_points, x_points, x_points), axis=1)
        y_points = np.concatenate((y_points, y_points, y_points), axis=1)

        z_points_lower = z_points - 2 * pcb_thickness
        z_points_upper = z_points + 2 * pcb_thickness
        z_points = np.concatenate((z_points_lower, z_points, z_points_upper), axis=1)

        return x_points, y_points, z_points
    else:
        return 'Error calculating points'



def _spiralRectCoilDimensionCalc(N, length1, length2, width, spacing, thickness, angle):
    n = N
    z_thick = thickness  # %pcb board thickness
    theta_rot = angle  # %default value is pi/2
    # define dimensions
    l1 = length1  # define side length of outer square
    l2 = length2
    w = width  # define width of each track
    s = spacing  # define spacing between tracks

    ll_s = w + s  # line to line spacing

    Points_total = (4 * n) + 1

    x_points = np.zeros(Points_total)
    y_points = np.zeros(Points_total)
    z_points = np.zeros(Points_total)

    z_points[0:(2 * n + 1)] = +z_thick / 2
    z_points[(2 * n + 1):Points_total] = -z_thick / 2

    x_points_new = np.zeros(Points_total + 1)
    y_points_new = np.zeros(Points_total + 1)
    z_points_new = np.zeros(Points_total + 1)

    line_segs = np.zeros(4 * n)
    line_segs[0] = l1
    line_segs[1] = l2
    line_segs[2] = l1

    i = 1  # increment the decrease in length

    for seg_move in range(3, 2 * n, 2):
        line_segs[seg_move] = (l2 - i * ll_s)
        line_segs[seg_move + 1] = (l1 - i * ll_s)
        i = i + 1

    i_smaller = i - 1

    for seg_move in range(2 * n - 1, (4 * n) - 1, 2):
        line_segs[seg_move] = (l2 - i_smaller * ll_s)
        line_segs[seg_move + 1] = (l1 - i_smaller * ll_s)
        i = 1 + i
        i_smaller = i_smaller - 1

    line_segs[4 * n - 3] = l2
    line_segs[4 * n - 2] = l1
    line_segs[4 * n - 1] = l2

    x_traj = np.cos([theta_rot, theta_rot + .5 * np.pi, theta_rot + np.pi, theta_rot + 1.5 * np.pi])
    y_traj = np.sin([theta_rot, theta_rot + .5 * np.pi, theta_rot + np.pi, theta_rot + 1.5 * np.pi])

    q = 0

    for p in range(1, Points_total):
        x_points[p] = x_traj[q] * line_segs[p - 1] + x_points[p - 1]
        y_points[p] = y_traj[q] * line_segs[p - 1] + y_points[p - 1]
        q = q + 1
        if q == 4:
            q = 0

    x_points_new[0:2 * n + 1] = x_points[0:2 * n + 1]
    x_points_new[2 * n + 1] = x_points[2 * n]
    x_points_new[2 * n + 2:Points_total + 1] = x_points[2 * n + 1:Points_total]

    y_points_new[0:2 * n + 1] = y_points[0:2 * n + 1]
    y_points_new[2 * n + 1] = y_points[2 * n]
    y_points_new[2 * n + 2:Points_total + 1] = y_points[2 * n + 1:Points_total]

    z_points_new[0:2 * n + 2] = z_points[0:2 * n + 2]
    z_points_new[2 * n + 2] = z_points[2 * n + 1]
    z_points_new[2 * n + 3:Points_total + 1] = z_points[2 * n + 2:Points_total]

    x_points_out = x_points_new
    y_points_out = y_points_new
    z_points_out = z_points_new

    x_points_out = x_points_out - length1 / 2
    y_points_out = y_points_out - length2 / 2

    return x_points_out, y_points_out, z_points_out


# User defined function for generating coil filaments.
def _spiralCoilDimensionCalc(N, length, width, spacing, thickness, angle):

    z_thick = thickness  # %pcb board thickness

    theta_rot = angle  # %default value is pi/2
    # define dimensions
    l = length  # define side length of outer square
    w = width  # define width of each track
    s = spacing  # define spacing between tracks

    ll_s = w + s  # line to line spacing

    Points_total = (4 * N) + 1

    x_points = np.zeros(Points_total)
    y_points = np.zeros(Points_total)
    z_points = np.zeros(Points_total)

    z_points[0:(2 * N + 1)] = +z_thick / 2
    z_points[(2 * N + 1):Points_total] = -z_thick / 2

    x_points_new = np.zeros(Points_total + 1)
    y_points_new = np.zeros(Points_total + 1)
    z_points_new = np.zeros(Points_total + 1)

    x_points_out = np.zeros(Points_total + 1)
    y_points_out = np.zeros(Points_total + 1)
    z_points_out = np.zeros(Points_total + 1)

    line_segs = np.zeros(4 * N)
    line_segs[0:3] = l

    i = 1
    # increment the decrease in length

    for seg_move in range(3, 2 * N, 2):
        line_segs[seg_move:(seg_move + 2)] = (l - i * ll_s)
        i = i + 1

    i_smaller = i - 1

    for seg_move in range(2 * N - 1, (4 * N) - 1, 2):
        line_segs[seg_move:(seg_move + 2)] = (l - i_smaller * ll_s)
        i = 1 + i
        i_smaller = i_smaller - 1

    line_segs[(4 * N - 3):(4 * N - 1)] = l
    line_segs[4 * N - 1] = l - ll_s

    x_traj = np.cos([theta_rot, theta_rot + .5 * np.pi, theta_rot + np.pi, theta_rot + 1.5 * np.pi])
    y_traj = np.sin([theta_rot, theta_rot + .5 * np.pi, theta_rot + np.pi, theta_rot + 1.5 * np.pi])

    q = 0

    for p in range(1, Points_total):
        x_points[p] = x_traj[q] * line_segs[p - 1] + x_points[p - 1]
        y_points[p] = y_traj[q] * line_segs[p - 1] + y_points[p - 1]
        q = q + 1
        if q == 4:
            q = 0

    x_points_new[0:2 * N + 1] = x_points[0:2 * N + 1]
    x_points_new[2 * N + 1] = x_points[2 * N]
    x_points_new[2 * N + 2:Points_total + 1] = x_points[2 * N + 1:Points_total]

    y_points_new[0:2 * N + 1] = y_points[0:2 * N + 1]
    y_points_new[2 * N + 1] = y_points[2 * N]
    y_points_new[2 * N + 2:Points_total + 1] = y_points[2 * N + 1:Points_total]

    z_points_new[0:2 * N + 2] = z_points[0:2 * N + 2]
    z_points_new[2 * N + 2] = z_points[2 * N + 1]
    z_points_new[2 * N + 3:Points_total + 1] = z_points[2 * N + 2:Points_total]

    x_points_out = x_points_new
    y_points_out = y_points_new
    z_points_out = z_points_new

    if angle == np.pi / 2:
        x_points_out = x_points_out + length / 2
        y_points_out = y_points_out - length / 2
    else:

        y_points_out = y_points_out - length / (np.sqrt(2))

    return x_points_out, y_points_out, z_points_out


if __name__ == '__main__':
    import matplotlib

    x_points_out, y_points_out, z_points_out = _
