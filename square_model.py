""" 
    Designer specified class definition for emitter coils used in the EM tracking system design
"""

import numpy as np
import matplotlib.pyplot as plt

class CoilModel:

    def __init__(self, module_config={'centers_x': None, 'centers_y': None}):
        '''
            The init function should initialise and prepare all data necessary for 'coil_field' to operate.
        '''
        x_centres = np.array(module_config['centers_x'])
        x_centres = x_centres * 1.0e-3
        y_centres = np.array(module_config['centers_y'])
        y_centres = y_centres * 1.0e-3
        
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
        
        self.x_points = np.array(
            [x_points_1, x_points_2, x_points_3, x_points_4, x_points_5, x_points_6, x_points_7, x_points_8])
        self.y_points = np.array(
            [y_points_1, y_points_2, y_points_3, y_points_4, y_points_5, y_points_6, y_points_7, y_points_8])
        self.z_points = np.array(
            [z_points_1, z_points_2, z_points_3, z_points_4, z_points_5, z_points_6, z_points_7, z_points_8])
        
    def coil_field_total(self, px, py, pz):
        
        '''
            must be defined. 
            A User defined function which returns the magnetic field intensity H at a point in space P.
            This function must be defined as accepting a cartesian coordinate (x,y,z),
            and returning a three magnetic intensity values Hx, Hy and Hz
        '''
        
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
    
x_centers = [-93.543, 0., 93.543, -68.55, 68.55, -93.543, 0., 93.543]
y_centers = [93.543, 68.55, 93.543, 0., 0., -93.543, -68.55, -93.543]
coil_model = CoilModel(module_config={'centers_x': x_centers, 
                                      'centers_y': y_centers})

h = coil_model.coil_field_total(0., 0., 0.01)
print(h)