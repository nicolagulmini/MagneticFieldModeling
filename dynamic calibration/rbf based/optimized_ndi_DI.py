import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from queue import Queue
import pyigtl
from sklearn.metrics.pairwise import rbf_kernel
import sys
from time import sleep
np.set_printoptions(threshold=sys.maxsize)
from numpy import zeros
import pyigtl
from PyDAQmx import *
import threading
import platform
from ctypes import byref

DrfToAxis7 = np.array([
    [1,	0,					-0.0349154595874112,	-8.54323644559691],
    [0,	0.0213641235902639,	0.999162169684586,	1.90874592349076],
    [0,	0.999771761065104,	-0.0213510972315487,	-4.35779932552848],
    [0,	0,					0,					1]
    ])

class NIDAQ(Task):
    """
    Class definition for national instruments data acquisition system
    """
    def __init__(self, dev_name='Dev1', channels=np.array([0]), data_len=2000, sampleFreq=100000.0, contSample=True):
        Task.__init__(self)
        if dev_name is None:
            dev_name = dev_name
        self.dev_name = dev_name
        self._data = zeros(data_len * channels.shape[0])
        self.channels = channels
        self.sampleFreq = sampleFreq
        self.data_len = data_len
        self.read = int32()
        self.contSample = contSample

        self._data_lock = threading.Lock()
        self._newdata_event = threading.Event()

    # Sets up analog inputs of the NI DAQ. This function must be called from a new taskj which does not contain digital outputs.
    def SetAnalogInputs(self):
        for i in range(self.channels.shape[0]):
            self.CreateAIVoltageChan(self.dev_name+"/ai"+str(self.channels[i]),"", DAQmx_Val_RSE, -10.0,10.0, DAQmx_Val_Volts, None)

        if self.contSample is True:
            self.CfgSampClkTiming("", self.sampleFreq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.data_len)
        else:
            self.CfgSampClkTiming("", self.sampleFreq, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, self.data_len)

        if platform.system() == 'Windows' and self.contSample is True:
            self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.data_len,0)
            self.AutoRegisterDoneEvent(0)
        elif (platform.system() == 'Linux' or platform.system() == 'Darwin'):
            pass

    # Sets up the counter output of the NI DAQ for frequency generation. Must be a called from a new task which does not contain analog input.
    def SetClockOutput(self, dst="PFI7"):
        self.CreateCOPulseChanFreq( "/" + self.dev_name + "/" + "ctr0", "", DAQmx_Val_Hz, DAQmx_Val_Low, 0.0, 1250000, 0.5)
        self.CfgImplicitTiming(DAQmx_Val_ContSamps, 1000)
        DAQmxConnectTerms("/" + self.dev_name + "/" + "PFI12", "/" + self.dev_name + "/" + dst, DAQmx_Val_DoNotInvertPolarity)


    def EveryNCallback(self):
        with self._data_lock:
            self.ReadAnalogF64(self.data_len, 10.0, DAQmx_Val_GroupByChannel, self._data, self.data_len*self.channels.shape[0], byref(self.read), None)
            self._newdata_event.set()
        return 0 # The function should return an integer

    def DoneCallback(self, status):
        print("Status",status.value)
        return 0 # The function should return an integer

    def get_data(self, blocking=True, timeout=None):
        if platform.system() == 'Windows' and self.contSample is True:
            if blocking:
                if not self._newdata_event.wait(timeout):
                    raise ValueError("timeout waiting for data from device")
            with self._data_lock:
                self._newdata_event.clear()
                return self._data.copy()
        elif platform.system() == 'Linux' or platform.system() == 'Darwin' or self.contSample is False:
            self.ReadAnalogF64(self.data_len, 10.0, DAQmx_Val_GroupByChannel, self._data, self.data_len*self.channels.shape[0], byref(self.read), None)
            return self._data.copy()

    def get_data_matrix(self, timeout=None):
        data = self.get_data(timeout=timeout)
        data_mat = np.matrix(np.reshape(data, (self.channels.shape[0], self.data_len)).transpose())
        return data_mat

    def resetDevice(self):
        DAQmxResetDevice(self.dev_name)

channeldict = {0: 4, 1: 0, 2: 8, 3: 1, 4: 9, 5: 2, 6: 10, 7: 11, 8: 3, 9: 8, 10: 12, 11: 13, 12: 5, 13: 14, 14: 6, 15: 15, 16: 7}
sampleFreq = 40000
noSamples = 4000
freqs = np.array([6000, 6200, 6400, 6600, 6800, 7000, 7200, 7400])
idx_signal = freqs/sampleFreq*noSamples+1
idx_signal = idx_signal.astype(int)
deviceID = 'Dev1'
sensor_channel = 7
sensor_channel = channeldict[sensor_channel]
PhaseOffset = 0

task = NIDAQ(dev_name = deviceID, channels = np.array([4, str(sensor_channel)]), sampleFreq = sampleFreq, data_len = noSamples)
task.SetAnalogInputs()
task.StartTask()

task1 = NIDAQ(dev_name=deviceID)
task1.SetClockOutput()
task1.StartTask()

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

global coil_model
coil_model = CoilModel(module_config={'centers_x': [-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], 
                                      'centers_y': [93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]}) # mm

def theoretical_field(model, point):
    return np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T # (3, 8)

def get_fft(idx_signal):
    ft = task.get_data_matrix(timeout = 10.0)
    yf = np.fft.fft(ft, axis = 0) / noSamples
    yf = yf[idx_signal, :]
    return yf
    
def get_flux(yf, PhaseOffset):
    yf_mag = 2 * abs(yf[:, 1])
    yf_phase = np.angle(yf)
    angleSignal = yf_phase[:, 0] - yf_phase[:, 1] + PhaseOffset
    flux = np.sin(angleSignal) * yf_mag    
    return flux
    
def produce_basis_vectors_for_prediction(n):
    to_pred_x = np.array([np.ones(shape=(n)), np.zeros(shape=(n)), np.zeros(shape=(n))])
    to_pred_y = np.array([np.zeros(shape=(n)), np.ones(shape=(n)), np.zeros(shape=(n))])
    to_pred_z = np.array([np.zeros(shape=(n)), np.zeros(shape=(n)), np.ones(shape=(n))])
    return to_pred_x, to_pred_y, to_pred_z

class uniaxial_to_calib:
    
    def __init__(self, gamma, sigma, stack_grid):
        self.gamma = gamma
        self.sigma = sigma
        self.points = np.array([]) # shape = (number of points, 6), which means 3 for position and 3 for orientation
        self.measures = np.array([]) # shape = (number of points, 8), which means one uniaxial measure per coil
        
        self.k = None
        self.w = None 
        self.pred_kernel = None # kernel between training points and stack grid
        
        self.stack_grid = stack_grid
        self.diag_on_grid_for_cholensky = np.diag(self.produce_kernel(self.stack_grid, self.stack_grid))
        
    def update_points(self, new_points, new_measures):
        if self.points.shape[0] == 0:
            self.points = new_points
            self.measures = new_measures
        else:
            self.points = np.concatenate((self.points, new_points))
            self.measures = np.concatenate((self.measures, new_measures))
        
    def produce_kernel(self, X, Y):
        return rbf_kernel(X[:, :3], Y[:, :3], gamma=self.gamma) * np.tensordot(X[:, 3:], Y[:, 3:], axes=(1, 1))
    
    def set_kernel(self):
        self.k = self.sigma*np.eye(self.points.shape[0])+self.produce_kernel(self.points, self.points)
        
    def update_kernel(self, new_points, new_measures):
        self.update_points(new_points, new_measures)
        if self.k is None:
            self.set_kernel()
            return
        self.k = self.k - self.sigma*np.eye(self.k.shape[0])
        K_ = self.produce_kernel(new_points, self.points)
        self.k = np.concatenate((self.k, K_[:, :self.k.shape[0]]))
        self.k = np.concatenate((self.k, K_.T), axis=1)
        self.k = self.k + self.sigma*np.eye(self.k.shape[0])
        
    def set_weights(self):
        if self.k.shape[0] != self.measures.shape[0]:
            print("Error. Return")
            return
        self.w = np.linalg.solve(self.k, self.measures) # it has to be (number_of_grid_points, 8)
        
    def predict(self):  
        return np.matmul(self.pred_kernel, self.w)
            
    def update_pred_kernel(self, new_points, new_measures):
        #self.update_points(new_points, new_measures) # otherwise the points are set twice
        pred_kernel_on_new_points = self.produce_kernel(self.stack_grid, new_points)
        if self.pred_kernel is None:
            self.pred_kernel = pred_kernel_on_new_points
        else:
            self.pred_kernel = np.concatenate((self.pred_kernel, pred_kernel_on_new_points), axis=1)
        
    def uncertainty(self, new_points, new_measures):
        self.update_pred_kernel(new_points, new_measures)
        L = np.linalg.cholesky(self.k)
        Lk = np.linalg.solve(L, np.transpose(self.pred_kernel))
        stdv = np.sqrt(self.diag_on_grid_for_cholensky-np.sum(Lk**2, axis=0))
        return stdv
    
class cube_to_calib:
    
    def __init__(self, origin, side_length=40., gamma=0.0005, sigma=0., point_density=10.):
        self.origin_corner = origin # numpy vector for position [x, y, z]
        # for the opposite corner, for example, it is sufficient to do self.origin_corner + self.side_length * numpy.ones(3)
        self.side_length = side_length # in centimeters
        
        x = np.linspace(self.origin_corner[0], self.origin_corner[0]+self.side_length, int(self.side_length/point_density)+1) # i think
        y = np.linspace(self.origin_corner[1], self.origin_corner[1]+self.side_length, int(self.side_length/point_density)+1)
        z = np.linspace(self.origin_corner[2], self.origin_corner[2]+self.side_length, int(self.side_length/point_density)+1)

        # is there a better method to do so?
        grid = np.zeros((int(x.shape[0]*y.shape[0]*z.shape[0]), 3))
        c = 0
        for i in z:
            for j in y:
                for k in x:
                    grid[c] = np.array([k, j, i])
                    c += 1
        self.grid = grid
        
        basis_vectors_x, basis_vectors_y, basis_vectors_z = produce_basis_vectors_for_prediction(self.grid.shape[0])
        high_dim_x = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_x)))
        high_dim_y = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_y)))
        high_dim_z = np.transpose(np.concatenate((np.transpose(self.grid), basis_vectors_z)))
        self.stack_grid = np.concatenate((high_dim_x, high_dim_y, high_dim_z)) 
        self.interpolator = uniaxial_to_calib(gamma, sigma, self.stack_grid)
    
    def update_uncert_vis(self, new_points, new_measures):
        self.interpolator.update_kernel(new_points, new_measures)
        return self.interpolator.uncertainty(new_points, new_measures)

def uniaxial_dynamic_cal(client, origin, side_length, GAMMA=.0005, SIGMA=(2.5e-3)**2, interval=100, EPSILON=1, AMOUNT_OF_NEW_POINTS=15, referenceToBoard=None):
    # EPSILON is mm of tolerance for the visualization of the cube
            
    plt.close('all')
    fig = plt.figure("Three components")
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ay = fig.add_subplot(1, 3, 2, projection='3d')
    az = fig.add_subplot(1, 3, 3, projection='3d')
    
    global cube
    cube = cube_to_calib(origin=origin, side_length=side_length, sigma=SIGMA, point_density=20.)
    
    zline = cube.grid.T[2]
    yline = cube.grid.T[1]
    xline = cube.grid.T[0]
        
    q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
    
    def animate(k):
        
        ax.clear()
        ax.set_title("\nx component")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        ax.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        ax.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        ay.clear()
        ay.set_title("\ny component")
        ay.set_xlabel("x (mm)")
        ay.set_ylabel("y (mm)")
        ay.set_zlabel("z (mm)")
        ay.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        ay.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        ay.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        az.clear()
        az.set_title("\nz component")
        az.set_xlabel("x (mm)")
        az.set_ylabel("y (mm)")
        az.set_zlabel("z (mm)")
        az.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        az.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        az.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        for _ in range(AMOUNT_OF_NEW_POINTS):
            message = client.wait_for_message("SensorToReference", timeout=5)
            if message is not None:
                mat = np.matmul(np.matmul(referenceToBoard, message.matrix), DrfToAxis7)
                pos = mat.T[3][:3]
                ori = mat.T[2][:3]
                tmp = get_flux(get_fft(idx_signal), PhaseOffset)
                # messaggio ottico, get flux e poi ancora messaggio ottico e faccio la media
                # per le posizioni faccio la media, per le orientazioni faccio la somma e la normalizzo
                q.put(np.concatenate((pos, ori, tmp), axis=0))
                ax.quiver(pos[0], pos[1], pos[2], 7*ori[0], 7*ori[1], 7*ori[2], color='blue')
                ay.quiver(pos[0], pos[1], pos[2], 7*ori[0], 7*ori[1], 7*ori[2], color='blue')
                az.quiver(pos[0], pos[1], pos[2], 7*ori[0], 7*ori[1], 7*ori[2], color='blue')

        if len(q.queue) >= AMOUNT_OF_NEW_POINTS: 

            new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
                        
            new_points = new_raw_points[:, :6] # shape = (AMOUNT_OF_NEW_POINTS, 6)
            new_measures = new_raw_points[:, 6:] # shape = (AMOUNT_OF_NEW_POINTS, 8)
            unc = cube.update_uncert_vis(new_points, new_measures)
            
            dims = int(unc.shape[0]/3)
            unc_x = unc[:dims]
            unc_y = unc[dims:int(2*dims)]
            unc_z = unc[int(2*dims):]
            unc_x_ = unc_x[np.newaxis] / max(unc_x)
            unc_y_ = unc_y[np.newaxis] / max(unc_y)
            unc_z_ = unc_z[np.newaxis] / max(unc_z)
            color_vec_x = np.concatenate((unc_x_, 1-unc_x_, np.zeros(unc_x_.shape)), axis=0).T
            color_vec_y = np.concatenate((unc_y_, 1-unc_y_, np.zeros(unc_y_.shape)), axis=0).T
            color_vec_z = np.concatenate((unc_z_, 1-unc_z_, np.zeros(unc_z_.shape)), axis=0).T
        
            for i in np.arange(.1, 1., .2):
                ax.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x_)**2, alpha = .05/i, c = color_vec_x)
                ay.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_y_)**2, alpha = .05/i, c = color_vec_y)
                az.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_z_)**2, alpha = .05/i, c = color_vec_z)
                            
            #if cube.interpolator.w is not None:
            #    if cube.interpolator.w.shape[0] % AMOUNT_OF_NEW_POINTS == 0:
            cube.interpolator.set_weights()
            grid_field = cube.interpolator.predict()

            # tmp = theoretical_field(coil_model, cube.grid[0])
            # first_coil_field_theoretical = np.array([tmp[0].A1[0], tmp[1].A1[0], tmp[2].A1[0]])
            # grid_field_predicted = np.array([grid_field[0][0], grid_field[0+125][0], grid_field[0+125*2][0]])
            
    global ani
    ani = FuncAnimation(plt.gcf(), animate, interval=interval)
    #plt.tight_layout()
    
client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18944)
message = client.wait_for_message("ReferenceToBoard", timeout=5)
referenceToBoard = message.matrix 
#client = None
uniaxial_dynamic_cal(client, AMOUNT_OF_NEW_POINTS=10, origin=np.array([0., 0., 100.]), side_length=100., SIGMA=1e-10, referenceToBoard=referenceToBoard)