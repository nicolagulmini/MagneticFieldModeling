from PyDAQmx import *
import numpy as np
import sys
from numpy import zeros
import threading
import platform
from ctypes import byref

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
