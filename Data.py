import numpy as np


class Data:
    def __init__(self):
        self.data = np.load('files/data.npy')
        self.address = 0
        self.function = 1
        self.length = 2
        self.setpoint = 3
        self.gain = 4
        self.reset_rate = 5
        self.deadband = 6
        self.cycle_time = 7
        self.rate = 8
        self.system_mode = 9
        self.control_scheme = 10
        self.pump = 11
        self.solenoid = 12
        self.pressure_measurement = 13
        self.crc_rate = 14
        self.command_response = 15
        self.time = 16
        self.binary_result = 17
        self.categorized_result = 18
        self.specific_result = 19
        self.time_interval = 20

    def load_data(self):
        return self.data


if __name__ == '__main__':
    pass
