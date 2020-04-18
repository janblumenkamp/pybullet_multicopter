import numpy as np

class PID():
    def __init__(self, kp, ki, kd):
        self.k = {'P': kp, 'I': ki, 'D': kd}
        self.reset()

    def reset(self):
        self.integral = 0
        self.last_error = 0
        self.error = 0
        
    def step(self, setpoint, feedback, dt):
        self.error = setpoint - feedback
        self.integral += self.error * dt
        p = self.k['P']*self.error
        i = self.k['I']*self.integral
        d = self.k['D']*((self.error - self.last_error)/dt)
        self.last_error = self.error
        return p+i+d

