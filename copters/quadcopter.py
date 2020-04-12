import numpy as np
import pybullet as p
from ..util.pid import PID

DEFAULT_CONFIG = {
    'step_freq': 240,
    'reset_pos': [0, 0, 1],
    'pid': {
        'roll': [0.2, 0.01, 0.1],
        'pitch': [0.2, 0.01, 0.1],
        'yaw': [1, 0, 0],
        'vx': [0.5, 0.2, 0.01],
        'vy': [0.5, 0.2, 0.01],
        'vz': [100, 1, 0.1]
    },
    'limits': {
        'thrust': (0, 6),
        'pitch': (-np.pi/8, np.pi/8),
        'roll': (-np.pi/8, np.pi/8),
        'yaw': (-2*np.pi, 2*np.pi),
        'pid_pitch_out': (-3, 3),
        'pid_roll_out': (-3, 3),
        'pid_yaw_out': (-3, 3),
        'motor_force': (0, 15),
        'yaw_torque': (-0.5, 0.5),
        'vx': (-12, 12),
        'vy': (-12, 12),
        'vz': (-7, 1.5), # if we allow higher rising speeds, there are some issues with the speed controllers
    },
    'wind_vector': [0.2, 0.1, 0.03],
    'force_noise_fac': 2,
    'torque_noise_fac': 0.1
}

class Quadcopter():
    def __init__(self, pybullet_client, cfg=DEFAULT_CONFIG):
        self.pybullet_client = pybullet_client
        self.cfg = cfg
        self.initial_pos = self.position = np.array(self.cfg['reset_pos'])
        import os
        p.setAdditionalSearchPath(os.path.dirname(os.path.abspath(__file__)))
        self.body_id = p.loadURDF("quadcopter.urdf", basePosition=self.initial_pos, physicsClientId=self.pybullet_client)

        self.simulated_wind_vector = np.array(self.cfg['wind_vector'])

        self.thrust = 0
        self.setpoint_roll = 0
        self.setpoint_pitch = 0
        self.setpoint_yaw = 0
        self.roll_pid = PID(*self.cfg['pid']['roll'])
        self.pitch_pid = PID(*self.cfg['pid']['pitch'])
        self.yaw_pid = PID(*self.cfg['pid']['yaw'])

        self.pid_vx = PID(*self.cfg['pid']['vx'])
        self.pid_vy = PID(*self.cfg['pid']['vy'])
        self.pid_vz = PID(*self.cfg['pid']['vz'])

        self.reset()	

    def reset(self):
        p.resetBasePositionAndOrientation(self.body_id, self.initial_pos, p.getQuaternionFromEuler([0,0,0*(np.pi/180)]), physicsClientId=self.pybullet_client)
        self.update_state()
        self.last_orientation = self.orientation
        self.last_position = self.position
        self.compute_speed()

    def update_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.pybullet_client)
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.orientation_euler = np.array(p.getEulerFromQuaternion(orientation))

    def get_rotation_matrix(self):
        return np.array(p.getMatrixFromQuaternion(self.orientation)).reshape(3,3)

    def compute_speed(self):
        m = self.get_rotation_matrix()
        self.absolute_speed = (self.position - self.last_position)*self.cfg['step_freq']
        self.lateral_speed = self.absolute_speed @ m
        delta_speed_quat = p.getDifferenceQuaternion(self.last_orientation, self.orientation)
        self.ang_speed = np.array(p.getEulerFromQuaternion(delta_speed_quat))*self.cfg['step_freq']
        self.last_orientation = self.orientation
        self.last_position = self.position

    def set_thrust(self, thrust):
        self.thrust = np.clip(thrust, *self.cfg['limits']['thrust'])

    def set_pitch(self, pitch):
        self.setpoint_pitch = np.clip(pitch, *self.cfg['limits']['pitch'])

    def set_roll(self, roll):
        self.setpoint_roll = np.clip(roll, *self.cfg['limits']['roll'])

    def set_yaw(self, yaw):
        self.setpoint_yaw = np.clip(yaw, *self.cfg['limits']['yaw'])

    def step(self):
        self.update_state()
        self.compute_speed()

        dt = 1/self.cfg['step_freq']
        ctrl_pitch = np.clip(self.pitch_pid.step(self.setpoint_pitch, self.orientation_euler[1], dt), *self.cfg['limits']['pid_pitch_out'])
        ctrl_roll = np.clip(self.roll_pid.step(self.setpoint_roll, self.orientation_euler[0], dt), *self.cfg['limits']['pid_roll_out'])
        ctrl_yaw = np.clip(self.yaw_pid.step(self.setpoint_yaw, self.ang_speed[2], dt), *self.cfg['limits']['pid_yaw_out'])
        translation_forces = np.clip(np.array([
            self.thrust + ctrl_roll - ctrl_pitch,
            self.thrust - ctrl_roll - ctrl_pitch,
            self.thrust + ctrl_roll + ctrl_pitch,
            self.thrust - ctrl_roll + ctrl_pitch,
        ]), *self.cfg['limits']['motor_force'])
        yaw_torque = np.clip(ctrl_yaw, *self.cfg['limits']['yaw_torque'])

        motor_points = np.array([[1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0]])*0.2
        for force_point, force in zip(motor_points, translation_forces):
            p.applyExternalForce(self.body_id, -1, [0,0,force], force_point, p.LINK_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalTorque(self.body_id, 0, [0,0,yaw_torque], p.LINK_FRAME, physicsClientId=self.pybullet_client)

        # apply noise and wind
        p.applyExternalForce(self.body_id, -1, (np.random.rand(3)-0.5)*self.cfg['force_noise_fac'], self.position, p.WORLD_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalTorque(self.body_id, 0, (np.random.rand(3)-0.5)*self.cfg['torque_noise_fac'], p.LINK_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalForce(self.body_id, -1, self.simulated_wind_vector, self.position, p.WORLD_FRAME, physicsClientId=self.pybullet_client)

    def step_speed(self, vx, vy, vz):
        dt = 1/self.cfg['step_freq']
        self.set_pitch(self.pid_vx.step(np.clip(vx, *self.cfg['limits']['vx']), self.lateral_speed[0], dt))
        self.set_roll(self.pid_vy.step(np.clip(vy, *self.cfg['limits']['vy']), -self.lateral_speed[1], dt))
        self.set_thrust(self.pid_vz.step(np.clip(vz, *self.cfg['limits']['vz']), self.lateral_speed[2], dt))

