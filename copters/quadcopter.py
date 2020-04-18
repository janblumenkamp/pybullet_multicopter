import numpy as np
import pybullet as p
from ..util.pid import PID

DEFAULT_CONFIG = {
    'step_freq': 240,
    'reset_pos': [0, 0, 1],
    'pid': {
        'roll_rate': [1, 0.1, 0.001],
        'pitch_rate': [1, 0.1, 0.001],
        'yaw_rate': [1, 0, 0],
        'roll_angle': [10, 0, 0],
        'pitch_angle': [10, 0, 0],
        'vx': [0.3, 0.01, 0.001],
        'vy': [0.3, 0.01, 0.001],
        'vz': [100, 1, 0.1]
    },
    'limits': {
        'thrust': (0, 6),
        'pitch_angle': (-np.pi/8, np.pi/8),
        'roll_angle': (-np.pi/8, np.pi/8),
        'pitch_rate': (-np.pi/2, np.pi/2),
        'roll_rate': (-np.pi/2, np.pi/2),
        'yaw_rate': (-2*np.pi, 2*np.pi),
        'pid_pitch_rate_out': (-3, 3),
        'pid_roll_rate_out': (-3, 3),
        'pid_yaw_rate_out': (-3, 3),
        'motor_force': (0, 15),
        'yaw_torque': (-0.5, 0.5),
        'vx': (-12, 12),
        'vy': (-12, 12),
        'vz': (-7, 2), # if we allow higher rising speeds, there are some issues with the speed controllers
    },
    'wind_vector': [0.2, 0.1, 0.03],
    #'wind_vector': [0, 0, 0],
    'force_motor_noise': 2,
    'force_noise_fac': 2,
    'torque_noise_fac': 0.1
}

class Quadcopter():
    def __init__(self, pybullet_client, cfg=DEFAULT_CONFIG):
        self.pybullet_client = pybullet_client
        self.cfg = cfg
        self.cfg['dt'] = 1/self.cfg['step_freq']
        self.initial_pos = self.position = np.array(self.cfg['reset_pos'])
        import os
        p.setAdditionalSearchPath(os.path.dirname(os.path.abspath(__file__)))
        self.body_id = p.loadURDF("quadcopter.urdf", basePosition=self.initial_pos, physicsClientId=self.pybullet_client)

        self.simulated_wind_vector = np.array(self.cfg['wind_vector'])

        self.thrust = 0
        self.setpoint_roll_rate = 0
        self.setpoint_pitch_rate = 0
        self.setpoint_yaw_rate = 0
        self.setpoint_roll_angle = 0
        self.setpoint_pitch_angle = 0
        self.pid_roll_rate = PID(*self.cfg['pid']['roll_rate'])
        self.pid_pitch_rate = PID(*self.cfg['pid']['pitch_rate'])
        self.pid_yaw_rate = PID(*self.cfg['pid']['yaw_rate'])
        self.pid_roll_angle = PID(*self.cfg['pid']['roll_angle'])
        self.pid_pitch_angle = PID(*self.cfg['pid']['pitch_angle'])

        self.pid_vx = PID(*self.cfg['pid']['vx'])
        self.pid_vy = PID(*self.cfg['pid']['vy'])
        self.pid_vz = PID(*self.cfg['pid']['vz'])

        self.reset()	

    def reset(self):
        p.resetBasePositionAndOrientation(self.body_id, self.initial_pos, p.getQuaternionFromEuler([0,0,0*(np.pi/180)]), physicsClientId=self.pybullet_client)
        self.update_state()

        self.pid_roll_rate.reset()
        self.pid_pitch_rate.reset()
        self.pid_yaw_rate.reset()
        self.pid_roll_angle.reset()
        self.pid_pitch_angle.reset()
        self.pid_vx.reset()
        self.pid_vy.reset()
        self.pid_vz.reset()

    def update_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.pybullet_client)
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.orientation_euler = np.array(p.getEulerFromQuaternion(orientation))

        m = self.get_rotation_matrix()
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id, self.pybullet_client)
        self.world_lateral_speed = np.array(lin_vel)
        self.lateral_speed = self.world_lateral_speed @ m
        self.world_ang_speed = np.array(ang_vel)
        self.ang_speed = self.world_ang_speed @ m

    def get_rotation_matrix(self, orientation=None):
        return np.array(p.getMatrixFromQuaternion(self.orientation if orientation is None else orientation)).reshape(3,3)

    def set_thrust(self, thrust):
        self.thrust = np.clip(thrust, *self.cfg['limits']['thrust'])

    def set_pitch_rate(self, pitch):
        self.setpoint_pitch_rate = np.clip(pitch, *self.cfg['limits']['pitch_rate'])

    def set_roll_rate(self, roll):
        self.setpoint_roll_rate = np.clip(roll, *self.cfg['limits']['roll_rate'])

    def set_pitch_angle(self, pitch):
        self.setpoint_pitch_angle = np.clip(pitch, *self.cfg['limits']['pitch_angle'])

    def set_roll_angle(self, roll):
        self.setpoint_roll_angle = np.clip(roll, *self.cfg['limits']['roll_angle'])

    def set_yaw_rate(self, yaw):
        self.setpoint_yaw_rate = np.clip(yaw, *self.cfg['limits']['yaw_rate'])

    def step(self):
        self.update_state()

        ctrl_pitch = np.clip(self.pid_pitch_rate.step(self.setpoint_pitch_rate, self.ang_speed[1], self.cfg['dt']), *self.cfg['limits']['pid_pitch_rate_out'])
        ctrl_roll = np.clip(self.pid_roll_rate.step(self.setpoint_roll_rate, self.ang_speed[0], self.cfg['dt']), *self.cfg['limits']['pid_roll_rate_out'])
        ctrl_yaw = np.clip(self.pid_yaw_rate.step(self.setpoint_yaw_rate, self.ang_speed[2], self.cfg['dt']), *self.cfg['limits']['pid_yaw_rate_out'])
        translation_forces = np.clip(np.array([
            self.thrust + ctrl_roll - ctrl_pitch,
            self.thrust - ctrl_roll - ctrl_pitch,
            self.thrust + ctrl_roll + ctrl_pitch,
            self.thrust - ctrl_roll + ctrl_pitch,
        ]) + np.random.uniform(-self.cfg['force_motor_noise'], self.cfg['force_motor_noise'], 4), *self.cfg['limits']['motor_force'])
        yaw_torque = np.clip(ctrl_yaw, *self.cfg['limits']['yaw_torque'])

        motor_points = np.array([[1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0]])*0.2
        for force_point, force in zip(motor_points, translation_forces):
            p.applyExternalForce(self.body_id, -1, [0,0,force], force_point, p.LINK_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalTorque(self.body_id, 0, [0,0,yaw_torque], p.LINK_FRAME, physicsClientId=self.pybullet_client)

        # apply noise and wind
        p.applyExternalForce(self.body_id, -1, (np.random.rand(3)-0.5)*self.cfg['force_noise_fac'], self.position, p.WORLD_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalTorque(self.body_id, 0, (np.random.rand(3)-0.5)*self.cfg['torque_noise_fac'], p.LINK_FRAME, physicsClientId=self.pybullet_client)
        p.applyExternalForce(self.body_id, -1, self.simulated_wind_vector, self.position, p.WORLD_FRAME, physicsClientId=self.pybullet_client)

    def step_angle(self):
        self.set_roll_rate(self.pid_roll_angle.step(self.setpoint_roll_angle, self.orientation_euler[0], self.cfg['dt']))
        self.set_pitch_rate(self.pid_pitch_angle.step(self.setpoint_pitch_angle, self.orientation_euler[1], self.cfg['dt']))

    def step_speed(self, vx, vy, vz):
        self.set_pitch_angle(self.pid_vx.step(np.clip(vx, *self.cfg['limits']['vx']), self.lateral_speed[0], self.cfg['dt']))
        self.set_roll_angle(self.pid_vy.step(np.clip(vy, *self.cfg['limits']['vy']), -self.lateral_speed[1], self.cfg['dt']))
        self.set_thrust(self.pid_vz.step(np.clip(vz, *self.cfg['limits']['vz']), self.lateral_speed[2], self.cfg['dt']))

