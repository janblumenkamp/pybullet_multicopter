from .copters.quadcopter import Quadcopter
from .util.pid import PID
import numpy as np
import time
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import os
import threading
import keyboard

control_limits = {
    'thrust': (0, 14, 50),
    'pitch': (-np.pi, 0, np.pi),
    'roll': (-np.pi, 0, np.pi),
    'yaw': (-np.pi, 0, np.pi),
    'v': (-4, 0, 4),
    'height': (0, 1, 10)
}
current_control = {key: limit[1] for key, limit in control_limits.items()}

def keyboard_command(command, change):
    global current_control
    limits = {
        'thrust': (0, 14, 50),
        'pitch': (-np.pi, 0, np.pi),
        'roll': (-np.pi, 0, np.pi),
        'yaw': (-np.pi, 0, np.pi),
        'v': (-4, 0, 4)
    }
    if command == "reset":
        for k in current_control.keys():
            current_control[k] = limits[k][1]
    else:
        current_control[command] = np.clip(current_control[command]+change, limits[command][0], limits[command][2])

def print_info():
    while True:
        print(current_control['roll'], drone.orientation[0])
        #print(drone.orientation)
        time.sleep(0.05)

#threading.Thread(target=print_info, args=()).start()

def eval_pid_rate_roll(client):
    pif_h = PID(2, 2.4, 0.84)
    desired_height = 3
    angle_sign = -1
    setpoints = []
    real_data = []
    debug = []
    timestep = []
    current_time = 0
    state = 0
    setpoint = 0
    state_change_time = 0
    while True:
        if state == 0:
            if current_time > 1:
                state = 1
        elif state == 1:
            setpoint = np.pi/2
            if drone.orientation_euler[0] > np.pi/7:
                state_change_time = current_time
                setpoint = 0
                state = 2
        elif state == 2:
            if current_time - state_change_time > 2:
                setpoint = -np.pi/4
                state = 3
        elif state == 3:
            if drone.orientation_euler[0] < -np.pi/7:
                setpoint = 0
                state_change_time = current_time
                state += 1
        elif state == 4:
            if current_time - state_change_time > 2:
                break

        drone.set_roll_rate(setpoint)
        drone.set_pitch_rate(0)
        drone.set_yaw_rate(0)
        drone.set_thrust(pif_h.step(desired_height, drone.position[2], 1/240))
        drone.step()

        setpoints.append(setpoint)
        real_data.append(drone.ang_speed[0])
        #debug.append(drone.roll_pid.error)
        timestep.append(current_time)
        current_time += 1/240

        if np.linalg.norm(drone.position, ord=2) > 100:
            drone.reset()

        p.stepSimulation(physicsClientId=client)
        p.resetDebugVisualizerCamera(5, 50, -35, drone.position, client)
        #time.sleep(1/240)

    plt.plot(timestep, setpoints, label="setpoint")
    plt.plot(timestep, real_data, label="real")
    #plt.plot(timestep, debug, label="debug")
    #plt.ylim(-0.3,1)
    plt.legend()
    plt.grid()
    plt.show()

def eval_pid_roll(client):
    pif_h = PID(2, 2.4, 0.84)
    desired_height = 3
    angle_sign = -1
    setpoints = []
    real_data = []
    debug = []
    timestep = []
    current_time = 0
    state = 0
    setpoint = 0
    while True:
        if state == 0:
            if drone.position[2] >= desired_height:
                setpoint = np.pi/8
                state += 1
        elif state == 1:
            if drone.position[1] < -10:
                state += 1
                print("N")
                setpoint = -np.pi/16
        elif state == 2:
            if drone.position[1] > 0:
                setpoint = -np.pi/32
                state += 1
        elif state == 3:
            if drone.position[1] > 10:
                break

        drone.set_pitch_angle(0)
        drone.set_roll_angle(setpoint)
        drone.set_yaw_rate(0)
        drone.set_thrust(pif_h.step(desired_height, drone.position[2], 1/240))
        drone.step_angle()
        drone.step()

        setpoints.append(setpoint)
        real_data.append(drone.orientation_euler[0])
        #debug.append(drone.roll_pid.error)
        timestep.append(current_time)
        current_time += 1/240

        '''
        if drone.position[2] >= desired_height:
            start_logging = True
        if drone.position[1] > 80:
            angle_sign = 1
        elif drone.position[1] < -10:
            break
        elif drone.position[1] > 3:
            angle_sign = 0.1
        elif drone.position[1] < -1:
            angle_sign = 0.5
        '''
        if np.linalg.norm(drone.position, ord=2) > 100:
            drone.reset()

        p.resetDebugVisualizerCamera(5, 50, -35, drone.position, client)
        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)

    plt.plot(timestep, setpoints, label="setpoint")
    plt.plot(timestep, real_data, label="real")
    #plt.plot(timestep, debug, label="debug")
    #plt.ylim(-0.3,1)
    plt.legend()
    plt.grid()
    plt.show()

def test_pid_roll(client):
    pif_h = PID(2, 2.4, 0.84)
    desired_height = 3
    current_time = 0
    setpoints = []
    real_data = []
    debug = []
    timestep = []
    while True:
        drone.set_pitch_angle(0)
        current_time += 1/240
        setpoint = np.random.uniform(-np.pi/4, np.pi/4)
        setpoints.append(setpoint)
        real_data.append(drone.orientation_euler[0])
        timestep.append(current_time)
        debug.append(drone.pid_roll_rate.integral)
        drone.set_roll_angle(setpoint)
        drone.set_yaw_rate(0)
        drone.set_thrust(pif_h.step(desired_height, drone.position[2], 1/240))
        drone.step_angle()
        drone.step()

        if np.linalg.norm(drone.position, ord=2) > 15:
            drone.reset()
        if current_time > 10:
            break

        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)

    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    plt.plot(timestep, debug)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def eval_pid_yaw(client):
    pif_h = PID(4, 2.4, 0.42)
    desired_height = 2
    angle_sign = 1
    setpoints = []
    debug = []
    real_data = []
    timestep = []
    current_time = 0
    state_change_time = 0
    state = 0
    setpoint = 0
    while True:
        if state == 0:
            if current_time > 0.1:
                setpoint = 6*np.pi
                state_change_time = current_time
                state += 1
        elif state == 1:
            if current_time - state_change_time > 1:
                state_change_time = current_time
                setpoint = -np.pi
                state += 1
        elif state == 2:
            if current_time - state_change_time > 1:
                break

        drone.set_pitch_rate(0)
        drone.set_roll_rate(0)
        drone.set_yaw_rate(setpoint)
        drone.set_thrust(pif_h.step(desired_height, drone.position[2], 1/240))
        drone.step()

        setpoints.append(setpoint)
        real_data.append(drone.ang_speed[2])
        debug.append(drone.pid_yaw_rate.error)
        timestep.append(current_time)
        current_time += 1/240

        if np.linalg.norm(drone.position, ord=2) > 15:
            drone.reset()

        p.resetDebugVisualizerCamera(5, 50, -35, drone.position, client)
        p.stepSimulation(physicsClientId=client)
        time.sleep(1/240)

    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    plt.plot(timestep, debug)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def remote_control(client):
    keyboard.add_hotkey('up', keyboard_command, args=('pitch', -np.pi/16))
    keyboard.add_hotkey('down', keyboard_command, args=('pitch', np.pi/16))
    keyboard.add_hotkey('left', keyboard_command, args=('roll', np.pi/16))
    keyboard.add_hotkey('right', keyboard_command, args=('roll', -np.pi/16))
    keyboard.add_hotkey('page up', keyboard_command, args=('thrust', 1))
    keyboard.add_hotkey('page down', keyboard_command, args=('thrust', -1))
    keyboard.add_hotkey('ctrl', keyboard_command, args=('reset', 0))
    keyboard.add_hotkey(',', keyboard_command, args=('yaw', -0.1))
    keyboard.add_hotkey('.', keyboard_command, args=('yaw', 0.1))

    pid_h = PID(3, 0.24, 0.625)
    desired_height = 2
    while True:
        drone.step_speed(0, 0, pid_h.step(desired_height, drone.position[2], 1/240))
        drone.set_pitch_angle(current_control['pitch'])
        drone.set_roll_angle(current_control['roll'])
        drone.set_yaw_rate(current_control['yaw'])
        drone.step_angle()
        drone.step()
        if np.linalg.norm(drone.position, ord=2) > 10:
            drone.reset()
        p.stepSimulation(physicsClientId=client)
        time.sleep(1/240)

def eval_pid_speed_hor(client):
    pid_h = PID(3, 0.24, 0.625)
    setpoint = 0
    setpoints = []
    real_data = []
    timestep = []
    current_time = 0
    last_time = 0
    state = 0
    while True:
        if state == 0:
            if current_time > 5:
                state = 1
                setpoint = -1
        elif state == 1:
            if current_time > 10:
                state = 2
                setpoint = 0.2
        elif state == 2:
            if current_time > 30:
                break

        p.resetDebugVisualizerCamera(5, 50, -35, drone.position, client)
        setpoints.append(setpoint)
        real_data.append(drone.lateral_speed[0])
        timestep.append(current_time)
        current_time += 1/240

        drone.step_speed(setpoint, 0, 7)
        drone.step_angle()
        drone.step()

        if np.linalg.norm(drone.position, ord=2) > 100:
            drone.reset()

        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)

    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def eval_pid_speed_ver(client):
    setpoints = []
    real_data = []
    timestep = []
    current_time = 0
    start_logging = True
    start_hover = 0
    vx = 0
    vz = 0
    state = 0
    while True:
        print(drone.lateral_speed)
        if state == 0:
            if current_time > 2:
                vz = 7
                state = 2
        elif state == 2:
            if drone.position[2] > 20:
                state = 3
                vz = -7
        elif state == 3:
            if drone.position[2] < 10:
                state = 4
                vz = -1
        elif state == 4:
            if drone.position[2] < 3:
                state = 5
                vz = 0
                vx = 5
                start_hover = current_time
        elif state == 5:
            if current_time - start_hover > 6:
                break

        setpoints.append(vz)
        real_data.append(drone.lateral_speed[2])
        timestep.append(current_time)
        current_time += 1/240

        drone.step_speed(vx, 0, vz)
        drone.step_angle()
        drone.step()

        if np.linalg.norm(drone.position, ord=2) > 100:
            drone.reset()

        p.resetDebugVisualizerCamera(5, 0, -35, drone.position, client)
        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)

    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def eval_pid_pos_ver():
    pif_h = PID(3, 0.24, 0.625)

    setpoints = []
    out = []
    real_data = []
    timestep = []
    current_time = 0
    start_hover = 0
    setpoint = 0
    while True:
        if current_time < 1:
            setpoint = 1
        elif current_time < 5:
            setpoint = 6
        elif current_time < 9:
            setpoint = 3
        elif current_time < 13:
            setpoint = 0
        elif current_time > 15:
            break

        setpoints.append(setpoint)
        real_data.append(drone.position[2])
        timestep.append(current_time)
        current_time += 1/240

        o = pif_h.step(setpoint, drone.position[2], 1/240)
        out.append(pif_h.integral)#-drone.lateral_speed[2])
        drone.step_speed(0, 0, o)
        drone.step_angle()
        drone.step()

        if np.linalg.norm(drone.position, ord=2) > 15:
            drone.reset()

        p.stepSimulation(physicsClientId=client)
        #time.sleep(1/240)

    plt.plot(timestep, setpoints)
    plt.plot(timestep, real_data)
    plt.plot(timestep, out)
    #plt.ylim(-0.3,1)
    plt.grid()
    plt.show()

def remote_control_nonholonomic(client):
    keyboard.add_hotkey('up', keyboard_command, args=('v', 0.1))
    keyboard.add_hotkey('down', keyboard_command, args=('v', -0.1))
    keyboard.add_hotkey('page up', keyboard_command, args=('height', 0.5))
    keyboard.add_hotkey('page down', keyboard_command, args=('height', -0.5))
    keyboard.add_hotkey('ctrl', keyboard_command, args=('reset', 0))
    keyboard.add_hotkey(',', keyboard_command, args=('yaw', 0.1))
    keyboard.add_hotkey('.', keyboard_command, args=('yaw', -0.1))

    pid_h = PID(3, 0.24, 0.625)
    while True:
        print(current_control, drone.lateral_speed)
        drone.step_speed(current_control['v'], 0, pid_h.step(current_control['height'], drone.position[2], 1/240))
        drone.set_yaw_rate(current_control['yaw'])
        drone.step_angle()
        drone.step()
        #if np.linalg.norm(drone.position, ord=2) > 5:
        #    drone.reset()
        p.resetDebugVisualizerCamera(5, 270+drone.orientation_euler[2]*(180/np.pi), -35, drone.position, client)
        p.stepSimulation(physicsClientId=client)
        time.sleep(1/240)

def test_pid_speed_hor(client):
    pid_h = PID(3, 0.24, 0.625)
    desired_height = 2
    setpoint = 0
    setpoints = []
    real_data = []
    timestep = []
    current_time = 0
    last_time = 0
    keyboard.add_hotkey(',', keyboard_command, args=('yaw', 0.1))
    keyboard.add_hotkey('.', keyboard_command, args=('yaw', -0.1))
    while True:
        setpoints.append(setpoint)
        real_data.append(drone.lateral_speed[0])
        timestep.append(current_time)
        current_time += 1/240

        drone.step_speed(np.random.uniform(-5, 5), 0, pid_h.step(desired_height, drone.position[2], 1/240))
        drone.set_yaw_rate(current_control['yaw'])
        drone.step_angle()
        drone.step()
        print(drone.pid_vx.integral)

        if np.linalg.norm(drone.position, ord=2) > 100:
            drone.reset()

        p.stepSimulation(physicsClientId=client)
        time.sleep(1/240)

    #plt.plot(timestep, setpoints)
    #plt.plot(timestep, real_data)
    #plt.ylim(-0.3,1)
    #plt.grid()
    #plt.show()

if __name__ == '__main__':
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -10, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client)
    drone = Quadcopter(client)
   # p.resetDebugVisualizerCamera(5, 0, -35, drone.position, client)

    #eval_pid_speed_ver(client)
    #eval_pid_speed_hor(client)
    #remote_control()
    #test_pid_roll()
    #test_pid_speed_hor(client)
    #remote_control_nonholonomic(client)
    #test_pid_speed_hor(client)
    #eval_pid_pos_ver()
    eval_pid_yaw(client)
    #eval_pid_roll()
    #eval_pid_yaw(client)
    #remote_control(client)
