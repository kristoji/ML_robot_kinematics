import sys
import time
import jacobian
import numpy as np
import tensorflow as tf
from envs.reacher_v6 import ReacherEnv
from envs.reacher3_v6 import Reacher3Env
from envs.marrtino_arm import MARRtinoArmEnv


def get_env(NJOINT, init_theta, goal_pos, model=None, final_theta=None, seed=1234):

    if final_theta is not None:
        reachable_pos = jacobian.fwd_kin_true(final_theta)
        wanted_pos = jacobian.FK(model, final_theta)

    file_xml = "reacher.xml" if NJOINT == 2 else "reacher3.xml" if NJOINT == 3 else "wrong_njoint"

    with open(f"envs.bak/assets/{file_xml}", "r") as f:
        data = f.read()
        if final_theta is not None:
            data = data.replace( '<geom type="sphere" name="goal_pos" pos="0.05 -0.05 0.01" size="0.015" rgba="1 0 0 1"/>', 
                                f'<geom type="sphere" pos="{goal_pos[0]} {goal_pos[1]} 0.01" size="0.015" rgba="1 0 0 1"/>\n		<geom type="sphere" pos="{reachable_pos[0]} {reachable_pos[1]} 0.01" size="0.015" rgba="0 1 0 1"/>\n		<geom type="sphere" pos="{wanted_pos[0]} {wanted_pos[1]} 0.01" size="0.015" rgba="0 0 1 1"/>')
        else:
            data = data.replace( '<geom type="sphere" name="goal_pos" pos="0.05 -0.05 0.01" size="0.015" rgba="1 0 0 1"/>', 
                                f'<geom type="sphere" pos="{goal_pos[0]} {goal_pos[1]} 0.01" size="0.015" rgba="1 0 0 1"/>')
    with open(f"envs/assets/{file_xml}", "w") as f:
        f.write(data)

    if NJOINT == 2:
        env = ReacherEnv(render_mode="human")
        env.set_state(np.reshape(np.concatenate((init_theta, np.zeros(2))), (4,)), np.zeros((4,)))
    elif NJOINT == 3:
        env = Reacher3Env(render_mode="human")
        env.set_state(np.reshape(np.concatenate((init_theta, np.zeros(2))), (5,)), np.zeros((5,)))
    # elif NJOINT == 5:
    #     env = MARRtinoArmEnv(render_mode="human")
    else:
        print(f"[E] {NJOINT}DOF Robot not implemented yet")
        sys.exit(1)

    # env.reset(seed=seed)
    env.action_space.seed(seed=seed)
    env.step([0]*NJOINT)

    return env

def get_rnd_theta(NJOINT):
    if NJOINT != 2 and NJOINT != 3:
        print(f"[E] {NJOINT}DOF Robot not implemented yet")
        sys.exit(1)
    bounds = [3.14, 3] if NJOINT == 2 else [3.14, 1.8, 1.8]
    rnd = np.random.random((NJOINT,)).astype(np.float32)
    
    theta = np.array([rnd[i] * 2 * bounds[i] - bounds[i] for i in range(NJOINT)])
    theta = tf.cast(theta, tf.float64)
    return theta

def get_rnd_pos_in_workspace(NJOINT, verbose=False):
    if NJOINT != 2 and NJOINT != 3:
        print(f"[E] {NJOINT}DOF Robot not implemented yet")
        sys.exit(1)

    theta = get_rnd_theta(NJOINT)

    pos = jacobian.fwd_kin_true(theta)
    pos = tf.cast(pos, tf.float64)
    
    if verbose:
        print(f"[THT] True Goal: {theta}\n")
        # print(f"Goal pos: {jacobian.fwd_kin_true(theta)}")
    return pos


class PID_Controller:
    def __init__(self, NJOINT, final_theta, Kp=0.1, Ki=0.0007, Kd=0.17, dt=0.1):
        self.NJOINT = NJOINT
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.int_err = np.zeros(NJOINT)
        self.prev_err = np.zeros(NJOINT)
        self.final_theta = final_theta

    def step(self, curr_theta):
        err = self.final_theta - curr_theta
        self.int_err += err * self.dt
        d_err = (err - self.prev_err) / self.dt
        action = self.Kp * err + self.Ki * self.int_err + self.Kd * d_err
        action = np.clip(action, -1, 1)
        self.prev_err = err
        return action

    def reset(self):
        self.int_err = np.zeros(self.NJOINT)
        self.prev_err = np.zeros(self.NJOINT)
    
    def set_params(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt


if __name__ == "__main__":
    NJOINT = 3

    curr_theta = get_rnd_theta(NJOINT)
    final_theta = get_rnd_theta(NJOINT)

    goal_pos = jacobian.fwd_kin_true(final_theta)

    print(f"Initial theta: {curr_theta}")
    print(f"Goal theta: {final_theta}")
    print(f"Goal pos: {goal_pos}")

    env = get_env(NJOINT, curr_theta, goal_pos)
    
    # input()
    pid_ctrl = PID_Controller(NJOINT, final_theta)


    # curr_theta = in_theta
    for _ in range(100):
        action = pid_ctrl.step(curr_theta)

        observation, reward, terminated, truncated, info = env.step(action)
        curr_theta = observation[:NJOINT]

        # print(f"curr: {curr_theta}, goal: {final_theta}, act: {action}")

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(pid_ctrl.dt)

    env.close()
    print(f"Final theta: {curr_theta}")
