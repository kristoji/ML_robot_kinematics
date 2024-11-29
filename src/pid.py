import sys
import time
import jacobian
import numpy as np
from envs.reacher_v6 import ReacherEnv
from envs.reacher3_v6 import Reacher3Env
from envs.marrtino_arm import MARRtinoArmEnv


def get_env(NJOINT, init_theta, goal_pos, seed=1234):

    with open("envs.bak/assets/reacher.xml", "r") as f:
        data = f.read()
        data = data.replace('<geom type="sphere" name="goal_pos" pos="0.05 -0.05 0.01" size="0.015" rgba="1 0 0 1"/>', f'<geom type="sphere" pos="{goal_pos[0]} {goal_pos[1]} 0.01" size="0.015" rgba="1 0 0 1"/>')
    with open("envs/assets/reacher.xml", "w") as f:
        f.write(data)

    if NJOINT == 2:
        env = ReacherEnv(render_mode="human")
        env.set_state(np.reshape(np.concatenate((init_theta, np.zeros(2))), (4,)), np.zeros((4,)))
    elif NJOINT == 3:
        env = Reacher3Env(render_mode="human")
    elif NJOINT == 5:
        env = MARRtinoArmEnv(render_mode="human")
    else:
        print(f"Unknown environment {NJOINT}")
        sys.exit(1)

    # env.reset(seed=seed)
    env.action_space.seed(seed=seed)
    env.step([0, 0])

    return env


class PID_Controller:
    def __init__(self, NJOINT, Kp=0.1, Ki=0.0007, Kd=0.17, dt=0.1):
        self.NJOINT = NJOINT
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.int_err = np.zeros(NJOINT)
        self.prev_err = np.zeros(NJOINT)

    def step(self, curr_theta, final_theta):
        err = final_theta - curr_theta
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
    NJOINT = 2

    curr_theta = np.random.random((NJOINT,)).astype(np.float32) * 2 * np.pi - np.pi
    final_theta = np.random.random((NJOINT,)).astype(np.float32) * 2 * np.pi - np.pi

    goal_pos = jacobian.fwd_kin_true(final_theta)

    print(f"Initial theta: {curr_theta}")
    print(f"Goal theta: {final_theta}")
    print(f"Goal pos: {goal_pos}")

    env = get_env(NJOINT, curr_theta, goal_pos)
    
    input()
    pid_ctrl = PID_Controller(NJOINT)


    # curr_theta = in_theta
    for _ in range(100):
        action = pid_ctrl.step(curr_theta, final_theta)

        observation, reward, terminated, truncated, info = env.step(action)
        curr_theta = observation[:NJOINT]

        # print(f"curr: {curr_theta}, goal: {final_theta}, err: {err}, act: {action}")

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(pid_ctrl.dt)

    env.close()
    print(f"Final theta: {curr_theta}")
