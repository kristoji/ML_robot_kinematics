import sys
import time
import jacobian
import numpy as np
from envs.reacher_v6 import ReacherEnv
from envs.reacher3_v6 import Reacher3Env
from envs.marrtino_arm import MARRtinoArmEnv

NJOINT = 2

curr_theta = np.random.random((NJOINT,)).astype(np.float32) * 2 * np.pi - np.pi
final_theta = np.random.random((NJOINT,)).astype(np.float32) * 2 * np.pi - np.pi

goal_pos = jacobian.fwd_kin_true(final_theta)

with open("envs.bak/assets/reacher.xml", "r") as f:
    data = f.read()
    data = data.replace('<geom type="sphere" name="goal_pos" pos="0.05 -0.05 0.01" size="0.015" rgba="1 0 0 1"/>', f'<geom type="sphere" pos="{goal_pos[0]} {goal_pos[1]} 0.01" size="0.015" rgba="1 0 0 1"/>')
with open("envs/assets/reacher.xml", "w") as f:
    f.write(data)

print("Goal position:", np.round(goal_pos[0], 3), np.round(goal_pos[1], 3))


if NJOINT == 2:
    env = ReacherEnv(render_mode="human")
elif NJOINT == 3:
    env = Reacher3Env(render_mode="human")
elif NJOINT == 5:
    env = MARRtinoArmEnv(render_mode="human")
else:
    print(f"Unknown environment {NJOINT}")
    sys.exit(1)


seed = 1234
observation, info = env.reset(seed=seed)
env.action_space.seed(seed=seed)


int_err = np.zeros(NJOINT)
prev_err = np.zeros(NJOINT)
Kp = 0.1
Ki = 0.0007
Kd = 0.17
dt = 0.1

for i in range(1, 100):
    err = final_theta - curr_theta 
    ref_theta = final_theta
    
    int_err += err * dt
    d_err = (err - prev_err) / dt
    action = Kp * err + Ki * int_err + Kd * d_err
    
    action = np.clip(action, -1, 1)
    observation, reward, terminated, truncated, info = env.step(action)
    curr_theta = observation[:NJOINT]
    # print(f"curr: {curr_theta}, goal: {final_theta}, err: {err}, act: {action}")
    if terminated or truncated:
        observation, info = env.reset()

    prev_err = err
    time.sleep(dt)

env.close()