import pid
import time
import parse
import models
import jacobian
import inverse_kin
import numpy as np
import tensorflow as tf

# TF_CPP_MIN_LOG_LEVEL=2 python main.py
if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # Globals

    DIM = 2
    NJOINT = 3
    IN_SINCOS = False
    OUT_ORIENTATION = False

    NN = (12,12)
    VALIDATION = False

    # ------------------------------------------------------------------------
    # Load the data and split it into train and test

    if VALIDATION:
        data, header = parse.parse_data(f"../Dataset/logfile_{DIM}_{NJOINT}_val.csv")
    else:
        data, header = parse.parse_data(f"../Dataset/logfile_{DIM}_{NJOINT}.csv")


    X_train, X_test, y_train, y_test = parse.split_data(data, 
                                                        njoint=NJOINT, 
                                                        dimensions=DIM,
                                                        consider_orientation=OUT_ORIENTATION,
                                                        consider_sincos=IN_SINCOS,
                                                        header=header)
    
    if VALIDATION:
        X_test = np.concatenate((X_train, X_test), axis=0)
        y_test = np.concatenate((y_train, y_test), axis=0)



    # ------------------------------------------------------------------------
    # Train the model
    
    nn = models.NeuralNetwork(dim=DIM, 
                              njoint=NJOINT, 
                              layers=NN, 
                              in_sincos=IN_SINCOS, 
                              out_orientation=OUT_ORIENTATION)
    
    model = nn.get_trained_model(X_train, y_train)



    # ------------------------------------------------------------------------
    # Evaluate the model and Plot the error

    err_pos, err_ori = nn.evaluate(X_test, y_test, verbose=2)

    nn.plot_error(err_pos)


    # ------------------------------------------------------------------------
    # Compare the Jacobian with the true Jacobian and Plot the difference

    thetas = np.random.random((100, NJOINT)).astype(np.float32) * 2 * np.pi
    
    diffs = nn.compare_jacobian(model, thetas)
    nn.plot_jac_diff(diffs)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Inverse Kinematics

    print("\n------------------------------")
    print("Inverse Kinematics\n")

    NEWTON = False
    NUM_IT = 500
    STRESS_TEST = False

    if STRESS_TEST:

        err_avg_break = 0
        err_avg_max_it = 0
        tot_max_it_reached = 0

        for i in range(100):

            in_theta = pid.get_rnd_theta(NJOINT)
            goal_pos = pid.get_rnd_pos_in_workspace(NJOINT)

            final_theta, err, max_it_reached = inverse_kin.inverse_kinematic(model, in_theta, goal_pos,
                                                                            newton=NEWTON, num_it=NUM_IT)

            if max_it_reached == 1:
                err_avg_max_it = (tot_max_it_reached*err_avg_max_it + err) / (tot_max_it_reached+1)
                tot_max_it_reached += 1
            else:
                break_reached = i - tot_max_it_reached
                err_avg_break = (break_reached*err_avg_break + err) / (break_reached+1)

            print(".", end="", flush=True)

        print(f"\n{NUM_IT} iterations,", "Newton-Raphson" if NEWTON else "Levenberg-Marquardt")
        print(f"Max iteration reached: {tot_max_it_reached} / 100")
        print("Average error at max iteration:", err_avg_max_it)
        print("Average error at break:", err_avg_break)
        exit()
    
    else:
        in_theta = pid.get_rnd_theta(NJOINT)
        goal_pos = pid.get_rnd_pos_in_workspace(NJOINT, verbose=True)

        final_theta, err, max_it_reached = inverse_kin.inverse_kinematic(model, in_theta, goal_pos,
                                                                        newton=NEWTON, num_it=NUM_IT, dbg=True)

    # TODO: prova a cambiare la FK e Jacobiana per tenere conto dei boundaries
    bounds = [3.14, 3] if NJOINT == 2 else [3.14, 1.85, 1.85]
    if any([y < x < np.pi or y < -x < np.pi for x, y in zip(final_theta, bounds)]):
        # this happens due to redundancy of the robot when not considering the orientation
        print("Position out of workspace")
        exit()
    final_theta = tf.where(final_theta > bounds, final_theta - 2*np.pi, final_theta)


    # ------------------------------------------------------------------------
    # PID Controller    
    
    print("\n------------------------------")
    print("PID Controller\n")
    print(f"[THT] Start:\t{in_theta}")
    print(f"[THT] Goal:\t{final_theta}")
    print(f"[POS] Wanted:\t{goal_pos}")

    env = pid.get_env(NJOINT, in_theta, goal_pos, model=model, final_theta=final_theta)
    pid_ctrl = pid.PID_Controller(NJOINT, final_theta)

    curr_theta = in_theta

    
    for _ in range(100):
        action = pid_ctrl.step(curr_theta)

        observation, reward, terminated, truncated, info = env.step(action)
        curr_theta = observation[:NJOINT]

        # print(f"curr: {curr_theta}, goal: {final_theta}, err: {err}, act: {action}")

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(pid_ctrl.dt)

    env.close()

    print()
    print(f"[THT] GOT:\t{curr_theta}")
    print(f"[POS] GOT:\t{jacobian.FK(model, curr_theta)}")