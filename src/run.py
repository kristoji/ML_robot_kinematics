import pid
import time
import parse
import models
import jacobian
import inverse_kin
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# TF_CPP_MIN_LOG_LEVEL=2 python run.py
if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # Globals

    DIM = 2
    NJOINT = 3
    IN_SINCOS = False
    OUT_ORIENTATION = False

    NN = (16,16)
    VALIDATION = False
    if VALIDATION:
        SET_MODEL_FILENAME = None
        GET_MODEL_FILENAME = f"../Models/{NJOINT}DOF/model_{NJOINT}dof_NN-{NN[0]}-{NN[1]}.keras"
    else:
        SET_MODEL_FILENAME = f"../Models/{NJOINT}DOF/model_{NJOINT}dof_NN-{NN[0]}-{NN[1]}.keras"
        GET_MODEL_FILENAME = None
        GET_MODEL_FILENAME = f"../Models/{NJOINT}DOF/model_{NJOINT}dof_NN-{NN[0]}-{NN[1]}.keras"


    # ------------------------------------------------------------------------
    # Load the data and split it into train and test

    if VALIDATION:
        data, header = parse.parse_data(f"../Dataset/logfile_{DIM}_{NJOINT}_val.csv")
    else:
        data, header = parse.parse_data(f"../Dataset/logfile_{DIM}_{NJOINT}.csv")

    if OUT_ORIENTATION:
        # dim + quaternions
        noutput = DIM + (2 if DIM == 2 else 4)
    else:
        noutput = DIM


    X_train, X_test, y_train, y_test = parse.split_data(data, 
                                                        njoint=NJOINT, 
                                                        dimensions=DIM,
                                                        consider_orientation=OUT_ORIENTATION,
                                                        consider_sincos=IN_SINCOS,
                                                        header=header)
    
    if VALIDATION:
        X_test = np.concatenate((X_train, X_test), axis=0)
        y_test = np.concatenate((y_train, y_test), axis=0)


    assert noutput == y_train.shape[1]
    print(f"Number of input: {X_train.shape[1]}")
    print(f"Number of output: {noutput}")

    # ------------------------------------------------------------------------
    # Train the model
    
    if GET_MODEL_FILENAME is not None:
        model = tf.keras.models.load_model(GET_MODEL_FILENAME)
    else:
        model = models.get_model(noutput, regularize=False, dropout=False)

        # print(model.summary())
        model.fit(X_train, y_train, epochs=10, verbose=0)

        if SET_MODEL_FILENAME is not None:
            model.save(SET_MODEL_FILENAME)


    # ------------------------------------------------------------------------
    # Evaluate the model

    model.evaluate(X_test, y_test, verbose=2)

    y_pred = model.predict(X_test)

    err_pos = np.linalg.norm(y_pred[:, :DIM] - y_test[:, :DIM], axis=1)

    if OUT_ORIENTATION:
        err_ori = np.linalg.norm(y_pred[:, DIM:] - y_test[:, DIM:], axis=1)
        print(f"Mean error in position: {np.mean(err_pos)}")
        print(f"Mean error in orientation: {np.mean(err_ori)}")

        distances = np.sqrt(err_pos**2 + err_ori**2)
        print(f"Mean error: {np.mean(distances)}")
    else:
        print(f"Mean error in position: {np.mean(err_pos)}")
        

    # ------------------------------------------------------------------------
    # Plot the error

    plt.figure()
    plt.plot(err_pos)
    plt.savefig("output.png")


    # ------------------------------------------------------------------------
    # Compare the Jacobian with the true Jacobian

    thetas = np.random.random((100, NJOINT)).astype(np.float32) * 2 * np.pi
    diffs = []
    for theta in thetas:
        J = jacobian.FK_Jacobian(model,theta)
        J_true = jacobian.fwd_kin_jacobian_true(theta)
        diff = tf.abs(J-J_true)
        df_sum = tf.reduce_sum(diff) / (NJOINT*2)
        diffs.append(df_sum)


    # ------------------------------------------------------------------------
    # Plot the difference

    plt.figure()
    plt.plot(diffs)
    plt.ylim(0, 0.3)
    plt.axhline(y=np.mean(diffs), color='r', linestyle='--')
    plt.xlabel("Random theta")
    plt.ylabel("Difference")
    plt.title("Difference between true Jacobian and Jacobian from model")
    plt.savefig(f"../Imgs/Jac_diffs/{NJOINT}DOF/NN_{NN[0]}-{NN[1]}_s100.png")

    print("\nJACOBIAN DIFFERENCE")
    print("Mean difference:", np.mean(diffs))
    print("Max difference:", np.max(diffs))
    print("Min difference:", np.min(diffs))
    print()

    # ------------------------------------------------------------------------
    # Inverse Kinematics

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
            in_theta = tf.cast(in_theta, tf.float64)
            goal_pos = tf.cast(goal_pos, tf.float64)

            final_theta, err, max_it_reached = inverse_kin.inverse_kinematic(model, in_theta, goal_pos,
                                                                            newton=NEWTON, num_it=NUM_IT)

            if max_it_reached == 1:
                err_avg_max_it = (tot_max_it_reached*err_avg_max_it + err) / (tot_max_it_reached+1)
                tot_max_it_reached += 1
            else:
                break_reached = i - tot_max_it_reached
                err_avg_break = (break_reached*err_avg_break + err) / (break_reached+1)


        print(f"Max iteration reached: {tot_max_it_reached} / 100")
        print("Average error at max iteration:", err_avg_max_it)
        print("Average error at break:", err_avg_break)

        # 100 iterations, Newton-Raphson
        # Max iteration reached: 64 / 100
        # Average error at max iteration: 0.318262
        # Average error at break: -4.885498e-07

        # 500 iterations, Newton-Raphson
        # Max iteration reached: 70 / 100
        # Average error at max iteration: 0.31624046
        # Average error at break: -1.8805721e-07

        # 100 iterations, Levenberg-Marquardt
        # Max iteration reached: 82 / 100 
        # Average error at max iteration: 0.06085123
        # Average error at break: 0.00099721

        # 500 iterations, Levenberg-Marquardt
        # Max iteration reached: 33 / 100
        # Average error at max iteration: 0.118702054
        # Average error at break: 0.00098943
    
    else:
        in_theta = pid.get_rnd_theta(NJOINT)
        goal_pos = pid.get_rnd_pos_in_workspace(NJOINT)
        in_theta = tf.cast(in_theta, tf.float64)
        goal_pos = tf.cast(goal_pos, tf.float64)
        goal_pos = goal_pos / np.linalg.norm(goal_pos) * 0.1*NJOINT * np.random.random()

        final_theta, err, max_it_reached = inverse_kin.inverse_kinematic(model, in_theta, goal_pos,
                                                                        newton=NEWTON, num_it=NUM_IT, dbg=True)


    # ------------------------------------------------------------------------
    # PID Controller
    
    # TODO: prova a cambiare la FK e Jacobiana per tenere conto dei boundaries
    minmax = [3.14, 3] if NJOINT == 2 else [3.14, 1.8, 1.8]
    if any([y < x < np.pi or y < -x < np.pi for x, y in zip(final_theta, minmax)]):
        # this happens due to redundancy of the robot when not considering the orientation
        print("Position out of workspace")
        exit()
    final_theta = tf.where(final_theta > minmax, final_theta - 2*np.pi, final_theta)
    
    env = pid.get_env(NJOINT, in_theta, goal_pos, model=model, final_theta=final_theta)
    pid_ctrl = pid.PID_Controller(NJOINT, final_theta)

    curr_theta = in_theta

    print()
    print("------------------------------")
    print("PID Controller")
    print(f"[THT] Start:\t{curr_theta}")
    print(f"[THT] Goal:\t{final_theta}")
    print(f"[POS] Wanted: {goal_pos}")
    
    # input()

    for _ in range(100):
        action = pid_ctrl.step(curr_theta)

        observation, reward, terminated, truncated, info = env.step(action)
        curr_theta = observation[:NJOINT]

        # print(f"curr: {curr_theta}, goal: {final_theta}, err: {err}, act: {action}")

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(pid_ctrl.dt)

    env.close()
    print("----")
    print(f"[THT] GOT: {curr_theta}")
    print(f"[POS] GOT: {jacobian.FK(model, curr_theta)}")


# 2 OUTPUT
# without anything:    0.0015946422756504893
# with regularization: 0.011633878173556093
# with dropout:        0.005296317128044236


# 3 OUTPUT
# -> same dataset
# Mean error in position: 0.004059377363403357
# Mean error in orientation: 0.0088411142930527
# Mean error: 0.010096040302040745

# -> different dataset
# Mean error in position: 0.0040647649606516826
# Mean error in orientation: 0.008697662813949849
# Mean error: 0.009969588794127567


# 5 OUTPUT
# -> same dataset
# Mean error in position: 0.013782319353915714
# Mean error in orientation: 0.020014377072758775
# Mean error: 0.02501448404778734

# -> different dataset
# Mean error in position: 0.014029994865092094
# Mean error in orientation: 0.02028842137822963
# Mean error: 0.02538335946761391



# TODO: check this, in which the 3 dots where far away from each other

# ------------------------------
# Goal theta true: [-0.34784153 -0.68790793]
# Goal pos: [ 0.1449992  -0.12011141]
# Initial theta: [0.51671116 1.14708817]
# Initial position: [0.05370678 0.14796942]
# Target position: [ 0.14033133 -0.11624474]
# Break at iteration 85

# Final theta: [4.14391478 3.88674853]
# Final position: [ 0.13993664 -0.11683822]
# True position: [-0.07141107  0.0141709 ]
# Target: [ 0.14033133 -0.11624474]

# Initial theta: [0.51671116 1.14708817]
# Goal theta: [-2.13927053 -2.39643678]
# Goal pos: [ 0.14033133 -0.11624474]
# Final theta: [-2.17000402 -2.43978244]