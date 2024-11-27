import parse
import models
import jacobian
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":


    # ------------------------------------------------------------------------
    # Globals

    DIM = 2
    NJOINT = 2
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

    newton = True
    dbg = False

    max_it_reached = 0
    err_avg_max_it = None
    err_avg_break = None

    for _ in range(100):

        curr_theta = np.random.random((NJOINT,)).astype(np.float32) * 2 * np.pi
        curr_theta = tf.Variable(curr_theta)
        goal_pos = np.random.random((DIM,)).astype(np.float32)
        goal_pos = goal_pos / np.linalg.norm(goal_pos) * 0.1*NJOINT * np.random.random()
        goal_pos = tf.Variable(goal_pos)

        if dbg:
            print("Initial theta:", curr_theta.numpy())
            print("Initial position:", jacobian.FK(model, curr_theta).numpy())
            print("Target position:", goal_pos.numpy())


        if newton:

            ##########################################################################
            # Newton-Raphson Method
            # x_{n+1} = x_n - J^-1 * f(x_n)
            ##########################################################################

            for i in range(100):
                J = jacobian.FK_Jacobian(model, curr_theta)
                J_inv = tf.linalg.pinv(J)
                err = jacobian.FK(model, curr_theta) - goal_pos  

                # From (DIM,) to (DIM, 1) and then back to (DIM,)
                delta_theta = -tf.matmul(J_inv, tf.reshape(err, (-1, 1)))
                delta_theta = tf.reshape(delta_theta, (-1,))

                curr_theta.assign_add(delta_theta)
                curr_theta.assign(tf.math.floormod(curr_theta, 2*np.pi))

                if tf.reduce_sum(tf.abs(err)) < 1e-3:
                    if dbg:
                        print(f"Break at iteration {i}")
                    err = tf.reduce_sum(tf.abs(err))
                    break_reached = i + 1 - max_it_reached
                    err_avg_break = err if err_avg_break is None else (break_reached*err_avg_break + err) / (break_reached+1)
                    break
            else:
                if dbg:
                    print("Max num iteration reached")
                err = tf.reduce_sum(tf.abs(err))
                err_avg_max_it = err if err_avg_max_it is None else (max_it_reached*err_avg_max_it + err) / (max_it_reached+1)
                max_it_reached += 1

        else:

            ##########################################################################
            # Levenberg-Marquardt
            # x_{n+1} = x_n - (J^T * J + lambda * I)^-1 * J^T * f(x_n)
            ##########################################################################

            for i in range(500):
                J = jacobian.FK_Jacobian(model, curr_theta)
                J_T = tf.transpose(J)
                err = jacobian.FK(model, curr_theta) - goal_pos  

                lambda_ = 0.1
                delta_theta = -tf.matmul(tf.linalg.inv(tf.matmul(J_T, J) + lambda_ * tf.eye(NJOINT)), tf.matmul(J_T, tf.reshape(err, (-1, 1))))
                delta_theta = tf.reshape(delta_theta, (-1,))

                curr_theta.assign_add(delta_theta)
                curr_theta.assign(tf.math.floormod(curr_theta, 2*np.pi))

                if tf.reduce_sum(tf.abs(err)) < 1e-3:
                    if dbg:
                        print(f"Break at iteration {i}")
                    err = tf.reduce_sum(tf.abs(err))
                    break_reached = i + 1 - max_it_reached
                    err_avg_break = err if err_avg_break is None else (break_reached*err_avg_break + err) / (break_reached+1)
                    break
            else:
                if dbg:
                    print("Max num iteration reached")
                err = tf.reduce_sum(tf.abs(err))
                err_avg_max_it = err if err_avg_max_it is None else (max_it_reached*err_avg_max_it + err) / (max_it_reached+1)
                max_it_reached += 1
                

        if dbg:
            print()
            print("Final theta:", curr_theta.numpy())
            print("Final position:", jacobian.FK(model, curr_theta).numpy())
            print("True position:", jacobian.fwd_kin_true(curr_theta).numpy())
            print("Target:", goal_pos.numpy())

    print(f"Max iteration reached: {max_it_reached} / 100")
    print("Average error at max iteration:", err_avg_max_it.numpy())
    print("Average error at break:", err_avg_break.numpy())

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

    # 500 iterations, Levenberg-Marquardt
    # Max iteration reached: 33 / 100
    # Average error at max iteration: 0.118702054
    # Average error at break: 0.00098943


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