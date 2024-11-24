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

    NN = (32,16)
    VALIDATION = False
    if VALIDATION:
        SET_MODEL_FILENAME = None
        GET_MODEL_FILENAME = f"../Models/model_{DIM}_{NJOINT}.keras"
    else:
        SET_MODEL_FILENAME = f"../Models/model_{DIM}_{NJOINT}.keras"
        GET_MODEL_FILENAME = None


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

    thetas = np.random.random((100,2))
    diffs = []
    for theta in thetas:
        J = jacobian.FK_Jacobian(model,theta)
        J_true = jacobian.fwd_kin_jacobian_true(theta)
        diff = tf.abs(J-J_true)
        df_sum = tf.reduce_sum(diff) / 4
        # if df_sum > 100:
        #     print("Theta:", theta)
        #     print("Jacobian from model:")
        #     print(J)
        #     print("True Jacobian:")
        #     print(J_true)
        #     print("Difference:")
        #     print(diff)
        diffs.append(df_sum)


    # ------------------------------------------------------------------------
    # Plot the difference

    plt.figure()
    plt.plot(diffs)
    plt.ylim(0, 0.6)
    # mean diffs as red horizontal line
    plt.axhline(y=np.mean(diffs), color='r', linestyle='--')
    plt.xlabel("Random theta")
    plt.ylabel("Difference")
    plt.title("Difference between true Jacobian and Jacobian from model")
    plt.savefig(f"../Imgs/Jac_diffs/NN_{NN[0]}-{NN[1]}_s100.png")

    print("\nJACOBIAN DIFFERENCE")
    print("Mean difference:", np.mean(diffs))
    print("Max difference:", np.max(diffs))
    print("Min difference:", np.min(diffs))


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