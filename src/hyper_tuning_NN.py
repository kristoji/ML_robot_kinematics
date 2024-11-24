import parse
import models
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import product

def run_experiment(neurons, activation, loss, optimizer, X_train, y_train, X_test, y_test, noutput):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(neurons[0], activation=activation),
        tf.keras.layers.Dense(neurons[1], activation=activation),
        tf.keras.layers.Dense(noutput)
    ])

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, y_train, epochs=10, verbose=0)  # Set verbose to 1 to see training output

    # Evaluate the model
    loss_value = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    err_pos = np.linalg.norm(y_pred[:, :DIM] - y_test[:, :DIM], axis=1)
    mean_err_pos = np.mean(err_pos)

    if ORIENTATION:
        err_ori = np.linalg.norm(y_pred[:, DIM:] - y_test[:, DIM:], axis=1)
        mean_err_ori = np.mean(err_ori)
        mean_total_error = np.mean(np.sqrt(err_pos**2 + err_ori**2))
    else:
        mean_err_ori = None
        mean_total_error = mean_err_pos

    return loss_value, mean_err_pos, mean_err_ori, mean_total_error, model

if __name__ == "__main__":

    DIM = 2
    NJOINT = 3
    ORIENTATION = True
    VALIDATION = False

    # ------------------------------------------------------------------------
    # Load the data

    noutput = DIM + (4 if ORIENTATION and DIM == 3 else 2) if ORIENTATION else DIM

    data, header = parse.parse_data(f"../Dataset/logfile_{DIM}_{NJOINT}.csv")
    X_train, X_test, y_train, y_test = parse.split_data(data, NJOINT, DIM, consider_orientation=ORIENTATION, header=header)

    assert noutput == y_train.shape[1]


    # -----------------------------------------------------------------------
    # Hyperparameter to tune
    neurons_options = [(x, y) for x,y in product([8, 16, 32, 48, 64], repeat=2)]
    activation_options = ['relu', 'tanh']
    loss_options = ['mean_squared_error', 'mean_absolute_error']
    optimizer_options = ['adam', 'sgd', 'rmsprop']


    # -----------------------------------------------------------------------
    # Hyperparameter tuning
    best_config = None
    best_error = float('inf')
    best_model = None

    for neurons, activation, loss, optimizer in product(neurons_options, activation_options, loss_options, optimizer_options):
        loss_value, mean_err_pos, mean_err_ori, mean_total_error, model = run_experiment(
            neurons, activation, loss, optimizer, X_train, y_train, X_test, y_test, noutput
        )

        print(f"Configuration: Neurons={neurons}, Activation={activation}, Loss={loss}, Optimizer={optimizer}")
        print(f"Loss: {loss_value}, Mean Pos Error: {mean_err_pos}, Mean Ori Error: {mean_err_ori}, Total Error: {mean_total_error}")

        if mean_total_error < best_error:
            best_error = mean_total_error
            best_config = (neurons, activation, loss, optimizer)
            best_model = model


    print("\nBest Configuration:")
    print(f"Neurons: {best_config[0]}, Activation: {best_config[1]}, Loss: {best_config[2]}, Optimizer: {best_config[3]}")
    print(f"Best Mean Total Error: {best_error}")

    # -----------------------------------------------------------------------
    # Save the best model
    best_model.save("best_model.keras")