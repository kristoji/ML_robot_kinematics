import parse
import models
import numpy as np
from itertools import product

def run_experiment(neurons, activation, loss, optimizer, dropout, regularize, X_train, y_train, X_test, y_test):

    global DIM, NJOINT
    
    nn = models.NeuralNetwork(DIM, NJOINT, neurons, regularize=regularize, dropout=dropout, optimizer=optimizer, activation=activation, loss=loss, save_model=False)
    model = nn.get_trained_model(X_train, y_train)
    err_pos, err_ori = nn.evaluate(X_test, y_test, verbose=0)

    mean_err_pos = np.mean(err_pos)

    if err_ori is not None:
        mean_err_ori = np.mean(err_ori)
        mean_total_error = np.mean(np.sqrt(err_pos**2 + err_ori**2))
    else:
        mean_err_ori = None
        mean_total_error = mean_err_pos

    return mean_err_pos, mean_err_ori, mean_total_error, model

if __name__ == "__main__":

    DIM = 2
    NJOINT = 3
    VALIDATION = False
    IN_SINCOS = False
    OUT_ORIENTATION = False

    # ------------------------------------------------------------------------
    # Load the data

    noutput = (DIM + (4 if OUT_ORIENTATION and DIM == 3 else 2)) if OUT_ORIENTATION else DIM

    data, header = parse.parse_data(f"../Dataset/logfile_{DIM}_{NJOINT}.csv")
    X_train, X_test, y_train, y_test = parse.split_data(data, NJOINT, DIM, consider_sincos=IN_SINCOS, consider_orientation=OUT_ORIENTATION, header=header)

    assert noutput == y_train.shape[1]


    # -----------------------------------------------------------------------
    # Hyperparameter to tune
    neurons_options = [(12,12), (16,16), (32,16), (32,32), (48,48)]
    activation_options = ['relu', 'tanh']
    loss_options = ['mean_squared_error', 'mean_absolute_error']
    optimizer_options = ['adam', 'sgd']
    dropout_options = [True, False]
    regularize_options = [True, False]


    # -----------------------------------------------------------------------
    # Hyperparameter tuning
    best_config = None
    best_error = float('inf')
    best_model = None

    for neurons, activation, loss, optimizer, dropout, regularize in product(
        neurons_options, activation_options, loss_options, optimizer_options, dropout_options, regularize_options):
        
        mean_err_pos, mean_err_ori, mean_total_error, model = run_experiment(
            neurons, activation, loss, optimizer, dropout, regularize, X_train, y_train, X_test, y_test
        )

        print(f"Configuration: Neurons={neurons}, Activation={activation}, Loss={loss}, Optimizer={optimizer}, Dropout={dropout}, Regularize={regularize}")
        print(f"Mean Pos Error: {mean_err_pos}, Mean Ori Error: {mean_err_ori}, Total Error: {mean_total_error}")

        if mean_total_error < best_error:
            best_error = mean_total_error
            best_config = (neurons, activation, loss, optimizer, dropout, regularize)
            best_model = model


    print("\nBest Configuration:")
    print(f"Neurons: {best_config[0]}, Activation: {best_config[1]}, Loss: {best_config[2]}, Optimizer: {best_config[3]}, Dropout: {best_config[4]}, Regularize: {best_config[5]}")
    print(f"Best Mean Total Error: {best_error}")

    # -----------------------------------------------------------------------
    # Save the best model
    end_str = ""
    if best_config[4] or best_config[5]:
        end_str = "_"
        if best_config[4]:
            end_str += "d"
        if best_config[5]:
            end_str += "r"

    best_model.save(f"../Models/{NJOINT}DOF/TUNED_model_{NJOINT}dof_NN-{best_config[0][0]}-{best_config[0][1]}" + end_str + ".keras")