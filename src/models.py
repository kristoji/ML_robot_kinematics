import tensorflow as tf

def get_model(noutput, regularize=False, dropout=False, layers=(8,8)) -> tf.keras.models.Sequential:
    first_layer = layers[0]
    second_layer = layers[1]

    if regularize and dropout:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(first_layer, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(second_layer, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(noutput)
        ])
    elif dropout:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(first_layer, activation='relu'),
            tf.keras.layers.Dense(second_layer, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(noutput)
        ])

    elif regularize:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(first_layer, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(second_layer, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(noutput)
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(first_layer, activation='relu'),
            tf.keras.layers.Dense(second_layer, activation='relu'),
            tf.keras.layers.Dense(noutput)
        ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

class NeuralNetwork:
    def __init__(self, dim, njoint, layers: tuple, in_sincos=False, out_orientation=False, regularize=False, dropout=False,
                 optimizer='adam', loss='mean_squared_error', activation='relu', save_model=True):
        self.dim = dim
        self.njoint = njoint
        self.layers = layers
        self.in_sincos = in_sincos
        self.out_orientation = out_orientation
        self.regularize = regularize
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation

        if out_orientation:
            self.noutput = dim + (2 if dim == 2 else 4)
        else:
            self.noutput = dim
        
        end_str = ""
        if dropout or regularize:
            end_str = "_"
            if dropout:
                end_str += "d"
            if regularize:
                end_str += "r"

        if save_model:
            from os import listdir
            self.get_model_filename = f"../Models/{njoint}DOF/model_{njoint}dof_NN-{layers[0]}-{layers[1]}" + end_str + ".keras"
            if self.get_model_filename not in listdir(f"../Models/{njoint}DOF/"):
                print("Already saved model found!")
                self.set_model_filename = self.get_model_filename
                self.get_model_filename = None
        else:
            self.get_model_filename = None
            self.set_model_filename = None
    
    def __get_model__(self):
        model_list = []
        for layer in self.layers:
            if self.regularize:
                model_list.append(tf.keras.layers.Dense(layer, activation=self.activation, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            else:
                model_list.append(tf.keras.layers.Dense(layer, activation=self.activation))
        if self.dropout:
            model_list.append(tf.keras.layers.Dropout(0.2))
        model_list.append(tf.keras.layers.Dense(self.noutput))
        self.model = tf.keras.models.Sequential(model_list)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        return self.model


    def get_trained_model(self, X_train, y_train, verbose=False):

        assert self.noutput == y_train.shape[1]
        if verbose:
            print(f"# input: {X_train.shape[1]}")
            print(f"# output: {y_train.shape[1]}")
            print()

        if self.get_model_filename is not None:
            self.model = tf.keras.models.load_model(self.get_model_filename)
        else:
            self.__get_model__()
            self.model.fit(X_train, y_train, epochs=10, verbose=0)
            if self.set_model_filename is not None:
                self.model.save(self.set_model_filename)

        return self.model
            
    def evaluate(self, X_test, y_test, verbose=2):
        self.model.evaluate(X_test, y_test, verbose=verbose)

        y_pred = self.model.predict(X_test)

        err_pos = tf.linalg.norm(y_pred[:, :self.dim] - y_test[:, :self.dim], axis=1)
        err_ori = None
        if self.out_orientation:
            err_ori = tf.linalg.norm(y_pred[:, self.dim:] - y_test[:, self.dim:], axis=1)

        if verbose:
            print(f"Mean error in position: {tf.reduce_mean(err_pos)}")
            if self.out_orientation:
                print(f"Mean error in orientation: {tf.reduce_mean(err_ori)}")

        return err_pos, err_ori
    
    def plot_error(self, err_pos):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(err_pos)
        plt.savefig("output.png")

    def compare_jacobian(self, model, thetas):
        import jacobian
        print("\n------------------------------")
        print("Jacobian comparison\n")

        diffs = []
        for theta in thetas:
            J = jacobian.FK_Jacobian(model,theta)
            J_true = jacobian.fwd_kin_jacobian_true(theta)
            diff = tf.abs(J-J_true)
            df_sum = tf.reduce_sum(diff) / (self.njoint*2)
            diffs.append(df_sum)

        return diffs
    
    def plot_jac_diff(self, diffs):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(diffs)
        plt.ylim(0, 0.3)
        plt.axhline(y=tf.reduce_mean(diffs), color='r', linestyle='--')
        plt.xlabel("Random theta")
        plt.ylabel("Difference")
        plt.title("Difference between true Jacobian and Jacobian from model")
        plt.savefig(f"../Imgs/Jac_diffs/{self.njoint}DOF/NN_{self.layers[0]}-{self.layers[1]}_s100.png")
        
        print("Mean difference:", np.mean(diffs))
        print("Max difference: ", np.max(diffs))
        print("Min difference: ", np.min(diffs))
