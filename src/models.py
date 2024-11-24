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
