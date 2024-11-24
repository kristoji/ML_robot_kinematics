import tensorflow as tf


def fwd_kin_true(theta):
    L1 = 0.1
    L2 = 0.1
    x = L1*tf.cos(theta[0]) + L2*tf.cos(theta[0] + theta[1])
    y = L1*tf.sin(theta[0]) + L2*tf.sin(theta[0] + theta[1])
    return tf.stack([x, y])

def fwd_kin_jacobian_true(theta):
    L1 = 0.1
    L2 = 0.1
    J11 = -L1*tf.sin(theta[0]) - L2*tf.sin(theta[0] + theta[1])
    J12 = -L2*tf.sin(theta[0] + theta[1])
    J21 = L1*tf.cos(theta[0]) + L2*tf.cos(theta[0] + theta[1])
    J22 = L2*tf.cos(theta[0] + theta[1])
    return tf.stack([[J11, J12], [J21, J22]])

def FK(model,theta):
    # reshape to batch size 1
    t = tf.reshape(theta, shape=(1,2))
    out = model(t)
    # reshape to 1d vector
    out = tf.reshape(out, shape=(2,))
    return out


@tf.function
def FK_Jacobian(model,x):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = FK(model,x)
  return tape.jacobian(y, x)

