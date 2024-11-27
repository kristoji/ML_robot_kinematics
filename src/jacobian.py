import tensorflow as tf


def fwd_kin_true(theta):
    L = 0.1
    x = L*tf.cos(theta[0]) + L*tf.cos(theta[0] + theta[1])
    y = L*tf.sin(theta[0]) + L*tf.sin(theta[0] + theta[1])
    if theta.shape[0] == 3:
      x += L*tf.cos(theta[0] + theta[1] + theta[2])
      y += L*tf.sin(theta[0] + theta[1] + theta[2])
    return tf.stack([x, y])

def fwd_kin_jacobian_true(theta):
    L = 0.1
    if theta.shape[0] == 2:
      J11 = -L*tf.sin(theta[0]) - L*tf.sin(theta[0] + theta[1])
      J12 = -L*tf.sin(theta[0] + theta[1])
      J21 = L*tf.cos(theta[0]) + L*tf.cos(theta[0] + theta[1])
      J22 = L*tf.cos(theta[0] + theta[1])
      return tf.stack([[J11, J12], [J21, J22]])
    elif theta.shape[0] == 3:
      J11 = -L*tf.sin(theta[0]) - L*tf.sin(theta[0] + theta[1]) - L*tf.sin(theta[0] + theta[1] + theta[2])
      J12 = -L*tf.sin(theta[0] + theta[1]) - L*tf.sin(theta[0] + theta[1] + theta[2])
      J13 = -L*tf.sin(theta[0] + theta[1] + theta[2])
      J21 = L*tf.cos(theta[0]) + L*tf.cos(theta[0] + theta[1]) + L*tf.cos(theta[0] + theta[1] + theta[2])
      J22 = L*tf.cos(theta[0] + theta[1]) + L*tf.cos(theta[0] + theta[1] + theta[2])
      J23 = L*tf.cos(theta[0] + theta[1] + theta[2])
      return tf.stack([[J11, J12, J13], [J21, J22, J23]])

def FK(model,theta):
    njoints = theta.shape[0]
    # reshape to batch size 1
    t = tf.reshape(theta, shape=(1,njoints))
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

