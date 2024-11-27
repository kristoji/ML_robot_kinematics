import jacobian
import numpy as np
import tensorflow as tf

def inverse_kinematic(model, in_theta, goal_pos, num_it=500, lambda_=0.1, dbg=False, newton=False):
    # curr_theta = np.random.random((NJOINT,)).astype(np.float32) * 2 * np.pi
    # goal_pos = np.random.random((DIM,)).astype(np.float32)
    # goal_pos = goal_pos / np.linalg.norm(goal_pos) * 0.1*NJOINT * np.random.random()

    curr_theta = tf.Variable(in_theta)
    goal_pos = tf.Variable(goal_pos)
    max_it_reached = -1

    njoint = in_theta.shape[0]

    if dbg:
        print("Initial theta:", curr_theta.numpy())
        print("Initial position:", jacobian.FK(model, curr_theta).numpy())
        print("Target position:", goal_pos.numpy())


    if newton:

        ##########################################################################
        # Newton-Raphson Method
        # x_{n+1} = x_n - J^-1 * f(x_n)
        ##########################################################################

        for i in range(num_it):
            err = jacobian.FK(model, curr_theta) - goal_pos  

            if tf.reduce_sum(tf.abs(err)) < 1e-3:
                if dbg:
                    print(f"Break at iteration {i}")
                err = tf.reduce_sum(tf.abs(err))
                max_it_reached = 0
                break
            
            J = jacobian.FK_Jacobian(model, curr_theta)
            J_inv = tf.linalg.pinv(J)

            # From (DIM,) to (DIM, 1) and then back to (DIM,)
            delta_theta = -tf.matmul(J_inv, tf.reshape(err, (-1, 1)))
            delta_theta = tf.reshape(delta_theta, (-1,))

            curr_theta.assign_add(delta_theta)
            curr_theta.assign(tf.math.floormod(curr_theta, 2*np.pi))
        else:
            if dbg:
                print("Max num iteration reached")
            err = tf.reduce_sum(tf.abs(jacobian.FK(model, curr_theta) - goal_pos))
            max_it_reached = 1

    else:

        ##########################################################################
        # Levenberg-Marquardt
        # x_{n+1} = x_n - (J^T * J + lambda * I)^-1 * J^T * f(x_n)
        ##########################################################################

        for i in range(num_it):
            err = jacobian.FK(model, curr_theta) - goal_pos  
            
            if tf.reduce_sum(tf.abs(err)) < 1e-3:
                if dbg:
                    print(f"Break at iteration {i}")
                err = tf.reduce_sum(tf.abs(err))
                max_it_reached = 0
                break
            
            J = jacobian.FK_Jacobian(model, curr_theta)
            J_T = tf.transpose(J)

            delta_theta = -tf.matmul(tf.linalg.inv(tf.matmul(J_T, J) + lambda_ * tf.eye(njoint)), tf.matmul(J_T, tf.reshape(err, (-1, 1))))
            delta_theta = tf.reshape(delta_theta, (-1,))

            curr_theta.assign_add(delta_theta)
            curr_theta.assign(tf.math.floormod(curr_theta, 2*np.pi))

        else:
            if dbg:
                print("Max num iteration reached")
            err = tf.reduce_sum(tf.abs(jacobian.FK(model, curr_theta) - goal_pos))
            max_it_reached = 1
            

    if dbg:
        print()
        print("Final theta:", curr_theta.numpy())
        print("Final position:", jacobian.FK(model, curr_theta).numpy())
        print("True position:", jacobian.fwd_kin_true(curr_theta).numpy())
        print("Target:", goal_pos.numpy())

    return curr_theta.numpy(), err.numpy(), max_it_reached