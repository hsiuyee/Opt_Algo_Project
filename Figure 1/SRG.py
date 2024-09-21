# refer to: Stochastic Reweighted Gradient Descent - https://proceedings.mlr.press/v162/hanchi22a.html
 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

eps = 1e-10

def set_seed(seed=69):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def f(x, n):
    # f_i(x) = (x - a_i) ^ 2 / 2
        # a_i = 1 if i = n - 1
        # a_i = 0 else
    a = tf.zeros(n, dtype=tf.float32)
    a = tf.tensor_scatter_nd_update(a, [[n - 1]], [1.0])
    fi = 0.5 * tf.square(x - a)
    return fi


def compute_gradient(x, i_k, n):
    with tf.GradientTape() as tape:
        tape.watch(x)
        fi = f(x, n)[i_k]
    gradient = tape.gradient(fi, x)
    return gradient


def SRG(seed=69, n=20, learning_rate=0.01):
    set_seed(seed)
    num_iterations = n * 30
    # default
    x_star = 1 / n
    
    # step 1: Parameters
    # alpha = 0.9 / (1 + 0.01 * np.arange(num_iterations))
    # alpha = learning_rate * np.arange(num_iterations)
    alpha = np.linspace(0.025, 0, num_iterations)
    # alpha = np.linspace(0.5, 0.0001, num_iterations)
    # alpha = 0.08 / (1 + 0.02 * np.arange(num_iterations))
    # alpha = np.linspace(0.1, 0.005, num_iterations)
    # alpha = learning_rate * np.arange(num_iterations)
    # alpha = 0.3 / np.sqrt(np.arange(1, num_iterations + 1))
    # alpha = 0.01 / (1 + 0.001 * np.arange(num_iterations))
    # alpha = 0.01 / (1 + 0.00005 * np.arange(num_iterations))

    theta = np.linspace(0.5, 0.5, num_iterations)
    # theta = 0.5 / (1 + 0.0001 * np.arange(num_iterations))

    # step 2: Initialization
    x_old = tf.Variable(tf.random.uniform([], minval=-100, maxval=100, dtype=tf.float32))
    x_0 = tf.constant(x_old)
    g_old_norm = tf.Variable(tf.ones(n, dtype=tf.float32))

    

    # step 3: iteration
    gradient_list = []
    error_list = [1]

    for k in range(num_iterations):
        # step 4: update pk
        q_k = g_old_norm / tf.reduce_sum(g_old_norm)
        p_k = (1 - theta[k]) * q_k + theta[k] / n

        # step 5: update bk
        b_k = np.random.binomial(n=1, p=theta[k])
        # print(b_k)

        # step 6: update ik
        if b_k == 1:
            i_k = np.random.randint(0, n)
        else:
            i_k = np.random.choice(np.arange(n), p=q_k.numpy())

        # step 7: update x_{k+1}
        gradient = compute_gradient(x_old, i_k, n)
        x_new = x_old - alpha[k] * gradient / (n * max(p_k[i_k], 1e-9))
        gradient_list.append(gradient)


        # step 8: update g_k_{j+1}
        # g_new_norm = tf.Variable(tf.zeros(n, dtype=tf.float32))
        # for i in range(n):
        #     if  b_k == 1 and i_k == i:
        #         g_new_norm[i].assign(tf.norm(compute_gradient(x_old, i_k, n)))
        #     else:
        #         g_new_norm[i].assign(g_old_norm[i])

        # step 8: optimize
        g_new_norm = g_old_norm
        if  b_k == 1:
            g_new_norm[i_k].assign(tf.norm(compute_gradient(x_old, i_k, n)))
        
        # update
        x_old.assign(x_new)
        g_old_norm.assign(g_new_norm)
        relative_error = tf.abs(x_new - x_star) / tf.abs(x_0 - x_star)
        if (k+1) % n == 0:
            error_list.append(relative_error.numpy())
    

    # draw
    # print("x_old", x_old)
    # # print("x_star", x_star)
    # print("error_list", error_list[-1])
    # plt.plot(np.arange(len(error_list)), error_list, label="SRG", marker="v", color="green")
    # plt.yscale('log')
    # plt.xlabel('oracle calls / n')
    # plt.ylabel('relative error')
    # plt.title('Figure 1')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return error_list


# if __name__ == '__main__':
#     set_seed(69)
#     SRG()