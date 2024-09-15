# refer to: Stochastic Reweighted Gradient Descent - https://proceedings.mlr.press/v162/hanchi22a.html
 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

eps = 1e-10

def f(x):
    # f_i(x) = (x - a_i) ^ 2 / 2
        # a_i = 1 if i = n - 1
        # a_i = 0 else
    n = tf.shape(x)[0]
    a = tf.zeros_like(x)
    a = tf.tensor_scatter_nd_update(a, [[n-1]], [1]) 
    diff = x - a
    return tf.square(diff) / 2


def compute_gradient(x_k, i_k):
    n = tf.shape(x_k)[0]
    a = tf.zeros_like(x_k)
    a = tf.tensor_scatter_nd_update(a, [[n-1]], [1]) 
    
    result_vector = tf.tensor_scatter_nd_update(
        tf.zeros_like(x_k),
        [[i_k]],
        [x_k[i_k] - a[i_k]]
    )
    return result_vector


def SRG(d=20, n=20):
    num_iterations = n * 30
    # default
    x_star = tf.constant([1 / n for _ in range(n)])
    
    # step 1: Parameters
    # alpha = np.linspace(0.5, 0.0001, num_iterations)
    alpha = 0.7 / (1 + 0.01 * np.arange(num_iterations))
    # alpha = 0.01 / (1 + 0.00005 * np.arange(num_iterations))
    # alpha = np.ones(num_iterations) * 0.01
    # alpha[1000:] = 0.01 / (1 + 0.0001 * np.arange(num_iterations - 1000))

    theta = np.linspace(0.5, 0.5, num_iterations) 
    # theta = 0.5 / (1 + 0.0001 * np.arange(num_iterations))

    # step 2: Initialization
    x_old = tf.Variable(tf.random.uniform([d], minval=-60, maxval=60, dtype=tf.float32))
    # x_old = tf.Variable(tf.random.uniform([d], minval=0, maxval=1, dtype=tf.float32))
    x_0 = tf.constant(x_old)
    print("x_0", x_0)
    g_old = [tf.random.normal([n], dtype=tf.float32) for _ in range(n)]
    g_old_norm = tf.Variable([tf.norm(g) for g in g_old], dtype=tf.float32)

    # step 3: iteration
    gradient_list = []
    error_list = []

    for k in range(num_iterations):
        # step 4: update pk
        q_k = tf.cast(g_old_norm / tf.reduce_sum(g_old_norm), tf.float32).numpy()
        # q_k_temp = tf.clip_by_value(q_k, 1e-12, 1.0)
        p_k = (1 - theta[k]) * q_k + theta[k] / n

        # step 5: update bk
        b_k = np.random.binomial(n=1, p=theta[k])
        # print(b_k)

        # step 6: update ik
        if b_k == 1:
            i_k = np.random.randint(0, n)
        else:
            i_k = np.random.choice(np.arange(n), p=q_k)
        # i_k = int(i_k)

        # step 7: update x_{k+1}
        gradient = compute_gradient(x_old, i_k)
        x_new = x_old - alpha[k] * gradient / (n * max(p_k[i_k], 1e-9))
        gradient_list.append(gradient)
        # error_list.append(tf.reduce_sum((x_new - x_star)))
        if tf.reduce_sum(tf.square(x_new - x_star)) / tf.reduce_sum(tf.square(x_0 - x_star)) < eps:
            break

        # step 8: update g_k_{j+1}
        g_new_norm = tf.Variable(tf.zeros(n, dtype=tf.float32))
        for i in range(n):
            if  b_k == 1 and i_k == i:
                g_new_norm[i].assign(tf.norm(compute_gradient(x_old, i_k)))
            else:
                g_new_norm[i].assign(g_old_norm[i])

        # update
        x_old.assign(x_new)
        g_old_norm.assign(g_new_norm)
        # error_list.append(tf.reduce_sum(tf.square(x_new - x_star)) / tf.reduce_sum(tf.square(x_0 - x_star)))
        if k % n == 0:
            error_list.append(tf.reduce_sum(tf.square(x_new - x_star)) / tf.reduce_sum(tf.square(x_0 - x_star)))
    

    # draw
    print("x_old", x_old)
    print("x_star", x_star)
    print("error_list", error_list[-1])
    plt.plot(np.arange(len(error_list)) * n / d, error_list, label="SRG", marker="v", color="green")
    plt.yscale('log')
    plt.xlabel('oracle calls / n')
    plt.ylabel('relative error')
    plt.title('Figure 1')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    random.seed(13)
    SRG()