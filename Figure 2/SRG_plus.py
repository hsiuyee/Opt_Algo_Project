# refer to: Stochastic Reweighted Gradient Descent - https://proceedings.mlr.press/v162/hanchi22a.html
#           Estimating Convergence of Markov chains with L-Lag Couplings - https://proceedings.neurips.cc/paper_files/paper/2019/file/aec851e565646f6835e915293381e20a-Paper.pdf

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator
import random

eps = 1e-10

def set_seed(seed=69):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def f(x, L, n):
    # f_i(x) = (x - a_i) ^ 2 / 2
        # a_i = 1 if i = n - 1
        # a_i = 0 else
    a = tf.zeros(n, dtype=tf.float32)
    a = tf.tensor_scatter_nd_update(a, [[n - 1]], [1.0])
    fi = 0.5 * tf.square(x - a) * L
    return fi


def compute_gradient(x, L, i_k, n):
    with tf.GradientTape() as tape:
        tape.watch(x)
        fi = f(x, L, n)[i_k]
    gradient = tape.gradient(fi, x)
    return gradient


def maximally_couples(p, q, n):
    # step 1: initial
    X_star = np.random.choice(np.arange(n), p=p)
    W = np.random.uniform(0, 1)

    #step 2
    if p[X_star] * W <= q[X_star]: # case 1
        Y_star = X_star
        return (X_star, Y_star)
    else:                          # case 2        
        Y_hat = np.random.choice(np.arange(n), p=q)
        W_hat = np.random.uniform(0, 1)
        while (q[Y_hat] * W_hat > p[Y_hat]):
            Y_hat = np.random.choice(np.arange(n), p=q)
            W_hat = np.random.uniform(0, 1)
        Y_star = Y_hat
        return X_star, Y_star
# def maximally_couples(p, q, n):
#     cdf_p = np.cumsum(p)
#     cdf_q = np.cumsum(q)
#     u = np.random.uniform(0, 1)
#     i_k = np.searchsorted(cdf_p, u)
#     j_k = np.searchsorted(cdf_q, u)
#     return i_k, j_k


def SRG_plus(d=20, n=20):
    set_seed(75) # 75
    num_iterations = n * 30
    # default
    x_star = 1 / (n ** 2)

    # step 1: Parameters
    alpha = np.linspace(0.04, 0, num_iterations)
    # alpha = 0.9 / (1 + 0.01 * np.arange(num_iterations))
    # alpha = np.linspace(2, 0.5, num_iterations) 
    # alpha = 0.9 / (1 + 0.01 * np.arange(num_iterations))
    # alpha = 0.08 / (1 + 0.032 * np.arange(num_iterations))
    theta = np.linspace(0.5, 0.5, num_iterations) 

    # step 2: Initialization
    x_old = tf.Variable(tf.random.uniform([], minval=-10, maxval=10, dtype=tf.float32))
    x_0 = tf.constant(x_old)
    g_old_norm = tf.Variable(tf.ones(n, dtype=tf.float32))


    # step 3: iteration
    gradient_list = []
    error_list = [1]

    
    # step 4: update pk
        # L_1 = n - 1
        # L_n = 1 / n
        # L_i = (n - 1) / (n * (n - 2))
        # L_bar = 1
        # L_max = n - 1
    L = [(n - 1) / (n * (n - 2)) for _ in range(n)]
    L[0] = n - 1
    L[-1] = 1 / n
    L = tf.constant(L)
    L_bar = 1
    
    for k in range(num_iterations):
        v = L / (n * L_bar)
        w = tf.Variable([1 / n for i in range(n)])

        q_k = tf.cast(g_old_norm / tf.reduce_sum(g_old_norm), tf.float32).numpy()
        p_k = (1 - theta[k]) * q_k + theta[k] * v

        # step 5: update bk
        b_k = np.random.binomial(n=1, p=theta[k])

        # step 6: update ik
        if b_k == 1:
            i_k, j_k = maximally_couples(v.numpy(), w.numpy(), n)
        else:
            i_k = np.random.choice(np.arange(n), p=q_k)


        # step 7: update x_{k+1}
        gradient = compute_gradient(x_old, L, i_k, n)
        x_new = x_old - alpha[k] * gradient / (n * max(p_k[i_k], 1e-9))
        # gradient_list.append(gradient)

        # step 8: update g_k_{j+1}
        # g_new_norm = tf.Variable(tf.zeros(n, dtype=tf.float32))
        # for j in range(n):
        #     if b_k == 1 and j == j_k:
        #         g_new_norm[j].assign(tf.norm(compute_gradient(x_old, j_k, n)))
        #     else:
        #         g_new_norm[j].assign(abs(g_old_norm[j]))
        # step 8: optimize
        g_new_norm = g_old_norm
        if b_k == 1:
            g_new_norm[j_k].assign(tf.norm(compute_gradient(x_old, L, j_k, n)))
        # update
        x_old.assign(x_new)
        g_old_norm.assign(g_new_norm)
        relative_error = tf.abs(x_new - x_star) / tf.abs(x_0 - x_star)
        if (k+1) % n == 0:
            error_list.append(relative_error.numpy())

    # draw
    print("x_old", x_old)
    print("x_star", x_star)
    print("error_list", error_list[-1])
    plt.plot(np.arange(len(error_list)), error_list, label="SRG+", marker="o", color="blue")
    plt.yscale('log')
    plt.xlabel('oracle calls / n')
    plt.ylabel('relative error')
    plt.title('Figure 2')
    plt.legend()
    plt.grid(True)

    plt.savefig('figure2.png', dpi=300) 
    plt.show()
    return error_list


if __name__ == '__main__':
    SRG_plus()