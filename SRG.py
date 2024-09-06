# refer to: Stochastic Reweighted Gradient Descent - https://proceedings.mlr.press/v162/hanchi22a.html
 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
    
def f(x):
    # f_i(x) = (x - a_i) ^ 2 / 2
        # a_i = 1 if i = n - 1
        # a_i = 0 else
    n = tf.shape(x)[0]
    a = tf.zeros_like(x)
    a = tf.tensor_scatter_nd_update(a, [[n-1]], [1]) 
    diff = x - a
    return tf.reduce_sum(tf.square(diff) / 2)


def compute_gradient(f, x_k, i_k):
    n = tf.shape(x_k)[0]
    a = tf.zeros_like(x_k)
    a = tf.tensor_scatter_nd_update(a, [[n-1]], [1]) 
    
    result_vector = tf.tensor_scatter_nd_update(
        tf.zeros_like(x_k),
        [[i_k]],
        [x_k[i_k] - a[i_k]]
    )
    return result_vector


def SRG(num_iterations=10000, d=20, n=20):
    # default
    x_star = tf.Variable([1 / n for i in range(n)])
    
    # step 1: Parameters
    alpha = np.linspace(0.01, 0.0001, num_iterations)
    theta = np.linspace(0.5, 0.5, num_iterations) 

    # step 2: Initialization
    x_old = tf.Variable(tf.random.normal([d], dtype=tf.float32))
    g_old = [tf.random.normal([n], dtype=tf.float32) for _ in range(n)]
    g_old_norm = tf.Variable([tf.norm(g) for g in g_old], dtype=tf.float32)

    # step 3: iteration
    gradient_list = []
    error_list = []

    for k in range(num_iterations):
        # step 4: update pk
        q_k = tf.cast(g_old_norm / tf.reduce_sum(g_old_norm), tf.float32).numpy()
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
        gradient = compute_gradient(f, x_old, i_k)
        x_new = x_old - alpha[k] * gradient / (n * p_k[i_k])
        # if k % 1000 == 0:
        #     print("gradient", gradient)
        gradient_list.append(gradient)
        error_list.append(tf.reduce_sum(tf.square(f(x_new) - f(x_star))))

        # step 8: update g_k_{j+1}
        g_new_norm = tf.Variable(tf.zeros(n, dtype=tf.float32))
        for i in range(n):
            if  b_k == 1 and i_k == i:
                g_new_norm[i].assign(tf.norm(compute_gradient(f, x_old, i_k)))
            else:
                g_new_norm[i].assign(g_old_norm[i])

        # update
        x_old.assign(x_new)
        g_old_norm.assign(g_new_norm)

    # draw
    print(x_old)
    print(x_star)
    plt.plot(range(num_iterations), error_list)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Iterations')
    plt.show()


if __name__ == '__main__':
    SRG()