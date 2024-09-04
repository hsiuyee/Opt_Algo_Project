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
    x_k = tf.convert_to_tensor(x_k, dtype=tf.float32)
    x_k = tf.Variable(x_k)
    
    with tf.GradientTape() as tape:
        tape.watch(x_k)
        value = f(x_k)
    
    grad = tape.gradient(value, x_k)
    gradient_directional = tf.reshape(grad, [-1]).numpy()
    directional_gradient = gradient_directional[i_k]
    
    return directional_gradient

def SRG(num_iterations=1000, d=16, n=16):
    # default
    x_star = tf.Variable([1 / n for i in range(n)])
    # step 1: Parameters
    alpha = np.linspace(0.1, 0.001, num_iterations)
    theta = np.linspace(0.5, 0.5, num_iterations) 

    # step 2: Initialization
    x_old = tf.Variable(tf.random.normal([d], dtype=tf.float32))
    g_old = [tf.random.normal([n], dtype=tf.float32) for _ in range(n)]
    g_old_norm = tf.Variable([tf.norm(g) for g in g_old], dtype=tf.float32)

    # step 3: iteration
    gradient_list = []
    error_list = []

    for k in range(num_iterations):
        # step 4
        q_k = tf.cast(g_old_norm / tf.reduce_sum(g_old_norm), tf.float32).numpy()
        p_k = (1 - theta[k]) * q_k + theta[k] / n

        # step 5
        b_k = np.random.binomial(n=1, p=theta[k])

        # step 6
        if b_k == 1:
            i_k = np.random.randint(0, n)
        else:
            i_k = np.random.choice(np.arange(n), p=q_k)
        i_k = int(i_k)

        # step 7
        gradient = compute_gradient(f, x_old, i_k)
        x_new = x_old - alpha[k] * gradient / (n * p_k[i_k])
        # if k % 1000 == 0:
        #     print("gradient", gradient)
        gradient_list.append(gradient)
        error_list.append(tf.reduce_sum(tf.square(x_new - x_old)))

        # # step 8
        g_new_norm = tf.Variable(tf.zeros(n, dtype=tf.float32))
        for i in range(n):
            if i_k == i and b_k == 1:
                g_new_norm[i].assign(tf.norm(compute_gradient(f, x_old, i_k)))
            else:
                g_new_norm[i].assign(g_old_norm[i])

        # update
        x_old.assign(x_new)
        g_old_norm.assign(g_new_norm)

    # draw
    plt.plot(range(num_iterations), error_list)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Iterations')
    plt.show()

if __name__ == '__main__':
    SRG()