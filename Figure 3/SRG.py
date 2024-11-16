import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


eps = 1e-9

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Logistic regression loss function f_i(x)
def f(x, a, y, mu):
    """
    Computes the logistic regression loss function f_i(x).
    Parameters:
    - x: Model parameter vector (TensorFlow variable).
    - a: Feature vector for a single data point.
    - y: Label for a single data point, y ∈ {-1, 1}.
    - mu: Regularization parameter.
    Returns:
    - f_i(x): Logistic regression loss value for a single data point.
    """
    logistic_loss = tf.math.log(1 + tf.exp(-y * tf.tensordot(a, x, axes=1)))
    l2_regularization = (mu / 2) * tf.reduce_sum(tf.square(x))
    return logistic_loss + l2_regularization


# Gradient computation for logistic regression
def compute_gradient(x, i_k, A, y, mu):
    a_i = A[i_k]  # Feature vector for the i_k-th data point
    y_i = y[i_k]  # Label for the i_k-th data point
    with tf.GradientTape() as tape:
        tape.watch(x)
        fi = f(x, a_i, y_i, mu)
    gradient = tape.gradient(fi, x)
    return gradient


def load_data(file_path1, file_path2, mu=None):
    """
    Load the dataset and compute necessary parameters.
    """
    df = pd.read_csv(file_path1)
    A = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)
    n, d = A.shape

    # Load x_star
    x_star = pd.read_csv(file_path2, header=None).iloc[0, :].to_numpy(dtype=np.float32)
    if len(x_star) != d:
        raise ValueError(f"x_star dimensions {len(x_star)} do not match feature dimensions {d}.")

    if mu is None:
        mu = 1 / n  # Default regularization parameter

    print(f"Data loaded: {n} samples, {d} features")
    print(f"Regularization parameter (mu): {mu}")

    return A, y, n, d, mu, x_star


def SRG(data_path1, data_path2, seed=42, mu=None):
    """
    Implements Stochastic Reweighted Gradient Descent (SRG).
    """
    set_seed(seed)
    A, y, n, d, mu, x_star = load_data(data_path1, data_path2, mu)
    num_iterations = n * 30
    L = 0.25 + mu
    theta = np.linspace(0.5, 0.5, num_iterations)
    alpha = theta / (2 * L)
    x_old = tf.Variable(tf.random.uniform([d], minval=-1, maxval=1, dtype=tf.float32))
    x_0 = tf.constant(x_old)
    g_old_norm = tf.Variable(tf.ones(n, dtype=tf.float32))

    error_list = [1]
    print(f"Starting SRG with {num_iterations} iterations...")
    for k in range(num_iterations):
        q_k = g_old_norm / tf.reduce_sum(g_old_norm)
        p_k = (1 - theta[k]) * q_k + theta[k] / n

        b_k = np.random.binomial(n=1, p=theta[k])
        if b_k == 1:
            i_k = np.random.randint(0, n)
        else:
            i_k = np.random.choice(np.arange(n), p=q_k.numpy())

        gradient = compute_gradient(x_old, i_k, A, y, mu)
        x_new = x_old - alpha[k] * gradient / (n * max(p_k[i_k], eps))

        g_new_norm = tf.Variable(g_old_norm)
        if b_k == 1:
            g_new_norm[i_k].assign(tf.norm(gradient))

        x_old.assign(x_new)
        g_old_norm.assign(g_new_norm)

        if (k + 1) % n == 0:  # Log progress after each epoch
            relative_error = tf.reduce_sum(tf.square(x_new - x_star)) / tf.reduce_sum(tf.square(x_0 - x_star))
            error_list.append(relative_error.numpy())
            print(f"Epoch {int((k + 1) / n)}: Relative error = {relative_error:.6f}")

    print("SRG completed.")
    return error_list


def SGD(data_path1, data_path2, seed=42, mu=None):
    """
    Implements Stochastic Gradient Descent (SGD).
    """
    set_seed(seed)
    A, y, n, d, mu, x_star = load_data(data_path1, data_path2, mu)
    num_iterations = n * 30
    L = 0.25 + mu
    alpha = 0.1 / L
    x_old = tf.Variable(tf.random.uniform([d], minval=-1, maxval=1, dtype=tf.float32))
    x_0 = tf.constant(x_old)

    error_list = [1]
    print(f"Starting SGD with {num_iterations} iterations...")
    for k in range(num_iterations):
        i_k = np.random.randint(0, n)
        gradient = compute_gradient(x_old, i_k, A, y, mu)
        x_new = x_old - alpha * gradient
        x_old.assign(x_new)

        if (k + 1) % n == 0:  # Log progress after each epoch
            relative_error = tf.reduce_sum(tf.square(x_new - x_star)) / tf.reduce_sum(tf.square(x_0 - x_star))
            error_list.append(relative_error.numpy())
            print(f"Epoch {int((k + 1) / n)}: Relative error = {relative_error:.6f}")

    print("SGD completed.")
    return error_list


def plot_results(srg_errors, sgd_errors, oracle_calls, output_path="comparison_plot.png"):
    """
    Plot the relative error for SRG and SGD and save it as a PNG file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(oracle_calls, srg_errors, label='SRG', marker='o', color='green')
    plt.plot(oracle_calls, sgd_errors, label='SGD', marker='v', color='blue')
    plt.yscale('log')
    plt.xlabel('oracle calls / n')
    plt.ylabel('relative error')
    plt.title('Comparison of SRG and SGD')
    plt.legend()
    plt.grid(True)

    # Save the plot to a PNG file
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Plot saved to {output_path}")

    plt.show()


# Main script
if __name__ == '__main__':
    set_seed(42)
    data_path1 = "data/ijcnn1.csv"
    data_path2 = "x_star_ijcnn1.csv"

    srg_errors = SRG(data_path1, data_path2)
    sgd_errors = SGD(data_path1, data_path2)
    oracle_calls = [i for i in range(len(srg_errors))]

    # Save the plot as PNG
    plot_results(srg_errors, sgd_errors, oracle_calls, output_path="srg_vs_sgd.png")

