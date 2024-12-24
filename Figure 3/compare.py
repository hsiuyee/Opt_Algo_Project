import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


eps = 1e-9

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Logistic regression loss function f_i(x)
def f(x, a, y, mu):
    logistic_loss = tf.math.log(1 + tf.exp(-y * tf.tensordot(a, x, axes=1)))
    l2_regularization = (mu / 2) * tf.reduce_sum(tf.square(x))
    return logistic_loss + l2_regularization


def compute_gradient(x, i_k, A, y, mu):
    a_i = A[i_k]
    y_i = y[i_k]
    with tf.GradientTape() as tape:
        tape.watch(x)
        fi = f(x, a_i, y_i, mu)
    gradient = tape.gradient(fi, x)
    return gradient


def parse_libsvm_line(line, num_features):
    """
    將 LIBSVM 格式轉換為稠密向量。
    """
    parts = line.strip().split()
    label = int(parts[0])  # 第一列為標籤
    features = [0.0] * num_features
    for item in parts[1:]:
        index, value = item.split(":")
        features[int(index) - 1] = float(value)  # LIBSVM 索引從 1 開始
    return label, features


def load_data(file_path1, num_features=22, sample_size=1000):
    """
    從 LIBSVM 格式的 .txt 文件中載入資料，並隨機抽取 sample_size 筆資料。
    """
    # 讀取 txt 文件
    labels = []
    features = []
    with open(file_path1, 'r') as f:
        for line in f:
            label, feature = parse_libsvm_line(line, num_features)
            labels.append(label)
            features.append(feature)

    # 轉換為 NumPy 陣列
    A = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)


    # 隨機抽樣 1000 筆資料
    np.random.seed(42)
    indices = np.random.choice(len(A), sample_size, replace=False)
    A = A[indices]
    y = y[indices]

    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    n, d = A.shape

    mu = 1 / n  # 預設正則化參數

    print(f"Data loaded: {n} samples, {d} features")
    print(f"Regularization parameter (mu): {mu}")

    return A, y, n, d, mu


def SRG(A, y, n, d, mu, x_star):
    num_iterations = n * 30
    L = 0.25 + mu
    theta = np.linspace(0.5, 0.5, num_iterations)
    alpha = theta / (2 * L)

    x_old = tf.Variable(tf.random.uniform([d], minval=-1, maxval=1, dtype=tf.float32))
    x_0 = tf.constant(x_old)
    g_old_norm = tf.Variable(tf.ones(n, dtype=tf.float32))

    error_list = [1]
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

        g_new_norm = g_old_norm.numpy()
        if b_k == 1:
            g_new_norm[i_k] = np.linalg.norm(gradient.numpy())

        x_old.assign(x_new)
        g_old_norm.assign(g_new_norm)

        if (k + 1) % n == 0:
            relative_error = tf.reduce_sum(tf.square(x_new - x_star)) / tf.reduce_sum(tf.square(x_0 - x_star))
            error_list.append(relative_error.numpy())
            print(f"Epoch {int((k + 1) / n)}: Relative error = {relative_error:.6f}")

    return error_list


def SGD(A, y, n, d, mu, x_star):
    # 初始點與 SRG 一致
    x_0 = tf.random.uniform([d], minval=-1, maxval=1, dtype=tf.float32)
    x_old = tf.Variable(x_0)

    num_iterations = n * 30
    L = 0.25 + mu
    alpha = 0.05 / L  # 更新學習率

    error_list = [1]
    print(f"Starting SGD with {num_iterations} iterations...")
    for k in range(num_iterations):
        i_k = np.random.randint(0, n)
        gradient = compute_gradient(x_old, i_k, A, y, mu)
        x_new = x_old - alpha * gradient
        x_old.assign(x_new)

        # 誤差計算
        if (k + 1) % n == 0:
            relative_error = tf.reduce_sum(tf.square(x_new - x_star)) / tf.reduce_sum(tf.square(x_0 - x_star))
            error_list.append(relative_error.numpy())
            print(f"Epoch {int((k + 1) / n)}: Relative error = {relative_error:.6f}")

    print("SGD completed.")
    return error_list


def compute_optimal_point_cvxpy(X, y, mu):
    """
    Computes the exact optimal point for L2-regularized logistic regression.
    """
    # Step 1: Data normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Step 2: Define variables and parameters
    n, d = X.shape
    x = cp.Variable(d)  # Model parameter
    mu = mu / n  # Regularization parameter scaled by number of samples

    # Step 3: Define logistic regression objective function
    loss = cp.sum(cp.logistic(-cp.multiply(y, X @ x))) / n  # Average logistic loss
    reg = (mu / 2) * cp.sum_squares(x)  # L2 regularization
    objective = cp.Minimize(loss + reg)

    # Step 4: Solve the problem with SCS solver
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=True)  # Switch to SCS solver

    # Step 5: Return the exact solution
    return x.value


def plot_results(srg_errors, sgd_errors, oracle_calls, output_path="comparison_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(oracle_calls, srg_errors, label='SRG', marker='o', color='green')
    plt.plot(oracle_calls, sgd_errors, label='SGD', marker='v', color='blue')  # 增加 SGD 曲線
    plt.yscale('log')
    plt.xlabel('oracle calls / n')
    plt.ylabel('relative error')
    plt.title('Comparison of SRG and SGD')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, format='png', dpi=300)
    plt.show()


if __name__ == '__main__':
    set_seed(42)
    data_txt = "Datasets/ijcnn1.txt"  # 更新為 txt 檔案
    x_star_csv = "x_star_ijcnn1_cvxpy.csv"

    A, y, n, d, mu = load_data(data_txt)
    x_star = compute_optimal_point_cvxpy(A, y, mu)

    srg_errors = SRG(A, y, n, d, mu, x_star)
    sgd_errors = SGD(A, y, n, d, mu, x_star)
    oracle_calls = [i for i in range(len(srg_errors))]
    plot_results(srg_errors, sgd_errors, oracle_calls)

