import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path, num_features):
    """
    載入 LIBSVM 資料並轉換為積密格式
    """
    labels = []
    features = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])
            dense_vector = [0.0] * num_features
            for item in parts[1:]:
                index, value = item.split(":")
                dense_vector[int(index) - 1] = float(value)
            labels.append(label)
            features.append(dense_vector)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y


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


if __name__ == '__main__':
    # 設定檔案與特屬數量
    file_path = 'Datasets/ijcnn1.txt'
    num_features = 22

    # 資料載入
    X, y = load_data(file_path, num_features)

    # 設定正則化參數
    mu = 1 / len(y)

    # 計算最優點
    x_star = compute_optimal_point_cvxpy(X, y, mu)
    print("Optimal point (CVXPY):", x_star)

    # 儲存結果
    x_star_df = pd.DataFrame(x_star.reshape(1, -1))  # 改為一行多列
    x_star_df.to_csv('x_star_ijcnn1_cvxpy.csv', index=False, header=False)
