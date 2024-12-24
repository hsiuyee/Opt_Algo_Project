import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


def sparse_to_dense(sparse_line, num_features):
    """
    將 LIBSVM 格式轉換為稠密向量。
    """
    parts = sparse_line.strip().split()
    label = int(parts[0])  # 第一列為標籤
    dense_vector = [0.0] * num_features
    for item in parts[1:]:
        index, value = item.split(":")
        dense_vector[int(index) - 1] = float(value)  # LIBSVM 索引從 1 開始
    return label, dense_vector


def load_libsvm_data(file_path, num_features):
    """
    從 LIBSVM 格式的 .txt 文件中載入資料。
    """
    labels = []
    features = []

    with open(file_path, 'r') as file:
        for line in file:
            label, dense_vector = sparse_to_dense(line, num_features)
            labels.append(label)
            features.append(dense_vector)

    # 轉換為 NumPy 陣列
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # 標籤轉換為 -1, 1 (若需要調整，可以改回 0, 1)
    y = np.where(y == 0, -1, y)

    return X, y


if __name__ == '__main__':
    # 設定檔案名稱和參數
    fileNames = ['ijcnn1', 'mushrooms', 'phishing', 'w8a']
    num_features_dict = {'ijcnn1': 22, 'mushrooms': 22, 'phishing': 22, 'w8a': 22}  # 特徵數量

    for fileName in fileNames:
        print(f"Processing {fileName}...")

        # 資料路徑與設定
        file_path = f'data/{fileName}.txt'  # 修改成 txt 格式
        num_features = num_features_dict[fileName]

        # 載入資料
        X, y = load_libsvm_data(file_path, num_features)

        # 標準化資料
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 設定 L2 正則化參數 μ
        mu = 1 / len(y)  # μ = 1/N
        model = LogisticRegression(penalty='l2', C=1/mu, solver='lbfgs')

        # 訓練模型
        model.fit(X_scaled, y)

        # 計算最優點 (optimal point)
        x_star = model.coef_
        print("x_star:", x_star)

        # 將最優點儲存到 CSV 檔案
        x_star_df = pd.DataFrame(x_star)
        x_star_df.to_csv(f'x_star_{fileName}.csv', index=False, header=False)

        # 計算準確率
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f'Accuracy for {fileName}: {accuracy:.6f}')
        print("=" * 50)
