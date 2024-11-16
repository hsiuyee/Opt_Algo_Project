import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def sparse_to_dense(sparse_line, num_features):
    """
    Convert a sparse representation line from LIBSVM format to a dense vector.

    Parameters:
    - sparse_line: A line from the LIBSVM file (e.g., "1 1:0.5 3:0.8 5:0.2").
    - num_features: Total number of features (dimensionality).

    Returns:
    - label: The label of the data point.
    - dense_vector: The dense representation of the feature vector.
    """
    parts = sparse_line.strip().split()
    label = int(parts[0])  # First value is the label
    dense_vector = [0.0] * num_features
    for item in parts[1:]:
        index, value = item.split(":")
        dense_vector[int(index) - 1] = float(value)  # Indices in LIBSVM start at 1
    return label, dense_vector


if __name__ == '__main__':
    fileNames = ['ijcnn1', 'mushrooms', 'phishing', 'w8a']
    for fileName in fileNames:
        file_path = 'data/' + fileName + '.csv'
        data = pd.read_csv(file_path)

        X = data.iloc[:, :-1].values
        y = data['label'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mu = 1 / len(y)
        model = LogisticRegression(penalty='l2', C=1/mu, solver='lbfgs')
        model.fit(X_scaled, y)

        x_star = model.coef_
        print("x_star:", x_star)

        x_star_df = pd.DataFrame(x_star)
        x_star_df.to_csv('x_star_' + fileName + '.csv', index=False, header=False)

        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f'accuracy: {accuracy}')