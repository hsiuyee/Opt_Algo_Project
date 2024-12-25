import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file

def split_data(fileName, n_features):
    input_file = fileName + '.txt'
    train_file = fileName + '.txt'
    test_file = fileName + '_t.txt'

    X, y = load_svmlight_file(input_file, n_features=n_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dump_svmlight_file(X_train, y_train, train_file)
    print(f'Training data saved to {train_file}')

    dump_svmlight_file(X_test, y_test, test_file)
    print(f'Testing data saved to {test_file}')

    print('Data split completed successfully!')

if __name__ == '__main__':
    split_data('phishing', 68)