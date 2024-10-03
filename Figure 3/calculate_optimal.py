import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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