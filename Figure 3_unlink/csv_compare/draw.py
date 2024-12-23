import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_optimal_gap(fileName, file1, file2, output_file):
    # 讀取資料
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 將字符串轉換為正確格式的數組
    def parse_array(array_str):
        array_str = array_str.replace('[', '').replace(']', '').replace('\n', ' ').strip()
        return np.array([float(x) for x in array_str.split()])

    # 提取 x^*（file1 的最後一行）
    x_star_weight = parse_array(df1['linear.weight'].iloc[-1])
    x_star_bias = parse_array(df1['linear.bias'].iloc[-1])

    # 計算 optimal-gap
    def calculate_gap(df, x_star_weight, x_star_bias):
        gaps = []
        eps = 1e-8  # 防止除以 0
        for index, row in df.iterrows():
            # 提取每一行參數
            weight = parse_array(row['linear.weight'])
            bias = parse_array(row['linear.bias'])

            # 特徵數量對齊
            if len(weight) > len(x_star_weight):
                weight = weight[:len(x_star_weight)]

            # 計算與 x^* 的距離 (L2 norm)
            gap = np.linalg.norm(weight - x_star_weight) ** 2 + np.linalg.norm(bias - x_star_bias) ** 2
            gaps.append(gap)

        # Normalize，讓 epoch 0 為 1，避免除以 0
        gaps = np.array(gaps)
        gaps /= (gaps[0] + eps)
        return gaps

    # 計算 gaps
    gap1 = calculate_gap(df1, x_star_weight, x_star_bias)
    gap2 = calculate_gap(df2, x_star_weight, x_star_bias)

    # 只畫出前 30 epoch 的結果
    epochs = 30
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(epochs + 1), gap1[:epochs + 1], label='SGD')
    plt.plot(np.arange(epochs + 1), gap2[:epochs + 1], label='SRD')
    plt.xlabel('oracle calls / n')
    plt.yscale('log')
    plt.ylabel('relative error')
    plt.title(fileName)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f'Optimal-gap plot saved to {output_file}')


if __name__ == '__main__':
    # fileName = 'ijcnn1'
    fileNames = ['phishing', 'mushrooms', 'w8a', 'ijcnn1']
    for fileName in fileNames:
        calculate_optimal_gap(fileName,
                              'SGD_' + fileName + '_params_info.csv', \
                              'SRG_' + fileName + '_params_info.csv', \
                                       fileName + '.png')
