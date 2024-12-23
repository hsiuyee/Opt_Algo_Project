from SRG import SRG
from SRG_plus import SRG_plus
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

def combine(seed=69):
    error_list_SRG = SRG(seed)
    error_list_SRG_plus = SRG_plus(seed)
    plt.plot(np.arange(len(error_list_SRG)) * 20 / 20, error_list_SRG, label="SRG", marker="v", color="green")
    plt.plot(np.arange(len(error_list_SRG_plus)) * 20 / 20, error_list_SRG_plus, label="SRG+", marker="o", color="blue")

    plt.yscale('log')
    plt.xlabel('oracle calls / n')
    plt.ylabel('relative error')
    plt.title('Figure 1')
    plt.legend()
    plt.grid(True)

    plt.savefig('figure1.png', dpi=300) 
    plt.show()

if __name__ == '__main__':
    combine(86)