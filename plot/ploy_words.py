import numpy as np
from utility import *
import seaborn as sns

U = np.load('U.npy')
S = np.load('S.npy')

vocab_map, t = build_vocabulary_map('words.txt')
words = ['復興航空', '飛機', '歐巴馬', '民主', '自由']
n = len(words)
indices = [vocab_map[i] for i in words]

print(indices)

word_vector = np.matmul(U, S)
corr_matrix = np.zeros([n, n])

for i in range(n):
    for j in range(n):
        corr_matrix[i][j] = np.matmul(word_vector[i], word_vector[j])

print(corr_matrix.shape)

ax = sns.heatmap(corr_matrix)
