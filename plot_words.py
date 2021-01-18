from utility import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.font_manager import FontProperties

U = np.load('U.npy')
sigma = np.load('sigma.npy')
U = U[:, :150]
S = np.zeros([150, 150])
for i in range(150):
    S[i][i] = sigma[i]
vocab_map, t = build_vocabulary_map('words.txt')
words = ['病毒', '流感', '歐巴馬', '民主', '自由', '輪胎', '交通', '明星', '藝人', '王建民', '棒球', 'H1N1', '中國國民黨', '馬英九']
n = len(words)
indices = [vocab_map[i] for i in words]

print(indices)

word_vector = np.matmul(U, S)
corr_matrix = np.zeros([n, n])

for i in range(n):
    for j in range(n):
        corr_matrix[i][j] = np.matmul(word_vector[indices[i]], word_vector[indices[j]]) / (np.linalg.norm(word_vector[indices[i]]) * np.linalg.norm(word_vector[indices[j]]))

print(corr_matrix.shape)

# ax = sns.heatmap(corr_matrix)
font = FontProperties(fname=r"/Users/kevin/Downloads/msj.ttf", size=14)
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)
ax.matshow(corr_matrix, cmap='viridis')

fontdict = {'fontsize': 14}
word = [""]+words
ax.set_xticklabels(word, fontdict=fontdict, rotation=45, fontproperties=font, color="black")
ax.set_yticklabels(word, fontdict=fontdict, fontproperties=font, color="black")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

for i in range(len(words)):
    for j in range(len(words)):
        ax.text(j, i, round(corr_matrix[i][j], 3), ha="center", va="center", color="w")

plt.title("Similarity between words\n\n")
plt.show()