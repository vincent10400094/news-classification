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

word_classes = []
with open("./word_cluster.txt", "r") as f:
    while True:
        x = f.readline()
        if not x:
            break
        word_classes.append(x.split()[1:])

vocab_map, t = build_vocabulary_map('words.txt')
# n = len(words)
indices = []
# for i in range(100):
#     indices.append([vocab_map[x] for x in word_classes[i]])
indices.append([vocab_map[i] for i in word_classes[6]])
indices.append([vocab_map[i] for i in word_classes[7]])
indices.append([vocab_map[i] for i in word_classes[25]])
indices.append([vocab_map[i] for i in word_classes[51]])
indices.append([vocab_map[i] for i in word_classes[79]])

word_n = []
word_n.append([i for i in word_classes[6]])
word_n.append([i for i in word_classes[7]])
word_n.append([i for i in word_classes[25]])
word_n.append([i for i in word_classes[51]])
word_n.append([i for i in word_classes[79]])


word_vector = np.matmul(U, S)
word_vector_first = np.transpose(word_vector[:, 5])
word_vector_second = np.transpose(word_vector[:, 6])

# plt.plot([word_vector[0][i] for i in indices], [word_vector[1][i] for i in indices], marker, label="marker='{0}'".format(marker))
font = FontProperties(fname=r"/Users/kevin/Downloads/msj.ttf", size=12)
fontdict = {'fontsize': 5}
fig, ax = plt.subplots()
for i in range(len(indices)):
    a = [word_vector_first[x] for x in indices[i]]
    b = [word_vector_second[x] for x in indices[i]]
    plt.scatter(a, b, alpha=0.6)
    for j in range(5):
        ax.annotate(word_n[i][j], (a[j], b[j]), fontproperties=font)
# plt.scatter([word_vector_first[i] for i in indices_second], [word_vector_second[i] for i in indices_second], alpha=0.6)
plt.title("Visualize 5 clusters of words in 2D")
plt.show()