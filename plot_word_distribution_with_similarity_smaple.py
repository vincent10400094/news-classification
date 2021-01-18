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

indices.append([vocab_map[i] for i in word_classes[4]])
indices.append([vocab_map[i] for i in word_classes[13]])
indices.append([vocab_map[i] for i in word_classes[3]])
indices.append([vocab_map[i] for i in word_classes[79]])
indices.append([vocab_map[i] for i in word_classes[61]])
indices.append([vocab_map[i] for i in word_classes[82]])
indices.append([vocab_map[i] for i in word_classes[9]])
indices.append([vocab_map[i] for i in word_classes[66]])
indices.append([vocab_map[i] for i in word_classes[19]])

# 4 H1N1 13 病毒、流感 3 民主、自由 79 輪胎、交通 61 明星 82 藝人 9 王建民 66 棒球 19 馬英九 中國國民黨
word_n = []
word_n.append(["H1N1"])
word_n.append(["病毒","流感"])
word_n.append(["民主", "自由"])
word_n.append(["輪胎", "交通"])
word_n.append(["明星"])
word_n.append(["藝人"])
word_n.append(["王建民"])
word_n.append(["棒球"])
word_n.append(["馬英九","中國國民黨"])

word_n_index = []
word_n_index.append([word_classes[4].index("H1N1")])
word_n_index.append([word_classes[13].index("病毒"), word_classes[13].index("流感")])
word_n_index.append([word_classes[3].index("民主"), word_classes[3].index("自由")])
word_n_index.append([word_classes[79].index("輪胎"), word_classes[79].index("交通")])
word_n_index.append([word_classes[61].index("明星")])
word_n_index.append([word_classes[82].index("藝人")])
word_n_index.append([word_classes[9].index("王建民")])
word_n_index.append([word_classes[66].index("棒球")])
word_n_index.append([word_classes[19].index("馬英九"), word_classes[19].index("中國國民黨")])


word_vector = np.matmul(U, S)
word_vector_first = np.transpose(word_vector[:, 5])
word_vector_second = np.transpose(word_vector[:, 6])

font = FontProperties(fname=r"/Users/kevin/Downloads/msj.ttf", size=12)
fontdict = {'fontsize': 5}
fig, ax = plt.subplots()
for i in range(len(indices)):
    a = [word_vector_first[x] for x in indices[i]]
    b = [word_vector_second[x] for x in indices[i]]
    plt.scatter(a, b, alpha=0.6)
    for j in range(len(word_n[i])):
        ax.annotate(word_n[i][j], (a[word_n_index[i][j]], b[word_n_index[i][j]]), fontproperties=font)
plt.title("Visualize clusters of words in 2D")
plt.show()