from utility import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.font_manager import FontProperties
K = 100
sigma = np.load("sigma.npy")
VT = np.load("VT.npy")
S = np.zeros([K, K])
for i in range(K):
    S[i][i] = sigma[i]
VT = VT[:K]
documents_matrix = np.transpose(S.dot(VT))

document_classes = []
for i in range(50):
    document_classes.append([])
with open("./document_cluster_.txt", "r") as f:
    i = 0
    while True:
        x = f.readline()
        if not x:
            break
        document_classes[int(x)].append(i)
        i += 1
        

# vocab_map, t = build_vocabulary_map('words.txt')
# n = len(words)
indices = []

# for i in range(100):
#     indices.append([vocab_map[x] for x in word_classes[i]])

indices.append([i for i in document_classes[40]])
indices.append([i for i in document_classes[11]])
indices.append([i for i in document_classes[22]])
indices.append([i for i in document_classes[25]])
indices.append([i for i in document_classes[33]])
labels = ["articles about \"交通事故\"", "articles about \"中國\"", "articles about \"犯罪\"", "articles about \"農業\"", "articles about \"媽祖\""]

# word_n = []
# word_n.append([i for i in word_classes[6]])
# word_n.append([i for i in word_classes[7]])
# word_n.append([i for i in word_classes[25]])
# word_n.append([i for i in word_classes[51]])
# word_n.append([i for i in word_classes[79]])


word_vector_first = documents_matrix[:, 5]
word_vector_second = documents_matrix[:, 6]

# plt.plot([word_vector[0][i] for i in indices], [word_vector[1][i] for i in indices], marker, label="marker='{0}'".format(marker))
font = FontProperties(fname=r"/Users/kevin/Downloads/msj.ttf", size=12)
fontdict = {'fontsize': 5}
fig, ax = plt.subplots()
for i in range(len(indices)):
    a = [word_vector_first[x] for x in indices[i]]
    b = [word_vector_second[x] for x in indices[i]]
    ax.scatter(a, b, alpha=0.6, label=labels[i])
    # for j in range(5):
        # ax.annotate(word_n[i][j], (a[j], b[j]), fontproperties=font)
# plt.scatter([word_vector_first[i] for i in indices_second], [word_vector_second[i] for i in indices_second], alpha=0.6)
plt.title("Visualize 5 clusters of documents in 2D")
plt.legend(prop=font)
plt.show()