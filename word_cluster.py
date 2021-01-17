import numpy as np
from numpy import linalg as la
from sklearn.cluster import MiniBatchKMeans
K = 150
cluster = 100
if __name__ == '__main__':
    # W = np.load("W.npy")
    # U, sigma, VT = la.svd(W)
    sigma = np.load("sigma.npy")
    U = np.load("U.npy")
    epsilon = np.load("word_entropy.npy")

    S = np.zeros([K, K])
    for i in range(K):
        S[i][i] = sigma[i]
    U = U[:,:K]
    words_matrix = U.dot(S)
    word_classes = MiniBatchKMeans(n_clusters=cluster, random_state=0).fit(words_matrix)
    word_classes = [int(i) for i in word_classes.labels_]
    
    with open("./words.txt") as f:
        words = f.readlines()

    class_dict = []
    for i in range(cluster):
        class_dict.append([])
    for i, word in enumerate(words):
        if epsilon[i] < 0.5:
            class_dict[word_classes[i]].append((word.split()[0], epsilon[i]))
    # print(class_dict)
    with open("word_cluster.txt", "w+") as f:
        for i, word_list in enumerate(class_dict):
            # word_list.sort(key = lambda x:(x[1]), reverse=True)
            f.write(str(i) + ' ')
            for x in word_list:
                f.write(x[0]+' ')
            f.write("\n")
    
    # with open("word_cluster.txt", "w+") as f:
    #     for c in word_classes:
    #         f.write(str(c)+"\n")

