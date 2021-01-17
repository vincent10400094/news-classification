import numpy as np
from numpy import linalg as la
from sklearn.cluster import MiniBatchKMeans
import json
K = 100
cluster = 50
if __name__ == '__main__':
    # W = np.load("W.npy")
    # U, sigma, VT = la.svd(W)
    sigma = np.load("sigma.npy")
    VT = np.load("VT.npy")

    S = np.zeros([K, K])
    for i in range(K):
        S[i][i] = sigma[i]
    VT = VT[:K]
    documents_matrix = np.transpose(S.dot(VT))
    document_classes = MiniBatchKMeans(n_clusters=cluster, random_state=0, max_iter=1000).fit(documents_matrix)
    document_classes = [int(i) for i in document_classes.labels_]
    print(document_classes)
    with open("./icorpus.json") as f:
        datas = json.load(f)
    class_dict = []
    
    for i in range(cluster):
        class_dict.append([])
    for i, _ in enumerate(datas):
        class_dict[document_classes[i]].append(i)

    with open("document_cluster.txt", "w+") as f:
        for i, word_list in enumerate(class_dict):
            f.write(str(i) + '\n')
            for x in word_list:
                f.write(datas[x]["華語"].replace("\n", " ")+'\n')
            # f.write("\n")

    # with open("document_cluster.txt", "w+") as f:
    #     for c in document_classes:
    #         f.write(str(c)+"\n")



