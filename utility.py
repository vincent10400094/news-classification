import math
import numpy as np

def build_vocabulary_map(vocabulary_path):
    print("Building vocabulary map...")
    vocab_map = {}
    t = []
    with open(vocabulary_path, "r") as f:
        i = 0
        while True:
            line = f.readline().split()
            if not line:
                break
            vocab_map[line[0]] = i
            t.append((float(line[1]), line[0]))
            i += 1
        f.close()
    print("[+] Number of words:", len(vocab_map))
    return vocab_map, t

def read_documents(document_path):
    print("Rading documents...")
    documents = []
    f = open(document_path, "r")
    documents = f.readlines()
    f.close()
    print("[+] Number of documents:", len(documents))
    return documents

def compute_normalized_entropy(W, M, N, t):
    e = []
    r = -1.0 / math.log(N)
    for i in range(M):
        s = 0.0
        for j in range(N):
            p = float(W[i][j]) / t[i][0]
            s += (p * math.log(p)) if p > 0.0001 else 0.0
        e.append(r * s)
    return e

def build_document_matrix(vocab_map, documents, t):
    print("Building matrix W...")
    M = len(vocab_map)
    N = len(documents)
    W = np.zeros((M, N))
    document_length = []
    print("Computing c_ij")
    # compute c_ij
    for i, d in enumerate(documents):
        words = d.split()
        length = 0
        for w in words:
            if w in vocab_map:
                W[vocab_map[w]][i] += 1
                length += 1
        document_length.append(length)
    # normalize to word entropy
    print("Compute entropy...")
    e = compute_normalized_entropy(W, M, N, t)
    print("Normalizing...")
    for i in range(M):
        for j in range(N):
            W[i][j] = (1-e[i]) * W[i][j] / document_length[j]
    print("[+] Completed")
    return W
