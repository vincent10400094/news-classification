import numpy as np
import argparse

from utility import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vocabulary_path', type=str, help='Input vocabulary file')
    parser.add_argument('document_path', type=str, help='Input document file')
    parser.add_argument('--sample_size', type=int, default=5, help='Sample size')
    args = parser.parse_args()

    vocab_map = build_vocabulary_map(args.vocabulary_path)
    documents = read_documents(args.document_path)

