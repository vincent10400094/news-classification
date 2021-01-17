
def build_vocabulary_map(vocabulary_path):
    print("Building vocabulary map...")
    vacab_map = {}
    with open(vocabulary_path, "r") as f:
        i = 0
        while True:
            line = f.readline().split()
            if not line:
                break
            vacab_map[line[0]] = i, line[1]
            i += 1
        f.close()
    return vacab_map

def read_documents(document_path):
    documents = []
    f = open(document_path, "r")
    documents = f.readlines()
    f.close()
    return documents