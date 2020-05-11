
def get_relation2id(dir):
    file = dir +"/relation2id.txt"
    re2id = {}
    with open(file) as f:
        for line in f:
            splits = line.strip().split()
            re2id[splits[0]] = int(splits[1])
    return re2id


if __name__ == "__main__":
    r2id = get_relation2id("D:/WSL/Intra-Bag-and-Inter-Bag-Attentions/NYT_data")
    print(r2id)