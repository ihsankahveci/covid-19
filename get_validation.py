import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

annotation_file = "./annotations.csv"
text_file = "./annotation_doccano.csv"
gold_label_file = "./annotation_gold.csv"
doccano_label_file = "./labels.json"
topics_file = "./annotation.txt"

plot_out_file = "./plot.png"

NUM_TOPICS = 5
NUM_USERS = 4


def read_gold(file):
    gold = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            gold.append(line.strip())
    return gold


def read_texts(file):
    texts = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            texts.append(line.strip().strip("\""))
    return texts


def read_topics(file):
    topics = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            topics.append(line.strip().strip("\""))
    return topics


def read_annotation(file, texts):
    # user to [user_label]
    annotations = {}

    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split(",")
            user = line[4]
            text = line[3]
            label = line[2]

            if user not in annotations:
                annotations[user] = ["" for _ in range(NUM_TOPICS)]

            assert text in texts
            text_id = texts.index(text)
            annotations[user][text_id] = label

    return annotations


def get_doccano_label_to_index(file):
    label_to_index = {}
    with open(file, 'r') as f:
        labels = json.load(f)
        for label in labels:
            label_to_index[str(label["id"])] = label["text"]
    return label_to_index


def get_model_precision(gold, user_labels):
    res = []
    user_labels = user_labels.values()
    for i in range(len(gold)):
        gold_i = gold[i]
        matches = sum([1 if gold_i == user[i] else 0 for user in user_labels])
        mp_i = matches / NUM_USERS
        res.append(mp_i)
    return res


if __name__ == "__main__":
    # read gold
    gold = read_gold(gold_label_file)

    # read user annotations
    texts = read_texts(text_file)
    label_to_index = get_doccano_label_to_index(doccano_label_file)
    annotated_labels = read_annotation(annotation_file, texts)
    annotated_labels = {u: [label_to_index[l] for l in labels] for u, labels in annotated_labels.items()}

    # show results
    mp = get_model_precision(gold, annotated_labels)
    topics = read_topics(topics_file)

    sorted_index = np.argsort(mp)[::-1]
    mp = [mp[i] for i in sorted_index]
    topics = [topics[i] for i in sorted_index]

    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x=topics, y=mp)

    xmin, xmax = plt.xlim()

    ax.axhline(y = 1/6, color='red')
    arrowprops = {'width': 1, 'headwidth': 1, 'headlength': 1, 'shrink': 0.05}
    ax.annotate("Random Baseline", xy=(xmax, 1/6), xytext=(-60, 15), textcoords='offset points',
                va='bottom', ha='center', annotation_clip=False, arrowprops=arrowprops)

    plt.title("Model Precision of Word Intrusion Task")
    plt.xlabel("Topic")
    plt.ylabel("Percision")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_out_file)
    print(label_to_index)