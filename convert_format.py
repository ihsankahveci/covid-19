import numpy as np

annotation_file = "./annotation.csv"
output_file = "./annotation_doccano.csv"
label_file = "./annotation_gold.csv"


def setup_word_intrusion(topics, seed=42):
    # reproducibility
    np.random.seed(seed)

    intruding_topics = []
    labels = []

    for i in range(len(topics)):
        topic_words = topics[i][::]

        # find candidate intriuders
        other_words = [words for j, topic in enumerate(topics) if j != i for words in topic]
        intruder = np.random.choice(other_words, 1)

        assert intruder not in topic_words
        topic_words.append(intruder[0])
        np.random.shuffle(topic_words)

        # build dataset
        intruding_topics.append(topic_words)
        labels.append(topic_words.index(intruder))

    return intruding_topics, labels


def export_to_doccano_format(intruding_topics):
    output_string = []
    for i in range(len(intruding_topics)):
        topic_string = []

        for j, w in enumerate(intruding_topics[i]):
            topic_string.append(f"{j}:{w}")

        text = " ".join(topic_string)

        output_string.append(f"\"{text}\"")

    return "\n".join(output_string)


if __name__ == "__main__":

    topics = []

    with open(annotation_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            topics.append(line.strip().split(","))

    intruding_topics, labels = setup_word_intrusion(topics)
    doccano_format = export_to_doccano_format(intruding_topics)

    with open(output_file, 'w') as f:
        f.write("text\n")
        f.write(doccano_format)

    with open(label_file, 'w') as f:
        f.write("label\n")
        for label in labels:
            f.write(f"{label}\n")