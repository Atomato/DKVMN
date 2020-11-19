# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

dataset = "assist2009_updated"
ground_truth_label = False

if dataset == "synthetic":
    n_question = 50
    if ground_truth_label:
        label_list = [3, 2, 3, 2, 3, 4, 0, 2, 3, 0, 1, 0, 1, 4, 3, 3, 0, 1, 4, 4, 0, 4, 4,
                      1, 1, 0, 0, 4, 2, 0, 4, 4, 1, 2, 4, 0, 3, 0, 1, 2, 2, 0, 3, 3, 2, 3, 2, 3, 2, 1]

        label_list = np.array(label_list)
elif dataset == "assist2009_updated":
    n_question = 110
    if ground_truth_label:
        label_dict = {}
        label_list = []
        with open("../../data/assist2009_updated/clustered_skill_name.txt", "r") as f:
            for line in f.readlines():
                row = line.strip().split("\t")
                label_dict[int(row[1])] = int(row[0])

            for i in range(1, n_question + 1):
                label_list.append(label_dict[i])

with open("correlation_weight.txt", "r") as f:
    lines = f.readlines()

data = []
for idx, line in enumerate(lines):
    if idx > n_question * 2:
        break

    if idx % 2 == 0:
        pass
    else:
        # print((idx + 1) / 2)
        data.append(
            np.array(list(map(lambda s: float(s), line.strip().split(',')))))
data = np.array(data)

if not ground_truth_label:
    label_list = []
    for d in data:
        label_list.append(np.argmax(d))

model = TSNE(n_components=2, perplexity=100)
transformed = model.fit_transform(data)

xs = transformed[:, 0]
ys = transformed[:, 1]

# %%
plt.scatter(xs, ys, c=label_list)

for i in range(n_question):
    plt.text(xs[i], ys[i], i+1)
plt.show()
# %%
cluster = {}
for skill_id, label in enumerate(label_list):
    skill_id = skill_id + 1
    if label not in cluster.keys():
        cluster[label] = []
    cluster[label].append(skill_id)

for label in cluster:
    print(cluster[label], "\n")
