# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.cluster import KMeans

from base import dunn

dataset = "assist2009_updated"
label_type = "k_means"  # k_means, argmax, ground_truth

if dataset == "synthetic":
    n_question = 50
else:  # dataset is assist2009_updated
    n_question = 110

with open("result/skill_embedding.txt", "r") as f:
    data = []
    for idx, line in enumerate(f.readlines()):
        if idx > n_question * 2:
            break

        if idx % 2 == 0:
            pass
        else:
            data.append(
                np.array(list(map(lambda s: float(s), line.strip().split(',')))))
    data = np.array(data)

if label_type == "k_means":
    kmeans_model = KMeans(n_clusters=4, random_state=1).fit(data)
    label_list = kmeans_model.labels_
elif label_type == "argmax":
    label_list = []
    for d in data:
        label_list.append(np.argmax(d))
    label_list = np.array(label_list)
else:  # label type is ground_truth
    if dataset == "synthetic":
        label_list = np.array([3, 2, 3, 2, 3, 4, 0, 2, 3, 0, 1, 0, 1, 4, 3, 3, 0, 1, 4, 4, 0, 4, 4,
                               1, 1, 0, 0, 4, 2, 0, 4, 4, 1, 2, 4, 0, 3, 0, 1, 2, 2, 0, 3, 3, 2, 3, 2, 3, 2, 1])
    else:  # dataset is assist2009_updated
        label_dict = {}
        label_list = []
        with open("data/assist2009_updated/clustered_skill_name.txt", "r") as f:
            for line in f.readlines():
                row = line.strip().split("\t")
                # row[0]: label, row[1]: skill_id, row[2]: skill_name
                label_dict[int(row[1])] = (int(row[0]), row[2])

            for i in range(1, n_question + 1):
                label_list.append(label_dict[i][0])  # lable
        label_list = np.array(label_list)

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
cluster_id = {}
for skill_id, label in enumerate(label_list):
    skill_id = skill_id + 1
    if label not in cluster_id:
        cluster_id[label] = []
    cluster_id[label].append(skill_id)

for label in cluster_id:
    print(cluster_id[label], "\n")

# %% [markdown]
# # Metrics

# %%
print("Dunn")
k_list = {}
for d, label in zip(data, label_list):
    if label not in k_list:
        k_list[label] = []
    k_list[label].append(d)
print(dunn(list(k_list.values())))

print("Silhouettes")
metrics.silhouette_score(data, label_list, metric='euclidean')
