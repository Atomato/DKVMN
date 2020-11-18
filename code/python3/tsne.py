# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

with open("correlation_weight.txt", "r") as f:
    lines = f.readlines()

data = []
n_question = 110
for idx, line in enumerate(lines):
    if idx > n_question * 2:
        break

    if idx % 2 == 0:
        pass
    else:
        # print((idx + 1) / 2)
        data.append(np.array(list(map(lambda s: float(s), line.strip().split(',')))))
data = np.array(data)

# label = []
# for d in data:
#     label.append(np.argmax(d))

# label = np.array(label)
# print(label)

# Ground truth
label = [3, 2, 3, 2, 3, 4, 0, 2, 3, 0, 1, 0, 1, 4, 3, 3, 0, 1, 4, 4, 0, 4, 4, 1, 1, 0, 0, 4, 2, 0, 4, 4, 1, 2, 4, 0, 3, 0, 1, 2, 2, 0, 3, 3, 2, 3, 2, 3, 2, 1]

model = TSNE(n_components=2)
transformed = model.fit_transform(data)

xs = transformed[:,0]
ys = transformed[:,1]

# %%
plt.scatter(xs,ys,c=label)

for i in range(n_question):
    plt.text(xs[i], ys[i], i+1)
plt.show()
# %%
cluster = {0: [], 1: [], 2: [], 3: [], 4: []}
for i, v in enumerate(label):
    cluster[v].append(i+1)

for k in cluster:
    print(cluster[k])
