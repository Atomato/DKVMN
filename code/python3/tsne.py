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

label = []
for d in data:
    label.append(np.argmax(d))

label = np.array(label)
print(label)

model = TSNE(n_components=2)
transformed = model.fit_transform(data)

xs = transformed[:,0]
ys = transformed[:,1]

# %%
plt.scatter(xs,ys,c=label)

plt.show()
# %%
cluster = {0: [], 1: [], 2: [], 3: [], 4: []}
for i, v in enumerate(label):
    cluster[v].append(i+1)

for k in cluster:
    print(cluster[k])
# %%
