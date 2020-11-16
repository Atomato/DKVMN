# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

if __name__ == '__main__':
    with open("correlation_weight.txt", "r") as f:
        lines = f.readlines()

    data = []
    for idx, line in enumerate(lines):
        if idx > 110 * 2:
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
