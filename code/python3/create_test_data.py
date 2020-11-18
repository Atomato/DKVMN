import csv

with open("../../data/synthetic/naive_c5_q50_s4000_v1_test_graph.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    for j in range(1, 51):
        for i in range(1, 51):
            if i == j: continue
            writer.writerow([2]) # The number of sequence
            writer.writerow([i, j]) # The sequence of exercies id
            writer.writerow([1, 1]) # The sequence of correctness
