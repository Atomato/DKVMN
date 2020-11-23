data_path = 'data/assist2009_updated/assist2009_updated_test.csv'
new_data_path = 'data/assist2009_updated/assist2009_updated_cp_test.csv'
with open(data_path, 'r') as f:
    acc = []
    for lineID, line in enumerate(f):
        if lineID % 3 == 2:
            acc.extend(list(map(lambda x: int(x), line.strip().split(','))))
    print('Test accuracy: ', sum(acc)/len(acc))


with open(data_path, 'r') as fr, \
        open(new_data_path, 'w') as fw:
    read_lines = fr.readlines()
    N = len(read_lines)
    for idx in range(int(N/3)):
        read_line = read_lines[idx*3 + 2].strip().split(',')

        lineAcc = list(map(lambda x: int(x), read_line))

        # Accuracy threshold
        if (sum(lineAcc) / len(lineAcc)) > 0.65:
            fw.write(read_lines[idx*3])
            fw.write(read_lines[idx*3 + 1])
            fw.write(read_lines[idx*3 + 2])

with open(new_data_path, 'r') as f:
    acc = []
    for lineID, line in enumerate(f):
        if lineID % 3 == 2:
            acc.extend(list(map(lambda x: int(x), line.strip().split(','))))
    print('Test accuracy (cherry-pick): ', sum(acc)/len(acc))
