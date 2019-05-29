from pathlib import Path

from br_data_reader import load_br_data
import numpy as np

# prob = [3, 2, 1, 1, 1]
# prob = [float(i)/sum(prob) for i in prob]
# aa = np.random.choice(5, 5, replace=False, p=prob)
from eval import eval_ranking

encoder = "bert"
train_type = "cross"
# train_type = "seq"
max_seq_len = 300
toy_data = False

PROJECT = "HIVE"
# DATA_ROOT = Path("../dataset/bugtriaging/") / PROJECT
DATA_ROOT = Path("../data") / PROJECT
ASSIGNEE_ID_FILE = DATA_ROOT / "assignee_id.txt"

reader, train_ds, test_ds = load_br_data(DATA_ROOT, ASSIGNEE_ID_FILE, encoder, train_type, max_seq_len, toy_data)

prob_dict = {}
# generate the frequency of each authors
for instance in train_ds:
    authors = instance["label"].metadata
    for id in authors:
        if id in prob_dict:
            prob_dict[id] += 1
        else:
            prob_dict[id] = 1
sorted_prob = sorted(prob_dict.items(), key=lambda item: item[0])
prob = [i[1] for i in sorted_prob]

prob_str_list = [str(i) for i in prob]
print("\t".join(prob_str_list))
prob = [float(i)/sum(prob) for i in prob]
num_authors = len(prob)

truth = [i["label"].metadata for i in test_ds]

for _ in range(100):
    test_preds = [np.random.choice(num_authors, num_authors, replace=False, p=prob).tolist() for i in test_ds]
    test_preds = np.array([np.array(xi) for xi in test_preds])
    eval_ranking(test_preds, truth)
pass