from surprise import KNNBaseline
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
from surprise.model_selection import GridSearchCV
import os

uitem_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.item')

name_to_rids = {}
rid_to_names = {}

with open(uitem_path, 'r', encoding='ISO-8859-1') as f:
  for line in f:
    words = line.split('|')
    name_to_rids[words[1]] = words[0]
    rid_to_names[words[0]] = words[1]
  
# load dataset
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()


sim_options = {
  'name': 'pearson_baseline',
  'user_based': False,
}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

def inner_iid_to_name(iid):
  r_iid = trainset.to_raw_iid(iid)
  return rid_to_names[r_iid]

def get_k_top(name, k):
  iid = name_to_rids[name]
  iid = trainset.to_inner_iid(iid)
  close_iids = algo.get_neighbors(iid, k)
  return map(inner_iid_to_name, close_iids)

move_names = get_k_top('Toy Story (1995)', k = 10)
print('The 10 nearest neighbors of Toy Story are:')
for name in move_names:
  print(name)
