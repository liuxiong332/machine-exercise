from surprise import SVD
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
from surprise.model_selection import cross_validate
import os

user_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
reader = Reader(line_format='user item rating timestamp', sep='\t')

# load dataset
data = Dataset.load_from_file(user_path, reader=reader)

algo = SVD()

print_perf(cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True))
