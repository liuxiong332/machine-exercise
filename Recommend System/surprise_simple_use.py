from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# load dataset
data = Dataset.load_builtin('ml-100k')

algo = SVD()

print(cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True))
