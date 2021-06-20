from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV
import pandas as pd


data = Dataset.load_builtin('ml-100k')

#param_grid = {
#    'k': [10, 40, 100, 500, 1000],
#    'min_k': [1, 10, 100],
#    'sim_options': {
#        'name': ['msd', 'cosine', 'pearson'],
#        'min_support': [1, 5, 10, 100],
#        'user_based': [True, False]
#    }
#}

param_grid = {
    'n_factors': [10, 100, 200],
    'n_epochs': [10, 20, 100],
    'biased': [True, False],
    'lr_all': [0.001, 0.005, 0.01, 0.1]
}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['mse', 'rmse', 'mae'],
                  cv=3,
                  n_jobs=8)

gs.fit(data)

df_results = pd.DataFrame.from_dict(gs.cv_results)
df_results.to_csv("gridsearch_svd.csv")

print("mse")
print(gs.best_score['mse'])
print(gs.best_params['mse'])

print("rmse")
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

print("mae")
print(gs.best_score['mae'])
print(gs.best_params['mae'])
