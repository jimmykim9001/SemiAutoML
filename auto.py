from autosklearn.regression import AutoSklearnRegressor
from autosklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, \
        GridSearchCV
import pandas as pd, numpy as np
import IPython
class Model:
    def __init__(self, df, cols, ycol, warn_cols = 100, save_cols = [], \
            str_action = 'dummies', preprocess_x = None, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5):
        assert(len(split) == 3)
        self.cv = cv
        self.split = split
        self.cols = cols
        self.df = df.loc[df[cols].dropna().index][cols + [ycol]] # remove na values only by cols used

        self.Y = self.df[ycol].to_numpy()
        if callable(preprocess_y):
            self.Y = preprocess_y(self.Y)
        self.Y = self.Y.reshape(self.Y.shape[0], 1)
        self.df.drop(ycol, axis = 1, inplace = True)

        self.df, self.saved_df = Model.select_cols(self.df, cols, warn_cols, \
                save_cols, str_action)
        self.X = self.df.to_numpy()
        if callable(preprocess_x):
            self.X = preprocess_x(self.X)

        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(self.X, \
                self.Y, test_size = split[-1])

    def select_cols(df, cols, warn_cols = 100, save_cols = [], str_action = 'dummies'):
        def throw_num_unique_warning(colname, pd_series):
            assert(pd_series.dtype == "object")
            if pd_series.nunique() > warn_cols:
                warnings.warn(f"{colname} has more than {warn_cols} unique vals")
        def get_object_cols():
            fin = []
            for colname, type_col in df.dtypes.iteritems():
                if type_col == "object":
                    fin.append(colname)
                    throw_num_unique_warning(colname, df[colname])
            return fin
        if len(set(save_cols).intersection(set(cols))) > 0:
            raise Exception("The arguments cols and save_cols should have no columns in common")
        saved_df = df[save_cols]
        df = df[cols]
        if str_action == 'dummies':
            df = pd.get_dummies(df, drop_first = True, prefix = get_object_cols())
        return df, saved_df
res = pd.read_csv("nbastats2018-2019.csv")
res = res[res["Salary"] != "-"]
res["Salary"] = res["Salary"].astype('int64')
colnames = [elem for elem in res.columns if elem != "Name" and elem != "Salary"]
model = Model(res, colnames, "Salary", preprocess_y = np.log)
regressor = AutoSklearnRegressor(time_left_for_this_task = 420, per_run_time_limit = 60)\
        .fit(model.Xtrain, model.Ytrain.flatten(), metric = metrics.mean_squared_error)
print('finished')
IPython.embed()
