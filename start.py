import numpy as np, matplotlib.pyplot as plt, pandas as pd, xgboost as xgb
import warnings, itertools, pickle
from sklearn.model_selection import train_test_split, cross_val_score, \
        GridSearchCV, KFold
# from sklearn.linear_model import LinearRegression
from sklearn import linear_model, svm, neighbors, ensemble, metrics, base
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from textwrap import wrap
from IPython import embed
from pytorch_tabnet.tab_model import TabNetRegressor
from joblib import dump, load


def remove_outliers(x):
    perc_25, perc_75 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = perc_75 - perc_25
    return np.min(x), np.min([perc_75 + iqr * 1.5, np.max(x)])

class ModifiedKFoldSearch:
    """
    General KFoldSearch Algorithm. Model_Func is a function that takes in a 
    dictionary of parameters, and generates a model. First used for TabNet.
    Saves best_estimator_, best_params_
    """
    def __init__(self, model, cv, parameters, input_valid = False, \
            metric_func = metrics.mean_squared_error, feat_params = []):
        """
        @param: model (sklearn model) must have fit/transform/set_params
        @param: cv (int) number of folds or kfold object
        @param: parameters (dict): dictionary of parameters. It'll generate cartesian product
            of all the values to generate all possible set of hyperparameters
        @param: input_valid (bool) if true, model.fit inputs validation data
        @param: metric (func) from sklearn.metrics
        @returns: instance
        """
        if isinstance(cv, int):
            self.kfold = KFold(cv)
        else:
            self.kfold = cv
        self.n_splits = self.kfold.get_n_splits()

        # feature selection
        if len(feat_params) > 0:
            assert(len(feat_params) == self.n_splits)
            self.feat_params = feat_params
        else:
            self.feat_params = [{} for i in range(self.n_splits)]

        self.metric_func = metric_func
        self.input_valid = input_valid
        self.parameters = parameters
        self.model = model

    def fit(self, X, Y):
        """
        Self explanatory
        @param: X (np.ndarray)
        @param: Y (np.ndarray)
        @returns: results (pd.DataFrame) with all cv results
        """
        assert(X.shape[0] == Y.shape[0])
        self.splits = list(self.kfold.split(X))

        # set up results dictionary
        results = {"params": [], "mean_test_score": []}
        for param_title in self.parameters.keys():
            results[f"param_{param_title}"] = []
        for i in range(self.kfold.get_n_splits()):
            results[f"test{i}_score"] = []

        self.best_estimator_, self.best_score_, self.best_params_ = None, np.inf, None
        for params in itertools.product(*self.parameters.values()):
            # set param 
            param_dict = {}
            for param_idx, param_title in enumerate(self.parameters.keys()):
                results[f"param_{param_title}"].append(params[param_idx])
                param_dict[param_title] = params[param_idx]

            results["params"].append(param_dict)
            # perform split
            models, metrics = [], []
            for split_idx, elem in enumerate(self.splits):
                train_idx, valid_idx = elem
                # model = self.model_func(param_dict, self.feat_params[split_idx])
                model = base.clone(self.model)
                model = model.set_params(**{**param_dict, **self.feat_params[split_idx]})

                Xtrain, Xtest = X[train_idx], X[valid_idx]
                Ytrain, Ytest = Y[train_idx], Y[valid_idx]

                if self.input_valid:
                    model.fit(Xtrain, Ytrain.flatten(), Xtest, Ytest.flatten())
                else:
                    model.fit(Xtrain, Ytrain.flatten())

                pred = model.predict(Xtest)
                metric = self.metric_func(Ytest, pred) 

                # bookkeeping
                models.append(model)
                metrics.append(metric)
                results[f"test{split_idx}_score"].append(metric)

            avg_score = np.average(metrics)
            results["mean_test_score"].append(avg_score)
            if avg_score < self.best_score_:
                self.best_score_ = avg_score
                self.best_estimator_ = models[np.argmin(metrics)]
                self.best_params_ = param_dict
        return pd.DataFrame(results)

class Model:
    has_feat_imp = False
    prefix = ''
    def __init__(self, df, cols, ycol, warn_cols = 100, save_cols = [], \
            str_action = 'dummies', preprocess_x = None, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, path = ''):
        assert(len(split) == 3)
        self.cv = cv
        self.split = split
        self.cols = cols
        self.cv_obj = None
        self.path = f"{path}{type(self).__name__}"
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
                self.Y, test_size = split[-1], random_state = 0)

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

    def predict(self, X):
        if 'predictor' not in self.__dict__.keys():
            raise Exception('Model needs a predictor')
        return self.predictor.predict(X)

    def score(self, post_process = lambda x: x):
        if self.cv_obj:
            return post_process(self.cv_obj.best_score_)
        else:
            raise Exception("Score function not written")

    def best_params(self):
        if self.cv_obj:
            return self.cv_obj.best_params_
        else:
            raise Exception("Best Params function not written")
    def get_param(self, param):
        if 'results' in self.__dict__.keys():
            return self.results[f'param_{self.prefix}__{param}']
        else:
            raise Exception("Results has not been creataed")

    def generate_type1_text(self):
        text = ""
        if not self.feat_sel:
            with open(f'analyze_models/{type(self).__name__}.md', 'r') as f:
                whole_text = f.read()
                all_strs = [f"{self.score():.4f}", f"{self.best_params()}", \
                        f"{self.path}"]
                text = whole_text.format(*all_strs)
        else:
            with open(f'analyze_models/{type(self).__name__}Feat.md', 'r') as f:
                whole_text = f.read()
                all_strs = [f"{type(self.feat_sel).__name__}", f"{self.score():.4f}", f"{self.best_params()}", \
                        f"{self.path}"]
                text = whole_text.format(*all_strs)
        return text

    def generate_code_md(self):
        with open("analyze_models/code_md.py", "r") as f:
            return f.read().format(self.path, self.path)[:-1]

    def set_paths(self):
        if self.feat_sel == None:
            self.paths = {
                'text_md': f'analyze_models/{type(self).__name__}.md',
                'results': f'{self.path}_cvresults.csv',
                'hyperparam':  f'{self.path}_hyperparam.png',
                'feat_imp': f'{self.path}_feature_importance.png',
                'estimator': f'{self.path}_bestestimator.joblib',
                'history': f'{self.path}_history.png'
            }
        else:
            self.paths = {
                'text_md': f'analyze_models/{type(self).__name__}Feat.md',
                'results': f'{self.path}_feat_cvresults.csv',
                'hyperparam':  f'{self.path}_feat_hyperparam.png',
                'feat_imp': f'{self.path}_feature_importance_feat.png',
                'estimator': f'{self.path}_feat_bestestimator.joblib',
                'history': f'{self.path}_feat_history.png'
            }


class LeastSquares(Model):
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, parameters = {}, feat_sel = None, \
            feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", None, \
                preprocess_y, split, path = path)
        self.feat_sel = feat_sel
        if feat_sel:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                ('feature_selection', SelectFromModel(feat_sel)),
                ('least_squares', linear_model.LinearRegression())
            ])
        else:
            self.pipeline = linear_model.LinearRegression()
        self.set_paths()

        self.cv_obj = ModifiedKFoldSearch(self.pipeline, cv, parameters, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain.flatten())

    def save(self):
        dump(self.cv_obj.best_estimator_, self.paths['estimator'])
        self.results.to_csv(self.paths['results'])

    def score(self):
        return np.min(self.results['mean_test_score'])

    def generate_text_md(self):
        text = ""
        format_strs = [f"{self.score():.4f}"]
        with open(self.paths['text_md'], 'r') as f:
            if self.feat_sel:
                formats = [f"{type(self.feat_sel).__name__}", f"{self.score():.4f}"]
                text = f.read().format(*formats)
            else:
                text = f.read().replace("{}", f"{self.score():.4f}")
        return text


class RidgeRegression(Model):
    prefix = 'ridge'
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, parameters = {}, feat_sel = None, \
            feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", \
                None, preprocess_y, split, path = path)
        self.feat_sel = feat_sel
        if feat_sel:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                ('feature_selection', SelectFromModel(feat_sel)),
                (self.prefix, linear_model.Ridge())
            ])
        else:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()), 
                (self.prefix, linear_model.Ridge())
            ])

        self.set_paths()
        self.cv_obj = ModifiedKFoldSearch(self.pipeline, cv, parameters, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain.flatten())

    def save(self):
        plt.title("RidgeRegression: Effect of Lambda on CV MSE")
        plt.plot(self.get_param('alpha'), self.results['mean_test_score'])
        plt.xscale('log')
        plt.xlabel("log lambda")
        plt.ylabel("CV MSE")
        plt.savefig(self.paths['hyperparam'])
        self.results.to_csv(self.paths['results'])
        dump(self.cv_obj.best_estimator_, self.paths['estimator'])
        plt.close()

    def generate_text_md(self):
        return self.generate_type1_text()

class LassoRegression(Model):
    prefix = 'lasso'
    has_feat_imp = True
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, max_iter = 3000, parameters = {}, \
            feat_sel = None, feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", \
                None, preprocess_y, split, path = path)
        self.max_iter = max_iter
        self.feat_sel = feat_sel
        if feat_sel:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                ('feature_selection', SelectFromModel(feat_sel)),
                (self.prefix, linear_model.Lasso())
            ])
        else:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                (self.prefix, linear_model.Lasso(max_iter = self.max_iter))
            ])
        self.set_paths()

        self.cv_obj = ModifiedKFoldSearch(self.pipeline, cv, parameters, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain.flatten())

        self.all_cols = np.array(self.df.columns)
        self.best_coeffs = self.cv_obj.best_estimator_['lasso'].coef_

    def save(self):
        plt.title("Lasso: Effect of Penalization Factor (lambda) on CV MSE")
        plt.plot(self.get_param('alpha'), self.results['mean_test_score'])
        plt.xscale('log')
        plt.xlabel("lambda")
        plt.ylabel("CV MSE")
        plt.savefig(self.paths['hyperparam'])
        self.results.to_csv(self.paths['results'])
        dump(self.cv_obj.best_estimator_, self.paths['estimator'])
        plt.close()

    def get_feature_importances(self):
        pd_dict = {'column': self.all_cols}
        one_zero = np.zeros((len(pd_dict['column']),))
        for idx in self.best_coeffs.nonzero():
            one_zero[idx] = 1
        pd_dict[f"{self.prefix}__feat_imp"] = one_zero
        df = pd.DataFrame(pd_dict)
        df.set_index('column', inplace = True)
        return df

    def get_used_columns(self):
        return self.all_cols[self.best_coeffs.nonzero()]

    def get_dropped_columns(self):
        return self.all_cols[np.where(self.best_coeffs == 0)[0]]

    def generate_feat_params(self):
        feat_params = []
        for i in range(self.cv_obj.n_splits):
            idx_best_params = np.argmin(self.results[f'test{i}_score'].to_numpy())
            params = self.results['params'].iloc[idx_best_params]

            # set new params (with feature_selection)
            new_params = {}
            for key in params:
                new_params[f"feature_selection__{key.replace(self.prefix, 'estimator')}"] = params[key]

            feat_params.append(new_params)
        return feat_params

    def generate_text_md(self):
        return self.generate_type1_text()

class SVR(Model):
    prefix = 'svr'
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, parameters = {}, feat_sel = None, \
            feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", \
                None, preprocess_y, split, path = path)
        # hyperparameters: kernel, C
        self.feat_sel = feat_sel
        if feat_sel:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                ('feature_selection', SelectFromModel(feat_sel)),
                (self.prefix, svm.SVR())
            ])
        else:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                (self.prefix, svm.SVR()) 
            ])
        self.set_paths()
        self.cv_obj = ModifiedKFoldSearch(self.pipeline, cv, parameters, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain.flatten())

    def save(self):
        kernels = self.get_param('kernel').unique()
        gammas = self.get_param('gamma').unique()
        fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
        fig.suptitle('SVR')
        for idx, kernel in enumerate(kernels):
            axes[idx].set_title(kernel)
            axes[idx].set_xscale('log')
            axes[idx].set_xlabel('C')
            axes[idx].set_ylabel('CV MSE')
            for gamma in gammas:
                selector = np.logical_and(self.get_param('kernel') == kernel, \
                        self.get_param('gamma') == gamma)
                axes[idx].plot(self.get_param('C')[selector], \
                        self.results[selector]['mean_test_score'], label = f"gamma {gamma}")
            axes[idx].legend()
        plt.savefig(self.paths['hyperparam'])
        self.results.to_csv(self.paths['results'])
        dump(self.cv_obj.best_estimator_, self.paths['estimator'])
        plt.close()

    def generate_text_md(self):
        return self.generate_type1_text()

class KNNRegression(Model):
    prefix = 'kneighborsregressor'
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, parameters = {}, feat_sel = None, \
            feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", \
                None, preprocess_y, split, path = path)
        self.feat_sel = feat_sel
        if feat_sel:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                ('feature_selection', SelectFromModel(feat_sel)),
                (self.prefix, neighbors.KNeighborsRegressor()) 
            ])
        else:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                (self.prefix, neighbors.KNeighborsRegressor())
            ])
        self.set_paths()
        self.cv_obj = ModifiedKFoldSearch(self.pipeline, cv, parameters, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain.flatten())

        #TODO: you should implement mahalanois distance this requires you to write
        # your own Cross Validation function however

    def save(self):
        weights = self.get_param('weights').unique()
        plt.ylabel("CV MSE")
        plt.xlabel("K")
        plt.title("KNN Hyperparameters")
        for weight in weights:
            selector = self.get_param('weights') == weight
            plt.plot(self.get_param('n_neighbors')[selector], \
                    self.results[selector]['mean_test_score'], label = weight)
        plt.legend()
        plt.savefig(self.paths['hyperparam'])
        self.results.to_csv(self.paths['results'])
        dump(self.cv_obj.best_estimator_, self.paths['estimator'])
        plt.close()
    def generate_text_md(self):
        return self.generate_type1_text()

class RandomForestRegression(Model):
    prefix = 'randomforestregressor'
    has_feat_imp = True
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, parameters = {}, feat_sel = None, \
            feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", \
                None, preprocess_y, split, path = path)
        self.feat_sel = feat_sel
        if feat_sel: # note you shouldn't be using this (not recommended)
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                ('feature_selection', SelectFromModel(feat_sel)),
                (self.prefix, ensemble.RandomForestRegressor(bootstrap = True, oob_score = True)) 
            ])
        else:
            self.pipeline = Pipeline([
                (self.prefix, ensemble.RandomForestRegressor(bootstrap = True, oob_score = True))
            ])
        self.set_paths()
        self.cv_obj = ModifiedKFoldSearch(self.pipeline, cv, parameters, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain.flatten())
        #TODO: rewrite function to calculate OOB for cvs

    def save(self, n = 10):
        max_features = self.get_param('max_features').unique()
        plt.title("Random Forest Hyperparameters")
        plt.xlabel("Number of Estimators")
        plt.ylabel("CV MSE")
        for mf in max_features:
            selector = self.get_param('max_features') == mf
            plt.plot(self.get_param('n_estimators')[selector], \
                    self.results[selector]['mean_test_score'], label = mf)
        plt.legend()
        plt.savefig(self.paths['hyperparam'])
        self.results.to_csv(self.paths['results'])
        dump(self.cv_obj.best_estimator_, self.paths['estimator'])
        plt.close()

        plt.title("Feature Importance")
        plt.ylabel("Feature Importance (Gini)")
        all_cols = list(self.df.columns)

        # TODO: fix this garbage
        feat_imp = np.sort(self.cv_obj.best_estimator_[self.prefix].feature_importances_)[::-1][:n]
        idx = self.cv_obj.best_estimator_[self.prefix].feature_importances_.argsort()[::-1][:n]
        plt.bar(list(range(0, n)), feat_imp)
        plt.xticks(list(range(0, n)), ['\n'.join(wrap(all_cols[elem], 5)) for elem in idx])
        plt.savefig(self.paths['feat_imp'])
        plt.close()

    def get_feature_importances(self):
        pd_dict = {
            'column': list(self.df.columns),
            'randfor__feat_imp': list(self.cv_obj.best_estimator_[self.prefix].feature_importances_)
        }
        importance_df = pd.DataFrame(pd_dict)
        importance_df.set_index('column', inplace = True)
        return importance_df

    def generate_text_md(self):
        text = ""
        with open(self.paths['text_md'], 'r') as f:
            whole_text = f.read()
            all_strs = []
            if self.feat_sel:
                all_strs.append(f"{type(self.feat_sel).__name}")
            all_strs += [f"{self.score():.4f}", f"{self.best_params()}", f"{self.path}"]
            all_strs += all_strs[-2:]
            text = whole_text.format(*all_strs)
        return text
class GradientBoostedTreeRegression(Model):
    has_feat_imp = True
    prefix = 'xgbregressor'
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, parameters = {}, feat_sel = None, \
            feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", \
                None, preprocess_y, split, path = path)
        self.dtrain = xgb.DMatrix(self.Xtrain, label = self.Ytrain)

        self.parameters = {f'{self.prefix}__n_estimators': [50, 80, 100, 200, 300, 500], \
                f'{self.prefix}__max_depth': [2, 3, 4, 5], \
                f'{self.prefix}__learning_rate': [0.01, 0.05, 0.1]}
        self.feat_sel = feat_sel
        if feat_sel:
            self.pipeline = Pipeline([
                ('preprocessing', MinMaxScaler()),
                ('feature_selection', SelectFromModel(feat_sel)),
                (self.prefix, xgb.XGBRegressor())
            ])
        else:
            self.pipeline = Pipeline([
                (self.prefix, xgb.XGBRegressor()) 
            ])
        self.set_paths()
        self.cv_obj = ModifiedKFoldSearch(self.pipeline, cv, self.parameters, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain.flatten())

    def save(self):
        n_rows = int(len(self.parameters[f'{self.prefix}__learning_rate']) / 2 + 0.5)
        fig, axes = plt.subplots(n_rows, 2, figsize = (10, n_rows * 5), sharex = True)
        fig.suptitle("GB Tree Hyperparameters")
        look_at = [f'param_{self.prefix}__max_depth', 'mean_test_score']
        all_lrs = self.get_param('learning_rate').unique()
        x_vals = self.get_param('n_estimators').unique()
        for lr_idx, lr in enumerate(all_lrs):
            res = self.results[self.get_param('learning_rate') == lr]
            res = res[look_at].groupby([f'param_{self.prefix}__max_depth'])\
                    ['mean_test_score'].apply(list).reset_index()
            ax = axes[lr_idx // 2, lr_idx % 2] 
            ax.set_title(f"lr{lr}")
            all_ys = []
            for depth_idx, elem in res.iterrows():
                curr_ys = np.array(elem['mean_test_score'])
                # lol this next line is atrocious LMAO
                ax.plot(x_vals, curr_ys, label = f"max depth {elem[f'param_{self.prefix}__max_depth']}")
                all_ys.append(curr_ys)
            ax.legend() 
            ax.set_xlabel("Number of Estimators")
            ax.set_ylabel("CV MSE")
            ax.set_ylim(remove_outliers(np.array(all_ys).flatten()))
        plt.savefig(self.paths['hyperparam'])
        self.results.to_csv(self.paths['results'])
        dump(self.cv_obj.best_estimator_, self.paths['estimator'])
        plt.close()

        #TODO: this assumes we're only looking at param_max_depth and n_estimators
        #TODO: this assumes that n_estimators is ordered 
        #TODO: fix this garbage

    def get_feature_importances(self):
        pd_dict = {
            'column': list(self.df.columns),
            'xgboost__feat_imp': list(self.cv_obj.best_estimator_[self.prefix].feature_importances_)
        }
        importance_df = pd.DataFrame(pd_dict)
        importance_df.set_index('column', inplace = True)
        return importance_df

    def generate_text_md(self):
        return self.generate_type1_text()

class TabNetRegression(Model):
    prefix = 'tabnet'
    def __init__(self, df, cols, ycol, warn_cols = 100, preprocess_y = None, \
            split = (0.7, 0.2, 0.1), cv = 5, parameters = {}, feat_sel = None, \
            feat_params = [], path = ''):
        super().__init__(df, cols, ycol, warn_cols, [], "dummies", \
                None, preprocess_y, split, path = path)
        self.feat_sel = feat_sel
        self.set_paths()
        self.cv_obj = ModifiedKFoldSearch(TabNetRegression.TabNetModel(feat_sel = feat_sel), 5,\
                parameters, input_valid = True, feat_params = feat_params)
        self.results = self.cv_obj.fit(self.Xtrain, self.Ytrain)

    def save(self):
        history = self.cv_obj.best_estimator_.tabnet.history
        fig, axes = plt.subplots(1, 2, figsize = (10, 5))
        fig.suptitle("History")
        axes[0].plot(history['train']['loss'], label = 'train_loss')
        axes[0].plot(history['valid']['loss'], label = 'valid_loss')
        axes[0].legend()
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[1].plot(history['train']['metric'], label = 'train_metric')
        axes[1].plot(history['valid']['metric'], label = 'valid_metric')
        axes[1].legend()
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Metric")
        plt.savefig(self.paths['history'])
        self.results.to_csv(self.paths['results'])
        pickle.dump(self.cv_obj.best_estimator_, open(self.paths['estimator'], 'wb'))
        plt.close()

    def generate_text_md(self):
        return self.generate_type1_text()

    class TabNetModel:
        def __init__(self, feat_sel = None):
            if feat_sel is not None:
                self.pipeline = Pipeline([
                    ('preprocessing', MinMaxScaler()),
                    ('feature_selection', SelectFromModel(feat_sel))
                ])
            else:
                self.pipeline = Pipeline([
                    ('preprocessing', MinMaxScaler())
                ])
            self.feat_sel = feat_sel
            self.params = {'feat_sel': feat_sel}
        def set_params(self, **kwargs):
            tabnet_params, proc_params = {}, {}
            for key in kwargs.keys():
                if key == 'feat_sel':
                    self = TabNetRegression.TabNetModel.__init__(kwargs[key])
                elif key.startswith(TabNetRegression.prefix):
                    tabnet_params[key.replace(TabNetRegression.prefix + "__", "")] = kwargs[key]
                else:
                    proc_params[key] = kwargs[key]
            self.tabnet = TabNetRegressor(**tabnet_params)
            self.pipeline.set_params(**proc_params)
            self.params = {**tabnet_params, **proc_params}
            return self
        def get_params(self, deep = False):
            return self.params
        def fit(self, Xtrain, Ytrain, Xvalid, Yvalid):
            if self.feat_sel is not None:
                self.pipeline.fit(Xtrain, Ytrain)
            else:
                self.pipeline.fit(Xtrain)
            Xtrain_scaled, Xvalid_scaled = self.pipeline.transform(Xtrain), self.pipeline.transform(Xvalid)
            self.tabnet.fit(Xtrain_scaled, Ytrain.flatten(), Xvalid_scaled, Yvalid.flatten())
        def predict(self, X):
            return self.tabnet.predict(self.pipeline.transform(X))

conf_to_models = {
    'LINEAREG': LeastSquares,
    'RIDGEREG': RidgeRegression,
    'LASSO': LassoRegression,
    'SVR': SVR,
    'KNN': KNNRegression,
    'RANDFOR': RandomForestRegression,
    'GBTREE': GradientBoostedTreeRegression,
    'TABNET': TabNetRegression
}

def read_config(filepath):
    config = configparser.ConfigParser()
    config.optionxform = str # keep uppercase letters
    config.read(filepath)

    selected_models = [conf_to_models[x] for x in config.sections()]
    fin = []
    for idx, section in enumerate(config.sections()):
        search_type = config[section]['search']
        assert(search_type == 'grid' or search_type == 'tpe')
        params = {}
        for key in config[section].keys():
            if key == 'search':
                continue
            if key + '_scale' in config[section]:
                np_func = np.logspace
                if config[section][key + '_scale'] == 'linear':
                    np_func = np.arange
                args = [int(x) for x in config[section][key].split(',')]
                params[f"{selected_models[idx].prefix}__{key}"] = np_func(*args)
            elif '_scale' in key:
                continue
            else:
                params[f"{selected_models[idx].prefix}__{key}"] = config[section][key].split(',')
        fin.append(params)
    return selected_models, fin

import nbformat as nbf
def generate_phase_two_jupyter(path, models, feat_imps):
    text1 = "# Phase Two (Preliminary Models)\n"
    cells = [nbf.v4.new_markdown_cell(text1)]
    with open('analyze_models/start_code.py', 'r') as f:
        cells.append(nbf.v4.new_code_cell(f.read()[:-1]))

    for model in models:
        cells += [nbf.v4.new_markdown_cell(model.generate_text_md()), \
                nbf.v4.new_code_cell(model.generate_code_md())]

    with open('analyze_models/feature_importances.py', 'r') as f:
        cells.append(nbf.v4.new_code_cell(f.read().format(f'{path}feat_imp.csv')[:-1]))

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nbf.write(nb, 'test.ipynb')

import argparse, os, datetime, configparser
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser("Run Regression Models")
    parser.add_argument('path', help = "path to csv", type=str, action = "store")
    parser.add_argument('column', help = "column to predict", type=str, action = "store")
    parser.add_argument('config', help = "path to config file", type = str, action = "store")
    parser.add_argument('--no_save', help = "don't save results", action = 'store_false')
    parser = parser.parse_args()

    path, column, config, save = parser.path, parser.column, parser.config, parser.no_save
    selected_models, hyper_params = read_config(config)

    # create output directory
    today_str = datetime.datetime.today().strftime("%m_%d_%Hh_%Mm")
    end_path = f"models/{path.split('/')[-1].replace('.csv', '')}_run_{today_str}/"
    if not os.path.exists(end_path) and save:
        os.mkdir(end_path)

    # preprocessing
    res = pd.read_csv(path)
    res = res[res[column] != "-"]
    res[column] = res[column].astype('int64')
    colnames = [elem for elem in res.columns if elem != "Name" and elem != column]

    cv = KFold(n_splits = 5, random_state = 0, shuffle = True)

    # running models/saving preliminary information
    models, feat_imps = [], []
    feat_params_lasso = []
    for idx, reg_class in enumerate(selected_models):
        rc = reg_class(res, colnames, column, preprocess_y = np.log, parameters = \
                hyper_params[idx], cv = cv, path = f"{end_path}")
        if reg_class == LassoRegression:
            feat_params_lasso = rc.generate_feat_params()
        if save:
            rc.save()
        if reg_class.has_feat_imp:
            feat_imps.append(rc.get_feature_importances())
        models.append(rc)
    # LASSO feature selection
    # lasso_models = []
    # for idx, reg_class in enumerate(selected_models):
    #     if reg_class is not GradientBoostedTreeRegression and reg_class is not RandomForestRegression \
    #             and reg_class is not LassoRegression:
    #         rc = reg_class(res, colnames, column , preprocess_y = np.log, parameters = \
    #                 hyper_params[idx], cv = cv, feat_sel = linear_model.Lasso(), feat_params = \
    #                 feat_params_lasso)
    #         if save:
    #             rc.save(f"{end_path}{type(rc).__name__}")
    #         lasso_models.append(rc)

    # feat_imps = pd.concat(feat_imps, axis = 1)
    # feat_imps.to_csv(f"{end_path}feat_imp.csv", index = False)
    # generate_phase_two_jupyter(end_path, models + lasso_models)
    generate_phase_two_jupyter(end_path, models, feat_imps)
