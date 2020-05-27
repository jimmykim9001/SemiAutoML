# class LinearModel(Model):
#     def __init__(self, df, cols, ycol, warn_cols = 100, save_cols = [], preprocess_y = None):
#         """
#         Initializes Linear Model (but does not run OLS)
#         @param: df (pd.DataFrame) including the ycolumn and all the predictors
#         @param: cols (list of str) of all cols you'd like to use
#         @param: ycol (str) name of col to predict
#         @param: warn_cols (int) if there's a string column, linearmodel will
#             create dummy vars for the string column. If there are more than 
#             warn_cols unique values in the column, a warning will be posted
#         @param: save_cols (list of str) columns you want to save to analyze
#             after OLS is run
#         """
#         super().__init__(df, cols, ycol, warn_cols, save_cols, str_action = 'dummies',\
#                 preprocess_y = preprocess_y)
#         self.yname = ycol
#         self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
#         self.k = self.X.shape[1] - 1 # number of predictors (not including intercept)
#         self.n = self.X.shape[0]
#
#     def OLS(self):
#         self.vif = self.X.T @ self.X
#         self.cov = inv(self.vif)
#         Hprime = self.cov @ self.X.T 
#         self.H = self.X @ Hprime
#         self.bhat = Hprime @ self.Y
#         self.yhat = self.H @ self.Y
#
#         # rss
#         self.residuals = (self.Y - self.yhat).flatten()
#         self.rss = np.inner(self.residuals, self.residuals)
#         # regss
#         regss_tmp = (self.yhat - np.mean(self.Y)).flatten()
#         self.regss = np.inner(regss_tmp, regss_tmp)
#         # tss
#         tss_tmp = (self.Y - np.mean(self.Y)).flatten()
#         self.tss = np.inner(tss_tmp, tss_tmp)
#
#         self.se = (self.rss / (self.n - self.k - 1)) ** (0.5) 
#         self.rsquared = self.regss / self.tss
#         self.coeff_se = ((self.se ** 2) * self.cov.diagonal().reshape((self.k + 1, 1))) ** 0.5
#          
#         # t tests
#         right_t = np.abs(self.bhat / self.coeff_se)
#         self.coeff_p = t.cdf(right_t, self.n - self.k - 1) - t.cdf(-1 * right_t, self.n - self.k - 1)
#
#         # adjusted residuals
#         self.stan_res = self.residuals / (self.se * (1 - self.H.diagonal()) ** 0.5)
#         self.stud_res = self.stan_res * ( (self.n - self.k - 2) / \
#                 (self.n - self.k - 1 - self.stan_res ** 2))
#
#     def predict(self, X):
#         return X @ self.bhat
#
#     def component_resid(self, col, path = 'comp_res.png'):
#         idx = self.cols.index(col) + 1
#         plt.figure(figsize = (5, 5))
#         plt.title(f"Component({col}) + Residual Plot")
#         plt.xlabel(col)
#         plt.ylabel("Component + Res")
#         xs, ys = self.X[:, idx], self.X[:, idx] * self.bhat[idx] + self.residuals
#         plt.scatter(xs, ys)
#         plt.savefig(path)
#         #TODO: lowess + linear regression on this
#     def studentized_residuals(self, path = 'studentized_residuals.png'):
#         plt.title("OLS Studentized Residual Plot")
#         plt.xlabel(self.yname)
#         plt.ylabel("Studentized Residual")
#         plt.scatter(self.yhat, self.stud_res)
#         plt.savefig(path)
