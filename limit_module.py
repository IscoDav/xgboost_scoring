
import pandas as pd
import numpy as np
import pickle

import warnings
# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

import xgboost as xgb
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
sns.set()


class limit_model():

    def __init__(self, xgb = 'XGBoost_V_02', lgb = 'LGBM_V_02'):

        # read the 'model', files which were saved
        self.xgb = pickle.load(open(xgb, 'rb'))
        self.lgb = pickle.load(open(lgb, 'rb'))
        self.before = None
        self.data = None
        self.cust_id = None

    # take a data file (*.csv) and preprocess it in the same way as preprocessing notebook

    def cleaning_and_prediction(self):

        # import the data
        df = pd.read_csv('data.csv')

        indexes = df[df['monthly_income'] < 1000].index
        # Customers that does not have income but got limits
        df.drop(df.index[indexes], inplace=True)
        self.cust_id = df['customer_id']

        df = df.drop('customer_id', axis = 1 )

        self.data = df.copy()

        self.X_test = df.drop(['total_limit'], axis = 1)
        self.y_test = df['total_limit']

        self.test_xgb = self.xgb.predict(xgb.DMatrix(self.X_test))
        self.test_lgb = self.lgb.predict(self.X_test)


        return df


    def compare_xgb(self):

        check = pd.DataFrame()
        check['customer_id'] = self.cust_id
        check['actual'] = self.y_test
        check['predicted'] = self.test_xgb
        check['differance'] = check['actual'] - check['predicted']

        return check

    def compare_lgb(self):

        check = pd.DataFrame()
        check['customer_id'] = self.cust_id
        check['actual'] = self.y_test
        check['predicted'] = self.test_lgb
        check['differance'] = check['actual'] - check['predicted']

        return check

    def metrics_xgb(self):

        y_test = self.data['total_limit']

        rmse = np.sqrt(mean_squared_error(y_test, self.test_xgb))
        mae = mean_absolute_error(y_test, self.test_xgb)

        X_addC = sm.add_constant(self.test_xgb)
        result = sm.OLS(y_test, X_addC).fit()

        results = pd.DataFrame({'R2' : [result.rsquared], "R2adj": [result.rsquared_adj],
                                'RMSE': [rmse] , 'MAE': [mae]})

        return results

    def metrics_lgb(self):

        y_test = self.data['total_limit']

        rmse = np.sqrt(mean_squared_error(y_test, self.test_lgb))
        mae = mean_absolute_error(y_test, self.test_lgb)

        X_addC = sm.add_constant(self.test_lgb)
        result = sm.OLS(y_test, X_addC).fit()

        results = pd.DataFrame({'R2' : [result.rsquared], "R2adj": [result.rsquared_adj],
                                'RMSE': [rmse] , 'MAE': [mae]})

        return results


#%%
