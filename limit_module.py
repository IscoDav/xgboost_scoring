
import pandas as pd
import numpy as np
import pickle
import pandas.io.sql as psql

import warnings
# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

import xgboost as xgb
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class limit_model():

    def __init__(self, model_file):

        # read the 'model', files which were saved
        self.reg = pickle.load(open(model_file, 'rb'))
        self.before = None
        self.data = None
        self.cust_id = None
        self.check = None


    # take a data file (*.csv) and preprocess it in the same way as preprocessing notebook

    def cleaning_and_prediction(self):

        # import the data
        df = pd.read_csv('data.csv')

        indexes = df[df['monthly_income'] < 1000].index
        # Customers that does not have income but got limits
        df.drop(df.index[indexes], inplace=True)

        df = df.drop('customer_id', axis = 1 )

        self.data = df.copy()

        self.X_test = df.drop(['total_limit'], axis = 1)
        self.y_test = df['total_limit']

        self.test_check = self.reg.predict(xgb.DMatrix(self.X_test))

        return df


    def compare(self):

        check = pd.DataFrame()
        check['customer_id'] = self.cust_id
        check['actual'] = self.y_test
        check['predicted'] = self.test_check
        check['differance'] = check['actual'] - check['predicted']

        self.check = check.copy()
        return check.round()

    def differance(self):

        diff = self.check
        diff = diff[diff['total_limit'] < -500000]


    def before_filtering(self):
        return self.before

    def viz(self):
        sns.regplot(x= self.y_test  ,y=self.test_check ,order=1,
                    scatter_kws={"color": "darkblue", 'alpha': 0.2},
                    line_kws={"color": "red"})
        plt.show()


    def metrics(self):

        y_test = self.data['total_limit']

        rmse = np.sqrt(mean_squared_error(y_test, self.test_check))
        mae = mean_absolute_error(y_test, self.test_check)

        X_addC = sm.add_constant(self.test_check)
        result = sm.OLS(y_test, X_addC).fit()

        results = pd.DataFrame({'R2' : [result.rsquared], "R2adj": [result.rsquared_adj],
                                'RMSE': [rmse] , 'MAE': [mae]})

        return results




#%%
