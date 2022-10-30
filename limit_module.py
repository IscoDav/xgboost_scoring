
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


    # take a data file (*.csv) and preprocess it in the same way as preprocessing notebook

    def cleaning_and_prediction(self):

        # import the data
        df = pd.read_csv('data.csv')

        indexes = df[df['monthly_income'] < 1000].index
        # Customers that does not have income but got limits
        df.drop(df.index[indexes], inplace=True)

        # Dropping all duplicate values
        df.drop_duplicates(subset='customer_id', inplace = True)

        # Adding amount_limit and installment to get total limit

        df['total_limit'] = df['amount_limit'] + df['installment_amount']

        #df = df.rename(columns={"expiry_month_count": "period_month_count"})

        # Filtering outliers from less than 20Ml limit and 15Ml income
        df_2 = df[(df['total_limit'] <= 20000000) & (df['monthly_income'] <= 15000000)]

    # Removing clients that has ability to pay from the % of > 60%
        df_2['return_ability'] = (df_2['total_limit'] / df_2['expiry_month_count']) / df_2['monthly_income']

    # Filtering with return ability < 50% for between expiry months 6 and 12
    # return ability  60% <> 80% for between expiry months 3 and 6
    # return ability  90% <= for between expiry months 1

        f_1 = df_2.query('return_ability <= 0.55 and 1 <= expiry_month_count <= 12')

        f_2 = df_2.query('0.55 <= return_ability <= 0.80 and  3 <= expiry_month_count <= 6')

        f_3 = df_2.query('return_ability <= 0.9 and expiry_month_count == 1')

        df_3 = pd.concat([f_1,f_2,f_3])


        df_3.drop(['customer_id',  'amount_limit', 'installment_amount', 'date', 'return_ability'], axis = 1, inplace = True)

        df_3.reset_index(drop=True, inplace=True)

        self.data = df_3.copy()

        self.X_test = df_3.drop(['total_limit'], axis = 1)
        self.y_test = df_3['total_limit']

        self.test_check = self.reg.predict(xgb.DMatrix(self.X_test))

        return df_3


    def compare(self):

        check = pd.DataFrame()
        check['actual'] = self.y_test
        check['predicted'] = self.test_check
        check['differance'] = check['actual'] - check['predicted']

        return check.round()

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
