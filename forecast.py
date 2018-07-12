import statsmodels.api as sm
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_excel('datasets/Sample - Superstore.xls')
furniture = df.loc[df.Category == 'Furniture']

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID',
        'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code',
        'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
        'Quantity', 'Discount', 'Profit']

furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

furniture.set_index('Order Date', inplace=True)
# furniture = furniture.resample('D').sum()
# furniture = furniture.resample('MS').mean()
#
y = furniture['Sales'].resample('MS').mean()

decomposition = sm.tsa.seasonal_decompose(y, model='additive')


p = d = q = range(0, 2)


pdq = list(itertools.product(p, q, d))
seasonal_pdq = [
    (x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))
]


aics = []

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y,
                order=param,
                seasonal_order=param_seasonal,
                enforce_stationary=False,
                enforce_invertibility=False)
            results = mod.fit()
            aics.append(results.aic)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal,
                                                 results.aic))
        except:
            continue


mod = sm.tsa.statespace.SARIMAX(
    y,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationary=False,
    enforce_invertibility=False
)

results = mod.fit()


print(results.summary().tables[1])
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0],
                                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
