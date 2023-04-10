#!/usr/bin/env python
# coding: utf-8

# In[316]:


#!pip install yfinance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from scipy.optimize import minimize
import seaborn as sns


# In[317]:


'''
start_date='2013-10-30'
end_date='2021-10-30'
UKPound = pd.DataFrame(yf.download('GBPUSD=X',start_date,end_date)['Adj Close'].pct_change()[1:])
Gold = pd.DataFrame(yf.download('GLD',start_date,end_date)['Adj Close'].pct_change()[1:])
WTI = pd.DataFrame(yf.download('WTI',start_date,end_date)['Adj Close'].pct_change()[1:])
SPY = pd.DataFrame(yf.download('SPY',start_date,end_date)['Adj Close'].pct_change()[1:])

df1 = UKPound.join(Gold,how = 'inner',rsuffix = '_GLD')\
    .join(WTI,how = 'inner', rsuffix = '_WTI')\
    .join(SPY,how = 'inner',rsuffix = '_SPY')\
    .fillna(method = 'ffill')

BitCoin =  pd.DataFrame(pd.read_csv('BTC_USD.csv'))
BitCoin = BitCoin.set_index('Date')
BitCoin = pd.DataFrame(BitCoin[['Change %']])

dfFIN6102 = df1.join(BitCoin,how = 'inner',rsuffix = '_BTC')
dfFIN6102.rename(columns={'Change %': 'Adj Close_BTC'}, inplace = True)
dfFIN6102.index = pd.to_datetime(dfFIN6102.index)
dfFIN6102
dfFIN6102.to_csv('FIN6102Portfolio.csv')
'''


# In[318]:


df = pd.DataFrame(pd.read_csv('FIN6102Portfolio.csv'))
df = df.set_index('Date')
df.index = pd.to_datetime(df.index)
df.info()


# In[319]:


F5Yr = df.index.get_loc('2017-10-30')
df_train = df[0:F5Yr] #data for first 5 years are set as training set
df_test = df[F5Yr:] #data for later 3 years are set as test set


# In[320]:


CorrF5Yr = df_train.corr()
CovF5Yr = np.cov(df_train.T)


# In[321]:


RetF5Yr = []
StdF5Yr = []
for i in range(5):
    RetF5Yr.append(df_train.iloc[:,i].mean())
    StdF5Yr.append(df_train.iloc[:,i].std())
CorrF5Yr = df_train.corr()

Portfolio = ['GBP','GLD','WTI','SPY','BTC']
F5YrSum = pd.DataFrame(list(zip(Portfolio, RetF5Yr, StdF5Yr)), columns =['Portfolio','Mean return in first 5 year', 'Std of return in first 5 year']) 
#print(RetF5Yr)
#print(StdF5Yr)
#print(CorrF5Yr)
#print(CovF5Yr)
print(F5YrSum)


# In[322]:


plt.figure(figsize=(8,4))
corrMatrix = df_train.corr()
sns.heatmap(corrMatrix, annot=True)
plt.title('Correlation Matrix')
plt.show()


# In[323]:


def calculate_risk_contribution(w,cov):
    sigma_train = np.sqrt(np.matmul(np.matmul(w.T, cov),w))
    risk_contrib = w * np.matmul(cov,w)/sigma_train
    risk_contrib_pct = risk_contrib/sigma_train
    return risk_contrib_pct


# In[324]:


def get_risk_parity_weight(cov):
    n=cov.shape[0]
    w=np.array([1/n]*n,ndmin=2)
    def cal_loss(w,cov):
        risk_contrib_pct = calculate_risk_contribution(w, cov)
        risk_contrib_pct = np.squeeze(risk_contrib_pct)
        eq_contrib = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        loss = np.sum(np.square(risk_contrib_pct - eq_contrib))
        return loss
    cons = [{'type':'eq','fun': lambda x: np.sum(x)-1}]
    bounds = [(0,1)]*5
    res = minimize(cal_loss,w,args=(CovF5Yr,),constraints=cons,bounds=bounds)
    return res.x


# In[325]:


def calculate_port_value(period_data, weight, port_value):
    cum_period_return = (1+period_data).cumprod()
    port_return_period = np.sum(cum_period_return * weight, axis=1)
    port_values_period = port_return_period * port_value.values[-1]
    port_values_period = port_values_period.to_frame("Portfolio")
    port_value = port_value.append(port_values_period)
    return port_value


# In[326]:


rebalance_days = ['2017-10-30', '2018-10-30', '2019-10-30']
rebalance_days=[pd.to_datetime(x) for x in rebalance_days]
equal_weight=np.array([1/5]*5,ndmin=2)
prev_rebalance_day  = None
weight = None


# In[327]:


port_value1 = pd.DataFrame(index=[rebalance_days[0]], data=[1.0], columns=['Portfolio'])

for date in rebalance_days:
    if prev_rebalance_day is not None:
        period_data = df.loc[prev_rebalance_day + pd.DateOffset(days=1):date]
        port_value1 = calculate_port_value(period_data,weight,port_value1)
    weight = equal_weight
    prev_rebalance_day = date


# In[328]:


port_value2 = pd.DataFrame(index=[rebalance_days[0]], data=[1.0], columns=['Portfolio'])

for date in rebalance_days:
    if prev_rebalance_day is not None:
        period_data = df.loc[prev_rebalance_day + pd.DateOffset(days=1):date]
        port_value2 = calculate_port_value(period_data,weight,port_value2)
    train_x = df.loc[:date,:].values
    cov = np.cov(train_x.T)
    weight = get_risk_parity_weight(cov)
    prev_rebalance_day = date


# In[329]:


final_period = df.loc[prev_rebalance_day:]
port_value1 = calculate_port_value(final_period, weight, port_value1)
port_value2 = calculate_port_value(final_period, weight, port_value2)
plt.figure(figsize=(9,6))
plt.plot(port_value1,label='Portfolio_EqualWeight',color='blue')
plt.plot(port_value2,label='Portfolio_RiskParity',color='red')
plt.legend()
plt.savefig('Portfolio Construction.png')

# In[330]:


print((port_value1['Portfolio'][-1]-port_value1['Portfolio'][0])/
      port_value1['Portfolio'][0])
print((port_value2['Portfolio'][-1]-port_value2['Portfolio'][0])/
      port_value2['Portfolio'][0])


# In[ ]:




