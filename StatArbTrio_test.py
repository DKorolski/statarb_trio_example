from finam import interval
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
from statsmodels.regression.linear_model import OLS
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
#%matplotlib inline

import statsmodels.api as sm

import datetime
import datetime
from finam import Exporter, Market, LookupComparator,Timeframe

tickers= ['RIH2','RIZ1','RIU1']
start_date = '2021-06-15'
fromdate=datetime.date(2021, 6, 15)
end_date = '2021-09-15'
exporter = Exporter()
asset = exporter.lookup(code=tickers[0], market=Market.FUTURES)
asset_id = asset[asset['code'] == tickers[0]].index[0]
data = exporter.download(asset_id, market=Market.FUTURES, timeframe=Timeframe.HOURLY, start_date=fromdate)
data['<DATE>'] = data['<DATE>'].apply(lambda x: str(x))
data['<TIME>'] = data['<TIME>'].apply(lambda x: str(x))
data['<DATE>'] =data['<DATE>']+' '+ data['<TIME>']
data['<DATE>'] = data['<DATE>'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d %H:%M:%S.%f'))
asset_df = data.set_index('<DATE>')
asset_df.index.names = ['Date']
asset_df.columns=['Time','Open','High','Low','Close','Volume']
asset_df=asset_df[['Open','High','Low','Close','Volume']]
asset_df['Adj Close'] = asset_df['Close']
asset_df=asset_df.dropna()
asset_df=pd.DataFrame(asset_df)
ins_w=asset_df['Adj Close']
asset = exporter.lookup(code=tickers[1], market=Market.FUTURES)
asset_id = asset[asset['code'] == tickers[1]].index[0]
data = exporter.download(asset_id, market=Market.FUTURES, timeframe=Timeframe.HOURLY, start_date=fromdate)

data['<DATE>'] = data['<DATE>'].apply(lambda x: str(x))
data['<TIME>'] = data['<TIME>'].apply(lambda x: str(x))
data['<DATE>'] =data['<DATE>']+' '+ data['<TIME>']
data['<DATE>'] = data['<DATE>'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d %H:%M:%S.%f'))
asset_df = data.set_index('<DATE>')
asset_df.index.names = ['Date']
asset_df.columns=['Time','Open','High','Low','Close','Volume']
asset_df=asset_df[['Open','High','Low','Close','Volume']]
asset_df['Adj Close'] = asset_df['Close']
asset_df=asset_df.dropna()
asset_df=pd.DataFrame(asset_df)
ins_x=asset_df['Adj Close']

asset = exporter.lookup(code=tickers[2], market=Market.FUTURES_ARCHIVE)
asset_id = asset[asset['code'] == tickers[2]].index[0]
data = exporter.download(asset_id, market=Market.FUTURES_ARCHIVE, timeframe=Timeframe.HOURLY, start_date=fromdate)
data['<DATE>'] = data['<DATE>'].apply(lambda x: str(x))
data['<TIME>'] = data['<TIME>'].apply(lambda x: str(x))
data['<DATE>'] =data['<DATE>']+' '+ data['<TIME>']
data['<DATE>'] = data['<DATE>'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d %H:%M:%S.%f'))
asset_df = data.set_index('<DATE>')
asset_df.index.names = ['Date']
asset_df.columns=['Time','Open','High','Low','Close','Volume']
asset_df=asset_df[['Open','High','Low','Close','Volume']]
asset_df['Adj Close'] = asset_df['Close']
asset_df=asset_df.dropna()
asset_df=pd.DataFrame(asset_df)
ins_y=asset_df['Adj Close']
ins_y=ins_y[ins_y.index.isin(ins_x.index)]
ins_w=ins_w[ins_w.index.isin(ins_x.index)]


w = ins_w#['Adj Close']
x = ins_x#['Adj Close']
y = ins_y#['Adj Close']
df = pd.DataFrame([w,x,y]).transpose().ffill()
df.columns = ['W','X','Y']
df1 = pd.DataFrame([w,x,y]).transpose().ffill()
df1.columns = ['ins_w','ins_x','ins_y']
df.dropna()
df1.dropna()
plt.figure(figsize=(8,5))
df.plot(figsize=(10,8))
plt.show()

from statsmodels.tsa.vector_ar.vecm import coint_johansen

y3 = df
df5 = df
plt.figure(figsize=(8,5))
df1.plot()
df1.plot.scatter(x='ins_x', y='ins_y')
plt.xlabel('ins_x share price')
plt.ylabel('ins_y share price')
plt.xlabel('April 4,2006, to April 9,2012')
plt.ylabel('Share price $')
plt.show()
results=smf.ols(formula="ins_y ~ ins_x", data=df1[['ins_x', 'ins_y']]).fit()
hedgeRatio=results.params[1]
print('hedgeRatio=%f' % hedgeRatio)
hedgeRatio=results.params[1]
print('hedgeRatio=%f' % hedgeRatio)
df4=df1
df4['spread'] = df1.ins_y - (df1.ins_x * hedgeRatio)
df4.dropna()
df4=df4.spread
df4.reset_index()
df4=pd.DataFrame(df4)
df4.plot()
plt.xlabel('Stationarity of Residuals of Linear ')
plt.ylabel('ins_y - hedgeRatio*ins_x')
plt.show()

print(ts.coint(df1['ins_x'], df1['ins_y']))
# cadf test
coint_t, pvalue, crit_value=ts.coint(df1['ins_x'], df1['ins_y'])
print('t-statistic=%f' % coint_t)
print('pvalue=%f' % pvalue)
print(crit_value)
df1=df1[['ins_w','ins_x','ins_y']]
# Johansen test
result=vm.coint_johansen(df1[['ins_x', 'ins_y']].values, det_order=0, k_ar_diff=1)
print('Johansen test')
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)

# Add ins_w for Johansen test
result=vm.coint_johansen(df1.values, det_order=0, k_ar_diff=1)
print('Add ins_w for Johansen test')
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)

print('eigenvalues')
print(result.eig)  # eigenvalues
print('eigenvectors')
print(result.evec)  # eigenvectors
y3=df1
j_results = coint_johansen(y3,0,1)
print(j_results.lr1)                           
print(j_results.cvt)                           
print(j_results.eig)
print(j_results.evec)
print(j_results.evec[:,0])
print('eigenvectors')
print(j_results.evec)  # eigenvectors

yport=pd.DataFrame(np.dot(df1.values, result.evec[:, 0])) #  (net) market value of portfolio
print(yport)

ylag=yport.shift()
deltaY=yport-ylag
df2=pd.concat([ylag, deltaY], axis=1)
df2.columns=['ylag', 'deltaY']
regress_results=smf.ols(formula="deltaY ~ ylag", data=df2).fit() # Note this can deal with NaN in top row
print(regress_results.params)

halflife=-np.log(2)/regress_results.params['ylag']
print('halflife=%f days' % halflife)

#  Apply a simple linear mean reversion strategy to ins_x-ins_y-ins_w
lookback=np.round(halflife).astype(int) #  setting lookback to the halflife found above
numUnits =-(yport-yport.rolling(lookback).mean())/yport.rolling(lookback).std() # capital invested in portfolio in dollars.  movingAvg and movingStd are functions from epchan.com/book2

print(numUnits)

positions=pd.DataFrame(np.dot(numUnits.values, np.expand_dims(result.evec[:, 0], axis=1).T)*df1.values) # results.evec(:, 1)' can be viewed as the capital allocation, while positions is the dollar capital in each ETF.
print(positions)

positions.to_csv('trio.csv')
pnl=np.sum((positions.shift().values)*(df1.pct_change().values), axis=1) # daily P&L of the strategy
print(df1)
#pnl=np.sum((pos1.shift().values)*(df5.pct_change().values), axis=1)
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
#(np.cumprod(1+ret)-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
ret=pnl / np.sum(np.abs(positions.shift(1)),axis=1)
print(ret)
# Kumulatif birlesik getiri
plt.plot(np.cumprod(1+ret)-1)
plt.show()
