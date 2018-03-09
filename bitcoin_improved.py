# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
#np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import *
from sklearn.externals import joblib
from keras import regularizers

import quandl

'''
Name:        The Self Learning Quant, Example 3

Author:      Daniel Zakrisson

Created:     30/03/2016
Copyright:   (c) Daniel Zakrisson 2016
Licence:     BSD

Requirements:
Numpy
Pandas
MatplotLib
scikit-learn
TA-Lib, instructions at https://mrjbq7.github.io/ta-lib/install.html
Keras, https://keras.io/
quandl, https://www.quandl.com/tools/python
backtest.py from the TWP library. Download backtest.py and put in the same folder

/plt create a subfolder in the same directory where plot files will be saved

'''

#Load data

def backtest(price, signal ,initialCash,to_sell=0.2):
    shares=[]
    cash=[]
    dif=np.diff(signal)
    dif=np.append([0],dif)
    for i,sig in enumerate(dif):
        if sig==100:
            shares.append(100/price[i])
            cash.append(-100)
        elif sig==-100:
            total_shares=sum(shares)*to_sell
            total_value=total_shares*price[i]
            cash.append(total_value)
            shares.append(-total_shares)
        else:
            cash.append(0)
            shares.append(0)
    data = pd.DataFrame(columns = ['price','total_shares','shares_trade','value','cash','pnl'])
    data['cash']=initialCash+np.cumsum(cash)
    data['price']=price
    data['signal']=signal
    data['shares_trade']=shares
    data['total_shares']=np.cumsum(shares)
    data['value']=data['total_shares']*price
    data['pnl']=data['cash']+data['value']-initialCash  
    
    return data
    
    

def load_data_(file='bitcoin_hourly_recent.csv',size=24*60):
    prices = pd.read_csv(file)
    prices.rename(columns={'Open': 'open', 'High': 'high', 
     'Low': 'low', 'Close': 'close', 'Volume (BTC)': 'volume'}, inplace=True)
    
    index=np.random.randint(0,int(prices.shape[0] - size))    
    prices_dummy=prices[index:index+size]
        
    return prices_dummy
    
    
def split_train_test(data,train=0.8):
    index2=int(data.shape[0]*0.8)
    prices_train = data[0:index2]
    prices_test = data[index2:]
    
    return prices_train, prices_test
        
  
def make_attributes(indata, test=False,indicators=[]):
    
    close = indata['close'].values
    diff = np.diff(close)
    diff = np.insert(diff, 0, 0)
    
    res={'close':close,'diff':diff}
    for key in indicators.keys():
        indic=indicators[key]
        print(key)
        if len(indic)==2:
            result = indic[0](indata,indic[1])
        if len(indic)==1:
            result = indic[0](indata)
        try:
            length=len(result.columns)
        except:
            length=1
            
        if length==1:   
            res[key]=result
        else:
            for col in result.columns:
                res[key+'_'+col]=result[col]
        
            
    variables=pd.DataFrame(res)
    
    return variables
    
#Initialize first state, all items are placed deterministically
def init_state(indata, test=False):    
    #--- Preprocess data    
    xdata = np.nan_to_num(indata.copy())
    xdata = np.column_stack((xdata,np.zeros(len(xdata))))
    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    elif test == True:
        scaler = joblib.load('data/scaler.pkl')
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    state = xdata[0:1, 0:1, :]
    
    return xdata, indata['close']

def make_state_steps(xdata,time_step,steps,cash,cash_norm):
    state = xdata[time_step-steps:time_step]
    state = state.reshape((1,steps,state.shape[2]))    
    state[0,0:steps,-1] = cash/cash_norm
    
    return state
    
#Take Action
def take_action(xdata, action, signal, time_step, steps, cash, cash_norm):
    #this should generate a list of trade signals that at evaluation time are fed to the backtester
    #the backtester should get a list of trade signals and a list of price data for the assett
    
    #make necessary adjustments to state and then return it
    time_step += 1
    new_state = make_state_steps(xdata,time_step,steps,cash,cash_norm)
    
    #move the market data window one step forward
    previous = signal.loc[time_step - 1]
    if np.isnan(previous):
        previous=0.0
        
        
    #if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step + 1 == xdata.shape[0]:
        terminal_state = 1
    else:
        terminal_state = 0  
        
    #take action
    if action == 1:
        #buy
        if cash>100:
            signal.iloc[time_step] = previous + 100
        else:
            signal.iloc[time_step] = signal.iloc[time_step - 1]
    elif action == 2:
        #do nothing
        signal.iloc[time_step] = signal.iloc[time_step - 1]
    elif action==0:
        #sell
        if previous>=100:
            signal.loc[time_step] = previous - 100
        elif previous<100 and previous>0:
            signal.loc[time_step] = 0
        else:
            pass
        
    return new_state, time_step, signal, terminal_state

#Get Reward, the reward is returned at the end of an episode
def get_reward(time_step, action, price_data, signal, initial_cash, terminal_state, eval=False, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)

    index1=None
    index2=None
    
    if eval == False:
        index1=time_step-2
        index1=0
        index2=time_step+1
        
    #bt = twp.Backtest(pd.Series(data=[x for x in price_data[index1:index2].values],index=signal[index1:index2].index.values), 
    #                            signal[index1:index2], initialCash=initial_cash)
    
    bt = backtest(price_data[index1:index2].values, signal[index1:index2].values, initialCash=initial_cash)
    #reward = bt.data.pnl.values[-1]
    
    reward = bt.pnl.values[-1] - bt.pnl.values[-2]
    cash = bt.cash.values[-1]
    if cash<0.0:
        asdasdf2=234234
        
    
#    dif = bt.data.price.diff(1)     
#    if dif.values[-1]>0 and action!=2:
#        reward=reward-10
#    if dif.values[-1]<0 and action==1:
#        reward=reward+100+price
#        
#    if action==0 and cash==initial_cash:
#        reward-=200
    if action==2:
        reward-=1
    
    return reward, cash, bt

def plotTradesBuySell(data):
    """ 
    visualise trades on the price chart 
    long entry : green triangle up
    short entry : red triangle down
    exit : black circle
    """
    l = ['price']
    
    p = data['price']
    p.plot(style='x-')
    
    d = data['signal'].diff()
    d[0] = 0
    idx = d > 0
    if idx.any():
        p[idx].plot(style='go')
        l.append('buy')
    
    #colored line for short positions    
    idx = d < 0
    if idx.any():
        p[idx].plot(style='ro',alpha=0.5)
        l.append('sell')

    plt.xlim([p.index[0],p.index[-1]]) # show full axis
    
    plt.legend(l,loc='best')
    plt.title('trades')    
    
def plotTrades(bt,epoch):
    #save a figure of the test set
    plt.figure(figsize=(6,8))
    plotTradesBuySell(bt)

    plt.suptitle(str(epoch))
    plt.savefig('plt/'+str(epoch)+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
    plt.close('all')
    
    
    
def run_Q(xdata, model, price_data, time_step, steps, signal, cash, cash_norm, epsilon):
    
    state = make_state_steps(xdata,time_step,steps,cash,cash_norm)

    if (random.random() < epsilon): #choose random action
        action = np.random.randint(0,3) #assumes 4 different actions
    else: #choose best action from Q(s,a) values
        qval = model.predict(state, batch_size=1)
        action = (np.argmax(qval))
    #Take action, observe new state S'
    
    new_state, time_step, signal, terminal_state = take_action(xdata=xdata, action=action, 
                                                               signal=signal, time_step=time_step, steps=steps,
                                                               cash=cash, cash_norm=cash_norm)
    #Observe reward
    reward, cash, bt = get_reward(time_step=time_step, action=action, 
                        price_data = price_data, signal=signal, 
                        terminal_state=terminal_state,initial_cash=initial_cash)
    new_state[0][0][-1] = cash/cash_norm
    
    return state, action, reward, new_state, cash, time_step, signal, terminal_state, bt
 
    
    
def evaluate_Q(eval_data, eval_model, price_data, cash, steps, epoch=0,epsilon=0):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    terminal_state = 0
    #initiliaze time_step to steps, so that the index won't go below the first element
    time_step = steps
    rewards=[]
    while(terminal_state != 1):
        state, action, reward, new_state, cash, time_step, signal, terminal_state, bt = run_Q(xdata=xdata, 
                                                                                              model=model, 
                                         price_data=price_data,time_step=time_step, 
                                         signal=signal,cash=cash, cash_norm=cash_norm,epsilon=epsilon, steps=steps)
        
        rewards.append(reward)
             
    plotTrades(bt,epoch)
            
    return bt.pnl.values[-1], rewards, signal, bt

#This neural network is the the Q-function, run it like this:
#model.predict(state.reshape(1,64), batch_size=1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam


def create_model(data, steps):
    model = Sequential()
    #the +1 in the input_shape is for the available money
    model.add(LSTM(data.shape[1],
                   input_shape=(steps, data.shape[1]+1),
                   return_sequences=False,
                   stateful=False,activation='relu',kernel_regularizer=regularizers.l1(0.01)))
#    model.add(Dropout(0.5))
#    
#    model.add(Dropout(0.5))
    #model.add(Dense(30,input_shape=(data.shape[1]+1,)))
    model.add(Dense(data.shape[1]*3))
    model.add(Dropout(0.5))    
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs
    
    model.compile(loss='mse', optimizer='nadam')
    return model


import random, timeit


start_time = timeit.default_timer()

#old
#{'sma15':(SMA,15),'sma60':(SMA,60)'rsi14':(RSI,14),'atr14':(ATR,14)}

data = load_data_()
data = make_attributes(data,indicators={'sma12':(SMA,6),'sma12':(SMA,12),'sma24':(SMA,24),'sma48':(SMA,48),
                                               'rsi6':(RSI,6),'rsi12':(RSI,12),'rsi24':(RSI,24),'rsi24':(RSI,48),
                                               'atr6':(ATR,6),'atr12':(ATR,12),'atr24':(ATR,24),'atr48':(ATR,48),
                                               'stoch6':(STOCH,6),'stoch12':(STOCH,12),'stoch24':(STOCH,24),'stoch48':(STOCH,48),
                                               'bop6':(BOP,6),'bop12':(BOP,12),'bop24':(BOP,24),'bop48':(BOP,48),
                                               'kama6':(KAMA,6),'kama12':(KAMA,12),'kama24':(KAMA,24),'kama48':(KAMA,48),
                                               #'MAVP':(MAVP,24),
                                               'MACD':(MACD,),
                                               'HT_TRENDLINE':(HT_TRENDLINE ,),
                                               'HT_DCPERIOD':(HT_DCPERIOD,),
                                               'HT_SINE':(HT_SINE ,),
                                               'HT_TRENDMODE':(HT_TRENDMODE,),
                                               'CDL2CROWS':(CDL2CROWS,),
                                               'CDLHAMMER':(CDLHAMMER ,),
                                               'CDLHANGINGMAN':(CDLHANGINGMAN,),
                                               'CDLENGULFING':(CDLENGULFING,),
                                               'CDL3BLACKCROWS':(CDL3BLACKCROWS,),
                                               'CDLDRAGONFLYDOJI':(CDLDRAGONFLYDOJI,),
                                               'CDLENGULFING':(CDLENGULFING,),
                                               'CDLMORNINGDOJISTAR':(CDLMORNINGDOJISTAR,),
                                               'CDLRISEFALL3METHODS':(CDLRISEFALL3METHODS,),
                                               'CDLUNIQUE3RIVER':(CDLUNIQUE3RIVER,)})

#data['close-sma15'] = data['close'] - data['sma15']
#data['sma15-sma60'] = data['sma15'] - data['sma60']
#data.dropna(inplace=True)

indata,test_data = split_train_test(data)
xdata, price_data = init_state(indata)
xdata_test, price_data_test = init_state(indata,test=True)



#main parameters
initial_cash=1000
epochs = 50
alpha=0.1
gamma = 0.3 #higher values benefit longer term rewards
epsilon = 0.5
batchSize = 20
buffer = 50
steps = 12
replay = []
learning_progress = []

#stores tuples of (S, A, R, S')

h = 0
#trading signal. All 0s in the beginning
signal = pd.Series(index=np.arange(len(indata)))
signal.fillna(value=0,inplace=True)

#just a quick and dirty way to normalise the amount of cash available.
cash_norm = initial_cash*2
cash = initial_cash

model=create_model(indata,steps)


for i in range(epochs):
    terminal_state = 0
    time_step = steps+1
    #while game still in progress
    while(terminal_state != 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        
        state, action, reward, new_state, cash, time_step, signal, terminal_state, _ = run_Q(xdata=xdata, model=model, 
                                                 price_data=price_data,time_step=time_step, 
                                                 signal=signal,cash=cash, cash_norm=cash_norm,epsilon=epsilon, steps=steps)

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
            #print(time_step, reward, terminal_state)
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state, batch_size=1)
                newQ = model.predict(new_state, batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,len(old_qval[0])))
                y[:] = old_qval[:]
                if terminal_state == 0: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = (1-alpha)*y[0][action] + alpha*update
                #print(time_step, reward, terminal_state)
                X_train.append(old_state)
                y_train.append(y.reshape(len(old_qval[0]),))

            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=1)           
            state = new_state

    eval_reward, rewards, signal, bt = evaluate_Q(eval_data = xdata_test, eval_model = model, price_data = price_data_test, 
                             epoch=i, cash=initial_cash, steps=steps)
    bt.to_csv('epoch_data/epoch_'+str(i)+'.csv')
    
    learning_progress.append((eval_reward))
    print("Epoch #: %s Reward: %f Epsilon: %f" % (i,eval_reward, epsilon))
    #learning_progress.append((reward))
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1.0/epochs)


elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))

#bt = twp.Backtest(pd.Series(data=[x[0,0] for x in xdata]), signal, signalType='shares')
#bt.data['delta'] = bt.data['shares'].diff().fillna(0)
#
#print(bt.data)
#unique, counts = np.unique(filter(lambda v: v==v, signal.values), return_counts=True)
#print(np.asarray((unique, counts)).T)
#
#plt.figure()
#plt.subplot(3,1,1)
#bt.plotTrades()
#plt.subplot(3,1,2)
#bt.pnl.plot(style='x-')
#plt.subplot(3,1,3)
plt.plot(learning_progress)

plt.savefig('plt/summary'+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
#plt.show()


