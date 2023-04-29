import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def changeIndexTodatetime(df):
    # 純粹針對若要畫圖有x軸的時間可以對齊
    
    df = df.reset_index()
    df["Date"] = df["Date"].values.astype("datetime64[D]")   # 從Timestamp轉成純粹日的datetime
    return df.set_index("Date")

def windows_data(seq,ws):           
    # 吐出過往每window size天的資料 + window size外的一筆資料    
    # 讓lstm吃進的input所需的型態(連續資料)
        
    out = []                              
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    return out

class LSTMnetwork(nn.Module):
    # 一層lstm + 一層fully-connected layer
    # 參數數量不要超過train的資料量太多，避免overfitting，所以hidden_size選定30 ~ 35
    
    def __init__(self,input_size=1, hidden_size = 35, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size,hidden_size)
        
        self.linear = nn.Linear(hidden_size,output_size)
        
        # 初始化LSTM長短期記憶的參數
        # layers(僅一層lstm), batch_size(每次投入的組數) 皆為1
        self.hidden = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))

    def forward(self,seq):
        
        # 投入的參數為 : 資料總數, batch_size, 資料維度(都是收盤價)
        lstm_out, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)
        
        # 吐出的lstm已經經過activation function，無須搭配relu
        # 此時維度應當為hidden_size，由.view(資料數, -1)控制shape
        pred = self.linear(lstm_out.view(len(seq),-1))
        return pred[-1] 

import time
def trainForModel(model, train_data, criterion, optimizer):

    epochs = 100
    start_time = time.time()

    for epoch in range(epochs):
        
        # 若有需要該次訓練是否有overfitting的跡象，可於同epoch中加入model.eval()進行檢驗
        model.train()                           
        for seq, y_train in train_data:
            
            # 初始化參數使用
            optimizer.zero_grad()
            model.hidden = (torch.zeros(1,1,model.hidden_size),
                            torch.zeros(1,1,model.hidden_size))
            
            y_pred = model(seq)
            
            loss = criterion(y_pred, y_train)                     # 不使用RMSE因為loss過小，造成梯度消失
            loss.backward()
            optimizer.step()
            
        print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}', flush = True)
        
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')


def evalRealLSTM(model, norm, criterion, optimizer, scaler):

    model.eval()
    future, start = 252
    
    preds = []
    for i in range(future):
        seq = norm[(-(start+90)+i):(-start+i)]
        with torch.no_grad():
            model.hidden = (torch.zeros(1,1,model.hidden_size),
                            torch.zeros(1,1,model.hidden_size))
            preds.append(model(seq).item())
        
    return scaler.inverse_transform(np.array(preds[-future:]).reshape(-1, 1))

def evalPredLSTM(model, norm, criterion, optimizer, scaler):

    model.eval()

    preds_ = qqq_norm_90.tolist()
    for i in range(future):
        seq = torch.FloatTensor(preds_[-90:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1,1,model.hidden_size),
                            torch.zeros(1,1,model.hidden_size))
            preds_.append(model(seq).item())
            
            
    fake_predictions = scaler.inverse_transform(np.array(preds_[-future:]).reshape(-1, 1))


def main():
    """
    此架構以jupyter notebook上的程式碼改寫而成
    無plt進行繪圖
    """   
    qqq_df = yf.Ticker("QQQ").history(period="20y")
    qqq_df = changeIndexTodatetime(qqq_df)
    
    qqq_price = qqq_df[["Close"]]
    
    # 抓過去20年~1年的交易總日作為train data
    train_df = qqq_price[:-252]                                         
    train_arr = train_df.values.astype(float)

    # 讓超參數收斂更快 + 有效檢查是否有數據洩漏
    scaler = MinMaxScaler(feature_range=(0, 1))    
    train_norm = scaler.fit_transform(train_arr.reshape(-1, 1))
    train_norm = torch.FloatTensor(train_norm).view(-1)

    window_size = 90
    train_data = windows_data(train_norm, window_size)
    
    model = LSTMnetwork()               
    print(f"訓練資料數量: {len(train_data)}")
    print(f"模型參數總量: {sum([param.numel() for param in model.parameters()]) }")

    criterion = nn.MSELoss()               
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    # lr為超參數，每次權重移動的步伐
    
    trainForModel(model, train_data, criterion, optimizer)
    
    qqq_arr = qqq_price[-(252+90):].values.astype(float)
    qqq_norm = scaler.fit_transform(qqq_arr.reshape(-1, 1))
    qqq_norm = torch.FloatTensor(qqq_norm).view(-1)
    
    # 每日的預測，皆拿"實際資料"去預測
    pred_data_T = evalRealLSTM(model, qqq_norm, criterion, optimizer, scaler)
    
    qqq_90 = qqq_price[-(252+90):-252].values.astype(float)
    qqq_norm_90 = scaler.fit_transform(qqq_90.reshape(-1, 1))
    qqq_norm_90 = torch.FloatTensor(qqq_norm_90).view(-1)
    
    # 每日的預測，後續則拿"預測資料"去預測" (誤差疊加)
    pred_data_P = evalPredLSTM(model, qqq_norm_90, criterion, optimizer, scaler)

if __name__ == "__main__":
    main()