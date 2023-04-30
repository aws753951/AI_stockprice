# Deep learning 預測股價

##### 前言: 由於近期 chatgpt 實在火紅，本以為十幾年後才能看到的人工智慧如今橫空出世，因此萌生想學習其背後的深度學習的框架

##### 透過學習其一的框架: pytorch 後，想以該技術，試圖以利用深度學習來預測股價走勢，因此有該 side project

![](https://i.imgur.com/J7uL7xD.png)![docker](https://i.imgur.com/GpxYNoe.png+) ![pytorch](https://i.imgur.com/LwzDTfH.png)![](https://i.imgur.com/cADpkK3.png) ![](https://i.imgur.com/ILx6OWB.png)

## 目的: 希望透過深度學習的技術，預測股價走勢

醜話說在前頭，股市的走勢，很重要的還是得透過**基本面、籌碼面、市場概況**得以預估，從不認為可以簡簡單單就可以達成

因此此次的 side project 純粹以練習該框架，學習神經網路等相關知識，就算最後預測失敗，也不失一個學習的經驗

## 主要步驟

#### 1. 透過 LSTM 觀察股價是否得以根據過往的價格進行有效預測

#### 2. 使用 ANN 搭配有意義的 feature 進行訓練與預測

#### 3. 使用步驟二產的預測結果進行評論是否有獲利的可能

#### 4. 若有效則設定相關自動化流程來通知進場或出場

---

## 1. LSTM: 長短期記憶模型

主要用於長序列的訓練，負責記憶過去訊息，並刪除無用的記憶，更新當前記憶及輸出預測值

可用於去處理語言句子進行前文分析及預測下一句該怎麼講，其實就是 chatgpt 的精隨，只是別人多了更多的語言監督跟更複雜的訓練而成

這裡取其中 LSTM 的模型來協助預測長序列的資料，如股價，因此這裡先用 LSTM 進行預測分析

#### 相關程式碼請參考: lstmModel.py

> 這裡以個人覺得值得投資的標的 QQQ(ETF)作為股價預測的標的
>
> 抓取時間為過去 20 年，利用過去 19 年預測近一年的股價表現

![lstm1](https://i.imgur.com/0665Ogn.png)

###### 注意: 此處每個預測的資料，皆是以滾動式，拿取過往"真實"的 90 天資料進行預測第 91 天的資料，只是記錄了 LSTM 的預測結果，並沒有拿該預測結果作為 input 進行預測

#### 看起來挺貼合真實的股價表現，但其實只是滯後的表現，也就是真實股價先行後，預測的結果才慢慢跟上

> 畢竟 LSTM 是根據過往的股價表現，去推估未來的股價，若過往的股價從 1 -> 2 -> 3 ，是很容易推估下一個可能是 4.01
>
> 但影響股價的因素眾多，很多時候都是價格優先反映真實市場狀況，而這個當下模型是不會知道的，得等到隔天後才有辦法納入
>
> 因此仔細來看 LSTM 預測的股價是遞延於真實股價 => 也就是追漲殺跌，對於投資上幫助不大

![lstm2](https://i.imgur.com/rfNWLkf.png)

#### 這裡則使用過往 90 天的股價，慢慢替換成預測的股價作為 input 去預測更久遠未來的股價

> 完全失效 => 不可能僅憑 LSTM 的模型去預測未來股價的走勢
>
> 後續的股價如心跳停止的表現，一來是訓練無效外，二來是預測誤差疊加所造成

### 結論: 使用 LSTM 去預測股價是不切實際的，因為股價他並不是單純透過過往歷史股價表現就能輕鬆預測，參雜了各種因素去影響股價，而使用 LSTM 去預測句子算行得通

> 如: 今晚，我想吃.... 若經過足夠的訓練，是可以回答得出來像是"泡麵" / "義大利麵" / "金黃火腿"等，因為這些大多可以依循著固定的**語言順序**進行訓練

#### 股價並沒有固定的表現，至於常常聽到的股價會歷史重演...，很多時候都是講這句話的人，大多擷取過往對他有利的股價片段，而忽略了整個股價的表現跟隨機走勢沒兩樣

> 我說明天會漲，50%的機率是對的我就拿那 50%的機率證明我很厲害..... 這種感覺

---

#### 既然使用 LSTM 做機器學習，對於股價沒有預測只是滯後的效果，那如果簡單一點，採用 ANN 的模型去預測呢?

> 畢竟 LSTM 很吃過往的股價表現，ANN 則吃你餵入了甚麼樣的特徵去訓練，或許能夠有效?

#### 那就要先選定，甚麼樣的特徵對於股價的預測是有幫助的?

> 就我個人的經驗，大概有以下幾點是常用來判斷市場未來的走向 (沒興趣者可以翻到後面的程式碼)

---

#### 講古時間

|

|

|

### 1. 聯準會的態度:

由於聯準會負責制定和實施貨幣政策，例如，聯準會通常會透過調整基準利率來影響市場借貸成本，而公司資金的主要來源分兩塊，一塊是股權，一塊是債權，通常債權佔資本的比重越高的，其受到利率調升的影響幅度就越大 (為何不乾脆讓資本大多來自股權呢? 因為通常股權的成本比債權高，也要考慮公司所有權分散的問題，因此發放債券籌措資金效益通常比較好，發放股票籌措資金也會讓投資人覺得公司自己也覺得自身股價高，所以發股籌資金，會影響投資人對公司股價的信心)，所以近期看到的通膨，讓聯準會不得不打破過往的經驗，直接升息以減緩通膨造成的民間消費的負面影響(不過也可能造成停滯性通膨，就不細說了)

此外，聯準會也控制貨幣供給，像是 2020 年 3 月的股市鎔斷，就是因為市場資金流動性匱乏，很多金融商品都是資金借資金去運作，那些金融資產都有一定的價值，但當資金流動性不足時，資產被拋售，讓健全的金融環境被不合理的對待，導致會有更多連鎖反應(你把它想成銀行擠兌 跟 2008 年連動債 )

所以投資人大多會觀察每一次聯準會開會結果，其政策方向大大影響市場表現

### 2. 公司法說會:

這跟剛剛聯準的態度很像，通常法說會會公布公司未來一季度或一整年的展望，尤其該產業相關的龍頭企業的法說會，更會被投資人視為極重要的 insight 資訊，此外公司的法說會也會展現每家公司董事長對於公司未來的發展態度，看是裁員縮減人力成本，還是擴展設備以因應訂單要求等，這些態度都是很多股票分析師會派人去法說會第一時間蒐集該公司對於未來看法，以提供第一手的重要資訊給投資人，畢竟投資玩的就是資訊落差，優先掌握到資訊的人大多才能夠賺到錢

當然也會提供公司的財務狀況等，這些都是對於該公司未來有重要的資訊

### 3. 產業狀況:

這補充剛剛公司法說會的內容，龍頭公司大多掌握自己產業的最新消息，比如市場消費力道足夠，以至於未來訂單大增，或者有最新的技術能夠大幅縮減成本，讓未來公司產能充足，又或者原料成本因 XXX 事情，可望減少公司進貨成本等，這些都是產業分析師會著重的要點，當然除了公司講的外，還可能有些資訊是產業分析師"發現"被低估的部分，如該產業所需的原物料價格具有景氣循環性，目前正處於哪個 cycle，又或者甚麼突發天災事件，對於相關重要的工廠會無法及時交付等，這些都是在公司法說會外，時常會發生的事情，需要產業分析師時刻盯著最新市場概況，提供相關資訊給投資者

```
以上幾點是我認為最重要，也是最多投資人意見分歧的要點，不然每個事件都只會有一種走勢的話，股市就不會拉扯，每個人都看對方向，只要做多跟放空就通通賺大錢了，但現實不太可能這樣發生

但以上幾點，通通無法作為特徵去訓練，畢竟質化的東西無法量化，況且這質化的東西只有掌握更多資訊的人才有辦法看對方向，所以就我這個小菜雞是很難用這些事件作為機器學習的特徵
```

#### 但我個人認為還是有東西可做為特徵丟入到機器學習裡面，不過這些資料大多是落後資訊，在投資界來講大多是供參考而已

### 1. 經濟數據:

你常聽到的 GDP / CPI / ISM / PPI / 失業率 等，這些都反映市場相關研究機構公布最新的資訊，反映的是"過去"一段時間真實的狀況，所以當你看到如美國 GDP 年年衰減，發現美國經濟不行了快逃阿的話，你往往就錯失相關投資機會，因為這些資訊都是"落後指標"，過往怎麼樣，不代表未來表現會影響，也有可能過往狀況壓抑的市場，你現在看到的都是負面消息反而有可能利空出盡，是進入市場的好時機，所以這些落後指標，往往是讓投資人判斷這些數字背後的細項有沒有甚麼是目前市場上未發現的資訊，還有發酵的空間可以操作等，誰先掌握這些資訊誰就可以有獲利的空間

但對於我而言，算是一個很好拿來讓電腦發現其中有用的特徵，畢竟機器學習就是協助人們從看不懂的數字找出其規律做預測，因此這裡算是我這次會放入的特徵資料做訓練

### 2. 技術指標:

有很多做交易的人，他們都會觀察技術指標，其核心理念在於"價格"反映了所有資訊，很多市場真實狀況價格往往都是先行，因此他們認為從價格的走勢加以分析，不管短線還是長線，比那些只能拿到二手、三手資訊的產業或股票分析者而言，有更快的時間因應市場未消化的操作空間。

這一塊我不是很熟，畢竟我上一份工作跟這塊比較無關聯，或許真的有所謂的 golden rule 可以獲利，但我目前也不知道誰說的是對的，在這塊技術指標往往百家爭鳴，有不同流派的人深信自己的技術能夠替自己獲利，比如 RSI / KD / MACD / 布林通道 等，都是由過往的"股價"組合成的技術指標，判斷當今的股價落於該指標的何方，來看是否進場或退場

因此，在這裡我就沒辦法拿技術指標作為特徵丟入到機器學習裡面，而且這些指標跟價格有密切的關係，作為未來預測股價很有可能發生共線性的問題，再加上自己也沒有相關的這塊經驗，因此沒辦法拿這些資訊胡亂餵入

#### 以上幾點，都是我個人的淺見，還有很多我沒有寫出來，畢竟很久沒碰這塊了，我就把我最有印象的東西寫出來，或許我花一天的時間問問 chatgpt 可以有更完整的內容，但目前還是趕快進入到跟機器學習有關的程式碼吧

|

|

|

#### 講古結束

---

## 2. ANN 模型訓練

> 這裡僅抓三個我個人有用的資訊: ISM 美國製造業指數 + 通膨 + 失業率

以下我廢話少說

#### 美國製造業指數: 市場調查數據，製造業採購經理人對於未來的展望而匯整的數據，50 代表榮枯線

#### 通膨: 不用多說，聯準會目前最主要判斷是否升息的重要依據，過往是 2%最能代表經濟正常發展的狀況

#### 失業率: 也是聯準會常說的數據，我記得好像是 4%以下算是聯準會認可的標準

> 基本上聯準會看中的就是經濟成長 + 通膨跟失業率，這都是比較粗略的講法，真的常關注聯準會的人不要打我，畢竟我很久沒 follow 了

一定還有其他資訊，但我這邊就先以這三個為主，有興趣的人可以上 investing.com 看看有甚麼資訊是市場時常關注的數據

https://www.investing.com/economic-calendar/

#### 因此我這邊會使用該三項特徵 + 股價 + 交易量 去預測未來的股價，相關程式碼請參考 : ANN.ipynb

```
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import yfinance as yf
import datetime

import matplotlib.pyplot as plt
%matplotlib inline
```

#### step1. 資料蒐集跟整理

##### ISM 數據

```
ISM_df = pd.read_excel("ISM.xlsx")
ISM_df["Release Date"].value_counts()
```

> 這裡我手動爬公開網站的數據資料，來源來自於 investing.com
>
> 不用 API 直接查是因為這些資料它沒有收錄在我認知的 API 裡面 (fredapi / quandl / pandas-datareader 都沒有= =)
>
> > 小知識補充: 每個大型金融機構，都會買 Bloomberg，它就是一個裝金融數據的資料庫，每年光一台，就得花費約莫 60 萬台幣，就代表這些金融數據很賣錢，不是你隨便想得到就可以得到，畢竟這些資料都是需要有人針對公開資訊進行彙整，誰做得越完整且即時，誰賣的金融數據這些資料就越有價值
>
> 因此若這個計畫有效，後續可以用爬蟲的方式自動化補充

![](https://i.imgur.com/QYFH6Lg.png)

```
date_list = []

# 發現比用pd.to_datetime or Series.apply還好用，直接依照邏輯更改成自己要的格式
for d in ISM_df["Release Date"].values:
    if type(d) == str:
        date_list.append(datetime.datetime.strptime(d[:-6], "%b %d, %Y"))
    else:
        date_list.append(d)

ISM_df["Date"] = date_list
ISM_df.rename(columns = {"Actual": "ISM"}, inplace = True)
ISM_df = ISM_df[["Date", "ISM"]]
ISM_df["Date"] = ISM_df["Date"].values.astype("datetime64[D]")
ISM_df.head()
```

> 資料不是統一的時間格式，因此做相關整理 => 通通改成 datetime64[D]

##### 失業率數據

```
unemp_df = pd.read_excel("unemploy.xlsx")

date_list_ = []
for d in unemp_df["Release Date"].values:
    if type(d) == str:
        date_list_.append(datetime.datetime.strptime(d[:-6], "%b %d, %Y"))
    else:
        date_list_.append(d)

unemp_df["Date"] = date_list_

unemp_df.rename(columns = {"Actual": "UNEMP"}, inplace = True)
unemp_df = unemp_df[["Date", "UNEMP"]]
unemp_df["Date"] = unemp_df["Date"].values.astype("datetime64[D]")
```

##### 通膨率

```
import quandl
import os
from dotenv import load_dotenv
load_dotenv()

quandl.ApiConfig.api_key = os.getenv("QUANDLAPI")

# 不是我一開始不用這些可以取得美國經濟數據的api，是相關我需要的api這邊無法提供，僅找到通膨的代碼
cpi_df = quandl.get("RATEINF/INFLATION_USA")

cpi_df.reset_index(inplace = True)
cpi_df["Date"] = cpi_df["Date"].values.astype("datetime64[D]")
cpi_df.rename(columns={"Value": "CPI"}, inplace = True)
```

##### 挑選要預測的標的"QQQ"

```
qqq_df = yf.Ticker("QQQ").history(period="20y")
qqq_df.head()
```

> 這裡挑選標的為"QQQ" ETF，是我個人認為長期投資回報獲利跟風險都最佳的標的

#### 其中我們會選取有用的資料: 收盤價 + 交易量

> Open: 開盤價，大多反應當天投資人不理性的行為，對於機器學習沒有幫助
>
> High/Low: 當天最高價/最低價，由於跟最主要的收盤價有密切關係，所以也不納入
>
> Close: 收盤價，這次判斷的主要特徵之一，很多技術分析都是根據收盤價去生出來的，因此一定要用到收盤價
>
> Volumn: 交易量，通常技術分析的人也常看交易量，如甚麼價漲量跌，價量背離都有其背後的邏輯在，因此納入給機器學習分析
>
> Dividends: 股利，你把它想成配息即可，後續分析後不納入
>
> Stock Splits: 股票分割，這標的沒有這個內容，所以不納入
>
> Capital Gains: 同上，所以不納入

![](https://i.imgur.com/BvbwzB1.png)

#### 股利需要加入嗎?

```
qqq_df.reset_index(inplace = True)
qqq_df["Date"] = qqq_df["Date"].values.astype("datetime64[D]")

# 在合併前先整理qqq_df  =>  擷取只要的欄位 + 整理一下價格
qqq_df = qqq_df[["Date", "Close", "Volume"]]

# 順移前一天的股價，比對看看是否隔天的收盤價因為發放股利而減少
qqq_df["former_Close"] = qqq_df["Close"].shift(1, axis = 0)
qqq_df["rational_Close"] = qqq_df["former_Close"] - qqq_df["Dividends"]
gap_list = qqq_df["Close"] - qqq_df["rational_Close"]
gap_list.describe()
```

> 這裡是因為該 yfiance 提供的標的資料有股利，因此需要確定股利發放是否需要補回股價
>
> 正常來講發放股利的隔天股價，平均來講會剛好少掉股利發放的額度，但目前股價沒減少，還增加(平均數，中位數皆為正)
>
> 這很大的可能是因為該標的本身很被投資人看好，因此就算發放股利也有所謂的"股利回補"，屬於市場給予正向的反應，所以這邊就不打算調整股價的資料了

![](https://i.imgur.com/KjKQFmu.png)

#### step2. 蒐集的資料進行整併

```
merge_df = pd.merge(qqq_df, ISM_df, on='Date', how='outer')
print(len(merge_df))
merge_df
```

> 這是使用 merge 後，兩邊的資料時間長短不一，需要做裁減
>
> 此外，由於經濟數據大多一個月公布一次，所以會用回補的方式補充該月的資料到每天的股價表現 => 機器學習不能有 NaN 的資料

![](https://i.imgur.com/pTbtnCs.png)

```
merge_df = merge_df[:len(qqq_df)]
merge_df["ISM"] = merge_df["ISM"].fillna(method = "ffill")

merge_df = pd.merge(merge_df, unemp_df, on='Date', how='outer')
merge_df = merge_df[:len(qqq_df)]
merge_df["UNEMP"] = merge_df["UNEMP"].fillna(method = "ffill")

merge_df = pd.merge(merge_df, cpi_df, on='Date', how='outer')
merge_df = merge_df[:len(qqq_df)]
merge_df["CPI"] = merge_df["CPI"].fillna(method = "ffill")

merge_df = merge_df.dropna()
merge_df = merge_df.set_index("Date")
```

> 失業率跟通膨率一樣的操作

#### 先讓大家看看這些經濟數據過往的表現

![](https://i.imgur.com/tCpWrDE.png)![](https://i.imgur.com/P7om6JA.png)![](https://i.imgur.com/oAESxFW.png)

> 除了失業率每月變化比較小外，其他的資料每月都有不定的隨機表現

#### 這裡決定刪除: 失業率

> 因為失業率每月的變化都蠻固定的，除了重大事件發生才會有異常狀況，不然這特徵太容易依循上個月的狀況增加或下滑，怕影響到機器學習的判斷，故刪掉
>
> > 很心痛阿... 找了很久的資料來源，最終還是覺得不適合做為機器學習的特徵

```
merge_df.drop(["UNEMP"], axis = 1, inplace = True)
```

#### step3. 把有數值資料進行壓縮，避免梯度爆炸的問題發生

```
merge_df.loc[merge_df["ISM"] < 50, "ISM"] = 0
merge_df.loc[merge_df["ISM"] >= 50, "ISM"] = 1
merge_df["ISM"] = merge_df["ISM"].astype(int)
```

> 剛剛有講到 ISM 的資料大多依據 50 為榮枯線，因此設定 50 以上為 1

#### step4. 在處理 CPI 跟交易量的資料壓縮前，先做出這次需要預測的重點，才能 train / validation / test 的資料切割

> 此次的目的是要預測股價，但怎麼預測? 像 LSTM 那樣預測明天的股價準確性相當不高，但如果幫我預測未來一段時間的股價呢?

#### 因此，這裡需要用特徵，幫我預測未來 10 天 / 30 天 / 60 天的平均股價，只要該預測值有效，就足以作為現在操作的方向

```
merge_df["avg_10_Close"] = merge_df["Close"].rolling(10).sum()/10
merge_df["avg_10_Close"] = merge_df["avg_10_Close"].shift(-10, axis = 0)

merge_df["avg_30_Close"] = merge_df["Close"].rolling(30).sum()/30
merge_df["avg_30_Close"] = merge_df["avg_30_Close"].shift(-30, axis = 0)

merge_df["avg_60_Close"] = merge_df["Close"].rolling(60).sum()/60
merge_df["avg_60_Close"] = merge_df["avg_60_Close"].shift(-60, axis = 0)

merge_df.dropna(inplace = True)          # 因最後的資料 10天 30天 60天會沒有未來的資料，所以需要清除掉NaN的資料
merge_df
```

> 如果我在 5/1 就知道未來 10 天 30 天 60 天的表現，高我就持有或加碼，低我就提前賣出
>
> 這裡的 10 30 60 天未來的平均股價資料，作為此機器學習的 label

![](https://i.imgur.com/lP0HJft.png)

#### Step5. 把跟價格有關的數值通通正規化: 一樣避免梯度爆掉

```
from sklearn.preprocessing import MinMaxScaler

scaler_c = MinMaxScaler(feature_range=(0, 1))
scaler_c_10 = MinMaxScaler(feature_range=(0, 1))
scaler_c_30 = MinMaxScaler(feature_range=(0, 1))
scaler_c_60 = MinMaxScaler(feature_range=(0, 1))

# 正常來講要使用train的，validation跟test配合該正規化的參數，同時忽略離群值
# 但由於2020/3月後是大牛市，train只到2019年，這些離群值不再離群了，因此採用"所有資料"做正規化
# 後續再分別train / validation / test套用同一正規化處理  =>  股市長期還是會漲，因此可能需要長期持續重新正規化才比較準確
merge_df["Close"] = scaler_c.fit_transform(merge_df["Close"].values.reshape(-1, 1))
merge_df["avg_10_Close"] = scaler_c_10.fit_transform(merge_df["avg_10_Close"].values.reshape(-1, 1))
merge_df["avg_30_Close"] = scaler_c_30.fit_transform(merge_df["avg_30_Close"].values.reshape(-1, 1))
merge_df["avg_60_Close"] = scaler_c_60.fit_transform(merge_df["avg_60_Close"].values.reshape(-1, 1))
```

![](https://i.imgur.com/1U2qkEn.png)

> 這些都做完了，我才有辦法針對 CPI 跟 Volume 進行正規化 => 避免 train validation test 沒有最後 60 天的資料

#### Step6. 把資料切成 training / validation / test + Step3 未做完的欄位

> training data: 作為訓練的資料
>
> validation data: 作為評估訓練的好壞
>
> test data: 真正查看模型是否有效

```
validation_size = test_size = int(len(merge_df)/10)

# 需特別注意是否有切對，避免使用到validation_data跟test_data進行訓練
train_size = len(merge_df) - validation_size - test_size
train_data = merge_df[:train_size]
validation_data = merge_df[train_size:train_size+validation_size]
test_data = merge_df[train_size+validation_size:]
```

```
train_data["Volume"] = train_data["Volume"].astype(int)
train_data["VolumeBand"] = pd.qcut(train_data["Volume"], 4)
train_data["VolumeBand"].value_counts().sort_values()
```

> 利用四分位把資料變成機器學習可增加學習效率的數值
>
> 作法參考 kaggle: 鐵達尼號針對數值調整的方式 https://www.kaggle.com/code/startupsci/titanic-data-science-solutions

```
combine = [train_data, validation_data, test_data]

for dataset in combine:
    dataset.loc[dataset["Volume"] <= 32560400, "Volume"] = 0
    dataset.loc[(dataset["Volume"] > 32560400) & (dataset["Volume"] <= 63623300) , "Volume"] = 1
    dataset.loc[(dataset["Volume"] > 63623300) & (dataset["Volume"] <= 101376750) , "Volume"] = 2
    dataset.loc[(dataset["Volume"] > 101376750) , "Volume"] = 3
    dataset["Volume"] = dataset["Volume"].astype(int)

train_data.drop("VolumeBand", axis = 1, inplace = True)
```

```
train_data["CPI"] = train_data["CPI"]
train_data["CPIBand"] = pd.qcut(train_data["CPI"], 4)
train_data["CPIBand"].value_counts().sort_values()

for dataset in combine:
    dataset.loc[dataset["CPI"] <= 1.464, "CPI"] = 0
    dataset.loc[(dataset["CPI"] > 1.464) & (dataset["CPI"] <= 2.076) , "CPI"] = 1
    dataset.loc[(dataset["CPI"] > 2.076) & (dataset["CPI"] <= 2.871) , "CPI"] = 2
    dataset.loc[(dataset["CPI"] > 2.871) , "CPI"] = 3
    dataset["CPI"] = dataset["CPI"].astype(int)

train_data.drop("CPIBand", axis = 1, inplace = True)
```

```
train_x = torch.tensor(train_data[["Close", "Volume", "ISM", "CPI"]].values, dtype=torch.float)
train_y = torch.tensor(train_data[["avg_10_Close", "avg_30_Close", "avg_60_Close"]].values, dtype=torch.float)

validation_x = torch.tensor(validation_data[["Close", "Volume", "ISM", "CPI"]].values, dtype=torch.float)
validation_y = torch.tensor(validation_data[["avg_10_Close", "avg_30_Close", "avg_60_Close"]].values, dtype=torch.float)
```

> 把資料分成特徵跟 label 後就可以開始學習了

#### Step7. 建立三層的 ANN 模型

```
class Model(nn.Module):

    def __init__(self, in_features = 4, h1 = 100, h2 = 100, h3 = 100, out_features = 3):
        # how many layer
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        # activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x

model = Model()
print(sum([param.numel() for param in model.parameters()]))
```

> 參數量 21003，避免太高造成 Overfitting

```
criterion_10 = nn.MSELoss()
criterion_30 = nn.MSELoss()
criterion_60 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
```

> 由於需要預測三個直，所以 loss function 用三個
>
> Adam 為目前較常使用的 Optimizer => 其公認梯度下降的速度最快，但仍有其他 optimizer 在不同場合有適合的機會，留待後續有空慢慢研究

![](https://i.imgur.com/OXpRYeK.png)

#### STEP8. 開始機器學習

```
epochs = 500
losses = []
losses_ = []

for i in range(epochs):

    model.train()

    # Forward and get a prediction
    y_pred = model.forward(train_x)

    # calculate loss
    loss_10 = criterion_10(y_pred[:,0], train_y[:, 0])
    loss_30 = criterion_30(y_pred[:,1], train_y[:, 1])
    loss_60 = criterion_60(y_pred[:,2], train_y[:, 2])

    loss = loss_10 + loss_30 + loss_60
    losses.append(loss)

    if i % 10 == 0:
        print(f"Epoch {i} and loss is: {loss}")

    # BACKPROPAGATION
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        y_pred_ = model.forward(validation_x)

        loss_10_ = criterion_10(y_pred_[:,0], validation_y[:, 0])
        loss_30_ = criterion_30(y_pred_[:,1], validation_y[:, 1])
        loss_60_ = criterion_60(y_pred_[:,2], validation_y[:, 2])

        loss_ = loss_10_ + loss_30_ + loss_60_

        losses_.append(loss_)
```

> 每一次學習都會進行 validation 資料的評估，看看是否有 overfitting

![](https://i.imgur.com/Bxr6LaN.png)

#### STEP9. 評估是否有 overfitting 的問題

```
train_losses = [loss.item() for loss in losses]
validation_losses = [loss_.item() for loss_ in losses_]

plt.plot(train_losses, label='training loss')
plt.plot(validation_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()
```

> 每一次好的訓練，誤差會持續收斂至某值，有這樣的現象就是個好的訓練 => 代表每次訓練都越來越準確
>
> 但當 training 的誤差下降時，validation 的誤差也下降才有用才有意義，否則就只是針對 training 找到最適合的模型，套用到新的資料就不準了
>
> 還好此次的模型以上兩點都有符合，看起來不用針對 overfitting 進行調整了 (early stopping)

![](https://i.imgur.com/2sbhZsE.png)

#### Step 10. 開始評估模型針對 test 的資料是否具有準確的預測

> training data: 作為訓練的資料
>
> validation data: 作為評估訓練的好壞
>
> test data: 真正查看模型是否有效 (validation 也可以，但通常用 test data)

```
test_x = torch.tensor(test_data[["Close", "Volume", "ISM", "CPI"]].values, dtype=torch.float)
test_y = torch.tensor(test_data[["avg_10_Close", "avg_30_Close", "avg_60_Close"]].values, dtype=torch.float)

model.eval()
with torch.no_grad():
    y_pred_t = model.forward(test_x)

    test_10 = y_pred_[:,0]
    test_30 = y_pred_[:,1]
    test_60 = y_pred_[:,2]

avg_c10_p = [c.item() for c in test_10]
avg_c10_t = [c.item() for c in test_y[:, 0]]

avg_10_p = scaler_c_10.inverse_transform(np.array(avg_c10_p).reshape(-1, 1))
avg_10_t = scaler_c_10.inverse_transform(np.array(avg_c10_t).reshape(-1, 1))

plt.plot(test_data.index, avg_10_p, label='predicted avg_close_10')
plt.plot(test_data.index, avg_10_t, label='true avg_close_10')
plt.title('avg_close_10')
plt.legend()
```

#### 這裡把預測的未來平均 10 天股價跟實際的未來平均 10 天股價做比較

![](https://i.imgur.com/KFC9Rq6.png)

# 這是甚麼鳥........

#### 不信邪查看 validation 的資料是否也是這樣

```
model.eval()
with torch.no_grad():
    y_pred_t = model.forward(validation_x)

    val_10 = y_pred_[:,0]
    val_30 = y_pred_[:,1]
    val_60 = y_pred_[:,2]

avg_c10_p = [c.item() for c in val_10]
avg_c10_t = [c.item() for c in validation_y[:, 0]]

avg_10_p = scaler_c_10.inverse_transform(np.array(avg_c10_p).reshape(-1, 1))
avg_10_t = scaler_c_10.inverse_transform(np.array(avg_c10_t).reshape(-1, 1))

plt.plot(validation_data.index, avg_10_p, label='predicted avg_close_10')
plt.plot(validation_data.index, avg_10_t, label='true avg_close_10')
plt.title('avg_close_10')
plt.legend()
```

![](https://i.imgur.com/5i3Vnj0.png)

#### 心安了一半，看起來要把訓練的資料，拿到很久的未來做預測比較容易失效，但當訓練的資料拿來做近期的預測，看似有效....吧?

#### training data 最後的 10 / 30 / 60 筆，都有拿到 validation 的資料 => 因為未來平均的股價，就是我們事後加工上去的

> 所以使用 validation 進行預測評估，需要扣掉後 10 筆 / 30 筆 / 60 筆才準確

```
plt.plot(validation_data.index[:20], avg_10_p[:20], label='predicted avg_close_10')
plt.plot(validation_data.index[:20], avg_10_t[:20], label='true avg_close_10')
plt.title('avg_close_10 - 10 days check')
plt.legend()
```

![](https://i.imgur.com/pHRgd0N.png)

#### 左半部為有吃進未來資料的預設，右半部則是真正的預測，由於這段時間的資料算平緩，所以看似有預測成功，但當我們檢驗 30 / 60 天的資料後會發現...

#### 30 天的預測資料

```
plt.plot(validation_data.index[:60], avg_30_p[:60], label='predicted avg_close_30')
plt.plot(validation_data.index[:60], avg_30_t[:60], label='true avg_close_30')
plt.title('avg_close_30 - 30 days check')
plt.legend()
```

![](https://i.imgur.com/UKdw424.png)

#### 60 天的預測資料

```
plt.plot(validation_data.index[:120], avg_60_p[:120], label='predicted avg_close_60')
plt.plot(validation_data.index[:120], avg_60_t[:120], label='true avg_close_60')
plt.title('avg_close_60 - 60 days check')
plt.legend()
```

![](https://i.imgur.com/5PQLdzq.png)

## 其實就是滯後的表現，跟 LSTM 預測的結果差不多......

#### 因此使用機器學習去學習有序的時間序列，效果真的不好，只是在模仿過往的股價，並沒有辦法做到有效的預測

#### 基本上機器學習，比較適合那種有標準答案的問題

> 如圖片判別，語言回答(大部分說話也是有主詞 動詞 受詞組合)

#### 真的要拿來預測某些值是不太合理的，畢竟他也只是從過往的經驗找到類似的跡象呈現，對於具有隨機性值的股市預測能力相當低

#### 因此妄想想透過機器學習在股市賺錢是不太可能的，影響市場的因素太多，只能拿來驗證某些看似有獲利空間的操作是否有效

#### 但要拿來預測的準確性卻是不足的....

## 透過初步了解機器學習及應用，應當注意其可用性，並不是所有的東西都可以靠機器學習來做預測，但試著嘗試，起碼之後就比別人更加知道那些可行，哪些不可行
