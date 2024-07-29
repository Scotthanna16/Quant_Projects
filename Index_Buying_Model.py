import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


def pred_index(parameters, model,training_data, testing_data):
    model.fit(training_data[parameters], training_data["Buy/Sell"])
    predictions=model.predict_proba(testing_data[parameters])[:,1]
    predictions[predictions>=.5] = 1
    predictions[predictions <.5] = 0
    predictions= pd.Series(predictions, index=testing_data.index, name = "Predictions")
    return pd.concat([testing_data["Open"],testing_data["Close"],testing_data["Next"],testing_data["Buy/Sell"],predictions], axis=1)
    

def backtest(parameters,Index, model, back_time=2500, jump_time=250):
    all_predictions = []
    for i in range(back_time, Index.shape[0], jump_time):
        training_data=Index.iloc[0:i].copy()
        testing_data = Index.iloc[i:(i+jump_time)].copy()
        predictions=pred_index(parameters,model, training_data,testing_data)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


def get_Profit(predictions):
    total_Profit=0
    for entry in range(len(predictions)):
        if predictions["Predictions"][entry]==1:
            total_Profit+=1
            total_Profit += (predictions["Next"][entry]-predictions["Close"][entry])
    return total_Profit
            
        

def Index_Model(Index, cutoff_Date,num_est, mss, Prominent_Stocks, rs=1, RA_time= [2,5,60,250,1000], ):
    Stock_Index = yf.Ticker(Index)
    Stock_Index=Stock_Index.history(period="max")
    Stock_Index["Next"]=Stock_Index["Close"].shift(-1)
    Stock_Index["Buy/Sell"] = (Stock_Index["Next"]>Stock_Index["Close"]).astype(int)
    Stock_Index=Stock_Index.loc[cutoff_Date:].copy()

    parameters = []

    
    for ticker in Prominent_Stocks:
        Stock = yf.Ticker(ticker)
        Stock =Stock.history(period="max")
        Stock["Next"]=Stock["Close"].shift(-1)
        Stock["Buy/Sell"] = (Stock["Next"]>Stock["Close"]).astype(int)
        Stock=Stock.loc["1990-01-01":].copy()
        for time in RA_time[:3]:
            ra = Stock_Index.rolling(time).mean()
            rc= f"{ticker}_Close_Ratio_{time}"
            Stock_Index[rc] = Stock["Close"]/ra["Close"]
            tc = f"{ticker}_Up/Down_{time}"
            Stock_Index[tc] = Stock.shift(1).rolling(time).sum()["Buy/Sell"]
            parameters+=[rc,tc]
        

    for time in RA_time:
        ##Rolling Average
        ra = Stock_Index.rolling(time).mean()
        ##Close Ratio
        rc= f"Close_Ratio_{time}"
        Stock_Index[rc] = Stock_Index["Close"]/ra["Close"]
        ##Trend
        tc = f"Up/Down_{time}"
        Stock_Index[tc] = Stock_Index.shift(1).rolling(time).sum()["Buy/Sell"]
        ##Add to Parameters
        parameters+=[rc,tc]
    
    
    
    ##Rid of NaN
    Stock_Index=Stock_Index.dropna()
    

    ##Build Model
    model=RandomForestClassifier(n_estimators = num_est, min_samples_split = mss, random_state=rs)

    #Backtest
    predictions = backtest( parameters,Stock_Index, model)
    print(predictions["Predictions"].value_counts())
    
    print(precision_score(predictions["Buy/Sell"], predictions["Predictions"]))
    print(predictions)
    print(get_Profit(predictions))
    
    

if __name__ == '__main__':
    Index_Model("^GSPC","1990-01-01",200,50, ["AMZN","AAPL","MSFT"])
    
    
