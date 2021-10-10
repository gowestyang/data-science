'''
Environment Infomation:
* Win10 x64
* Python 3.8.8 (Anaconda)
* numpy 1.20.3
* pandas 1.3.2
* fastparquet 0.7.1
* keras 2.4.3
* scikit-learn 0.24.2
* joblib 1.0.1
* pydantic 1.8.2
* fastapi 0.68.1
* uvicorn 0.15.0
'''

# Load trained model
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

model = load_model('model/merchant_recommender')
encUser = load('model/user_encoder.joblib')
encMerchant = load('model/merchant_encoder.joblib')

dfTop5TrendingMerchant = pd.read_parquet('model/top_5_trending_merchant.parquet', engine='fastparquet')
dfTop5NewMerchant = pd.read_parquet('model/top_5_new_merchant.parquet', engine='fastparquet')
dfTop10PopularMerchant = pd.read_parquet('model/top_10_popular_merchant.parquet', engine='fastparquet')

TOP_N = 3

class FixedMinMaxScaler():
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value
    
    def transform(self, ar):
        return (ar - self.min) / (self.max - self.min)
    
    def inverse_transform(self, ar):
        return ar*(self.max - self.min) + self.min
    
scaler = FixedMinMaxScaler(0, 9)

# Function to make recommendations to a given user Id
def MakeRecommendation(userId):
    if userId in encUser.classes_:
        sUserType = 'existing'
        
        # Prepare input array
        arUserIndex = encUser.transform([userId])
        arAllMerchantIndex = encMerchant.transform(encMerchant.classes_)

        arUserMerchantIndex = np.hstack((
            [arUserIndex]*len(arAllMerchantIndex),
            arAllMerchantIndex.reshape(-1,1)
        ))

        # Predict
        arPred = model.predict(arUserMerchantIndex).flatten()

        arTopMerchantPredIndex = arPred.argsort()[-TOP_N:][::-1]
        arTopMerchantPredRating = scaler.inverse_transform(arPred[arTopMerchantPredIndex])
        arTopMerchantPredRating = ["%.3f" % f for f in arTopMerchantPredRating]

        arTopMerchantIndex = arAllMerchantIndex[arTopMerchantPredIndex]
        arTopMerchantId = encMerchant.inverse_transform(arTopMerchantIndex)

        # Also recommend a trending merchant and a new merchant
        iTrendingMerchant = dfTop5TrendingMerchant[~dfTop5TrendingMerchant['merchant_id'].isin(arTopMerchantId)]['merchant_id'].sample(1).iloc[0]
        arTopMerchantId = np.append(arTopMerchantId, iTrendingMerchant)

        iNewMerchant = dfTop5NewMerchant[~dfTop5NewMerchant['merchant_id'].isin(arTopMerchantId)]['merchant_id'].sample(1).iloc[0]
        arTopMerchantId = np.append(arTopMerchantId, iNewMerchant)

        arTopMerchantPredRating = np.append(arTopMerchantPredRating, ['nan']*2)

    else:
        sUserType = 'new'
        
        # recommend 3 popular merchants, one trending merchant and one new merchant
        arTopMerchantId = dfTop10PopularMerchant['merchant_id'].sample(3)
        
        iTrendingMerchant = dfTop5TrendingMerchant[~dfTop5TrendingMerchant['merchant_id'].isin(arTopMerchantId)]['merchant_id'].sample(1).iloc[0]
        arTopMerchantId = np.append(arTopMerchantId, iTrendingMerchant)

        iNewMerchant = dfTop5NewMerchant[~dfTop5NewMerchant['merchant_id'].isin(arTopMerchantId)]['merchant_id'].sample(1).iloc[0]
        arTopMerchantId = np.append(arTopMerchantId, iNewMerchant)
        
        arTopMerchantPredRating = ['nan']*5

    return {
        'user_type':sUserType,
        'merchant_id':str(arTopMerchantId[0]),
        'ar_merchant_id':'|'.join(arTopMerchantId.astype(str)),
        'ar_rating':'|'.join(arTopMerchantPredRating)
    }

# Function to make batch recommendations given a list of user Ids
def MakeBatchRecommendation(lUserId):
    lRecommendations = list(map(MakeRecommendation, lUserId))
    keys = lRecommendations[0].keys()
    return {k: ",".join([dic[k] for dic in lRecommendations]) for k in keys}

# API
from pydantic import BaseModel
from fastapi import FastAPI

class Inputs(BaseModel):
    user_id: int

class BatchInputs(BaseModel):
    list_user_id: list

app = FastAPI()

@app.get("/")
async def root():
    return {'message': 'Merchant Recommender'}

@app.post('/predict')
def predcit(data: Inputs):
    data = data.dict()
    userId = data['user_id']
    return MakeRecommendation(userId)

@app.post('/batch_predict')
def predcit(data: BatchInputs):
    data = data.dict()
    lUserId = data['list_user_id']
    return MakeBatchRecommendation(lUserId)