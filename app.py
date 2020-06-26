from flask import Flask, jsonify, request
from tensorflow import keras
import yaml
from datetime import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import zscore

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('config.yaml', 'r') as read_config:
    config = yaml.load(read_config)
    file = config['file']
    model = config['model']

def makeUsefulDf(df, noise=2.5, hours_prior=0):
    """
    Turn a dataframe of datetime and load data into a dataframe useful for
    machine learning. Normalize values.
    """
    def _isHoliday(holiday, df):
        m1 = None
        if holiday == "New Year's Day":
            m1 = ((df["dates"].dt.month == 1) & (df["dates"].dt.day == 1))
        if holiday == "Independence Day":
            m1 = (df["dates"].dt.month == 12) & (df["dates"].dt.day == 16)
        if holiday == "Womens Day":
            m1 = (df["dates"].dt.month == 3) & (df["dates"].dt.day == 8)
        if holiday == "Victory Day":
            m1 = (df["dates"].dt.month == 5) & (df["dates"].dt.day == 9)
        if holiday == "Astana Day":
            m1 = (df["dates"].dt.month == 7) & (df["dates"].dt.day == 6)
        if holiday == "Constitution Day":
            m1 = (df["dates"].dt.month == 8) & (df["dates"].dt.day == 30)
        if holiday == "Labor Day":
            m1 = (df["dates"].dt.month == 5) & (df["dates"].dt.day == 1)
        return m1 #| m2
    
    if 'dates' not in df.columns:
        df['dates'] = df.apply(lambda x: dt(int(x['year']), int(x['month']), int(x['day']), int(x['hour'])), axis=1)

    r_df = pd.DataFrame()
    # LOAD
    r_df["load_n"] = zscore(df["load"])
    r_df["load_prev_n"] = r_df["load_n"].shift(hours_prior)
    r_df["load_prev_n"].bfill(inplace=True)

    # LOAD PREV
    def _chunks(l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]
    n = np.array([val for val in _chunks(list(r_df["load_n"]), 24) for _ in range(24)])
    l = ["l" + str(i) for i in range(24)]
    for i, s in enumerate(l):
        r_df[s] = n[:, i]
        r_df[s] = r_df[s].shift(hours_prior)
        r_df[s] = r_df[s].bfill()
    r_df.drop(['load_n'], axis=1, inplace=True)
        
        # DATE
    r_df["years_n"] = df["dates"].dt.year/2019
    r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.hour, prefix='hour')], axis=1)
    r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.dayofweek, prefix='day')], axis=1)
    r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.month, prefix='month')], axis=1)
    for holiday in ["New Year's Day", "Womens Day", "Independence Day", "Astana Day", "Victory Day", "Labor Day","Constitution Day"]:
        r_df[holiday] = _isHoliday(holiday, df)
    
	# TEMP
    temp_noise = df['tempc'] + np.random.normal(0, noise, df.shape[0])
    return r_df

app = Flask(__name__)
model = keras.models.load_model(model)


@app.route('/health')
def health():
    return('Health ok')


@app.route("/", methods=["POST"])
def index():
   
    data = request.json
    
    df = pd.read_csv(file)
    df1 = pd.DataFrame.from_dict(data)
    all_X=makeUsefulDf(df.append(df1,ignore_index=True))
    all_X = all_X.astype('float64')
    predictions = [float(f) for f in model.predict(all_X[-24:])]
   
    return jsonify({"data": predictions})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)