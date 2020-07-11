from flask import Flask, jsonify, request, g, render_template
from tensorflow import keras
import yaml
from datetime import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import zscore
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging, sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__, static_url_path='/health', static_folder='./public')

class Model:

    def __init__(self, holidays, executable_model, **kwargs):

        self.model = None
        self.holidays = holidays
        self.noise = kwargs['noise']#2.5
        self.hours_prior = kwargs['hours_prior']#0

        self.executable_model = executable_model

    def _isHoliday(self, df):

        if "New Year's Day" in self.holidays:
            self.model = ((df["dates"].dt.month == 1) & (df["dates"].dt.day == 1))
        if "Independence Day" in self.holidays:
            self.model = (df["dates"].dt.month == 12) & (df["dates"].dt.day == 16)
        if "Womens Day" in self.holidays:
            self.model = (df["dates"].dt.month == 3) & (df["dates"].dt.day == 8)
        if "Victory Day" in self.holidays:
            self.model = (df["dates"].dt.month == 5) & (df["dates"].dt.day == 9)
        if "Astana Day" in self.holidays:
            self.model = (df["dates"].dt.month == 7) & (df["dates"].dt.day == 6)
        if "Constitution Day" in self.holidays:
            self.model = (df["dates"].dt.month == 8) & (df["dates"].dt.day == 30)
        if "Labor Day" in self.holidays:
            self.model = (df["dates"].dt.month == 5) & (df["dates"].dt.day == 1)

        for holiday in self.holidays:
            df[holiday] = self._isHoliday(df)

    def _chunks(self, l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]

    def _makeUsefulDf(self, df):
        """
        Turn a dataframe of datetime and load data into a dataframe useful for
        machine learning. Normalize values.
        """
        if 'dates' not in df.columns:
            df['dates'] = df.apply(lambda x: dt(int(x['year']), int(x['month']), int(x['day']), int(x['hour'])), axis=1)

        r_df = pd.DataFrame()

        # LOAD
        r_df["load_n"] = zscore(df["load"])
        r_df["load_prev_n"] = r_df["load_n"].shift(self.hours_prior)
        r_df["load_prev_n"].bfill(inplace=True)

        # LOAD PREV
        n = np.array([val for val in self._chunks(list(r_df["load_n"]), 24) for _ in range(24)])
        l = ["l" + str(i) for i in range(24)]

        for i, s in enumerate(l):
            r_df[s] = n[:, i]
            r_df[s] = r_df[s].shift(self.hours_prior)
            r_df[s] = r_df[s].bfill()
        r_df.drop(['load_n'], axis=1, inplace=True)

        # DATE
        r_df["years_n"] = df["dates"].dt.year/2019

        r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.hour, prefix='hour')], axis=1)
        r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.dayofweek, prefix='day')], axis=1)
        r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.month, prefix='month')], axis=1)

        self._isHoliday(r_df)
        logger.info(r_df.head())

        # TEMP
        temp_noise = df['tempc'] + np.random.normal(0, self.noise, df.shape[0])

        return r_df

    def execute(self, static, live):

        df = static.append(live, ignore_index=True)

        df_all_x = self._makeUsefulDf(df).astype('float64')

        predictions = [float(f) for f in self.executable_model.predict(df_all_x[-24:])]

        return predictions

@app.before_request
def before_request():
    g.request_start_time = time.time()
    g.request_time = lambda: "%.5fs" % (time.time() - g.request_start_time)


@app.route('/health')
def health():
    t = request.values.get('t', 0)
    time.sleep(float(t)) #just to show it works...
    return render_template("index.html")


@app.route("/", methods=["POST"])
def index():

    predictions = None

    try:

        data = request.json
        df_from_request = pd.DataFrame.from_dict(data)
        # columns=['load','tempc','year','month','day','hour']
        logger.info(f'Data as df recieved from request: {df_from_request.head()}')
        logger.info(f'Data as df recieved from file: {df_from_file.head()}')

        predictions = model_body.execute(static=df_from_file, live=df_from_request)

        logger.info(f'Prediction: {predictions}')

    except Exception as error:
        print(f'index() rise exception: {error}')

    return jsonify({"data": predictions})


if __name__ == "__main__":

    with open('models/config.yaml', 'r') as read_config:
        config = yaml.full_load(read_config)

    file = config['file']
    model = config['model']
    noise = config['noise']
    hours_prior = config['hours_prior']
    holidays = config['holidays']

    df_from_file = pd.read_csv(file)

    exec_model = keras.models.load_model(model)

    model_body = Model(holidays, exec_model, noise=noise, hours_prior=hours_prior)

    logger.info(f'Start webserver')

    app.run(host='0.0.0.0', port=3020, use_reloader=True)