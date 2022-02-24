import plotly.express as px
import pandas as pd
from datetime import datetime
import numpy as np
from pandas import DatetimeIndex
from pycaret.regression import *
import matplotlib

# read csv file
data = pd.read_csv('C:\\Users\\Best\\Documents\\AirPassengers.csv')
#print(data)

data['Date'] = pd.to_datetime(data['Month'])
data.head()

# create 12 month moving average
data['MA12'] = data['Passengers'].rolling(12).mean()

# plot the data and MA
fig = px.line(data, x="Date", y=["Passengers", "MA12"], template = 'plotly_dark')
fig.show()

# extract month and year from dates
month = [i.month for i in data['Date']]
year = [i.year for i in data['Date']]

data['Month'] = DatetimeIndex(data['Date']).month
data['Year'] = DatetimeIndex(data['Date']).year

# create a sequence of numbers
data['Series'] = np.arange(1, len(data)+1)
# drop unnecessary columns and re-arrange
data.drop(['Date', 'MA12'], axis=1, inplace=True)
data = data[['Series', 'Year', 'Month', 'Passengers']]
#print(data)
# check the head of the dataset
data.head()


# split data into train-test set
train = data[data['Year'] < 1960]
test = data[data['Year'] >= 1960]
# check shape

a = train.shape, test.shape
#>>>((132, 4), (12, 4))


# import the regression module
# initialize setup
s = setup(data = train, test_data = test, target = 'Passengers', fold_strategy = 'timeseries', numeric_features = ['Year', 'Series'], fold = 3, transform_target = True, session_id = 123)

best = compare_models(sort = 'MAE')


prediction_holdout = predict_model(best);

# generate predictions on the original dataset
predictions = predict_model(best, data=data)
# add a date column in the dataset
predictions['Date'] = pd.date_range(start='1949-01-01', end = '1960-12-01', freq = 'MS')
# line plot
fig = px.line(predictions, x='Date', y=["Passengers", "Label"], template = 'plotly_dark')
# add a vertical rectangle for test-set separation
fig.add_vrect(x0="1960-01-01", x1="1960-12-01", fillcolor="grey", opacity=0.25, line_width=0)
fig.show()


final_best = finalize_model(best)

future_dates = pd.date_range(start = '1961-01-01', end = '1965-01-01', freq = 'MS')
future_df = pd.DataFrame()
future_df['Month'] = [i.month for i in future_dates]
future_df['Year'] = [i.year for i in future_dates]
future_df['Series'] = np.arange(145,(145+len(future_dates)))
future_df.head()


predictions_future = predict_model(final_best, data=future_df)
predictions_future.head()


concat_df = pd.concat([data,predictions_future], axis=0)
concat_df_i = pd.date_range(start='1949-01-01', end = '1965-01-01', freq = 'MS')
concat_df.set_index(concat_df_i, inplace=True)
fig = px.line(concat_df, x=concat_df.index, y=["Passengers", "Label"], template = 'plotly_dark')
fig.show()

