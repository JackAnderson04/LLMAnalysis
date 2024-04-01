import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

data = pd.read_feather('house_sales.ftr')

columns = ['Sold Price', 'Sold On', 'Type', 'Year built', 'Bedrooms','Total spaces',
            'Bathrooms', 'Zip Code', 'Cooling features', 'Lot size'] #Selected attributes
df0 = data[columns].copy()
#uncomment the below line to save memory
#del data


#change this attribute to numerical type instead of object type
df0['Sold Price'] = np.log10(df0['Sold Price'].replace(
        r'[$,-]', '', regex=True).replace(
        r'^\s*$', np.nan, regex=True).astype(float))
df1 = df0[(df0['Sold Price'] >= 4 ) & (df0['Sold Price'] <= 8 )] # use the prices between 10^4 and 10^8 only

#handle attribute type
df1 = df1[df1['Type'].isin(['SingleFamily', 'Condo', 'MultiFamily', 'Townhouse'])]
df1['Type'].value_counts()
df1['Type'] = df1['Type'].astype('category').cat.codes # convert to categorical and then conver to integers
df1['Type'].value_counts()

#handle attribute Year built
df1['Year built'].value_counts()
df1 = df1[df1['Year built'] != 'No Data']#remove "No Data" -- do any missing-value imputation methods work here?
df1['Year built'] = df1['Year built'].astype('int')
df1 = df1[(df1['Year built']>1900)  & (df1['Year built'] < 2023)]# get rid of too old or irregular value 
df1['Year built'].value_counts()

#handle attribute Bedrooms
df1['Bedrooms'].value_counts()
df1  = df1[df1['Bedrooms'].isin([str(i) for i in range(11)])] # we see a lot of noises; let's keep only the reasonable ones[0, 10] bedrooms
df1['Bedrooms'] = df1['Bedrooms'].astype('int')
df1['Bedrooms'].value_counts()

#handle attribute Total spaces
df1['Total spaces'] = df1['Total spaces'].astype(float)
df1.dropna(subset=['Total spaces'], inplace=True)
df1['Total spaces'] = df1['Total spaces'].astype(int)

#handle attribute Bathrooms
df1['Bathrooms'] = df1['Bathrooms'].replace(['No Data', 'None','NaN', ''], np.nan)
df1['Bathrooms'] = df1['Bathrooms'].astype(float)
df1.dropna(subset=['Bathrooms'], inplace=True)
df1['Bathrooms'] = df1['Bathrooms'].astype(int)

#handle attribute Zip Code
df1['Zip Code'] = df1['Zip Code'].astype(str)
df1['Zip Code'] = df1['Zip Code'].replace(['None', ''], np.nan)
df1['Zip Code'] = df1['Zip Code'].astype('category').cat.codes

#handle attribute Cooling Features
df1 = df1[df1['Cooling features'].isin(['Central Air, Dual', 'Central', 'Central Air, Electric',
                             'Wall/Window Unit(s)', 'Ceiling Fan(s), Whole House Fan',
                             'None', 'Evaporative', 'No Air Conditioning',
                             'Central, Solar', 'Electric', 'Wall',
                             'Ceiling Fan(s), Wall/Window'])]
df1['Cooling features'] = df1['Type'].astype('category').cat.codes
df1['Type'].value_counts()

#handle attribute Lot size
acre_threshold = 10  
# Convert values to square feet
def convert_to_sqft(value):
    try:
        if pd.isna(value):
            return np.nan
        elif 'sqft' in value:
            return float(value.replace('sqft', '').replace(',', ''))
        else:
            # Assuming values below the threshold are in acres
            num = float(value)
            return num * 43560 if num < acre_threshold else num
    except ValueError:
        return np.nan
df1['Lot size'] = df1['Lot size'].apply(convert_to_sqft)
df1.dropna(subset=['Lot size'], inplace=True)
df1['Lot size'] = df1['Lot size'].astype(int)

print(df1)

def split_data(df, train_timestamp):
    test_start, test_end = pd.Timestamp(2021, 2, 15), pd.Timestamp(2021, 3, 1)
    train_start = train_timestamp  
    df['Sold On'] = pd.to_datetime(df['Sold On'], errors='coerce')
    train = df[(df['Sold On'] >= train_start) & (df['Sold On'] < test_start)]
    test = df[(df['Sold On'] >= test_start) & (df['Sold On'] < test_end)]
    return train, test

def prepare_train_test(train, test, features):
    target = "Sold Price"
    X_train = train.loc[:, features ]
    y_train = train.loc[:, [target]]
    X_test = test.loc[:, features]
    y_test = test.loc[:, [target]]
    return X_train, y_train, X_test, y_test

def modeling (x_train, y_train, x_test, y_test):
    #training and test data as input
    print('Starting training...')
    #train
    gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=100)
    gbm.fit(x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='l2',
        callbacks=[lgb.early_stopping(5)])
    print('Starting predicting...') #predict
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)#eval
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5 #use root mean squared error (RMSE)
    print(f'The RMSE of prediction is: {rmse_test}')
    return gbm


train, test = split_data(df1, pd.Timestamp(2021, 1, 1))
features = ["Type", "Year built", 'Bedrooms','Bathrooms', 'Total spaces', 'Zip Code',
             'Cooling features', 'Lot size'] #specify the selected features
X_train, y_train, X_test, y_test = prepare_train_test(train, test, features)

predictor = modeling(X_train, y_train, X_test, y_test) #model
X_train.shape


lgb.plot_importance(predictor)
plt.show()

#allcollumns = data.columns.tolist()
#print(allcollumns)