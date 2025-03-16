import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df1 = pd.read_csv('archive123/nsw_property_data.csv')
df2 = df1.drop(['property_id', 'download_date', 'council_name', 'strata_lot_number', 'property_name', 'contract_date', 'legal_description'], axis=1)
df3 = df2.dropna()
df5 = df3.copy()
df5["bhk"] = np.random.randint(1, 6, df5.shape[0])
df5['bath'] = np.random.randint(1, 6, df5.shape[0])
df6 = df5.copy()
df6.loc[df6['area_type'] == 'H', 'area_type'] = 107639
df6.loc[df6['area_type'] == 'M', 'area_type'] = 10.7639
df6['area-sqft'] = df6['area']*df6['area_type']
df7 = df6.drop(['area', 'area_type'], axis=1)
df8 = df7.groupby('address')['address'].agg('count').sort_values(ascending=True)
df9 = df7.copy()
le = LabelEncoder()
df10 = df9.copy()
df10['address'] = le.fit_transform(df10['address'])
df10['property_type'] = le.fit_transform(df10['property_type'])
df10['settlement_date'] = le.fit_transform(df10['settlement_date'])
df10['zoning'] = le.fit_transform(df10['zoning'])
df10['nature_of_property'] = le.fit_transform(df10['nature_of_property'])
df10['primary_purpose'] = le.fit_transform(df10['primary_purpose'])


X = df10.drop('purchase_price', axis=1)
Y = df10.purchase_price
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train.values, Y_train.values)
print(lr.score(X_test, Y_test))

# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import cross_val_score
# cv =  ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
# cross_val_score(LinearRegression(), X, Y, cv=cv)


def predict_price(arr):
    x = np.zeros(len(X.columns))
    x[0] = arr[0]
    x[1] = arr[1]
    x[2] = arr[2]
    x[3] = arr[3]
    x[4] = arr[4]
    x[5] = arr[5]
    x[6] = arr[6]
    x[7] = arr[7]
    x[8] = arr[8]
    x[9] = arr[9]
    # if loc_index >= 0:
        # x[loc_index] = 1
    return lr.predict([x])[0]
    # address	post_code	property_type	settlement_date	zoning	nature_of_property	primary_purpose	bhk	bath	area-sqft


print(predict_price([261722, 2482.0, 0, 9285, 47, 1, 5002, 4, 4, 368017.741]))


