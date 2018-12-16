import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import xgboost as xgb
import numpy as np

dtype_for_train={'date_time':np.str_,
       'site_name':np.str_,
       'posa_continent':np.str_,
       'user_location_country':np.str_,
       'user_location_region':np.str_,
       'user_location_city':np.str_,
       'orig_destination_distance':np.float64,
       'user_id':np.str_,
       'is_mobile':np.str_,
       'is_package':np.str_,
       'channel':np.str_,
       'srch_ci':np.str_,
       'srch_co':np.str_,
       'srch_adults_cnt':np.int32,
       'srch_children_cnt':np.int32,
       'srch_rm_cnt':np.int32,
       'srch_destination_id':np.str_,
       'srch_destination_type_id':np.str_,
       'is_booking':bool,
       'cnt':np.str_,
       'hotel_continent':np.str_,
       'hotel_country':np.str_,
       'hotel_market':np.str_,
       'hotel_cluster':np.str_}

dtype_for_test = {'date_time':np.str_,
       'site_name':np.str_,
       'posa_continent':np.str_,
       'user_location_country':np.str_,
       'user_location_region':np.str_,
       'user_location_city':np.str_,
       'orig_destination_distance':np.float64,
       'user_id':np.str_,
       'is_mobile':np.str_,
       'is_package':np.str_,
       'channel':np.str_,
       'srch_ci':np.str_,
       'srch_co':np.str_,
       'srch_adults_cnt':np.int32,
       'srch_children_cnt':np.int32,
       'srch_rm_cnt':np.int32,
       'srch_destination_id':np.str_,
       'srch_destination_type_id':np.str_,
       'hotel_continent':np.str_,
       'hotel_country':np.str_,
       'hotel_market':np.str_}
       
train = pd.read_csv("/Users/seungminbaek/Desktop/expedia/train.csv", dtype=dtype_for_train, usecols=dtype_for_train, parse_dates=['date_time'], sep=',')
test = pd.read_csv("/Users/seungminbaek/Desktop/expedia/test.csv", dtype=dtype_for_test, usecols=dtype_for_test, parse_dates=['date_time'], sep=',')

## Before dealing with this data, I want to convert date_time's float type to
# panda's datetime type. And I will create year and month column

train['date_time'] = pd.to_datetime(train['date_time'])
train['srch_ci'] = pd.to_datetime(train['srch_ci'], infer_datetime_format = True, errors='coerce')
train['srch_co'] = pd.to_datetime(train['srch_co'], infer_datetime_format = True, errors='coerce')

train['year'] = train['date_time'].dt.year
train['month'] = train['date_time'].dt.month

train = train[train.srch_ci.isnull() == False]

train['checkin_month'] = train['srch_ci'].dt.month
train['checkin_year'] = train['srch_ci'].dt.year
train['checkout_month'] = train['srch_co'].dt.month
train['checkout_year'] = train['srch_co'].dt.year

test['date_time'] = pd.to_datetime(test['date_time'])
test['srch_ci'] = pd.to_datetime(test['srch_ci'], infer_datetime_format = True, errors='coerce')
test['srch_co'] = pd.to_datetime(test['srch_co'], infer_datetime_format = True, errors='coerce')

test['year'] = test['date_time'].dt.year
test['month'] = test['date_time'].dt.month

test = test[test.srch_ci.isnull() == False]
test['checkin_year'] = test['srch_ci'].dt.year
test['checkin_month'] = test['srch_ci'].dt.month
test['checkout_year'] = test['srch_co'].dt.year
test['checkout_month'] = test['srch_co'].dt.month

# It returns (37670293, 24). It has too many data. So we have to minimize the data.
# Our plan is eliminating ids in test.csv from train.csv
sample_train = train[~train.user_id.isin(test.user_id)]

# I want to check our sampling works well. So I will make a unique_sample_train_ids to compare it really doesn't have
# test.unique_ids
unique_sample_train_ids = set(sample_train.user_id.unique())
unique_orig_train_ids = set(train.user_id.unique())
print(len(unique_orig_train_ids))
## It returns unique ids in original train. It returns 1198786
print(len(unique_sample_train_ids))
# It returns unique ids in sample train. It returns 17209
print(len(sample_train.user_id))
# It returns 456131.
 

# And Now, I want to deal with the null data points
#Now, we have to check how many users booked in sample train data
booked_sample_train = sample_train[sample_train.is_booking == 1]
print(len(booked_sample_train))
# In sample train data, there are 31908

# I want to check how many null data points in original train data
print(train.isnull().sum(axis=0))
##date_time                           0
##site_name                           0
##posa_continent                      0
##user_location_country               0
##user_location_region                0
##user_location_city                  0
##orig_destination_distance    13525001
##user_id                             0
##is_mobile                           0
##is_package                          0
##channel                             0
##srch_ci                         47083
##srch_co                         47084
##srch_adults_cnt                     0
##srch_children_cnt                   0
##srch_rm_cnt                         0
##srch_destination_id                 0
##srch_destination_type_id            0
##is_booking                          0
##cnt                                 0
##hotel_continent                     0
##hotel_country                       0
##hotel_market                        0
##hotel_cluster                       0
##dtype: int64

# Now, I want to check how many null data points in sampled data.
print(sample_train.isnull().sum(axis=0))
##date_time                         0
##site_name                         0
##posa_continent                    0
##user_location_country             0
##user_location_region              0
##user_location_city                0
##orig_destination_distance    183290
##user_id                           0
##is_mobile                         0
##is_package                        0
##channel                           0
##srch_ci                         570
##srch_co                         570
##srch_adults_cnt                   0
##srch_children_cnt                 0
##srch_rm_cnt                       0
##srch_destination_id               0
##srch_destination_type_id          0
##is_booking                        0
##cnt                               0
##hotel_continent                   0
##hotel_country                     0
##hotel_market                      0
##hotel_cluster                     0
##dtype: int64

## So, orig_destination_distance, srch_ci, srch_co are attributes which contain null data
# We will eliminate the row which srch_ci and srch_co are empty
#sample_train = sample_train[sample_train.srch_ci.isnull() == False]
#print(sample_train.isnull().sum(axis=0))
##date_time                         0
##site_name                         0
##posa_continent                    0
##user_location_country             0
##user_location_region              0
##user_location_city                0
##orig_destination_distance    183046
##user_id                           0
##is_mobile                         0
##is_package                        0
##channel                           0
##srch_ci                           0
##srch_co                           0
##srch_adults_cnt                   0
##srch_children_cnt                 0
##srch_rm_cnt                       0
##srch_destination_id               0
##srch_destination_type_id          0
##is_booking                        0
##cnt                               0
##hotel_continent                   0
##hotel_country                     0
##hotel_market                      0
##hotel_cluster                     0
##dtype: int64

##Now, we have to deal with orig_destination_distance.
#We have two option.
#First: Replace all null data with average distance from original destination distance
#Second: Using multiple amputation
#Even though, we are programming with sample_train, we be more precise,
# I will calculate the average from train, not sample_train
avg = train.orig_destination_distance.mean()
print(avg)
# It returns 1970.090026720705
sample_train['orig_destination_distance']= sample_train.orig_destination_distance.fillna(avg)
print(sample_train.isnull().sum(axis=0))
##date_time                    0
##site_name                    0
##posa_continent               0
##user_location_country        0
##user_location_region         0
##user_location_city           0
##orig_destination_distance    0
##user_id                      0
##is_mobile                    0
##is_package                   0
##channel                      0
##srch_ci                      0
##srch_co                      0
##srch_adults_cnt              0
##srch_children_cnt            0
##srch_rm_cnt                  0
##srch_destination_id          0
##srch_destination_type_id     0
##is_booking                   0
##cnt                          0
##hotel_continent              0
##hotel_country                0
##hotel_market                 0
##hotel_cluster                0
##dtype: int64


# Now, I will divide target(y) and information(x)
# Since test data doesn't have hotel_cluster, is_booking, and cnt,
# we will drop them from sample_train
# And also, since I created year and month, I will drop date_time
sample_train = sample_train.drop(['date_time', 'srch_ci', 'srch_co'], axis=1)
x = sample_train.drop(['hotel_cluster', 'is_booking', 'cnt'], axis=1)
y = sample_train['hotel_cluster']

from sklearn.model_selection import train_test_split
# We chose test_size = 0.2, because we want to divide into 80% and 20%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# And now, I want to use RandomForestClassifier Algorithm for machine learning
from sklearn.ensemble import RandomForestClassifier
rftree = RandomForestClassifier(n_estimators=31, max_depth=10, random_state = 123)
rftree.fit(x_train, y_train)

features = rftree.feature_importances_
# Because I want to see top 20 features
i = np.argsort(features)[::-1][:20]
features[i]

# I wanted to see top 20, so I set range 20
plt.barh(range(20), features[i], color='r')
plt.yticks(range(20), x_train.columns[20])
plt.xlabel('features importances')
plt.show()

clusters = {}
for (k,v) in enumerate(rftree.classes_):
    clusters[k] = v

prediction = rftree.predict_proba(x_test)
# I want top 5 hotel cluster
sorted_prediction = prediction.argsort(axis=1)[:,-5:]

#c_prediction will be used for cluster prediction
c_prediction = []
for i in sorted_prediction.flatten():
    c_prediction.append(clusters.get(i))
    
my_clusters = np.array(c_prediction).reshape(sorted_prediction.shape)

## Now we have to deal with test data. Since it has more value than sample_train data, we have
# to simplify the data first
# I want 2% of the data from the test. Before that we have to deal with the null data first

test_dis_avg = test.orig_destination_distance.mean()
test['orig_destination_distance'] = test.orig_destination_distance.fillna(test_dis_avg)

# Now, pick 2% of the data
sample_test = test.sample(frac=0.02)
sample_test = sample_test.drop(['date_time', 'srch_ci', 'srch_co'], axis=1)

#Now, we need target which is the hotel cluster in train
my_hotel_cluster = sample_train['hotel_cluster']
learned_train = sample_train.drop(['hotel_cluster', 'is_booking', 'cnt'], axis=1)

mlearning = RandomForestClassifier(n_estimators=31, max_depth =10, random_state=123)
mlearning.fit(learned_train, my_hotel_cluster)

features = mlearning.feature_importances_
i = np.argsort(features)[::-1][:20]
features[i]

plt.barh(range(20), features[i], color='r')
plt.yticks(range(20), learned_train.columns[i])
plt.xlabel('Feature Importances')
plt.show()

prediction = mlearning.predict_proba(sample_test)
sorted_prediction = prediction.argsort(axis=1)[:,-5:]
clusters = {}
for (k,v) in enumerate(rftree.classes_):
    clusters[k] = v
    
c_prediction = []
for i in sorted_prediction.flatten():
    c_prediction.append(clusters.get(i))
my_clusters = np.array(c_prediction).reshape(sorted_prediction.shape)
my_clusters = list(my_clusters)
data = {'id': sample_test['user_id'], 'hotel_cluster': my_clusters}
submission = pd.DataFrame(data)
submission.to_csv("mission_complete.csv")
