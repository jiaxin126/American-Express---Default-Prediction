#!/usr/bin/env python
# coding: utf-8

# # import packages

# In[2]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt, gc, os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#train_data=pd.read_parquet('/Users/wangjiaxin/Desktop/study program/data analysis/project/amex/train.parquet')
#train_labels=pd.read_csv('/Users/wangjiaxin/Desktop/study program/data analysis/project/amex/train_labels.csv')
#train = train_data.merge(train_labels,how='inner',on="customer_ID")


# # load train data

# In[4]:


def read_file(path = '', usecols = None):
    
    # LOAD DATAFRAME
    if usecols is not None: df = pd.read_parquet(path, columns=usecols)
    else: df = pd.read_parquet(path)
    
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    #df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df['customer_ID'] = df['customer_ID'].str[-16:].apply(lambda x: int(x, 16))
    df.S_2 = pd.to_datetime( df.S_2 )
    df = df.sort_values(['customer_ID','S_2'])
                    
    # Compute date based features
    
    df['S_2_dayofweek'] = df['S_2'].dt.weekday
    df['S_2_dayofmonth'] = df['S_2'].dt.day
    
    df['days_since_1970'] = df.S_2.astype('int64')/1e9/(60*60*24)
    df['S_2_diff'] = df.days_since_1970.diff()
    df['x'] = df.groupby('customer_ID').S_2.agg('cumcount')
    df.loc[df.x==0,'S_2_diff'] = 0
    df = df.drop(['days_since_1970','x'], axis=1)
    
    # Compute "after pay" features
    
    for bcol in [f'B_{i}' for i in [1,2,3,4,5,9,11,14,17,24]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]
    
    # Null columns handling
    
    nullvals = df.isnull().sum() / df.shape[0]
    #nullCols = nullvals[nullvals>0.3].index.to_arrow().to_pylist()
    nullCols = nullvals[nullvals>0.3].index.tolist()
    
    for col in nullCols:
        df[col+'_null'] = df[col].isnull().astype(int)
    
    # Drop raw date column
    df = df.drop(columns=['S_2'])
    
    print('shape of data:', df.shape)
    
    return df

print('Reading train data...')
TRAIN_PATH = '/Users/wangjiaxin/Desktop/study program/data analysis/project/amex/train.parquet'
train = read_file(path = TRAIN_PATH)


print(train.shape)


# # train data process and feature engineering

# In[5]:


def process_and_feature_engineer(df):
    
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
    cat1_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    cat2_features = [
                    'B_31','B_32','B_33','D_103','D_109','D_111','D_127',
                    'D_129','D_135','D_137','D_139','D_140','D_143','D_86',
                    'D_87','D_92','D_93','D_94','D_96','R_15','R_19','R_2','R_21',
                    'R_22','R_23','R_24','R_25','R_28','R_4','S_18','S_20','S_6'
                       ]
    cat3_features = [
                    'R_9','R_18','R_10','R_11','D_89','D_91','D_81','D_82','D_136',
                    'D_138','D_51','D_123','D_125','D_108','B_41','B_22',
                       ]
    
    nullvals = df.isnull().sum() / df.shape[0]
    exclnullCols = nullvals[nullvals>0.9].index.tolist()
    nullCols = nullvals[nullvals>0.3].index.tolist()
    nullAggCols = [col + "_null" for col in nullCols]
    
    cat_features = cat1_features + cat2_features + cat3_features + exclnullCols + nullAggCols
    
    num_features = [col for col in all_cols if col not in cat_features]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['first','mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
        
    # Diff/Div columns
    for col in test_num_agg.columns:
        
        # Last/First
        if 'last' in col and col.replace('last', 'first') in test_num_agg.columns:
            test_num_agg[col + '_life_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'first')]
  #             test_num_agg[col + '_life_div'] = cupy.where((test_num_agg[col.replace('last', 'first')].isnull()), 0, 
#                                                          cupy.where((test_num_agg[col.replace('last', 'first')]==0), 0, test_num_agg[col] / test_num_agg[col.replace('last', 'first')]))
        # Last/Mean
        if 'last' in col and col.replace('last', 'mean') in test_num_agg.columns:
            test_num_agg[col + '_lmean_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'mean')]
#             test_num_agg[col + '_lmean_div'] = cupy.where((test_num_agg[col.replace('last', 'first')].isnull()) | (test_num_agg[col.replace('last', 'first')]==0), 0, test_num_agg[col] / test_num_agg[col.replace('last', 'first')])
    
    test_cat1_agg = df.groupby("customer_ID")[cat1_features].agg(['first', 'last', 'nunique'])
    test_cat1_agg.columns = ['_'.join(x) for x in test_cat1_agg.columns]
    
    test_cat2_agg = df.groupby("customer_ID")[cat2_features].agg(['first', 'last', 'nunique'])
    test_cat2_agg.columns = ['_'.join(x) for x in test_cat2_agg.columns]
    
    test_cat3_agg = df.groupby("customer_ID")[cat3_features].agg(['first', 'last', 'nunique','min', 'max','mean', 'std'])
    test_cat3_agg.columns = ['_'.join(x) for x in test_cat3_agg.columns]
    
    test_null_agg = df.groupby("customer_ID")[nullAggCols].agg(['count'])
    test_null_agg.columns = ['_'.join(x) for x in test_null_agg.columns]
    
    test_exclnull_agg = df.groupby("customer_ID")[exclnullCols].agg(['last'])
    test_exclnull_agg.columns = ['_'.join(x) for x in test_exclnull_agg.columns]
         
    temp1 = df.groupby(['customer_ID'])['P_2'].count()
    temp1 = temp1.reset_index()
    temp1.columns = ['customer_ID','num_statements']
    temp1 = temp1.set_index('customer_ID')
 
    df = pd.concat([test_num_agg, test_cat1_agg, test_cat2_agg, test_cat3_agg, temp1, test_null_agg, test_exclnull_agg], axis=1) #test_bal_agg
    del test_num_agg, test_cat1_agg, test_cat2_agg, test_cat3_agg, temp1, test_null_agg, test_exclnull_agg
    _ = gc.collect()
     
    print('shape after engineering', df.shape )
    
    return df

train = process_and_feature_engineer(train)

print(train.shape)


# # merge train dataframe

# In[6]:


#add target
targets = pd.read_csv('/Users/wangjiaxin/Desktop/study program/data analysis/project/amex/train_labels.csv')
#targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets['customer_ID'] = targets['customer_ID'].str[-16:].apply(lambda x: int(x, 16))
targets = targets.set_index('customer_ID')

train = train.merge(targets, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del targets
_ = gc.collect()

# NEEDED TO MAKE CV DETERMINISTIC 
train = train.sort_index().reset_index()

# FEATURES
print(f'There are {len(train.columns[1:-1])} features!')


# In[7]:


train = train.set_index('customer_ID')


# # information of dataset

# In[8]:


train.info()


# In[9]:


train.head()


# # machine learning model

# In[11]:


y = train['target']
X = train.drop(["target"],axis=1)


# In[51]:


#test set unoffical
test=X[152972:305944]
test


# # lightgbm

# In[13]:


X_lgbm=X[0:152971]
y_lgbm =y[0:152971]
X_lgbm.shape,y_lgbm.shape


# In[14]:


from sklearn.model_selection import train_test_split
import lightgbm as lgb
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping


# In[15]:


X_train_lgbm,X_valid_lgbm,y_train_lgbm,y_valid_lgbm = train_test_split(X_lgbm, y_lgbm, test_size=0.25,stratify=y_lgbm)


# In[16]:


param_lgbm = {
            'metric': "binary_logloss",
            'boosting_type': "dart",
            'n_estimators':1000,
            'verbosity': -1,
            'lambda_l1': 3.1412416493672213e-06,
             'lambda_l2': 1.919550703890871,
             'num_leaves': 53,
             'feature_fraction': 0.8361911823947347,
             'bagging_fraction': 0.5246353885003125,
             'bagging_freq': 7,
             'min_child_samples': 91}


# In[17]:


lgbm =LGBMClassifier(**param_lgbm).fit(X_train_lgbm, y_train_lgbm, 
                                       eval_set=[(X_train_lgbm, y_train_lgbm), (X_valid_lgbm, y_valid_lgbm)],
                                       
                                       eval_metric=['auc','binary_logloss'],verbose=0)


# In[18]:


#prediction
prdeictions_lgbm = lgbm.predict_proba(test)

preds_lgbm = pd.DataFrame(prdeictions_lgbm)
pred_final_lgbm = np.array(preds_lgbm[1])
pred_final_lgbm


# # xgbm

# In[19]:


import xgboost as xgb
from xgboost import XGBClassifier


# In[20]:


X_xgbm=X[0:152971]
y_xgbm =y[0:152971]
X_xgbm.shape,y_xgbm.shape


# In[21]:


X_train_xgbm,X_valid_xgbm,y_train_xgbm,y_valid_xgbm = train_test_split(X_xgbm, y_xgbm, test_size=0.25,stratify=y_xgbm)


# In[22]:


xgb_parms ={
    'booster': 'dart',
     'n_jobs':4,
     'n_estimators':500,
    'lambda': 4.091409953463271e-08,
    'alpha': 3.6353429991712695e-08,
    'subsample': 0.6423675532438815,
    'colsample_bytree': 0.7830450413657872,
    'max_depth': 9,
    'min_child_weight': 5,
    'eta': 0.3749337530972536,
    'gamma': 0.0745370910451703,
    'grow_policy': 'depthwise',
    'sample_type': 'uniform',
    'normalize_type': 'tree',
    'rate_drop': 0.0723975209176045,
    'skip_drop': 0.9026367296518939}


# In[23]:


xgbm = XGBClassifier(**xgb_parms)
xgbm.fit(X_train_xgbm, y_train_xgbm, 
             early_stopping_rounds=10, 
             eval_set=[(X_valid_xgbm, y_valid_xgbm)],
             verbose=0)  


# In[24]:


#prediction
prdeictions_xgbm = xgbm.predict_proba(test)
preds_xgbm = pd.DataFrame(prdeictions_xgbm)
pred_final_xgbm = np.array(preds_xgbm[1])


# # catboost

# In[25]:


import catboost as cb
from catboost import CatBoostClassifier


# In[26]:


X_cat=X[0:152971]
y_cat =y[0:152971]
X_cat.shape,y_cat.shape


# In[27]:


X_train_cat,X_valid_cat,y_train_cat,y_valid_cat = train_test_split(X_cat, y_cat, test_size=0.25,stratify=y_cat)


# In[28]:


Params_cat={ 
    'objective': 'CrossEntropy',
    'n_estimators':1000,
    'colsample_bylevel': 0.07868805912943484,
    'depth': 9,
    'boosting_type': 'Plain',
    'bootstrap_type': 'MVS',
    }


# In[29]:


cbm =CatBoostClassifier(**Params_cat).fit(X_train_cat, y_train_cat, 
                                       eval_set=[(X_train_cat, y_train_cat), (X_valid_cat, y_valid_cat)],
                                      verbose=0,
                                       
                                       )


# In[30]:


#prediction
prdeictions_cbm = cbm.predict_proba(test)
preds_cbm = pd.DataFrame(prdeictions_cbm)
pred_final_cbm = np.array(preds_cbm[1])


# # combine model

# In[54]:


pred_df= pd.DataFrame({'lgbm':pred_final_lgbm,'xgbm':pred_final_xgbm,'catboost':pred_final_cbm})

pred_df['prediction'] = (pred_df['lgbm'] + pred_df['xgbm'] + pred_df['catboost'])/3

pred_df.head()


# # evaluation

# In[98]:


y['target'] = train['target']
test_y_true=y[152972:305944]
test_y_predict=pred_df['prediction']


# In[99]:


test_y_true_=test_y_true.reset_index()
test_y_true_


# In[100]:


test_y_predict.head()


# In[101]:


from pathlib import Path

input_path = Path('/kaggle/input/amex-default-prediction/')


# In[102]:


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


# In[104]:


#score_amex
print(amex_metric(test_y_true_,test_y_predict))


# # prediction using test data

# In[ ]:


#load test data
print('Reading test data...')
TEST_PATH = '/Users/wangjiaxin/Desktop/study program/data analysis/project/amex/test.parquet'
test = read_file(path = TEST_PATH)


print(test.shape)


# In[ ]:


#test data process and feature engineering
test = process_and_feature_engineer(test)

print(test.shape)

