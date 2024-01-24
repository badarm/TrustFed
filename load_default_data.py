import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
from sklearn.neighbors import NearestNeighbors

def load_default(url):
    data = pd.read_csv(url)
    #data = shuffle(data)
    # Encode categorical columns
    categorical_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Split the data into features and labels
    X = data.drop('y', axis=1)
    y = LabelEncoder().fit_transform(data['y'])
    #print("bismillah")
    return X, y 

def load_default_age(url, sensitive_feature):
    data = pd.read_csv(url)
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    data = shuffle(data)
    
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #data = train_df
    mask = (data['AGE'] >=0) & (data['AGE'] <=29)
    df1 = data[mask]
    mask = (data['AGE'] >=30) & (data['AGE'] <=39)
    df2 = data[mask]
    mask = (data['AGE'] >=40)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
   
    #scalrising the 'age' column as well
    numerical_columns = ['AGE']
    scaler = StandardScaler()
    df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
    df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])
    df3[numerical_columns] = scaler.fit_transform(df3[numerical_columns])
    
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)
    # Split the data into features and labels
    test_df = result = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_client1 = df1.drop('y', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['y'])
    
    X_client2 = df2.drop('y', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['y'])
    
    X_client3 = df3.drop('y', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['y'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    #y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    y_potential_client1 =y_client1 
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    y_potential_client1 =y_client1 
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
   
    
    
    
    
    s_client2 = X_client2[sensitive_feature]
    #y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    y_potential_client2 =y_client2 
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    #y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    y_potential_client3 =y_client3 
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('y', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['y'])
    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    #ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential =y_test 
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential