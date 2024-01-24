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
from collections import defaultdict

def load_acs():
    #states = ['ca', 'il', 'ny', 'tx', 'fl', 'pa', 'oh', 'mi', 'ga', 'nc'] #10 states
     #all states
    states = [
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
    'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
    'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
    'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
    ]
    
    categorical_columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'RELP', 'SEX']
    numerical_columns = ['OCCP', 'POBP', 'WKHP']
    sensitive_feature = 'RAC1P'
    #print("bismillah")
    data_dict = defaultdict(list)
    scaler = StandardScaler()
    test_dfs = defaultdict(list)
    client_num = 1
    for state in states:
        data = pd.read_csv(f'./acs_dataset/{state}_data.csv')
        data['RAC1P'] = data['RAC1P'].apply(lambda x: 1 if x == 1 else 0)
        for col in categorical_columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
        # Normalize numerical columns
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        data, test_df = train_test_split(data, test_size=0.1, random_state=42)
        client_name = "client_"+str(client_num)
        test_dfs[client_name]=test_df
        
        # Split the data into features and labels
        X_client = data.drop('PINCP', axis=1)
        y_client = LabelEncoder().fit_transform(data['PINCP'])
        s_client = X_client[sensitive_feature]
        #y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
        X_client = torch.tensor(X_client.values, dtype=torch.float32)
        y_client = torch.tensor(y_client, dtype=torch.float32)
        s_client = torch.from_numpy(s_client.values).float()
        #y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
        y_pot = torch.zeros_like(y_client)
        update_data = {"X": X_client, "y": y_client, "s": s_client, "y_pot": y_pot}
        data_dict[client_name]=update_data
        client_num +=1
    
    # Concatenate the dataframes into a single dataframe
    test_df = pd.concat(test_dfs.values(), ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_test = test_df.drop('PINCP', axis=1)
    y_test = LabelEncoder().fit_transform(test_df['PINCP'])
    sex_column = X_test[sensitive_feature]
    sex_list = sex_column.tolist()
    column_names_list = X_test.columns.tolist()
    #ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    #ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_pot = torch.zeros_like(y_test)
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,y_pot