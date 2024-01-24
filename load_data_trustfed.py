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

from load_adult_data import *
from load_bank_data import *
from load_default_data import *
from load_law_data import *
from load_acs_data import *



def load_dataset(url, dataset_name, num_clients, sensitive_feature):
    if dataset_name == 'adult':
        X, y = load_adult(url)
    elif dataset_name == 'bank':
        X, y  = load_bank(url)
    elif dataset_name == 'default':
        X, y = load_default(url)
    elif dataset_name == 'law':
        X, y = load_law(url)
    elif dataset_name == 'law-income':
        data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_law_income(url, sensitive_feature)
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    elif dataset_name == 'adult-age':
        data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_adult_age(url, sensitive_feature)
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    elif dataset_name == 'bank-age':
        data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_bank_age(url, sensitive_feature)
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    elif dataset_name == 'bank-age-5':
        data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_bank_age_5(url, sensitive_feature)
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    elif dataset_name == 'default-age':
        data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_default_age(url, sensitive_feature)
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    elif dataset_name == 'acs':
        data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential= load_acs()
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential
    k = 0
    if k==0:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        sex_column = X_test[sensitive_feature]
        column_names_list = X_temp.columns.tolist()
        # Convert the pandas Series to a Python list
        sex_list = sex_column.tolist()
        data_dict = {}
        sensitive_index = X_temp.columns.get_loc(sensitive_feature)
        for i in range(num_clients):
            if i == num_clients - 1:
                X_client, y_client = X_temp, y_temp
            else:
                X_temp, X_client, y_temp, y_client = train_test_split(X_temp, y_temp, test_size=1/(num_clients-i), random_state=42)
            
            s_client = X_client[sensitive_feature]
            #compute potential outcomes
            #y_potential_client = find_potential_outcomes(X_client,y_client, sensitive_feature)
            y_potential_client = y_client
            # Convert to PyTorch tensors
            X_client = torch.tensor(X_client.values, dtype=torch.float32)
            y_client = torch.tensor(y_client, dtype=torch.float32)
            s_client = torch.from_numpy(s_client.values).float()
            y_potential_client = torch.tensor(y_potential_client, dtype=torch.float32)
            
            # Store the client data in the dictionary
            data_dict[f"client_{i+1}"] = {"X": X_client, "y": y_client, "s": s_client, "y_pot": y_potential_client}
        ytest_potential = y_test
        
       
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential

def get_data(client_name, data_dict):
    client_data = data_dict.get(client_name, {})
    X = client_data.get("X")
    y = client_data.get("y")
    s = client_data.get("s")
    y_pot = client_data.get("y_pot")
    return X, y, s, y_pot

 