import torch
import argparse
from utilities_trustfed import find_statistical_parity_score, find_eqop_score, all_metrics
from load_data_trustfed import get_data, load_dataset
from constraint_trustfed import DemographicParityLoss, EqualOpportunityLoss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from torch import nn
from torch import optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from itertools import chain
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch import fit_gpytorch_mll
import math
from botorch.utils.transforms import unnormalize, normalize
from sklearn.metrics import confusion_matrix
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.models.transforms import Standardize
parser = argparse.ArgumentParser(description="pass the following arguments: dataset_name, number of clients, fairness notion, number of communication rounds.")

device = torch.device('cpu')


# Add arguments
parser.add_argument("--with_noise", type=str, default='yes', 
                    choices=['yes', 'no'], 
                    help="Add Differential Privacy noise. Options: 'yes', 'no'. Default is 'yes'.")
parser.add_argument("--fairness_notion", type=str, default='stat_parity', 
                    choices=['eqop', 'stat_parity'], 
                    help="Fairness notion to use. Options: 'ate', 'stat_parity'. Default is 'stat_parity'.")
parser.add_argument("--num_clients", type=int, default=3, 
                    choices=[3, 5, 10, 15], 
                    help="Number of clients for random distribution. Options: 3, 5, 10, 15. Default is 3.")
parser.add_argument("--dataset_name", type=str, default='bank', 
                    choices=['adult', 'default', 'acs', 'bank', 'law'], 
                    help="Name of the dataset. Options: 'adult', 'default', 'acs', 'bank', 'law'. Default is 'bank'.")
parser.add_argument("--epochs", type=int, default=15, 
                   help="Client training epochs. Default is 15.")
parser.add_argument("--communication_rounds", type=int, default=50, 
                    help="Number of communication rounds. Default is 50.")
parser.add_argument("--mobo_optimization_rounds", type=int, default=10, 
                    help="Number of MOBO optimization rounds. Default is 10.")
parser.add_argument("--noise_type", type=str, default='gaussian', 
                    choices=['gaussian', 'laplace'], 
                    help="Differential privacy noise type to be used. Options: 'gaussian', 'laplace'. Default is 'gaussian'.")
parser.add_argument("--epsilon", type=float, default=3, 
                    choices=[0.1,2,3,4,10], 
                    help="Value of privacy budget (epsilon). Options: 0.1,2,3,4,10. Default is 3.")

#parser.add_argument("--distribution_type", type=str, default='random', 
#                    choices=['random', 'attribute-based'], 
#                    help="Data distribution type. Options: 'random', 'attribute-based'. Default is 'random'.")

# Parse the arguments
args = parser.parse_args()

# Store them in respective variables
with_noise = args.with_noise
fairness_notion = args.fairness_notion
num_clients = args.num_clients
dataset_name = args.dataset_name
epochs = args.epochs
communication_rounds = args.communication_rounds
mobo_optimization_rounds = args.mobo_optimization_rounds
noise_type = args.noise_type
delta = 1  # Example value; typically a small value
clip_norm = 1  # This is the L2 norm to which we want to clip the gradients. This can be adjusted based on your needs.
clip_norm_2 = 1
epsilon = args.epsilon
#distribution_type = args.distribution_type

def create_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    return model

bounds = torch.tensor([[100.0,0.0], [2000.0,0.01]])  # bounds on learning rate
if dataset_name == 'adult':
    url = './datasets/adult.csv'
    sensitive_feature = 'sex' #'sex': 0 ->female, 1-> male
elif dataset_name == 'adult-age':
    url = './datasets/adult.csv'
    sensitive_feature = 'sex' #'sex': 0 ->female, 1-> male
elif dataset_name == 'bank':
    url = './datasets/bank-full.csv'
    sensitive_feature = 'marital' #'marital': 0->married, 1-> single
elif dataset_name == 'bank-age':
    url = './datasets/bank-full.csv'
    sensitive_feature = 'marital' #'marital': 0->married, 1-> single
elif dataset_name == 'default':
    url = './datasets/default.csv'
    sensitive_feature = 'SEX' #'sex': 0 ->female, 1-> male
elif dataset_name == 'default-age':
    url = './datasets/default.csv'
    sensitive_feature = 'SEX' #'sex': 0 ->female, 1-> male
elif dataset_name == 'law':
    url = './datasets/law.csv'
    sensitive_feature = 'sex' #'sex': 0 ->female, 1-> male
elif dataset_name == 'law-income':
    url = './datasets/law.csv'
    sensitive_feature = 'sex' #'sex': 0 ->female, 1-> male
elif dataset_name == 'acs':
    url = './datasets/acs/'
    sensitive_feature = 'sex' #'sex': 0 ->female, 1-> male
else:
    print("dataset not supported, please update file load_data.py")
    exit()
  # Example value; .

noise_scale = clip_norm_2 / epsilon * np.sqrt(2 * np.log(1.25 / delta))

bal_acc_list = []
fairness_notion_list = []
#clients_data,X_test, y_test, sex_list, column_names_list, ytest_potential, non_member_data, member_data  = load_dataset(url,dataset_name, num_clients, sensitive_feature)
clients_data,X_test, y_test, sex_list, column_names_list, ytest_potential  = load_dataset(url,dataset_name, num_clients, sensitive_feature)

X_test = X_test.to(device)
y_test = y_test.to(device)
global_model = create_model(X_test.shape[1])
global_model = global_model.to(device)
# Define a loss function and optimizer
criterion = nn.BCELoss()

def laplace_noise(scale, shape, device):
    # Generate two uniform random variables
    u1 = torch.rand(shape).to(device) - 0.5
    u2 = torch.sign(u1)
    # Sample from the Laplace distribution
    noise = -scale * u2 * torch.log(1.0 - 2.0 * torch.abs(u1))
    return noise

def initialize_model(train_x, train_y):
    # Normalize the input data
    train_x = normalize(train_x, bounds)
    
    models = []
    for i in range(train_y.shape[-1]):
        train_objective = train_y[:,i].double()
        
        # Standardize the outcome
        
        outcome_transform = Standardize(m=1)
        models.append(
          SingleTaskGP(train_x, train_objective.unsqueeze(-1), outcome_transform=outcome_transform)
        )
        
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll
def initialize_model_1(train_x, train_y):
    models = []
    for i in range(train_y.shape[-1]):
        train_objective = train_y[:,i]
        #print("bismillah")
        models.append(
          SingleTaskGP(train_x, train_objective.unsqueeze(-1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model,mll
#NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs) #(2**0.5) * noise_scale_metric

cost_false_negatives = 10.0 #15 for adult-age
cost_false_positives = 5.0

def calculate_weights(targets, cost_false_negatives=5):
    cost_false_negatives = 10
    # Give higher weight to the positive samples because false negatives cost more
    return torch.where(targets == 1, cost_false_negatives, cost_false_positives)

def evaluate(alpha = 100, lr=0.001, cost_false_negatives=5):
    # Initialize a list to store the parameters of each model
    params = [torch.zeros_like(param.data) for param in global_model.parameters()]
    for client_name in clients_data.keys():
        print(client_name)
        X1,y1,s1,y1_potential = get_data(client_name, clients_data)
        X1 = X1.to(device)
        y1 = y1.to(device)
        y1_potential = y1_potential.to(device)
        s1 = s1.to(device)
        model1 = create_model(X1.shape[1])
        model1 = model1.to(device)
        model1.load_state_dict(global_model.state_dict())
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        if fairness_notion == 'stat_parity':
            dp_loss = DemographicParityLoss(alpha=alpha)
        elif fairness_notion == 'eqop':
            dp_loss = EqualOpportunityLoss(alpha=alpha)
        
        for epoch in range(epochs):
            criterion = nn.BCEWithLogitsLoss(pos_weight=None)
            # Training on Client 1
            optimizer1.zero_grad()
            y_pred = model1(X1)
            
            weights = calculate_weights(y1,cost_false_negatives)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
            X1_cpu = X1.cpu()
            X1_dataframe = pd.DataFrame(X1_cpu.numpy(), columns=column_names_list)
            y_pred_numpy = y_pred.clone().cpu()
            
            
            fairness_loss = dp_loss(X1, y_pred, s1,y1)
            fairness_loss = fairness_loss.to(device)
            loss = criterion(y_pred.view(-1), y1) + fairness_loss
            loss.backward()
            
            # Add noise to gradients
            if with_noise == 'yes':
                torch.nn.utils.clip_grad_norm_(model1.parameters(), clip_norm)
                if noise_type == 'gaussian':
                #gaussian noise
                    for param in model1.parameters():
                        noise = torch.normal(0, noise_scale, size=param.grad.shape).to(device)
                        param.grad.add_(noise)
                #laplace noise
                elif noise_type == 'laplace':
                    for param in model1.parameters():
                        noise = laplace_noise(noise_scale, param.grad.shape, device)
                        param.grad.add_(noise)
                    
            optimizer1.step()
        print(f'- Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        # After training, add the trained model's parameters to the list
        
        for param, param_sum in zip(model1.parameters(), params):
            param_sum.add_(param.data)

    # Compute average of parameters and update global model
    average_params = [param_sum / len(clients_data) for param_sum in params]
    # Copy the averaged parameters into the global model
    with torch.no_grad():
        for param_global, param_avg in zip(global_model.parameters(), average_params):
            param_global.copy_(param_avg)
    global_model.eval()
    
    # Average the model parameters (weights and biases) and set the averaged parameters to both models
    with torch.no_grad():
        y_pred = global_model(X_test).squeeze()
        y_pred_cls = y_pred.round()
        sensitivity,specificity,bal_acc,G_mean,FN_rate,FP_rate,Precision,f1_sc, acc, auc = all_metrics(y_test.cpu(),y_pred.cpu())
        stat_parity = find_statistical_parity_score(sex_list, y_test,y_pred_cls)
        eqop = find_eqop_score(sex_list, y_test,y_pred_cls)
        X_test_cpu = X_test.cpu()
        Xtest_dataframe = pd.DataFrame(X_test_cpu.numpy(), columns=column_names_list)
        y_pred_numpy = y_pred.clone().cpu()
        #ytest_potential = find_potential_outcomes(Xtest_dataframe,y_pred_numpy.round().detach().numpy())
        #acc = (y_pred_cls == y_test).float().mean()
        auprc = average_precision_score(y_test.cpu(), y_pred.cpu())
        print(f'Communication round {round+1}/{communication_rounds}')
        if communication_rounds % 10 == 0:
            print(f'Test accuracy: {acc.item()}')
            print("BalanceACC: %s" % bal_acc)
            if fairness_notion == 'stat_parity':
                print("statistical parity: %s" % stat_parity)
            else:
                print("eqop: %s" % eqop)
    bal_acc_orig = bal_acc
    if fairness_notion == 'stat_parity':
        fairness_notion_orig = stat_parity
    else:
        fairness_notion_orig = eqop
    if fairness_notion == 'stat_parity':
        if with_noise=='yes':
            print("bismillah*************")
            if noise_type == 'gaussian':
                noise = torch.normal(0, noise_scale, size=param.grad.shape).to(device)
                noise_bal_acc = bal_acc+noise
                noise_stat_parity = stat_parity+noise
                objectives = torch.tensor([[np.float64(-noise_stat_parity), np.float64(noise_bal_acc)]]) #the two objectives
            elif noise_type == 'laplace':
                noise_bal_acc = bal_acc+laplace_noise(noise_scale, (1,), device)
                noise_stat_parity = stat_parity+laplace_noise(noise_scale, (1,), device)
                objectives = torch.tensor([[np.float64(-noise_stat_parity), np.float64(noise_bal_acc)]]) #the two objectives
        else:
            objectives = torch.tensor([[-stat_parity, bal_acc]]) #the two objectives
    elif fairness_notion == 'eqop':
        if with_noise=='yes':
            print("bismillah*************")
            if noise_type == 'gaussian':
                noise = torch.normal(0, noise_scale, size=param.grad.shape).to(device)
                noise_bal_acc = bal_acc+noise
                noise_eqop = eqop+noise
                objectives = torch.tensor([[np.float64(-noise_eqop), np.float64(noise_bal_acc)]]) #the two objectives
            elif noise_type == 'laplace':
                noise_bal_acc = bal_acc+laplace_noise(noise_scale, (1,), device)
                noise_eqop = eqop+laplace_noise(noise_scale, (1,), device)
                objectives = torch.tensor([[np.float64(-noise_eqop), np.float64(noise_bal_acc)]]) #the two objectives
        else:
            objectives = torch.tensor([[-eqop, bal_acc]]) #the two objectives
    return objectives, bal_acc_orig, fairness_notion_orig

def models_have_same_parameters(model1, model2):
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    #print(params1)
    #print("bismillah")
    #print(params2)
    if len(params1) != len(params2):
        return False
    
    for p1, p2 in zip(params1, params2):
        if not torch.allclose(p1, p2):
            return False

    return True


# Train the model on both clients

for round in range(communication_rounds):
    
    print(f'Communication round {round+1}/{communication_rounds}')

    bounds = torch.tensor([[100.0,0.0], [2000.0,0.01]])  # bounds on learning rate
    #bounds = torch.tensor([[0.0], [350.0]])
    alpha = torch.tensor([100], dtype=torch.float64)
    alpha = alpha.view(1, -1)

    #objectives = evaluate(alpha)
    #objectives = evaluate(alpha)
    if round == 0:
        objectives, bal_acc_, fairness_notion_ = evaluate(alpha)
    else:
        objectives, bal_acc_, fairness_notion_ = evaluate(updated_alpha, updated_lr)
    fairness_notion_list.append(fairness_notion_)
    bal_acc_list.append(bal_acc_)
    
    #alpha = normalize(alpha, bounds)
    x_input =  torch.tensor([100,0.001], dtype=torch.float64)#input to the optimization process
    x_input = x_input.view(1, -1)
    #model, mll = initialize_model(alpha, objectives.float())
    model, mll = initialize_model(x_input, objectives)
   
    for i in range(mobo_optimization_rounds):  # number of rounds of optimization
        print("Global optimization round:", i)
        fit_gpytorch_mll(mll)
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model.double(),  # Convert model to double
            ref_point=torch.tensor([-0.1, 0.1], dtype=torch.float64),  # Explicitly set ref_point to double
            X_baseline=x_input.double(),  # Convert X_baseline to double
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),  # Set sampler dtype to double
            prune_baseline=True,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]).double(),  # Convert objective to double
    
        )
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=300,
            raw_samples=1024,
            options ={"batch_limit": 10, "maxiter": 200}
        )
    

        #new_candidate = unnormalize(candidate, bounds)
        new_objectives, a, b= evaluate(candidate[0,0].item(), candidate[0,1].item())#evaluate(candidate.item())

        for i, m in enumerate(model.models):
                train_x = torch.cat([m.train_inputs[0], candidate])
                #print("bismillah")
                train_y = torch.cat([m.train_targets, new_objectives[:,i]])
                m.set_train_data(train_x, train_y, strict=False)
                if i == 0:
                    train_y_0 = train_y
                else:
                    train_y_1 = train_y
        train_y_all = torch.stack((train_y_0, train_y_1), dim=1)
        weights = torch.tensor([0.6, 0.4])
        weighted_sums = (train_y_all * weights).sum(dim=1)

        # Find the index of the best solution based on the highest weighted sum
        best_solution_idx = weighted_sums.argmax()
        best_solution = train_y_all[best_solution_idx]

        print(f"Best solution based on weighted sum: {best_solution}")

        # Select the corresponding row in train_x using the best_solution_idx
        best_train_x = train_x[best_solution_idx]
        updated_alpha, updated_lr = best_train_x.tolist()

    # Now both models have the same parameters and we can continue with the next round of communication


global_model.eval()
    
# Average the model parameters (weights and biases) and set the averaged parameters to both models
with torch.no_grad():
        y_pred = global_model(X_test).squeeze()
        y_pred_cls = y_pred.round()
        sensitivity,specificity,bal_acc,G_mean,FN_rate,FP_rate,Precision,f1_sc, acc, auc = all_metrics(y_test.cpu(),y_pred.cpu())
        stat_parity = find_statistical_parity_score(sex_list, y_test,y_pred_cls)
        X_test_cpu = X_test.cpu()
        Xtest_dataframe = pd.DataFrame(X_test_cpu.numpy(), columns=column_names_list)
        y_pred_numpy = y_pred.clone().cpu()
        eqop = find_eqop_score(sex_list, y_test,y_pred_cls)
        #ytest_potential = find_potential_outcomes(Xtest_dataframe,y_pred_numpy.round().detach().numpy())
        
        auprc = average_precision_score(y_test.cpu(), y_pred.cpu())
        print(f'Test accuracy: {acc.item()}')
        print("BalanceACC: %s" % bal_acc)
        if fairness_notion == 'stat_parity':
            print("statistical parity: %s" % stat_parity)
        else:
            print("eqop: %s" % eqop)
        

destination = './results/'+dataset_name+'/'
if with_noise == 'no':
    if dataset_name == 'adult' or dataset_name == 'bank' or dataset_name == 'law'or dataset_name == 'default'or dataset_name == 'kdd':
        if fairness_notion == 'stat_parity':
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_bal_acc_stat_parity.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_stat_parity.npy', np.array(fairness_notion_list))
        else:
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_bal_acc_ate.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_ate.npy', np.array(fairness_notion_list))
    else:
        if fairness_notion == 'stat_parity':
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_bal_acc_stat_parity.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_stat_parity.npy', np.array(fairness_notion_list))
        else:
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_bal_acc_ate.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_ate.npy', np.array(fairness_notion_list))
else:
    if dataset_name == 'adult' or dataset_name == 'bank' or dataset_name == 'law'or dataset_name == 'default'or dataset_name == 'kdd':
        if fairness_notion == 'stat_parity':
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_bal_acc_stat_parity_noisy.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_stat_parity_noisy.npy', np.array(fairness_notion_list))
        
        else:
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_bal_acc_eqop_noisy.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_eqop_noisy.npy', np.array(fairness_notion_list))
        
    else:
        if fairness_notion == 'stat_parity':
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_bal_acc_stat_parity_noisy.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_stat_parity_noisy.npy', np.array(fairness_notion_list))
        
        else:
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_bal_acc_eqop_noisy.npy', np.array(bal_acc_list))
            np.save(destination+str(num_clients)+'epsilon_'+str(epsilon)+'_attr_eqop_noisy.npy', np.array(fairness_notion_list))
        