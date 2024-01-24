# TrustFed: Optimizing Predictive Performance and Fairness in Federated Learning with Privacy Protection
As Federated Learning (FL) gains prominence in secure machine learning applications, achieving trustworthy predictions without compromising predictive performance becomes paramount. While differential privacy (DP) is extensively used for its effective privacy protection, yet its application as a lossy protection method can lower the predictive performance of the machine learning model. Also, the data being gathered from distributed clients in an FL environment often leads to class imbalance making traditional accuracy less reflective of the true performance of model. In this context, we introduce a fairness-aware federated learning framework (TrustFed) based on Gaussian differential privacy and Multi-Objective Optimization (MOO), which effectively protects privacy while providing fair and accurate predictions. This work is the first effort towards achieving Pareto-optimal trade-offs between balanced accuracy and fairness in a federated environment while safeguarding the privacy of individual clients. The framework's flexible design adeptly accommodates both statistical parity and equal opportunity fairness notions, ensuring its applicability in various FL scenarios. We demonstrate our framework's effectiveness through comprehensive experiments on five real-world datasets and comparisons with six baseline models. The empirical results highlight our framework's capability to enhance the trade-off between fairness and balanced accuracy while preserving the anonymization rights of users in FL applications.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* [Law School](https://github.com/iosifidisvasileios/FABBOO/blob/master/Data/law_dataset.arff)
* [ACS Income Dataset](https://github.com/socialfoundations/folktables): Use the repository given at the URL to download ACS dataset.
## Code
### Dataset Processing Scripts

The `datasets` directory contains all the datasets used in this project. Below is a description of python scripts written to process datasets:

- `load_data_trustfed.py`: Utility script for loading all the datasets (Adult, Bank, Default, Law, ACS).
- `load_data_adult.py`: Utility script for loading and preprocessing Adult dataset.
- `load_data_bank.py`: Utility script for loading and preprocessing Bank dataset.
- `load_data_law.py`: Utility script for loading and preprocessing Law School dataset.
- `load_data_default.py`: Utility script for loading and preprocessing Default dataset.
- `load_data_acs.py`: Utility script for loading and preprocessing ACS Income dataset.


### Utility Scripts
- `utilities_trustfed.py`: Utility script for computing evaluation metrics including 'statistical parity', Equal Opportunity (Eqop), balanced accuracy, and accuracy.

### TrustFed main scripts
The following scripts constitute the complete methodology of TrustFed
- `Trustfed.py`: Main script for the 'TrustFed' framework that orchestrates the fairness aware federated learning process on different datasets with privacy guarantees.
- `constraint_trsutfed.py`: The script contains the implementation of fairness constraints for discrimination mitigation.
  
## Running the Trustfed.py Script
To run the `Trustfed.py` script with the default settings, you can use the following command:

```bash
python Trustfed.py --with_noise 'yes' --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --noise_type 'gaussian' --epsilon 3
```
## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

- torch==2.0.1
- torchvision==0.15.2
- scikit-learn==0.24.2
- pandas==1.5.3
- gpytorch==1.10
- botorch==0.8.5
- crypten==0.4.1
- cvxopt==1.3.1
- cvxpy==1.3.2
