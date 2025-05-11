#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from CMI_torch import compute_all_cmi_methods, estimate_CMI_soft_kronecker_gaussian, estimate_CMI_gumbel_softmax_kernel,     estimate_CMI_separate_kernel
import sys
import torch


# In[2]:


adult_data_train = pd.read_csv(r'C:\Users\admin\Desktop\projects\Imputation_Fairness\Data\adult\adult.data',
                         header=None, 
                        delimiter=',')
adult_data_test= pd.read_csv(r'C:\Users\admin\Desktop\projects\Imputation_Fairness\Data\adult\adult.test',
                         header=None, 
                        delimiter=',', skiprows=1)
adult_data = pd.concat([adult_data_train, adult_data_test], axis=0).reset_index(drop=True)


# In[3]:


adult_data.head(5)


# In[4]:


# Replace all ' ?' values with np.nan
adult_data = adult_data.replace(' ?', np.nan)
adult_data.isna().sum()


# In[5]:


missing_percentage = (adult_data.isnull().sum() / len(adult_data)) * 100
print(missing_percentage)


# In[6]:


adult_data = adult_data.dropna()


# In[7]:


adult_data.info()


# In[8]:


columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
adult_data.columns = columns


# In[9]:


adult_data.nunique()


# In[10]:


print("age",adult_data['age'].unique())
print("workclass",adult_data['workclass'].unique())
print("fnlwgt",adult_data['fnlwgt'].unique())
print("education",adult_data['education'].unique())
print("education-num",adult_data['education-num'].unique())
print("marital-status",adult_data['marital-status'].unique())
print("occupation",adult_data['occupation'].unique())
print("relationship",adult_data['relationship'].unique())
print("race",adult_data['race'].unique())
print("sex",adult_data['sex'].unique())
print("capital-gain",adult_data['capital-gain'].unique())
print("capital-loss",adult_data['capital-loss'].unique())
print("hours-per-week",adult_data['hours-per-week'].unique())
print("native-country",adult_data['native-country'].unique())
print("income",adult_data['income'].unique())


# In[11]:


# Drop unnecessary columns:
# 'education' is redundant because 'education-num' already encodes it numerically.
# 'fnlwgt' represents sampling weights and does not contribute to feature relationships.

adult_data = adult_data.drop(columns=['education', 'fnlwgt'])


# In[12]:


adult_data.head(5)


# In[13]:


categorical_columns = [
    'workclass', 
    'marital-status', 
    'occupation', 
    'relationship', 
    'race', 
    'sex', 
    'native-country', 
    'income'
] 
for col in categorical_columns:
    encoder = LabelEncoder()
    not_null_idx = adult_data[col].notnull()
    adult_data.loc[not_null_idx, col] = encoder.fit_transform(adult_data.loc[not_null_idx, col])


# In[14]:


adult_data.head(5)


# In[15]:


# List of (X_index, Y_index, Z_index) triplets for CMI calculation
# Column indices after dropping 'education' and 'fnlwgt':

# 0: age
# 1: workclass
# 2: education-num
# 3: marital-status
# 4: occupation
# 5: relationship
# 6: race
# 7: sex
# 8: capital-gain
# 9: capital-loss
# 10: hours-per-week
# 11: native-country
# 12: income

cmi_triplets = [
    (0, 12, 2),  # (age, income, education-num)
    (6, 12, 2),  # (race, income, education-num)
    (7, 12, 2),  # (sex, income, education-num)
    (11, 12, 2), # (native-country, income, education-num)
    (0, 10, 4),  # (age, hours-per-week, occupation)
    (6, 10, 4),  # (race, hours-per-week, occupation)
    (7, 10, 3),  # (sex, hours-per-week, marital-status)
    (11, 10, 4), # (native-country, hours-per-week, occupation)
]


# In[16]:


# continuous and discrete column indices
discrete_columns = [1, 3, 4, 5, 6, 7, 11, 12]
continuous_columns = [0, 2, 8, 9, 10]

# Create a copy of the original data to preserve the raw values
adult_data_scaled = adult_data.copy()

# Normalize only the continuous columns using MinMaxScaler
scaler = MinMaxScaler()
adult_data_scaled.iloc[:, continuous_columns] = scaler.fit_transform(adult_data_scaled.iloc[:, continuous_columns])


# In[17]:


# Sampling strategy (Test vs Full)

#adult_data_sampled = adult_data_scaled

sample_indices = np.random.choice(adult_data_scaled.shape[0], size=100, replace=False)

adult_data_sampled = adult_data_scaled.iloc[sample_indices]
data_np = adult_data_sampled.values.astype(float)
data_tensor= torch.tensor(data_np,dtype=torch.float32)


# In[18]:


#Encode and combine data

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
one_hot_encoded = encoder.fit_transform(adult_data_sampled.iloc[:, discrete_columns])
continuous_part = adult_data_sampled.iloc[:, continuous_columns].values
data_processed = np.concatenate([one_hot_encoded, continuous_part], axis=1)

one_hot_encoded_tensor = torch.tensor(one_hot_encoded, dtype=torch.float32)
continuous_part_tensor = torch.tensor(continuous_part, dtype=torch.float32)
data_processed_tensor = torch.cat([one_hot_encoded_tensor, continuous_part_tensor], dim=1)


# In[19]:


# GRID SERACH
#BOOTSTRAP


# In[20]:


sys.path.append(r"C:\Users\admin\Desktop\Imputation_Fairness") 

import Utils
import Inject_Missing_Values
import RR_imputer
import Sinkhorn_CMI
import SinkhornImputation
import SoftImpute


# In[21]:


for triplet in cmi_triplets:
    cmi1, cmi2, cmi3 = compute_all_cmi_methods(data_tensor, data_processed_tensor, one_hot_encoded_tensor, continuous_part_tensor,
                            triplet, encoder, discrete_columns, continuous_columns
    )

    print(f"Triplet {triplet} :")
    print("  CMI (Soft Kronecker + Gaussian):", cmi1)
    print("  CMI (Gumbel-Softmax + Gaussian):", cmi2)
    print("  CMI (Gumbel on Discrete + Separate Kernels):", cmi3)
    print()


# # Injection of 25% MCAR Missingness

# In[22]:


from Inject_Missing_Values import Inject_Missing_Values

# Step 1: Split the full dataset into features (X) and target (Y)
X = adult_data_sampled.iloc[:, :-1]
Y = adult_data_sampled.iloc[:, -1]

# Step 2: Inject 25% MCAR missingness into the feature set X
generator_mcar25 = Inject_Missing_Values()
X_miss_mcar25, index_mcar25 = generator_mcar25.MCAR(X, missing_rate=25)

# Step 3: Report total missing percentage
total_missing_percentage = X_miss_mcar25.isnull().sum().sum() / X_miss_mcar25.size * 100
print(f"Total Missing Percentage (MCAR 25%): {total_missing_percentage:.2f}%")

# Step 4: Report missing percentage per column
missing_per_column = (X_miss_mcar25.isnull().sum() / len(X_miss_mcar25)) * 100
print(missing_per_column)


# In[23]:


X_miss_with_Y = X_miss_mcar25.copy()
X_miss_with_Y["target"] = Y


X_numpy = X.to_numpy(dtype=float)
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)


Y_numpy = Y.to_numpy(dtype=int)
Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)

X_miss_mcar25_numpy = X_miss_mcar25.to_numpy(dtype=float)
X_miss_mcar25_tensor = torch.tensor(X_miss_mcar25_numpy, dtype=torch.float32) #converting to tensor

X_miss_with_Y_numpy = X_miss_with_Y.to_numpy(dtype=float)
X_miss_with_Y_tensor = torch.tensor(X_miss_with_Y_numpy, dtype=torch.float32)


# In[24]:


#Sinkhorn 

from Utils import pick_epsilon, MAE, RMSE, nanmean
from SinkhornImputation import SinkhornImputation

# Get shape of the data
n_mcar25, d_mcar25 = X_miss_mcar25_tensor.shape

# Set Sinkhorn hyperparameters
batchsize = 28  #128
lr = 1e-2
epsilon_mcar25 = pick_epsilon(X_miss_mcar25_tensor) # Determines Sinkhorn regularization strength

# Create a binary mask indicating missing positions (1.0 for missing, 0.0 otherwise)
mask_mcar25 = torch.isnan(X_miss_mcar25_tensor).double()

# Initialize and run Sinkhorn-based imputation
sinkhorn_imputer = SinkhornImputation(eps=epsilon_mcar25, batchsize=batchsize, lr=lr, niter=50)
sinkhorn_filled_tensor, maes_mcar25, rmses_mcar25 = sinkhorn_imputer.fit_transform(
    X_miss_mcar25_tensor,
    verbose=True,
    report_interval=50,
    X_true= X_tensor # for tracking MAE/RMSE during training   
)

# Convert imputed tensor to NumPy format for CMI analysis
#sinkhorn_filled_np = sinkhorn_filled_tensor.detach().cpu().numpy()

# Evaluate final MAE and RMSE only on originally missing positions
mae_final = MAE(sinkhorn_filled_tensor, X_tensor, mask_mcar25)
rmse_final = RMSE(sinkhorn_filled_tensor, X_tensor, mask_mcar25)
print(f"Final MAE (Sinkhorn, MCAR-25): {mae_final:.4f}")
print(f"Final RMSE (Sinkhorn, MCAR-25): {rmse_final:.4f}")


#Y_numpy = Y.to_numpy(dtype=int).reshape(-1,1)
Y_tensor_reshaped = Y_tensor.view(-1, 1) 
full_data_with_y_sinkhorn = torch.cat([sinkhorn_filled_tensor, Y_tensor_reshaped], dim=1)
#full_data_with_y_sinkhorn = np.concatenate([sinkhorn_filled_tensor, Y_tensor], axis=1)

# Estimate (CMI) after imputation
# for each predefined triplet using all three CMI estimation methods
for triplet in cmi_triplets:
    cmi1, cmi2, cmi3 = compute_all_cmi_methods(full_data_with_y_sinkhorn,
        data_processed_tensor, # One-hot encoded + scaled continuous
        one_hot_encoded_tensor, # Only the discrete part encoded
        continuous_part_tensor, # Only the continuous part
        triplet, # (X, Y, Z)
        encoder,
        discrete_columns,
        continuous_columns
    )

    print(f"[Sinkhorn Imputation] Triplet {triplet} :")
    print(" CMI (Soft Kronecker + Gaussian):", cmi1)
    print(" CMI (Gumbel-Softmax + Gaussian):", cmi2)
    print(" CMI (Gumbel on Discrete + Separate Kernels):", cmi3)
    print()


# In[25]:


#sinkhorn cmi

from Sinkhorn_CMI import SinkhornImputation_CMI
from Utils import pick_epsilon, MAE, RMSE

n, d = X_miss_with_Y_tensor.shape
batchsize = 28
lr = 1e-2
epsilon = pick_epsilon(X_miss_with_Y_tensor)
mask_mcar25 = torch.isnan(X_miss_with_Y_tensor).double()

results = []

for cmi_index in [0, 1, 2]:
    print(f"\n\n=========================== Running Sinkhorn_CMI with CMI Method {cmi_index + 1} ===========================")

    for triplet_index in [-1] + list(range(len(cmi_triplets))):
        sk_imputer = SinkhornImputation_CMI(
            eps=epsilon,
            batchsize=batchsize,
            lr=lr,
            niter=10,
            highest_lamda_cmi=500,
            cmi_index=cmi_index
        )

        fit_kwargs = dict(
            X=X_miss_with_Y_tensor,
            verbose=False,
            report_interval=50,
            X_true=torch.cat([X_tensor, Y_tensor.reshape(-1, 1)], dim=1),
            X_cols=[cmi_triplets[triplet_index][0]] if triplet_index != -1 else [t[0] for t in cmi_triplets],
            Y_cols=[cmi_triplets[triplet_index][1]] if triplet_index != -1 else [t[1] for t in cmi_triplets],
            Z_cols=[cmi_triplets[triplet_index][2]] if triplet_index != -1 else [t[2] for t in cmi_triplets],
            encoder=encoder,
            discrete_columns=discrete_columns,
            continuous_columns=continuous_columns
        )

        if cmi_index in [1, 2]:
            fit_kwargs["Y"] = Y_tensor

        sk_imp, maes, rmses, history = sk_imputer.fit_transform(**fit_kwargs)
        
        # --- Compute CMI for each triplet after imputation ---
        print(f"\n[CMI Calculation after Sinkhorn_CMI - Method {cmi_index + 1}]")
        full_data_with_y = sk_imp  

        for triplet in cmi_triplets:
            cmi1, cmi2, cmi3 = compute_all_cmi_methods(
            full_data_with_y,
            data_processed_tensor,
            one_hot_encoded_tensor,
            continuous_part_tensor,
            triplet,
            encoder,
            discrete_columns,
            continuous_columns
            )

            print(f"[Sinkhorn_CMI - CMI Method {cmi_index + 1}] Triplet {triplet} :")
            print(" CMI (Soft Kronecker + Gaussian):", cmi1)
            print(" CMI (Gumbel-Softmax + Gaussian):", cmi2)
            print(" CMI (Gumbel + Separate Kernels):", cmi3)
            print()

        print("\n--- Final MAE / RMSE ---")
        with torch.no_grad():
            mask = torch.isnan(X_miss_with_Y_tensor).double()
            final_mae = MAE(sk_imp, torch.cat([X_tensor, Y_tensor.reshape(-1, 1)], dim=1), mask).item()
            final_rmse = RMSE(sk_imp, torch.cat([X_tensor, Y_tensor.reshape(-1, 1)], dim=1), mask).item()
            print(f"Final MAE: {final_mae:.4f}")
            print(f"Final RMSE: {final_rmse:.4f}")


# In[26]:


# Mean Imputation

# --- Step 1: Mean Imputation on X only (without Y) ---
from sklearn.impute import SimpleImputer

# X only (without Y), as NumPy and Tensor
X_miss_np = X_miss_mcar25.to_numpy(dtype=float)
X_miss_tensor = torch.tensor(X_miss_np, dtype=torch.float32)

# Groundtruth X only (without Y)
X_true_only = X_tensor[:, :X_miss_tensor.shape[1]]

# Mask over X only
mask_mcar25 = torch.isnan(X_miss_tensor).double()

# Run Mean Imputation
mean_imp_np = SimpleImputer().fit_transform(X_miss_tensor)
mean_imp_torch = torch.tensor(mean_imp_np, dtype=torch.float32)

# Step 2: Evaluate MAE only on X (exclude Y from groundtruth comparison)
mean_mcae_mcar25 = MAE(mean_imp_torch, X_true_only, mask_mcar25)
print("Final MAE (Mean Imputation):", mean_mcae_mcar25.item())

# Step 3: Append Y to imputed X for CMI computation
Y_tensor_reshaped = Y_tensor.view(-1, 1)
full_data_with_y_mean = torch.cat([mean_imp_torch, Y_tensor_reshaped], dim=1)

# --- Step 4: Compute CMI for all triplets ---
for triplet in cmi_triplets:
    cmi1, cmi2, cmi3 = compute_all_cmi_methods(
        full_data_with_y_mean,
        data_processed_tensor,           # One-hot + scaled continuous
        one_hot_encoded_tensor,          # Encoded discrete
        continuous_part_tensor,          # Continuous only
        triplet,
        encoder,
        discrete_columns,
        continuous_columns
    )

    print(f"[Mean Imputation] Triplet {triplet} :")
    print(" CMI (Soft Kronecker + Gaussian):", cmi1)
    print(" CMI (Gumbel-Softmax + Gaussian):", cmi2)
    print(" CMI (Gumbel + Separate Kernels):", cmi3)
    print()


# In[27]:


# --- Step 1: Imputation by Chained Equations on X only (without Y) ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


X_miss_np = X_miss_mcar25.to_numpy(dtype=float)
X_miss_tensor = torch.tensor(X_miss_np, dtype=torch.float32)
X_true_only = X_tensor[:, :X_miss_tensor.shape[1]]
mask_mcar25 = torch.isnan(X_miss_tensor).double()

# Run MICE / ICE
ice_imp_np = IterativeImputer(random_state=0, max_iter=500).fit_transform(X_miss_tensor)
ice_imp_torch = torch.tensor(ice_imp_np, dtype=torch.float32)

# Step 2: Evaluate MAE only on X
ice_mae_mcar25 = MAE(ice_imp_torch, X_true_only, mask_mcar25)
print("Final MAE (Imputation by Chained Equations):", ice_mae_mcar25.item())

# Step 3: Append Y to imputed X
Y_tensor_reshaped = Y_tensor.view(-1, 1)
full_data_with_y_Chained_Equations = torch.cat([ice_imp_torch, Y_tensor_reshaped], dim=1)

# --- Step 4: Compute CMI for all triplets ---
for triplet in cmi_triplets:
    cmi1, cmi2, cmi3 = compute_all_cmi_methods(
        full_data_with_y_Chained_Equations,
        data_processed_tensor,           # One-hot + scaled continuous
        one_hot_encoded_tensor,          # Encoded discrete
        continuous_part_tensor,          # Continuous only
        triplet,
        encoder,
        discrete_columns,
        continuous_columns
    )

    print(f"[Imputation by Chained Equations] Triplet {triplet} :")
    print(" CMI (Soft Kronecker + Gaussian):", cmi1)
    print(" CMI (Gumbel-Softmax + Gaussian):", cmi2)
    print(" CMI (Gumbel + Separate Kernels):", cmi3)
    print()


# In[28]:


# --- Step 1: SoftImpute only on X (exclude Y) ---
from SoftImpute import cv_softimpute, softimpute


X_miss_np = X_miss_mcar25.to_numpy(dtype=float)
X_true_only = X_tensor[:, :X_miss_np.shape[1]]
mask_mcar25 = torch.isnan(torch.tensor(X_miss_np, dtype=torch.float32)).double()

# Run CV to select lambda
cv_error_mcar25, grid_lambda_mcar25 = cv_softimpute(X_miss_np, grid_len=15)
lbda_mcar25 = grid_lambda_mcar25[np.argmin(cv_error_mcar25)]

# Run soft imputation
soft_imp_mcar25 = softimpute(X_miss_np, lbda_mcar25)[1]
soft_imp_mcar25_torch = torch.tensor(soft_imp_mcar25, dtype=torch.float32)

# Step 2: Evaluate MAE
soft_mae_mcar25 = MAE(soft_imp_mcar25_torch, X_true_only, mask_mcar25)
print("Final MAE (Soft Imputation):", soft_mae_mcar25.item())

# Step 3: Concatenate Y
Y_tensor_reshaped = Y_tensor.view(-1, 1)
full_data_with_y_soft = torch.cat([soft_imp_mcar25_torch, Y_tensor_reshaped], dim=1)

# --- Step 4: Compute CMI for all triplets ---
for triplet in cmi_triplets:
    cmi1, cmi2, cmi3 = compute_all_cmi_methods(
        full_data_with_y_soft,
        data_processed_tensor,
        one_hot_encoded_tensor,
        continuous_part_tensor,
        triplet,
        encoder,
        discrete_columns,
        continuous_columns
    )

    print(f"[Soft Imputation] Triplet {triplet} :")
    print(" CMI (Soft Kronecker + Gaussian):", cmi1)
    print(" CMI (Gumbel-Softmax + Gaussian):", cmi2)
    print(" CMI (Gumbel + Separate Kernels):", cmi3)
    print()


# # ______________________________________

# In[ ]:




