
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from geomloss import SamplesLoss

from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

from Utils import *
from SoftImpute import softimpute, cv_softimpute
from SinkhornImputation import SinkhornImputation
from Sinkhorn_CMI import *
from RR_imputer import RRimputer
import matplotlib.pyplot as plt
from CMI import *

from Inject_Missing_Values import *

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")
import pandas as pd

torch.set_default_tensor_type('torch.DoubleTensor')

def generateMissingValues(X,Y,missing_type,missing_rate,dependencies=None):

    if missing_type == "MCAR":
        generator_mcar = Inject_Missing_Values()
        miss_mcar,index_mcar = generator_mcar.MCAR(X,dependencies,missing_rate)
        total_missing_percentage_mcar= miss_mcar.isnull().sum().sum() / miss_mcar.size * 100
        #print(f"Total Missing Percentage MCAR: {total_missing_percentage_mcar:.2f}%")
        miss_mcar = pd.concat([miss_mcar, Y], axis=1) #adding the target coloumn
        missing_percentage = (miss_mcar.isnull().sum() / len(miss_mcar)) * 100
        #print("Coloumn Wise Missing Percentage: ", missing_percentage)
        miss_mcar_numpy = scale(miss_mcar)
        miss_mcar_tensor = torch.tensor(miss_mcar_numpy)
        mask  = torch.isnan(miss_mcar_tensor).double()
        return miss_mcar_tensor, index_mcar, mask
    
    if missing_type == "MAR":
        generator_mar = Inject_Missing_Values()
        miss_mar,index_mar = generator_mar.MAR(X,dependencies,missing_rate)
        miss_mar = pd.concat([miss_mar, Y], axis=1) #adding the target coloumn

        total_missing_percentage_mar = miss_mar.isnull().sum().sum() / miss_mar.size * 100
        #print(f"Total Missing Percentage MAR25: {total_missing_percentage_mar:.2f}%")
        missing_percentage = (miss_mar.isnull().sum() / len(miss_mar)) * 100
        #print("Coloumn Wise Missing Percentage: ",missing_percentage)
        miss_mar_numpy = scale(miss_mar)
        miss_mar_tensor = torch.tensor(miss_mar_numpy)
        mask  = torch.isnan(miss_mar_tensor).double()
        return miss_mar_tensor, index_mar, mask
 
    if missing_type == "MNAR":
        generator_mnar = Inject_Missing_Values()
        miss_mnar,index_mnar = generator_mnar.MNAR(X,dependencies,missing_rate)

        miss_mnar = pd.concat([miss_mnar, Y], axis=1) #adding the target coloumn

        total_missing_percentage_mnar = miss_mnar.isnull().sum().sum() / miss_mnar.size * 100
        #print(f"Total Missing Percentage MNAR: {total_missing_percentage_mnar:.2f}%")
        missing_percentage = (miss_mnar.isnull().sum() / len(miss_mnar)) * 100
        #print("Coloumn Wise Missing Percentage: ",missing_percentage)
        miss_mnar_numpy = scale(miss_mnar)
        miss_mnar_tensor = torch.tensor(miss_mnar_numpy)
        mask  = torch.isnan(miss_mnar_tensor).double()
        return miss_mnar_tensor, index_mnar, mask


#only Sinkhorn
def sinkhor(groundTruth_tensor,miss_data_tensor, mask, niter,bucket_specs, X_cols, Y_cols, Z_cols):
   #only Sinkhorn
    n_data, d_data = miss_data_tensor.shape
    batchsize = 128 # If the batch size is larger than half the dataset's size,
                    # it will be redefined in the imputation methods.
    lr = 1e-2
    epsilon = pick_epsilon(miss_data_tensor)
    #print(epsilon)
    mask = torch.isnan(miss_data_tensor).double()


    sk_imputer = SinkhornImputation(eps=epsilon, batchsize=batchsize, lr=lr, niter=niter)
    sk_imp_data, sk_mae_track, sk_rmse_track = sk_imputer.fit_transform(miss_data_tensor, verbose=True, report_interval=50, X_true=groundTruth_tensor)

    #using numpy version of data
    sk_imp_data_numpy = sk_imp_data.detach().cpu().numpy()
    sk_mae = MAE(sk_imp_data,groundTruth_tensor , mask)
    #sk_rmse = RMSE(sk_imp_data, groundTruth_tensor, mask)
    #print("MAE:", sk_mae)

    sk_cmi = CMI.c_m_i(sk_imp_data, bucket_specs, X_cols, Y_cols, Z_cols)
    #print("CMI:", sk_cmi)

    return sk_imp_data, sk_mae_track, sk_mae, sk_cmi

#sinkhorn_CMI
def sinkhornCMI(groundTruth_tensor,miss_data_tensor, mask, niter,highest_lamda_cmi,bucket_specs, X_cols, Y_cols, Z_cols):
   
    n, d = miss_data_tensor.shape
    batchsize = 128 # If the batch size is larger than half the dataset's size,
                    # it will be redefined in the imputation methods.
    lr = 1e-2
    epsilon = pick_epsilon(miss_data_tensor)
    mask  = torch.isnan(miss_data_tensor).double()


    sk_cmi_imputer = SinkhornImputation_CMI(eps=epsilon, batchsize=batchsize, lr=lr, niter = niter,highest_lamda_cmi = highest_lamda_cmi)
    sk_cmi_imp, sk_cmi_mae_track, sk_cmi_rmse_track, cmi_loss_track, sinkhorn_loss_track, lamda_cmi_track = sk_cmi_imputer.fit_transform(miss_data_tensor, True, 50, groundTruth_tensor, X_cols, Y_cols, Z_cols, bucket_specs)
    #using numpy version of data
    sk_imp_data_numpy = sk_cmi_imp.detach().cpu().numpy()

    sk_cmi_mae = MAE(sk_cmi_imp, groundTruth_tensor , mask)
    sk_cmi_rmse = RMSE(sk_cmi_imp, groundTruth_tensor, mask)
    
    sk_cmi = CMI.c_m_i(sk_cmi_imp, bucket_specs, X_cols, Y_cols, Z_cols)


    return sk_cmi_imp, sk_cmi_mae_track, sk_cmi_rmse_track, cmi_loss_track, sinkhorn_loss_track, lamda_cmi_track, sk_cmi_mae, sk_cmi


def mean(groundTruth_tensor,miss_data_tensor, mask, bucket_specs, X_cols, Y_cols, Z_cols):

    mask = torch.isnan(miss_data_tensor).double()
    mean_imp_data = SimpleImputer().fit_transform(miss_data_tensor) #numpy
    mean_imp_data_tensor = torch.tensor(mean_imp_data) #tensor
    mean_mae = MAE(mean_imp_data_tensor, groundTruth_tensor , mask)

    #print("MAE",mean_mae)

    mean_cmi = CMI.c_m_i(mean_imp_data_tensor, bucket_specs, X_cols, Y_cols, Z_cols)
    #print("CMI",cmi_mean)

    return mean_imp_data_tensor, mean_mae, mean_cmi



def ice(groundTruth_tensor, miss_data_tensor, niter, mask, bucket_specs, X_cols, Y_cols, Z_cols):
   
    mask = torch.isnan(miss_data_tensor).double()
    ice_imp_data = IterativeImputer(random_state=0, max_iter = 500).fit_transform(miss_data_tensor) #numpy
    ice_imp_data_tensor = torch.tensor(ice_imp_data) #tensor
    ice_mae = MAE(ice_imp_data_tensor, groundTruth_tensor , mask)
    #print("MAE",ice_mae)

    ice_cmi = CMI.c_m_i(ice_imp_data_tensor, bucket_specs, X_cols, Y_cols, Z_cols)
    #print("CMI",ice_cmi)

    return ice_imp_data_tensor, ice_mae, ice_cmi

def softImpute(groundTruth_tensor, miss_data_tensor, mask, bucket_specs, X_cols, Y_cols, Z_cols):
    
    
    miss_data_numpy = miss_data_tensor.detach().cpu().numpy()
    cv_error, grid_lambda = cv_softimpute(miss_data_numpy, grid_len=15) #data is numpy
    lbda = grid_lambda[np.argmin(cv_error)]
    soft_imp_data_numpy = softimpute((miss_data_numpy), lbda)[1]
    soft_imp_data_tensor = torch.tensor(soft_imp_data_numpy)

    soft_mae = MAE(soft_imp_data_tensor, groundTruth_tensor , mask)
    #print("MAE",soft_mae)

    soft_cmi = CMI.c_m_i(soft_imp_data_tensor, bucket_specs, X_cols, Y_cols, Z_cols)
    #print("CMI",soft_cmi)

    return soft_imp_data_tensor, soft_mae, soft_cmi



def run(groundTruth_tensor, X, Y, cycle, missing_type, missing_rate, dependencies, highest_lamda_cmi, niter,bucket_specs, X_cols, Y_cols, Z_cols):

    sk_mae = []
    sk_cmi = []
    skCmi_mae = []
    skCmi_cmi = []
    mean_mae = []
    mean_cmi = []
    ice_mae = []
    ice_cmi = []
    soft_mae = []
    soft_cmi = []
    miss_data_cmi = []


    for i in range(cycle):
        miss_data_tensor, index_mnar, mask = generateMissingValues(X, Y, missing_type, missing_rate, dependencies=dependencies)
        row_mask = mask.bool().all(dim=1)
        ugly_data = miss_data_tensor[row_mask]
        miss_data_tensor_cmi =0
        #miss_data_tensor_cmi = CMI.c_m_i(ugly_data, bucket_specs, X_cols, Y_cols, Z_cols)

        miss_data_cmi.append(miss_data_tensor_cmi)

        sk_imp_data, sk_mae_track, sk_imp_mae, sk_imp_cmi = sinkhor(groundTruth_tensor,miss_data_tensor, mask, niter,bucket_specs, X_cols, Y_cols, Z_cols)
        sk_mae.append(sk_imp_mae.detach())
        sk_cmi.append(sk_imp_cmi.detach())
        
        sk_cmi_imp, sk_cmi_mae_track, sk_cmi_rmse_track, cmi_loss_track, sinkhorn_loss_track, lamda_cmi_track, skCmi_imp_mae, skCmi_imp_cmi = sinkhornCMI(groundTruth_tensor,miss_data_tensor, mask, niter, highest_lamda_cmi, bucket_specs, X_cols, Y_cols, Z_cols)
        skCmi_mae.append(skCmi_imp_mae.detach())
        skCmi_cmi.append(skCmi_imp_cmi.detach())

        mean_imp_data_tensor, mean_imp_mae, mean_imp_cmi = mean(groundTruth_tensor,miss_data_tensor, mask, bucket_specs, X_cols, Y_cols, Z_cols)
        mean_mae.append(mean_imp_mae.detach())
        mean_cmi.append(mean_imp_cmi.detach())

        ice_imp_data_tensor, ice_imp_mae, ice_imp_cmi = ice(groundTruth_tensor, miss_data_tensor, niter, mask, bucket_specs, X_cols, Y_cols, Z_cols)
        ice_mae.append(ice_imp_mae.detach())
        ice_cmi.append(ice_imp_cmi.detach())

        soft_imp_data_tensor, soft_imp_mae, soft_imp_cmi = softImpute(groundTruth_tensor, miss_data_tensor, mask, bucket_specs, X_cols, Y_cols, Z_cols)
        soft_mae.append(soft_imp_mae.detach())
        soft_cmi.append(soft_imp_cmi.detach())

    return sk_mae, sk_cmi, skCmi_mae, skCmi_cmi, mean_mae, mean_cmi, ice_mae, ice_cmi, soft_mae,  soft_cmi


def visualize():
 print("abc")

