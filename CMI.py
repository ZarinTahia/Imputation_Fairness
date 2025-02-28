

import torch
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")
import pandas as pd

torch.set_default_tensor_type('torch.DoubleTensor')

class CMI():

    def bucketize_columns(data, bucket_specs):
        """
        Bucketizes specific columns in a dataset using different binning strategies.

        :param data: PyTorch tensor (N, D) (continuous values)
        :param bucket_specs: Dictionary {column_index: (method, num_bins or bin_edges)}
                            Example: {0: ('uniform', 3), 1: ('quantile', 4)}
        :return: Discretized tensor

        """
        
        data_buc = data.detach().clone()  # Ensure no gradients

        for col, bins in bucket_specs.items():
            feature = data_buc[:, col]  # Extract column
            #print(feature)

            
            min_val, max_val = feature.min(), feature.max()
            bin_edges = torch.linspace(min_val, max_val, bins + 1)
            #print(bin_edges)
            bin_edges[-1] += 1e-6 
            
            # Apply bucketization
            data_buc[:, col] = torch.bucketize(feature, bin_edges,right=True) # Start bins from 0
            #print(data_buc[:,col].long().unique())

        return data_buc.long()  # Convert to integer values

    def compute_probabilities_torch(data, columns):
        """
        Compute probability distributions for given feature columns using PyTorch.

        :param data: PyTorch tensor of shape (N, D) (discretized)
        :param columns: List of column indices to compute probabilities
        :return: Unique values and probability tensor
        """
        unique_vals, counts = torch.unique(data[:, columns], dim=0, return_counts=True)
        probs = counts.float() / data.shape[0]
        return unique_vals, probs

    def conditional_mutual_information(data, X_cols, Y_cols, Z_cols, bucket_specs, delta=1):
        """
        Compute Conditional Mutual Information I(X;Y|Z) with bucketization inside.

        :param data: PyTorch tensor (N, D) (continuous values)
        :param X_cols: List of column indices for X
        :param Y_cols: List of column indices for Y
        :param Z_cols: List of column indices for Z
        :param bucket_specs: Dictionary specifying bucketization for each column
        :param delta: Smoothing parameter to avoid log(0)
        :return: Conditional Mutual Information (scalar)
        """
        # Apply bucketization inside CMI function
        bucketized_data = CMI.bucketize_columns(data, bucket_specs)

        # Compute probability distributions
        unique_Z, P_Z = CMI.compute_probabilities_torch(bucketized_data, Z_cols)
        unique_XZ, P_XZ = CMI.compute_probabilities_torch(bucketized_data, X_cols + Z_cols)
        unique_YZ, P_YZ = CMI.compute_probabilities_torch(bucketized_data, Y_cols + Z_cols)
        unique_XYZ, P_XYZ = CMI.compute_probabilities_torch(bucketized_data, X_cols + Y_cols + Z_cols)

        cmi = 0
        for i in range(len(unique_XYZ)):
            xyz = unique_XYZ[i]
            z = xyz[len(X_cols + Y_cols):]  # Extract Z values
            xz = torch.cat((xyz[:len(X_cols)], z))  # Concatenate X and Z
            yz = torch.cat((xyz[len(X_cols):len(X_cols + Y_cols)], z))  # Concatenate Y and Z

            mask_z = (unique_Z == z).all(dim=1)
            mask_xz = (unique_XZ == xz).all(dim=1)
            mask_yz = (unique_YZ == yz).all(dim=1)

            if mask_z.any() and mask_xz.any() and mask_yz.any():
                P_z = P_Z[mask_z].sum()
                P_xyz = P_XYZ[i]
                P_xz = P_XZ[mask_xz].sum()
                P_yz = P_YZ[mask_yz].sum()

                # Compute CMI (avoid log(0) by adding a small constant)
                cmi += delta * P_xyz * torch.log2((P_z * P_xyz) / (P_xz * P_yz + 1e-10))

        return cmi.item()

