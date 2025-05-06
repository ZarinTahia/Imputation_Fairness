import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import logging
import torch
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
logger = logging.getLogger()
logger.setLevel(logging.INFO)
torch.set_default_tensor_type('torch.DoubleTensor')

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint




class CMI:

    @staticmethod
    def bucketize_columns(data, bucket_specs):
        
        
        data_buc = data.clone()  # Ensure no gradients
        
        for col, bins in bucket_specs.items():
            feature = data_buc[:, col]
            min_val, max_val = feature.min(), feature.max()
            bin_edges = torch.linspace(min_val, max_val, bins + 1, device=data.device)
            bin_edges[-1] += 1e-6  # Ensure max value inclusion
            
            data_buc[:, col] = torch.searchsorted(bin_edges, feature, right=True) - 1
            #print(torch.unique(data_buc[:,col]))
        
        
        
        return data_buc.long()

    @staticmethod
    def compute_probabilities_torch(data, columns):
        
        unique_vals, counts = torch.unique(data[:, columns], dim=0, return_counts=True)
        return unique_vals, counts / data.shape[0]

   
    @staticmethod
    def c_m_i(data, bucket_specs, X_cols, Y_cols, Z_cols):

        bucketized_data = CMI.bucketize_columns(data, bucket_specs)
            
        prob_cache = {}
        columns_list = [Z_cols, X_cols + Z_cols, Y_cols + Z_cols, X_cols + Y_cols + Z_cols]
        keys = ['Z', 'XZ', 'YZ', 'XYZ']
            
        for key, cols in zip(keys, columns_list):
            prob_cache[key] = CMI.compute_probabilities_torch(bucketized_data, cols)
            
        unique_XYZ, P_XYZ = prob_cache['XYZ']
        unique_Z, P_Z = prob_cache['Z']
        unique_XZ, P_XZ = prob_cache['XZ']
        unique_YZ, P_YZ = prob_cache['YZ']
            
        cmi = torch.tensor(0.0, device=data.device, dtype=torch.float64)
           

            
        for i, xyz in enumerate(unique_XYZ):
                z = xyz[len(X_cols + Y_cols):]
                xz = torch.cat((xyz[:len(X_cols)], z))
                yz = torch.cat((xyz[len(X_cols):len(X_cols + Y_cols)], z))
                
                mask_z = (unique_Z == z).all(dim=1)
                mask_xz = (unique_XZ == xz).all(dim=1)
                mask_yz = (unique_YZ == yz).all(dim=1)
                
                if mask_z.any() and mask_xz.any() and mask_yz.any():
                    P_z = P_Z[mask_z].sum()
                    P_xyz = P_XYZ[i]
                    P_xz = P_XZ[mask_xz].sum()
                    P_yz = P_YZ[mask_yz].sum()
                    cmi =  cmi + P_xyz * torch.log((P_z * P_xyz) / (P_xz * P_yz + 1e-10))
        
        
        return torch.clamp(cmi, min=0.00001)

        
""""""""""""""""



class CMI:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self.eps = 1e-20  # Small constant for numerical stability

    def gumbel_softmax_sample(self, logits):
        
        bucketed_data = torch.zeros_like(data, dtype=torch.long)
        for col, bins in bucket_specs.items():
            col_data = data[:, col]
            # Normalize to [0, 1] range
            normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min() + self.eps)
            # Convert to bucket indices (0 to bins-1)
            bucketed_data[:, col] = torch.floor(normalized * bins).clamp(0, bins-1)
        return bucketed_data

    def compute_joint_probs(self, data, cols, bucket_specs):
        
        bucketed_data = self.bucketize_data(data, bucket_specs)
        
        # Get bucket counts for each column
        bucket_counts = [bucket_specs[col] for col in cols]
        total_buckets = torch.prod(torch.tensor(bucket_counts))
        
        # Compute linear indices
        strides = torch.cumprod(torch.tensor([1] + bucket_counts[:-1]), dim=0)
        indices = torch.sum(bucketed_data[:, cols] * strides, dim=1)
        
        # Convert to one-hot and compute probabilities
        one_hot = F.one_hot(indices, num_classes=int(total_buckets)).float()
        counts = torch.sum(one_hot, dim=0)
        logits = torch.log(counts + self.eps)
        probs = self.gumbel_softmax_sample(logits)
        
        return probs.reshape(*bucket_counts)

    def c_m_i(self, data, bucket_specs, X_cols, Y_cols, Z_cols):
       
        # Verify all columns exist in bucket_specs
        all_cols = X_cols + Y_cols + Z_cols
        assert all(col in bucket_specs for col in all_cols), \
               "All specified columns must be in bucket_specs"
        
        # Compute joint and marginal probabilities
        P_XYZ = self.compute_joint_probs(data, X_cols+Y_cols+Z_cols, bucket_specs)
        P_XZ = self.compute_joint_probs(data, X_cols+Z_cols, bucket_specs)
        P_YZ = self.compute_joint_probs(data, Y_cols+Z_cols, bucket_specs)
        P_Z = self.compute_joint_probs(data, Z_cols, bucket_specs)
        
        # Reshape for proper broadcasting
        # P_Z should have shape (Z_dims)
        # P_XYZ should have shape (X_dims, Y_dims, Z_dims)
        # P_XZ should have shape (X_dims, Z_dims)
        # P_YZ should have shape (Y_dims, Z_dims)
        
        # Add dimensions for proper broadcasting
        P_Z = P_Z.reshape((1,)*(len(X_cols)+len(Y_cols)) + P_Z.shape)
        P_XZ = P_XZ.reshape(P_XZ.shape + (1,)*len(Y_cols))
        P_YZ = P_YZ.reshape((1,)*len(X_cols) + P_YZ.shape)
        
        # Compute CMI with proper gradient flow
        log_ratio = (torch.log(P_Z + self.eps) + 
                   torch.log(P_XYZ + self.eps) - 
                   torch.log(P_XZ + self.eps) - 
                   torch.log(P_YZ + self.eps))
        cmi = torch.sum(P_XYZ * log_ratio)
        
        return cmi
"""""""""""