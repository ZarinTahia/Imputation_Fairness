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
    def conditional_mutual_information(data, bucket_specs, X_cols, Y_cols, Z_cols):

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


"""""""""""
class CMI:


    @staticmethod
    def kde_entropy_torch(samples):
        
        num_samples = samples.shape[0]

        # Bandwidth selection using Silverman's rule
        sigma = torch.std(samples) * (4 / (3 * num_samples)) ** (1 / 5)

        # Batch processing for large datasets
        batch_size = min(1000, num_samples)
        total_entropy = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_samples = samples[start_idx:end_idx]
            pairwise_distances = torch.cdist(batch_samples, batch_samples, p=2)  # Euclidean distance
            kernel_matrix = torch.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))
            density_estimates = kernel_matrix.mean(dim=1)

            # Add epsilon to avoid log(0)
            entropy = -torch.mean(torch.log(density_estimates + 1e-10))
            total_entropy += entropy * (end_idx - start_idx) / num_samples  # Weighted by batch size

        return total_entropy
    

    @staticmethod
    def conditional_differential_entropy_torch(X, Z):
        
        XZ = torch.cat([X, Z], dim=1)  # Concatenate X and Z
        h_xz = CMI.kde_entropy_torch(XZ)
        h_z = CMI.kde_entropy_torch(Z)
        return h_xz - h_z  # Conditional entropy h(X|Z)

    @staticmethod
    def conditional_mutual_information(data, bucket_specs, X_cols, Y_cols, Z_cols):
        
       
        #bucketized_tensor = CMI.bucketization(data, bucket_specs) # Skip bucketization for now
        bucketized_tensor = data

        # Select features for X, Y, Z
        X_torch = bucketized_tensor[:, X_cols]
        Y_torch = bucketized_tensor[:, Y_cols]
        Z_torch = bucketized_tensor[:, Z_cols]

        # Compute conditional entropies
        h_x_given_z = CMI.conditional_differential_entropy_torch(X_torch, Z_torch)
        h_y_given_z = CMI.conditional_differential_entropy_torch(Y_torch, Z_torch)
        h_xy_given_z = CMI.conditional_differential_entropy_torch(torch.cat([X_torch, Y_torch], dim=1), Z_torch)

        # Debugging: Print entropies
        #print(f"H(X|Z): {h_x_given_z.item()}, H(Y|Z): {h_y_given_z.item()}, H(X,Y|Z): {h_xy_given_z.item()}")

        # Compute CMI and ensure non-negativity
        cmi = h_x_given_z + h_y_given_z - h_xy_given_z
        #print("cmi",cmi)
        cmi = torch.clamp(cmi, min=0.0001)  # Ensure non-negativity


        return cmi
    

    @staticmethod
    def c_m_i(data, bucket_specs, X_cols, Y_cols, Z_cols):
        
        #bucketized_tensor = CMI.bucketization(data, bucket_specs)
        bucketized_tensor = data

        # Select features for X, Y, Z
        X_torch = bucketized_tensor[:, X_cols].clone().detach().requires_grad_(True)
        Y_torch = bucketized_tensor[:, Y_cols].clone().detach().requires_grad_(True)
        Z_torch = bucketized_tensor[:, Z_cols].clone().detach().requires_grad_(True)

        h_x_given_z = CMI.conditional_differential_entropy_torch(X_torch, Z_torch)
        h_y_given_z = CMI.conditional_differential_entropy_torch(Y_torch, Z_torch)
        h_xy_given_z = CMI.conditional_differential_entropy_torch(torch.cat([X_torch, Y_torch], dim=1), Z_torch)
        #print(f"H(X|Z): {h_x_given_z.item()}, H(Y|Z): {h_y_given_z.item()}, H(X,Y|Z): {h_xy_given_z.item()}")  # Debug
        # Compute CMI
        cmi =  h_x_given_z + h_y_given_z - h_xy_given_z
        cmi = torch.clamp(cmi, min=0.0001)

        return cmi
    







class CMI:


    @staticmethod
    def compute_probabilities_torch(data, columns):
        # Alternative to torch.unique() that preserves gradients
        selected_data = data[:, columns]
        
        # Compute probabilities using soft histogram
        if len(columns) == 1:
            # For single column, use simpler approach
            min_val = selected_data.min()
            max_val = selected_data.max()
            bins = 20  # Adjust based on your needs
            bin_edges = torch.linspace(min_val, max_val, bins + 1, device=data.device)
            bin_edges[-1] += 1e-6
            
            # Soft assignment to bins
            distances = torch.abs(selected_data.unsqueeze(-1) - bin_edges)
            soft_assignments = torch.softmax(-distances * 1e3, dim=-1)
            probs = torch.mean(soft_assignments, dim=0)
            
            # Use bin centers as "unique" values
            unique_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
            return unique_vals, probs
        else:
            # For multiple columns, use a more complex approach
            # This is a simplified version - you might need to adjust for your specific case
            probs = torch.ones(1, device=data.device) / data.shape[0]
            return selected_data[0:1], probs  # Simplified - returns first row and uniform probs

    @staticmethod
    def bucketize_columns(data, bucket_specs):
        # Keep original data for gradient flow
        data_buc = data.clone()
        
        for col, bins in bucket_specs.items():
            feature = data[:, col]
            min_val, max_val = feature.min(), feature.max()
            bin_edges = torch.linspace(min_val, max_val, bins + 1, device=data.device)
            bin_edges[-1] += 1e-6
            
            # Soft bucket assignment
            distances = torch.abs(feature.unsqueeze(-1) - bin_edges)
            soft_assignments = torch.softmax(-distances * 1e3, dim=-1)
            bucket_indices = torch.sum(soft_assignments * torch.arange(bins + 1, device=data.device), dim=-1) - 1
            
            data_buc[:, col] = bucket_indices
        
        return data_buc

     @staticmethod
    def bucketization(data, bucket_specs):
       
        device = data.device
        bucketized = data.clone() # Clone to avoid modifying the original tensor

        for col, n_bins in bucket_specs.items():
            col_data = bucketized[:, col]
        
        # Compute bin edges using quantiles (equal-frequency bins)
            quantiles = torch.linspace(0, 1, n_bins + 1, device=data.device)
            bin_edges = torch.quantile(col_data, quantiles)
            
            # Digitize: assign bin indices (0 to n_bins-1)
            bin_indices = torch.bucketize(col_data, bin_edges[1:-1])  # exclude first and last edge
            
            # Replace the column with bucket indices
            bucketized[:, col] = bin_indices

        return bucketized


    @staticmethod
    def c_m_i(data, bucket_specs, X_cols, Y_cols, Z_cols):

        bucketized_data = CMI.bucketize_columns(data, bucket_specs)
        
        # Compute joint and marginal probabilities
        def get_probs(cols):
            if not cols:
                return torch.tensor([1.0], device=data.device), None
            
            # Get the subset of data we care about
            sub_data = bucketized_data[:, cols]
            
            # For simplicity, we'll use a kernel density estimate
            # This is a placeholder - you may need a more sophisticated approach
            if len(cols) == 1:
                # 1D KDE
                x = torch.linspace(sub_data.min(), sub_data.max(), 20, device=data.device)
                sigma = 0.1 * (sub_data.max() - sub_data.min())
                diffs = (sub_data.unsqueeze(1) - x.unsqueeze(0)).pow(2)
                kde = torch.exp(-diffs / (2 * sigma**2)).mean(dim=0)
                kde = kde / kde.sum()  # Normalize
                return x, kde
            else:
                # Multi-dimensional case - simplified
                return sub_data[0:1], torch.tensor([1.0], device=data.device)
        
        # Get probabilities for each combination
        _, P_Z = get_probs(Z_cols)
        _, P_XZ = get_probs(X_cols + Z_cols)
        _, P_YZ = get_probs(Y_cols + Z_cols)
        _, P_XYZ = get_probs(X_cols + Y_cols + Z_cols)
        
        # Compute CMI using the probabilities
        # Note: This is a simplified version - you'll need to adapt your exact calculation
        cmi = torch.sum(P_XYZ * torch.log((P_Z * P_XYZ) / (P_XZ * P_YZ + 1e-10) + 1e-10))
        
        return torch.clamp(cmi, min=0.0)
    
    @staticmethod
    def conditional_mutual_information(data, bucket_specs, X_cols, Y_cols, Z_cols):
        bucketized_data = CMI.bucketize_columns(data, bucket_specs)
        
        # Compute joint and marginal probabilities
        def get_probs(cols):
            if not cols:
                return torch.tensor([1.0], device=data.device), None
            
            # Get the subset of data we care about
            sub_data = bucketized_data[:, cols]
            
            # For simplicity, we'll use a kernel density estimate
            # This is a placeholder - you may need a more sophisticated approach
            if len(cols) == 1:
                # 1D KDE
                x = torch.linspace(sub_data.min(), sub_data.max(), 20, device=data.device)
                sigma = 0.1 * (sub_data.max() - sub_data.min())
                diffs = (sub_data.unsqueeze(1) - x.unsqueeze(0)).pow(2)
                kde = torch.exp(-diffs / (2 * sigma**2)).mean(dim=0)
                kde = kde / kde.sum()  # Normalize
                return x, kde
            else:
                # Multi-dimensional case - simplified
                return sub_data[0:1], torch.tensor([1.0], device=data.device)
        
        # Get probabilities for each combination
        _, P_Z = get_probs(Z_cols)
        _, P_XZ = get_probs(X_cols + Z_cols)
        _, P_YZ = get_probs(Y_cols + Z_cols)
        _, P_XYZ = get_probs(X_cols + Y_cols + Z_cols)
        
        # Compute CMI using the probabilities
        # Note: This is a simplified version - you'll need to adapt your exact calculation
        cmi = torch.sum(P_XYZ * torch.log((P_Z * P_XYZ) / (P_XZ * P_YZ + 1e-10) + 1e-10))
        
        return torch.clamp(cmi, min=0.0)
    







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
        
        return data_buc.long()

    @staticmethod
    def compute_probabilities_torch(data, columns):
        
        unique_vals, counts = torch.unique(data[:, columns], dim=0, return_counts=True)
        return unique_vals, counts / data.shape[0]
    
    @staticmethod
    def bucketize_tensor(data, bucket_specs):
       
        device = data.device
        bucketized_data = data.clone()  # Clone to avoid modifying the original tensor

        for col, bins in bucket_specs.items():
            feature = bucketized_data[:, col]  # Select column

            # Compute quantiles for bin edges
            quantiles = torch.linspace(0, 1, bins + 1, device=data.device)
            bin_edges = torch.quantile(feature, quantiles)

            # Compute soft bin assignments using Gaussian kernel
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Use bin centers as reference points
            distances = torch.abs(feature.unsqueeze(1) - bin_centers.unsqueeze(0))
            sigma = (bin_edges[-1] - bin_edges[0]) / bins  # Bandwidth
            soft_assignments = torch.exp(-0.5 * (distances / sigma) ** 2)
            soft_assignments = soft_assignments / soft_assignments.sum(dim=1, keepdim=True)

            # Compute differentiable bin values
            new_feature = (soft_assignments @ bin_centers).squeeze()

            # Replace the original column with the bucketized feature
            bucketized_data = torch.cat([
                bucketized_data[:, :col], 
                new_feature.unsqueeze(1),  
                bucketized_data[:, col+1:]
            ], dim=1)  

        return bucketized_data  # Convert back to tensor

    @staticmethod
    def kde_entropy_torch(samples):
        
        num_samples = samples.shape[0]

        # Bandwidth selection using Silverman's rule
        sigma = torch.std(samples) * (4 / (3 * num_samples)) ** (1 / 5)

        # Batch processing for large datasets
        batch_size = min(1000, num_samples)
        total_entropy = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_samples = samples[start_idx:end_idx]
            pairwise_distances = torch.cdist(batch_samples, batch_samples, p=2)  # Euclidean distance
            kernel_matrix = torch.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))
            density_estimates = kernel_matrix.mean(dim=1)

            # Add epsilon to avoid log(0)
            entropy = -torch.mean(torch.log(density_estimates + 1e-10))
            total_entropy += entropy * (end_idx - start_idx) / num_samples  # Weighted by batch size

        return total_entropy

    @staticmethod
    def conditional_differential_entropy_torch(X, Z):
        
        XZ = torch.cat([X, Z], dim=1)  # Concatenate X and Z
        h_xz = CMI.kde_entropy_torch(XZ)
        h_z = CMI.kde_entropy_torch(Z)
        return h_xz - h_z  # Conditional entropy h(X|Z)

    @staticmethod
    def conditional_mutual_information(data_tensor, bucket_specs, X_cols, Y_cols, Z_cols):
        
        Computes differentiable CMI using KDE.
        
        Args:
        - data_tensor: Input PyTorch tensor.
        - bucket_specs: Dictionary specifying number of bins per column.
        - X_cols, Y_cols, Z_cols: Lists of column indices for X, Y, Z.
        
        Returns:
        - cmi_value: Estimated CMI.
        
        # Apply bucketization (if needed)
        bucketized_tensor = data_tensor  # Skip bucketization for now

        # Select features for X, Y, Z
        X_torch = bucketized_tensor[:, X_cols]
        Y_torch = bucketized_tensor[:, Y_cols]
        Z_torch = bucketized_tensor[:, Z_cols]

        # Compute conditional entropies
        h_x_given_z = CMI.conditional_differential_entropy_torch(X_torch, Z_torch)
        h_y_given_z = CMI.conditional_differential_entropy_torch(Y_torch, Z_torch)
        h_xy_given_z = CMI.conditional_differential_entropy_torch(torch.cat([X_torch, Y_torch], dim=1), Z_torch)

        # Debugging: Print entropies
        #print(f"H(X|Z): {h_x_given_z.item()}, H(Y|Z): {h_y_given_z.item()}, H(X,Y|Z): {h_xy_given_z.item()}")

        # Compute CMI and ensure non-negativity
        cmi = h_x_given_z + h_y_given_z - h_xy_given_z
        #print("cmi",cmi)
        cmi = torch.clamp(cmi, min=0.001)  # Ensure non-negativity


        return cmi
   
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
        
        return torch.clamp(cmi, min=0.0)
        
    
    


    
    @staticmethod
    def c_m_i(data, bucket_specs, X_cols, Y_cols, Z_cols):
        
        bucketized_tensor = data

        # Select features for X, Y, Z
        X_torch = bucketized_tensor[:, X_cols].clone().detach().requires_grad_(True)
        Y_torch = bucketized_tensor[:, Y_cols].clone().detach().requires_grad_(True)
        Z_torch = bucketized_tensor[:, Z_cols].clone().detach().requires_grad_(True)

        h_x_given_z = CMI.conditional_differential_entropy_torch(X_torch, Z_torch)
        h_y_given_z = CMI.conditional_differential_entropy_torch(Y_torch, Z_torch)
        h_xy_given_z = CMI.conditional_differential_entropy_torch(torch.cat([X_torch, Y_torch], dim=1), Z_torch)
        print(f"H(X|Z): {h_x_given_z.item()}, H(Y|Z): {h_y_given_z.item()}, H(X,Y|Z): {h_xy_given_z.item()}")  # Debug
        # Compute CMI
        cmi =  h_x_given_z + h_y_given_z - h_xy_given_z

        return cmi
        

import torch

class CMI:
    @staticmethod
    def _gaussian_kernel(x, y, bandwidth):
        
        Compute Gaussian kernel density estimate between x and y
        x: (n, d)
        y: (m, d)
        bandwidth: float
        Returns: (n,) tensor of density estimates
    
        n, d = x.shape
        x = x.unsqueeze(1)  # (n, 1, d)
        y = y.unsqueeze(0)  # (1, m, d)

        diff = (x - y) / bandwidth
        exponent = -0.5 * torch.sum(diff ** 2, dim=-1)  # (n, m)
        kernel_vals = torch.exp(exponent) / ((2 * torch.pi) ** (d / 2) * bandwidth ** d)
        return kernel_vals.mean(dim=1)  # (n,)

    @staticmethod
    def estimate_cmi_kde(X, Y, Z, bandwidth):
       
        XYZ = torch.cat([X, Y, Z], dim=1)
        XZ = torch.cat([X, Z], dim=1)
        YZ = torch.cat([Y, Z], dim=1)

        p_xyz = CMI._gaussian_kernel(XYZ, XYZ, bandwidth)
        p_xz  = CMI._gaussian_kernel(XZ, XZ, bandwidth)
        p_yz  = CMI._gaussian_kernel(YZ, YZ, bandwidth)
        p_z   = CMI._gaussian_kernel(Z, Z, bandwidth)

        eps = 1e-10
        p_x_given_z = p_xz / (p_z + eps)
        p_y_given_z = p_yz / (p_z + eps)
        p_xy_given_z = p_xyz / (p_z + eps)

        ratio = p_xy_given_z / (p_x_given_z * p_y_given_z + eps)
        log_fraction = torch.log(ratio + eps)

        return torch.mean(log_fraction)

    @staticmethod
    def c_m_i(data,b, X_cols, Y_cols, Z_cols, bandwidth=0.3):
       
       
        X = data[:, X_cols]
        Y = data[:, Y_cols]
        Z = data[:, Z_cols]
        return CMI.estimate_cmi_kde(X, Y, Z, bandwidth)

    @staticmethod
    def conditional_mutual_information(data,b, X_cols, Y_cols, Z_cols, bandwidth=0.3):
        
        X = data[:, X_cols]
        Y = data[:, Y_cols]
        Z = data[:, Z_cols]
        return CMI.estimate_cmi_kde(X, Y, Z, bandwidth)

"""""
        
 