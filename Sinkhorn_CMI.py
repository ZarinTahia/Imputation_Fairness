import numpy as np
import torch
from geomloss import SamplesLoss

from utils import nanmean, MAE, RMSE

import logging 
class SinkhornImputation_CMI():
    # Initialization and other code...

    def __init__(self, 
                 eps=0.01, 
                 lr=1e-2, 
                 opt=torch.optim.RMSprop, 
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9,
                 lambda_cmi=0.1):  # Added lambda_cmi for controlling the CMI penalty strength
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
        self.lambda_cmi = lambda_cmi  # Store the CMI penalty trade-off parameter

    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None):
        """
        Imputes missing values using a batched OT loss and a weighted CMI penalty.
        """
        X = X.clone()
        n, d = X.shape

        sinkhorn_loss_history = []
        cmi_penalty_history = []

        imps_evolution = []  # To store imps values at each iteration
        grad_evolution = []

        mask = torch.isnan(X).double()
        imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
        #print(imps)
        initial_missing = imps.clone()
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):

            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0

            # Sinkhorn Loss
            for _ in range(self.n_pairs):
                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)
        
                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
        
                loss = loss + self.sk(X1, X2)
            

            # Compute CMI penalty and apply the trade-off with lambda_cmi
            #cmi_penalty = self.compute_cmi_penalty(X_filled)
            #loss += self.lambda_cmi * cmi_penalty  # Apply the trade-off here

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.info("Nan or inf loss")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #imps_evolution.append(imps.clone().detach().cpu().numpy())
            #grad_evolution.append(imps.grad.clone().detach().cpu().numpy())
            #logging.info(f'Iteration {i}:\t imps matrix: {imps}\t')
        

            if X_true is not None:
                maes[i] = MAE(X_filled, X_true, mask).item()
                rmses[i] = RMSE(X_filled, X_true, mask).item()
            if(i% report_interval == 0):
                sinkhorn_loss_history.append(loss.item() / self.n_pairs)

            if verbose and (i % report_interval == 0):
                if X_true is not None:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')
                    
                else:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}')
                    
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, maes, rmses, sinkhorn_loss_history
        else:
            return X_filled, sinkhorn_loss_history
    '''''
    def compute_cmi_penalty(self, X_filled):
        """
        Compute the CMI penalty term for the imputed data.

        This penalty is designed to reduce the CMI between the imputed features.
        """
        cmi_penalty = 0.0
        # Iterate through pairs of features
        for i in range(X_filled.shape[1] - 1):  # Example: iterate through pairs of features
            for j in range(i + 1, X_filled.shape[1]):
                cmi = compute_cmi(X_filled[:, i], X_filled[:, j], X_filled[:, -1])  # Example: condition on the last column
                cmi_penalty += cmi

        return cmi_penalty
'''''

