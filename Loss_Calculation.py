import numpy as np
import torch
from geomloss import SamplesLoss

from utils import nanmean, MAE, RMSE

import logging


class Loss_Calculation():
    """
    'One parameter equals one imputed value' model (Algorithm 1. in the paper)

    Parameters
    ----------

    eps: float, default=0.01
        Sinkhorn regularization parameter.
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing 
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"


    """
    def __init__(self, 
                 eps=0.01, 
                 lr=1e-2, 
                 opt=torch.optim.RMSprop, 
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9, seed = 42):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
        self.seed = seed

    def transform(self, X, verbose=True, report_interval=500, X_true=None):
        """
        Imputes missing values using a batched OT loss

        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a
            validation score during training, and return score arrays.
            For validation/debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).


        """
        np.random.seed(self.seed)
        X = X.clone()
        n, d = X.shape
        
        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(X).double()
        imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]


        loss_history = []
        sinkhorn_divergence_history = []

        if verbose:
            logging.info(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        for i in range(self.niter):
            X_filled = X.detach().clone()
            if torch.isnan(X_filled).any():
                X_filled[mask.bool()] = imps
            loss = 0
          
            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)
                
                
               
                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                print(idx1)
                print(idx2)
                
                
    
                loss = loss + self.sk(X1, X2)
                sinkhorn_divergence_history.append(self.sk(X1, X2))
                print("loss",loss)
            
            loss_history.append(loss.item())
           
                

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                ### Catch numerical errors/overflows (should not happen)
                logging.info("Nan or inf loss")
                break

            
            
            

            

        if X_true is not None:
            return loss_history
        else:
            return loss_history
