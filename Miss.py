# -*- coding: utf-8 -*

# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

import warnings

import numpy as np
import pandas as pd
import random
from collections import defaultdict


# ==========================================================================
class Miss:

    missing_log = defaultdict(set)
    """
    A class to generate missing data in a dataset based on the Missing Completely At Random (MCAR) mechanism for multiple features simultaneously.

    Args:
        X (pd.DataFrame): The dataset to receive the missing data.
        y (np.array): The label values from dataset
        missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
        missTarget (bool, optional): A flag to generate missing into the target.
        seed (int, optional): The seed for the random number generator.

    Example Usage:
    ```python
    # Create an instance of the MCAR class
    generator = MCAR(X, y, missing_rate=20)

    # Generate missing values using the random strategy
    data_md = generator.random()
    ```
    """
    

    def MCAR(self, X: pd.DataFrame, missing_rate: int = 10, missTarget: bool = False, seed: int = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas DataFrame')

        if missing_rate >= 100:
            raise ValueError('Missing Rate is too high, the whole dataset will be deleted!')
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        self.X = X
        self.dataset = self.X.copy()
        self.missing_rate = missing_rate
        self.mcar_missing_indices = []

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        """
        Function to randomly generate missing data in the entire dataset.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MCAR mechanism.
        """
        original_shape = self.dataset.shape
        mr = self.missing_rate / 100
        n = self.dataset.shape[0]
        p = self.dataset.shape[1]
        N = round(n * p * mr)

        # Convert the dataset values to float to allow NaN
        array_values = self.dataset.values.astype(float)

        # Select random positions for missingness
        pos_miss = np.random.choice(
            self.dataset.shape[0] * self.dataset.shape[1], N, replace=False
        )

        # Track missing indices
        self.mcar_missing_indices = [np.unravel_index(idx, original_shape) for idx in pos_miss]
        self.mcar_missing_indices = sorted(self.mcar_missing_indices, key=lambda x: (x[0], x[1]))
        for row, col in self.mcar_missing_indices:
            self.missing_log[(row, col)].add("MCAR")
        missing_log_formatted = {f"{row},{col}" : ",".join(types) for (row, col), types in self.missing_log.items()}
        

        # Flatten the array, insert NaN, and reshape
        array_values = array_values.flatten()
        array_values[pos_miss] = np.nan
        array_values = array_values.reshape(original_shape)

        return pd.DataFrame(array_values, columns=self.dataset.columns), missing_log_formatted





    def MAR(self,data, dependencies=None, missing_rate=15, random_seed=42):
        """
        Generate MAR missingness in the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset, which may already have missing values.
        missing_rate : float
            Percentage of new missing values to introduce, calculated on non-missing data.
        dependencies : dict, optional
            Dependencies for MAR missingness. Keys are target columns, values are:
            {"influencers": list of influencing columns, "condition": callable condition}.
            Example:
            {
                "B": {"influencers": ["A"], "condition": lambda row: row["A"] > 80.0},
                "C": {"influencers": ["A", "B"], "condition": lambda row: row["A"] + row["B"] > 90.0}
            }
        random_seed : int, optional
            Seed for reproducibility.

        Returns:
        --------
        data_with_missing : pd.DataFrame
            Dataset with MAR missingness introduced.
        new_missing_indices : list
            List of tuples (row, column) representing new missing cells.
        """
       
        np.random.seed(random_seed)

        # Copy the data to avoid modifying the original dataset
        data_mar = data.copy()

        # Find indices of already missing values
        existing_missing = data_mar.isna()

        # Initialize list to track new missing indices
        mar_missing_indices = []
        #influencing_cells = set()

        #calculating total number of cells
        total_cells = data_mar.size

        total_non_missing = total_cells - existing_missing.sum().sum()
        n_missing = int(total_non_missing * (missing_rate / 100))
        

        # If no dependencies are provided, apply default behavior
        if dependencies is None:
            print("no condition given")
            dependencies = {}
            for col in data_mar.columns:
                influencers = [c for c in data_mar.columns if c != col]
                chosen_influencer = random.choice(influencers)  # Randomly select an influencer
                dependencies[col] = {
                    "influencers": [chosen_influencer],
                    "condition": lambda row, chosen_col=chosen_influencer: row[chosen_col] > data_mar[chosen_col].mean()
                } 

               

            
        eligible_indices = []
        influencing_indices=[]        # Iterate over dependencies to introduce missingness
        influence_map = {col: [] for col in data_mar.columns} 
        for target, dependency in dependencies.items():
            #print("target",target)
            influencers = dependency.get("influencers", [])
            #print("influencer",influencers)
            chosen_influencer = np.random.choice(influencers) #in case no condition is given
            #print("if no condition given",chosen_influencer)
           
            condition = dependency.get("condition", lambda row, chosen_col=chosen_influencer: row[chosen_col] > data_mar[chosen_col].mean())
            #print(influencers)
            #print(condition)
            #print(target)
            condition_result = data_mar.apply(condition, axis=1)
            #print(f"Condition Result:\n{condition_result}")
            # Identify rows eligible for missingness based on the condition
            eligible_rows = data_mar.loc[
                data_mar.apply(condition, axis=1) & ~existing_missing[target]
            ]
            #print(eligible_rows)

            for i in influencers:
                influence_map[i].append(target)
            # If no rows are eligible, skip this column
            if eligible_rows.empty:
                continue
            

            eligible_indices.extend([(row, data_mar.columns.get_loc(target)) for row in eligible_rows.index])
           
            for row in eligible_rows.index:
                influencing_indices.extend([(row, data_mar.columns.get_loc(inf)) for inf in influencers])
        #eligible_indices =  sorted(eligible_indices, key=lambda x: (x[0], x[1]))
        #influencing_indices = sorted(influencing_indices, key=lambda x: (x[0], x[1]))           
        #print("eligible_indices",eligible_indices)
        #print("influencing_indices",influencing_indices)
        #print("influence_map",influence_map)

        while eligible_indices and len(mar_missing_indices) < n_missing:
        
            target_index = random.choice(eligible_indices) #randomly selected index
            print("target index",target_index)
            # Check if the selected index is an influencing cell for any other index
            if target_index in influencing_indices:
                # Find dependent indices for this influencing cell
                print("Influencing cell")

                row, col = target_index  # Unpack the target index
                influencing_col = data_mar.columns[col]  # Find the column name of the influencing cell

                # Get the columns influenced by this column using the influence map
                influenced_columns = influence_map.get(influencing_col, [])

                # Convert influenced column names to their corresponding column indices
                influenced_col_indices = [data_mar.columns.get_loc(col_name) for col_name in influenced_columns]

                # Combine the row with the influenced column indices to form dependent cell indices
                dependent_cells = [(row, col_idx) for col_idx in influenced_col_indices]

                print("all dependednt indices",dependent_cells)


                # Check if at least one dependent cell is None
                # Check if at least one dependent cell is null
                any_dependent_null = any(pd.isna(data_mar.iloc[i, j]) for i, j in dependent_cells)

                # Debugging: Print the dependent cells and their values
                print("Dependent Indices:", dependent_cells)
                for i, j in dependent_cells:
                    value = data_mar.iloc[i, j]
                    print(f"Cell ({i}, {j}):", value, "| Null:", pd.isna(value))

                print("Any Dependent Null:", any_dependent_null)




                if any_dependent_null==True:
                    # Step 8: Remove only the influencing cell from eligible_indices
                    eligible_indices.remove(target_index)
                    for dep_idx in dependent_cells:
                        if dep_idx in eligible_indices:
                            eligible_indices.remove(dep_idx)
                else:
                    # Step 7: Make the influencing cell null and remove both dependent and influencing cells
                    data_mar.iloc[target_index[0], target_index[1]] = None
                    mar_missing_indices.append(target_index)
                    eligible_indices.remove(target_index)
                    for dep_idx in dependent_cells:
                        print("dep_idx removing",dep_idx)
                        if dep_idx in eligible_indices:
                            eligible_indices.remove(dep_idx)
                    
            else:
                # Make the target index null and remove it from eligible_indices
                data_mar.iloc[target_index[0], target_index[1]] = None
                mar_missing_indices.append(target_index)
                eligible_indices.remove(target_index)
            print("after iteration Eligible indices",eligible_indices)


        return data_mar, mar_missing_indices

            
            