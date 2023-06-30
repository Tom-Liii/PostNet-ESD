#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#os.chdir('..')
print("Current Working Directory " , os.getcwd())
import sys
sys.path.append("../") # Add directory containing src/data to path


# In[2]:


import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from torch import nn
from matplotlib import pyplot as plt


# In[3]:


from src.posterior_networks.run import run


# ## Parameters

# In[4]:


# Dataset parameters
seed_dataset=123
directory_dataset='../data'
dataset_name='segment_scale_sky_missing'
ood_dataset_names=['segment_scale_sky_only']
unscaled_ood=False
split=[.6, .8]
transform_min=0.
transform_max=1.

# Architecture parameters
seed_model=123
directory_model='../saved_models'
architecture='linear'
input_dims=[18]
output_dim=6
hidden_dims=[64, 64, 64]
kernel_dim=None
latent_dim=6
no_density=False
density_type='normal_mixture'
n_density=6
k_lipschitz=None
budget_function='id'

# Training parameters
directory_results='../saved_results'
max_epochs=200
patience=10
frequency=2
batch_size=64
lr=1e-3
loss='UCE'
training_mode='joint'
regr=1e-5


# ## Training

# In[5]:


results_metrics = run(# Dataset parameters
                        seed_dataset,  # Seed to shuffle dataset. int
                        directory_dataset,  # Path to dataset. string
                        dataset_name,  # Dataset name. string
                        ood_dataset_names,  # OOD dataset names.  list of strings
                        unscaled_ood,  # If true consider also unscaled versions of ood datasets. boolean
                        split,  # Split for train/val/test sets. list of floats
                        transform_min,  # Minimum value for rescaling input data. float
                        transform_max,  # Maximum value for rescaling input data. float

                        # Architecture parameters
                        seed_model,  # Seed to init model. int
                        directory_model,  # Path to save model. string
                        architecture,  # Encoder architecture name. string
                        input_dims,  # Input dimension. List of ints
                        output_dim,  # Output dimension. int
                        hidden_dims,  # Hidden dimensions. list of ints
                        kernel_dim,  # Input dimension. int
                        latent_dim,  # Latent dimension. int
                        no_density,  # Use density estimation or not. boolean
                        density_type,  # Density type. string
                        n_density,  # Number of density components. int
                        k_lipschitz,  # Lipschitz constant. float or None (if no lipschitz)
                        budget_function,  # Budget function name applied on class count. name

                        # Training parameters
                        directory_results,  # Path to save resutls. string
                        max_epochs,  # Maximum number of epochs for training
                        patience,  # Patience for early stopping. int
                        frequency,  # Frequency for early stopping test. int
                        batch_size,  # Batch size. int
                        lr,  # Learning rate. float
                        loss,  # Loss name. string
                        training_mode,  # 'joint' or 'sequential' training. string
                        regr)


# In[6]:


print(results_metrics['model_path'])
print(results_metrics['result_path'])


# ## Evaluation

# In[7]:


no_show = ['model_path', 'result_path', 'train_losses', 'val_losses', 'test_losses', 'train_accuracies', 'val_accuracies', 'test_accuracies', 'fail_trace']
for k, v in results_metrics.items():
    if k not in no_show:
        print(k, v)


# In[ ]:




