import numpy as np
from collections import defaultdict
import jax.numpy
import neural_tangents as nt
from neural_tangents import stax
from jax import random
import math

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import *
from sklearn.linear_model import *


def MI_NNGP1(input, number_of_imputation=1, W_std=1.0, b_std=0.0):
    # W_std is standard deviation of weight parameters
    # b_std is standard deviation of bias parameters
    n, p = input.shape
    mask = np.isnan(input)
    pattern = defaultdict(list)
    for i in range(n):
        pattern[tuple(mask[i])].append(i)
    
    if tuple([False]*p) in pattern:
        complete_cases_indicator = True
        complete_cases = pattern[tuple([False]*p)]
    else:
        complete_cases_indicator = False
    try:
        assert complete_cases_indicator == True
    except:
        print('no complete cases found, please use MI-NNGP2')
        return 
    
    W_std = W_std
    b_std = b_std
    init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(300, W_std=math.sqrt(W_std), b_std=b_std, parameterization="standard"), stax.Relu(),  #stax.Relu()  stax.Erf()
    stax.Dense(300, W_std=math.sqrt(W_std), b_std=b_std, parameterization="standard"), stax.Relu(),
    stax.Dense(1, W_std=math.sqrt(W_std), b_std=b_std, parameterization="standard")
    )

    imputation_list = []
    for _ in range(number_of_imputation):
        key = random.PRNGKey(i*71)
        imputation = input.copy()
        for mask in list(pattern.keys()):
            if list(mask) != [False]*p:
                incomplete_cases = pattern[mask]
                mask = np.array(list(mask))
                train_input = jax.numpy.array(np.transpose(input[complete_cases][:,mask==False]))
                test_input = jax.numpy.array(np.transpose(input[complete_cases][:,mask==True]))
                train_target = jax.numpy.array(np.transpose(input[incomplete_cases][:,mask==False]))

                predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_input, train_target)
                nngp_mean, nngp_covariance = predict_fn(x_test=test_input, get='nngp',compute_cov=True)
                
                intermidate = imputation[incomplete_cases]
                if number_of_imputation==1:
                    # for single imputation, use mean value as imputation
                    intermidate[:,mask==True] = jax.numpy.transpose(nngp_mean)
                else:
                    # for multiple imputation, draw imputation from posterior distribution
                    sampling = np.zeros(nngp_mean.shape)
                    for j in range(nngp_mean.shape[1]):
                        sampling[:,j] = jax.random.multivariate_normal(key, nngp_mean[:,j], nngp_covariance)
                    intermidate[:,mask==True] = jax.numpy.transpose(sampling)
                imputation[incomplete_cases] = intermidate

        imputation_list.append(imputation)
    if number_of_imputation==1:
        return imputation_list[0]
    else:
        return imputation_list


def MI_NNGP2(input, number_of_imputation=1, burn_in=2, interval=1, W_std=1.0, b_std=0.0):
    # W_std is standard deviation of weight parameters
    # b_std is standard deviation of bias parameters
    # burn_in is burn in period
    # interval is sampling interval
    n, p = input.shape
    mask = np.isnan(input)
    pattern = defaultdict(list)
    for i in range(n):
        pattern[tuple(mask[i])].append(i)
    
    if tuple([False]*p) in pattern:
        initial_imputation = MI_NNGP1(input)
    else:
        MICE_imputer=IterativeImputer(estimator=BayesianRidge(),skip_complete=True,max_iter=20, tol=0.01,sample_posterior=False,random_state=42)
        initial_imputation=MICE_imputer.fit_transform(input)
    print('finish initial imputation!')
    
    W_std = W_std
    b_std = b_std
    init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(300, W_std=math.sqrt(W_std), b_std=b_std, parameterization="standard"), stax.Relu(),  #stax.Relu()  stax.Erf()
    stax.Dense(300, W_std=math.sqrt(W_std), b_std=b_std, parameterization="standard"), stax.Relu(),
    stax.Dense(1, W_std=math.sqrt(W_std), b_std=b_std, parameterization="standard")
    )

    imputation = initial_imputation.copy()
    imputation_list = []
    for i in range(burn_in+number_of_imputation*interval):
        key = random.PRNGKey(i*71)
        for mask in list(pattern.keys()):
            if list(mask) != [False]*p:
                incomplete_cases = pattern[mask]
                complement_cases = [i for i in list(range(n)) if i not in incomplete_cases]
                mask = np.array(list(mask))
                train_input = jax.numpy.array(np.transpose(imputation[complement_cases][:,mask==False]))
                test_input = jax.numpy.array(np.transpose(imputation[complement_cases][:,mask==True]))
                train_target = jax.numpy.array(np.transpose(imputation[incomplete_cases][:,mask==False]))

                predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_input, train_target)
                nngp_mean, nngp_covariance = predict_fn(x_test=test_input, get='nngp',compute_cov=True)

                intermidate = imputation[incomplete_cases]
                if number_of_imputation==1:
                    # for single imputation, use mean value as imputation
                    intermidate[:,mask==True] = jax.numpy.transpose(nngp_mean)
                else:
                    # for multiple imputation, draw imputation from posterior distribution
                    sampling = np.zeros(nngp_mean.shape)
                    for j in range(nngp_mean.shape[1]):
                        sampling[:,j] = jax.random.multivariate_normal(key, nngp_mean[:,j], nngp_covariance)
                    intermidate[:,mask==True] = jax.numpy.transpose(sampling)
                imputation[incomplete_cases] = intermidate   

        if i>=burn_in and (i+1-burn_in)%interval==0:
            imputation_list.append(imputation.copy()) 
        print('finish epoch {}!'.format(i))

    if number_of_imputation==1:
        return imputation_list[0]
    else:
        return imputation_list