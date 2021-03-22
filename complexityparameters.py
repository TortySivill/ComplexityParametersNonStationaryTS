import numpy as np 
import scipy as sp
import pandas as pd

from lempel_ziv_complexity import lempel_ziv_complexity
from entropy import *
from pyentrp import entropy as ent

import statsmodels.api as sm

import pyinform as pyinf
import tsfeatures as tf



def binarize(x):
	mean_value = np.mean(x)
	binarized_string = str([1 if value < mean_value else 0 for value in x])
	return(binarized_string)

def central_lempel_ziv(x):
	return lempel_ziv_complexity(binarize(x))

def central_approximate_entropy(x):
	# default order is 2, default metric is chebyshev
	return app_entropy(x)

def central_sample_entropy(x):
	# default order is 2, default metric is chebyshev
	return sample_entropy(x)

def central_permutation_entropy(x):
	# default order is 3, default delay (lag) is 1
	return perm_entropy(x)

def central_MS_entropy(x):
	#Multiscale Entropy
	#sample length paramter, default is 2
	return ent.multiscale_entropy(x,2)

def central_spectral_entropy(x):
	#sf = sampling frequency (samples per second) default is 1 
	return(spectral_entropy(x,1))

#def forbidden_patterns(x):
	#TODO

def central_skewness(x):
	return(sp.stats.skew(x))

def central_kurtosis(x):
	return(sp.stats.kurtosis(x))

def central_SVD(x):
	#default order = 3, default lag = 1
	return(svd_entropy(x))

def normalise(X):
    return [(x - min(X))/(max(X)-min(X)) for x in X]

def complexity_features(X):
	lempel_ziv_feature = [central_lempel_ziv(instance) for instance in X]
	approximate_entropy_feature = [central_approximate_entropy(instance) for instance in X]
	sample_entropy_feature = [central_sample_entropy(instance) for instance in X]
	permutation_entropy_feature = [central_permutation_entropy(instance) for instance in X]
	#MS_entropy_feature = [central_MS_entropy(instance) for instance in X]
	spectral_entropy_feature = [central_spectral_entropy(instance) for instance in X]
	skewness_feature = [central_skewness(instance) for instance in X]
	kurtosis_feature = [central_kurtosis(instance) for instance in X]
	SVD_entropy_feature = [central_SVD(instance) for instance in X]
	
	complexity_feature_df = pd.DataFrame()
	complexity_feature_df['lempel_ziv'] = lempel_ziv_feature
	complexity_feature_df['approx'] = approximate_entropy_feature
	complexity_feature_df['sample'] = sample_entropy_feature
	complexity_feature_df['permutation'] = permutation_entropy_feature
	#complexity_feature_df['MS'] = MS_entropy_feature
	complexity_feature_df['spectral'] = spectral_entropy_feature
	complexity_feature_df['skewness'] = skewness_feature
	complexity_feature_df['kurtosis'] = kurtosis_feature
	complexity_feature_df['SVD'] = SVD_entropy_feature


	return complexity_feature_df 

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def normalised_complexity_features(X):
	complexity_feature_df = complexity_features(X)
	normalised_df = pd.DataFrame()

	for column in complexity_feature_df:
		if is_unique(complexity_feature_df[column]):
			continue
		else:
			normalised_df[column] = normalise(complexity_feature_df[column])

	return normalised_df

def central_acf_10(x):
	#Time series must be of length longer than 10
	return np.sum([x.autocorr(lag=value)**2 for value in range(1,11)])

def central_pacf_5(x):
	return np.sum([sm.tsa.stattools.pacf(x,nlags=6)])

def central_hurst(x):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(1, 10)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def summary_features(X):
	summary_features_df = pd.DataFrame()
	summary_features_df['length'] = [len(instance) for instance in X]
	summary_features_df['ACF_1'] = [pd.Series(instance).autocorr(lag=1) for instance in X]
	summary_features_df['ACF_10'] = [central_acf_10(pd.Series(instance)) for instance in X]
	summary_features_df['differenced_ACF_1'] = [pd.Series(instance).diff().autocorr(lag=1) for instance in X]
	summary_features_df['differenced_ACF_10'] = [central_acf_10(pd.Series(instance).diff()) for instance in X]
	summary_features_df['twice_differenced_ACF_1'] = [pd.Series(instance).diff(periods=2).autocorr(lag=1) for instance in X]
	summary_features_df['twice_differenced_ACF_10'] = [central_acf_10(pd.Series(instance).diff(periods=2)) for instance in X]
	summary_features_df['PACF_5'] = [tf.pacf_features(instance)['x_pacf5'] for instance in X]
	summary_features_df['differenced_PACF_5'] = [tf.pacf_features(instance)['diff1x_pacf5'] for instance in X]
	summary_features_df['twice_differenced_PACF_5'] = [tf.pacf_features(instance)['diff2x_pacf5'] for instance in X]
	summary_features_df['nonlinearity'] = [tf.nonlinearity(instance)['nonlinearity'] for instance in X]
	summary_features_df['hurst'] = [tf.hurst(instance)['hurst'] for instance in X]
	summary_features_df['stability'] = [tf.stability(instance)['stability'] for instance in X]
	summary_features_df['lumpiness'] = [tf.lumpiness(instance)['lumpiness']for instance in X]
	summary_features_df['unit_root_kpss'] = [tf.unitroot_kpss(instance)['unitroot_kpss'] for instance in X]
	summary_features_df['unit_root_pp'] = [tf.unitroot_pp(instance)['unitroot_pp']for instance in X]
	#summary_features_df['nperiods'] = [tf.stl_features(instance)['nperiods'] for instance in X]
	#summary_features_df['seasonal_period'] = [tf.stl_features(instance)['seasonal_period'] for instance in X]
	summary_features_df['trend'] = [tf.stl_features(instance)['trend'] for instance in X]
	summary_features_df['spike'] = [tf.stl_features(instance)['spike'] for instance in X]
	summary_features_df['curvature'] = [tf.stl_features(instance)['curvature'] for instance in X]
	summary_features_df['eacf_1'] = [tf.stl_features(instance)['e_acf1'] for instance in X]
	summary_features_df['eacf_10'] = [tf.stl_features(instance)['e_acf10'] for instance in X]
	#summary_features_df['max_level_shift'] = [tf.max_level_shift(instance)['max_level_shift'] for instance in X]
	#summary_features_df['time_level_shift'] = [tf.max_level_shift(instance)['max_time_shift'] for instance in X]
	#summary_features_df['max_var_shift'] = [tf.max_var_shift(instance)['max_var_shift'] for instance in X]
	#summary_features_df['time_var_shift'] = [tf.max_var_shift(instance)['time_var_shift'] for instance in X]
	#summary_features_df['max_kl_shift'] = [tf.max_kl_shift(instance)['max_kl_shift'] for instance in X]
	#summary_features_df['time_kl_shift'] = [tf.max_kl_shift(instance)['time_kl_shift'] for instance in X]





	return summary_features_df

def normalised_summary_features(X):
	summary_feature_df = summary_features(X)
	normalised_df = pd.DataFrame()

	for column in summary_feature_df:
		if is_unique(summary_feature_df[column]):
			continue
		else:
			normalised_df[column] = normalise(summary_feature_df[column])

	return normalised_df

def all_features(X):
	complexity_df = complexity_features(X)
	summary_df = summary_features(X)
	all_features = pd.concat([complexity_df,summary_df],axis=1)
	return all_features

def normalised_all_features(X):
	complexity_df = normalised_complexity_features(X)
	summary_df = normalised_summary_features(X)
	all_features = pd.concat([complexity_df,summary_df],axis=1)
	return all_features
