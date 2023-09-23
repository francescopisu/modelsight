"""
This file deals with the implementation of the DeLong test for the comparison of
pairs of correlated areas under the receiver-operating characteristics curves.
"""

import pandas as pd
import numpy as np
import scipy.stats
from typing import Tuple

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x: np.ndarray) -> np.ndarray:
   """
   Computes midranks.
    
   Parameters
   ----------
   x : np.ndarray
     a 1-d array of predicted probabilities.
    
   Returns
   -------
   T2 : np.ndarray
      array of midranks
   """
   J = np.argsort(x)
   Z = x[J]
   N = len(x)
   T = np.zeros(N, dtype=np.float64)
   i = 0
   while i < N:
      j = i
      while j < N and Z[j] == Z[i]:
         j += 1
      T[i:j] = 0.5*(i + j - 1)
      i = j
   T2 = np.empty(N, dtype=np.float64)
   # Note(kazeevn) +1 is due to Python using 0-based indexing
   # instead of 1-based in the AUC formula in the paper
   T2[J] = T + 1
   return T2


def fastDeLong(predictions_sorted_transposed: np.ndarray, 
               label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
   """
   The fast version of DeLong's method for computing the covariance of
   unadjusted AUC.
   
   Parameters
   ----------
   predictions_sorted_transposed : a (n_classifiers, n_obs) numpy array containing
      the predicted probabilities by the two classifiers in the comparison. 
      These probabilities are sorted such that the examples with label "1" come first.
   
   Returns
   -------
   aucs, delongcov : Tuple[np.ndarray, np.ndarray]
      aucs: array of AUC values 
      delongcov: array of DeLong covariance
      
   Reference
   ---------
   @article{sun2014fast,
      title={Fast Implementation of DeLong's Algorithm for
            Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
      author={Xu Sun and Weichao Xu},
      journal={IEEE Signal Processing Letters},
      volume={21},
      number={11},
      pages={1389--1393},
      year={2014},
      publisher={IEEE}
   }
   """
   # Short variables are named as they are in the paper
   m = label_1_count
   n = predictions_sorted_transposed.shape[1] - m
   positive_examples = predictions_sorted_transposed[:, :m]
   negative_examples = predictions_sorted_transposed[:, m:]
   k = predictions_sorted_transposed.shape[0]

   tx = np.empty([k, m], dtype=np.float64)
   ty = np.empty([k, n], dtype=np.float64)
   tz = np.empty([k, m + n], dtype=np.float64)
   for r in range(k):
      tx[r, :] = compute_midrank(positive_examples[r, :])
      ty[r, :] = compute_midrank(negative_examples[r, :])
      tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
   aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
   v01 = (tz[:, :m] - tx[:, :]) / n
   v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
   sx = np.cov(v01)
   sy = np.cov(v10)
   delongcov = sx / m + sy / n
   return aucs, delongcov


def calc_pvalue(aucs: np.ndarray, sigma: np.ndarray) -> float:
   """
   Computes log(10) of p-values.
   
   Parameters
   ----------
   aucs : np.array
      a 1-d array of AUCs
   sigma : np.array
      an array AUC DeLong covariances
   
   Returns
   -------
   p : float
      log10(pvalue)
   """
   l = np.array([[1, -1]])
   z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
   p = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
   return p


def compute_ground_truth_statistics(ground_truth: np.ndarray) -> Tuple[np.ndarray, int]:
   """
   Compute statistics of ground-truth array.

   Parameters
   ----------
   ground_truth : np.ndarray
      a (n_obs,) array of 0 and 1 values representing the ground-truth.

   Returns
   -------
   order, label_1_count : Tuple[np.ndarray, int]
       order is a numpy array of sorted indexes
       label_1_count is the count of data points of the positive class.
   """
   assert np.array_equal(np.unique(ground_truth), [0, 1])
   order = (-ground_truth).argsort()
   label_1_count = int(ground_truth.sum())
   return order, label_1_count


def delong_roc_test(ground_truth: np.ndarray, 
                    predictions_one: np.ndarray, 
                    predictions_two: np.ndarray) -> float:
   """
   Compare areas-under-curve of two estimators using the DeLong test.
   Concretely, it computes the pvalue for hypothesis that two ROC AUCs are different.
   
   Parameters
   ----------
   ground_truth : np.ndarray
      a (n_obs,) array of 0 and 1 representing ground-truths.
   predictions_one : np.ndarray
      a (n_obs,) array of probabilities of class 1 predicted by the first model.
   predictions_two : np.ndarray
      a (n_obs,) array of probabilities of class 1 predicted by the second model.
      
   Returns
   -------
   p : float
      the p-value for hypothesis that two ROC AUCs are different.
   """
   order, label_1_count = compute_ground_truth_statistics(ground_truth)
   predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
   aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
   
   p = 10**calc_pvalue(aucs, delongcov).item()
   return p