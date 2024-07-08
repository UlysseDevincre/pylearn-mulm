# -*- coding: utf-8 -*-
##########################################################################
# Created on Created on Thu Feb  6 15:15:14 2020
# Copyright (c) 2013-2021, CEA/DRF/Joliot/NeuroSpin. All rights reserved.
# @author:  Edouard Duchesnay
# @email:   edouard.duchesnay@cea.fr
# @license: BSD 3-clause.
##########################################################################

"""
Module that contains utility functions.
"""

import numpy as np
import scipy.stats
from scipy.ndimage import label

def ttest_pval(df, tstat, two_tailed=True):
    """Calculate p-values.


    Parameters
    ----------
    df : float
        Degrees of freedom.
    tstat : array (p, ), optional
        T-statistics. The default is None.
    two_tailed : bool, optional
        two-tailed (two-sided) test. The default is True.

    Returns
    -------
    pval : array (p, )
        P-values.

    Example
    -------
    >>> import numpy as np
    >>> from mulm import ttest_pval
    >>> x = [.1, .2, .3, -.1, .1, .2, .3]
    >>> pval = ttest_pval(df=len(x)-1, tstat=2.9755097944025275)
    >>> np.allclose(pval, 0.02478)
    True
    """
    if two_tailed:
        pval = 2 * scipy.stats.t.sf(np.abs(tstat), df=df)
    else:
        pval = scipy.stats.t.sf(tstat, df=df)

    return pval


def ttest_ci(df, estimate=None, tstat=None, se=None, mu=0, alpha=0.05,
             two_tailed=True):
    """Calculate confidence interval given at least two of the values within estimate, tstat, se.

    See confidence intervals: https://en.wikipedia.org/wiki/Confidence_interval

    Parameters
    ----------
    df : float
        Degrees of freedom.
    estimate : array (p, ), optional
        Estimate of the parameter. The default is None.
    tstat : array (p, ), optional
        T-statistics. The default is None.
    se : array (p, ), optional
        Standard error. The default is None.
    mu : float, optional
        Null hypothesis. The default is 0.
    alpha : float, optional
        1 - confidence level. The default is 0.05.
    two_tailed : bool, optional
        two-tailed (two-sided) test. The default is True.

    Returns
    -------
    estimate : array (p, )
        Estimates.
    tstat : array (p, )
        T-statistics.
    se : array (p, )
        Standard errors.
    pval : array (p, )
        P-values.
    ci : (array (p, ) array (p, ))
        Confidence intervals.

    Example
    -------
    >>> import numpy as np
    >>> from mulm import ttest_ci
    >>> x = [.1, .2, .3, -.1, .1, .2, .3]
    >>> estimate, se, tstat, ci = ttest_ci(df=len(x)-1, estimate=np.mean(x),
    ...                                    se=np.std(x, ddof=1)/np.sqrt(len(x)))
    >>> np.allclose((estimate, tstat, ci[0], ci[1]),
    ...             (0.15714286, 2.9755, 0.02791636, 0.28636936))
    True
    """

    # tstat = (estimate - mu) / se
    assert np.sum([s is not None for s in (estimate, tstat, se)]) >= 2,\
        "Provide at least two values within estimate, tstat, se"
    if se is None:
        se = (estimate - mu) / tstat
    elif tstat is None:
        tstat = (estimate - mu) / se
    elif estimate is None:
        estimate = tstat * se  + mu

    if two_tailed:
        cint = scipy.stats.t.ppf(1 - alpha / 2, df)
        ci = estimate - cint * se, estimate + cint * se
    else:
        cint = scipy.stats.t.ppf(1 - alpha, df)
        ci = estimate - cint * se, np.inf

    return estimate, se, tstat, ci





def ttest_tfce(
    arr4d,
    bin_struct,
    E=0.5,
    H=2,
    dh="auto",
    two_sided_test=True,
):
    """Calculate threshold-free cluster enhancement values for scores maps.

    The :term:`TFCE` calculation is mostly implemented as described in [1]_,
    with minor modifications to produce similar results to fslmaths, as well
    as to support two-sided testing.

    Parameters
    ----------
    arr4d : :obj:`numpy.ndarray` of shape (X, Y, Z, R)
        Unthresholded 4D array of 3D t-statistic maps.
        R = regressor.
    bin_struct : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.
    E : :obj:`float`, default=0.5
        Extent weight.
    H : :obj:`float`, default=2
        Height weight.
    dh : 'auto' or :obj:`float`, default='auto'
        Step size for TFCE calculation.
        If set to 'auto', use 100 steps, as is done in fslmaths.
        A good alternative is 0.1 for z and t maps, as in [1]_.
    two_sided_test : :obj:`bool`, default=False
        Whether to assess both positive and negative clusters (True) or just
        positive ones (False).

    Returns
    -------
    tfce_arr : :obj:`numpy.ndarray`, shape=(n_descriptors, n_regressors)
        :term:`TFCE` values.

    Notes
    -----
    In [1]_, each threshold's partial TFCE score is multiplied by dh,
    which makes directly comparing TFCE values across different thresholds
    possible.
    However, in fslmaths, this is not done.
    In the interest of maximizing similarity between nilearn and established
    tools, we chose to follow fslmaths' approach.

    Additionally, we have modified the method to support two-sided testing.
    In fslmaths, only positive clusters are considered.

    References
    ----------
    .. [1] Smith, S. M., & Nichols, T. E. (2009).
       Threshold-free cluster enhancement: addressing problems of smoothing,
       threshold dependence and localisation in cluster inference.
       Neuroimage, 44(1), 83-98.
    """
    tfce_4d = np.zeros_like(arr4d)

    # For each passed t map
    for i_regressor in range(arr4d.shape[3]):
        arr3d = arr4d[..., i_regressor]

        # Get signs / threshs
        if two_sided_test:
            signs = [-1, 1]
            max_score = np.max(np.abs(arr3d))
        else:
            signs = [1]
            max_score = np.max(arr3d)

        step = max_score / 100 if dh == "auto" else dh

        # Set based on determined step size
        score_threshs = np.arange(step, max_score + step, step)

        # If we apply the sign first...
        for sign in signs:
            # Init a temp copy of arr3d with the current sign applied,
            # which can then be re-used by incrementally setting more
            # voxel's to background, by taking advantage that each score_thresh
            # is incrementally larger
            temp_arr3d = arr3d * sign

            # Prep step
            for score_thresh in score_threshs:
                temp_arr3d[temp_arr3d < score_thresh] = 0

                # Label into clusters - importantly (for the next step)
                # this returns clusters labelled ordinally
                # from 1 to n_clusters+1,
                # which allows us to use bincount to count
                # frequencies directly.
                labeled_arr3d, _ = label(temp_arr3d, bin_struct)

                # Next, we want to replace each label with its cluster
                # extent, that is, the size of the cluster it is part of
                # To do this, we will first compute a flattened version of
                # only the non-zero cluster labels.
                labeled_arr3d_flat = labeled_arr3d.flatten()
                non_zero_inds = np.where(labeled_arr3d_flat != 0)[0]
                labeled_non_zero = labeled_arr3d_flat[non_zero_inds]

                # Count the size of each unique cluster, via its label.
                # The reason why we pass only the non-zero labels to bincount
                # is because it includes a bin for zeros, and in our labels
                # zero represents the background,
                # which we want to have a TFCE value of 0.
                cluster_counts = np.bincount(labeled_non_zero)

                # Next, we convert each unique cluster count to its TFCE value.
                # Where each cluster's tfce value is based
                # on both its cluster extent and z-value
                # (via the current score_thresh)
                # NOTE: We do not multiply by dh, based on fslmaths'
                # implementation. This differs from the original paper.
                cluster_tfces = sign * (cluster_counts**E) * (score_thresh**H)

                # Before we can add these values to tfce_4d, we need to
                # map cluster-wise tfce values back to a voxel-wise array,
                # including any zero / background voxels.
                tfce_step_values = np.zeros(labeled_arr3d_flat.shape)
                tfce_step_values[non_zero_inds] = cluster_tfces[
                    labeled_non_zero
                ]

                # Now, we just need to reshape these values back to 3D
                # and they can be incremented to tfce_4d.
                tfce_4d[..., i_regressor] += tfce_step_values.reshape(
                    temp_arr3d.shape
                )

    return tfce_4d
