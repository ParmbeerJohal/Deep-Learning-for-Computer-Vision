# features.py ---
#
# Filename: features.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 21:06:57 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import numpy as np
from skimage.color import rgb2hsv
from skimage.feature import hog



def extract_h_histogram(data):
    """Extract Hue Histograms from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    h_hist : ndarray (float)
        Hue histgram per image, extracted and reshaped to be NxD, where D is
        the number of bins of each histogram.

    """
    #shape h_hist
    h_hist = np.ndarray(shape=(data.shape[0],16))
    
    #create bins
    bins = np.linspace(0,1,17)

    #loop through each data value and get the histgram of the hue value
    for x in range(len(data)):
        h_hist[x] = np.histogram(rgb2hsv(data[x])[:,:,0], bins)[0]

    # Assertion to help you check if implementation is correct
    assert h_hist.shape == (data.shape[0], 16)

    return h_hist


def extract_hog(data):
    """Extract HOG from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    hog_feat : ndarray (float)
        HOG features per image, extracted and reshaped to be NxD, where D is
        the dimension of HOG features.

    """

    # Using HOG
    print("Extracting HOG...")
    
    # TODO: Implement the method
    hog_feat = np.asarray([hog(x.mean(axis=-1)) for x in data])
    hog_feat = hog_feat.astype(float).reshape(len(data), -1)
    
    # Assertion to help you check if implementation is correct
    assert hog_feat.shape == (data.shape[0], 324)
    
    return hog_feat


#
# features.py ends here
