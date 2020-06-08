import PIL.Image as im
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltim
import h5py
import scipy.ndimage



def read_image(image_path, *args, **kwargs):
    """Read an image into a numpy array given the path of the file.

    Parameters
    ----------
    image_path : string
        Path to the image file.

    Returns
    -------
    image : ndarray
        HWC formatted image.
    """
    img = im.open(image_path)
	im.load()

    return np.asarray(img, dtype="int32")


def invert_image(image, *args, **kwargs):
    """Invert the image color by subtracting the image from a white image.

    Parameters
    ----------
    image : ndarray
        Original image.

    Returns
    -------
    inv_image : numpy.ndarray
        Inverted image.
    """
    inv_image = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype="int32") * 255 - image

    return inv_image


def save_image_to_h5(image, h5_path, *args, **kwargs):
    """Save an image to H5 file.

    Parameters
    ----------
    image : ndarray
        The image to be saved.

    h5_path: string
        Path to the H5 file.
    """
    save_image = h5py.File(h5_path, 'w')
	save_image.create_dataset('image', data = image)
	save_image.close()

    #raise NotImplementedError


def read_image_from_h5(h5_path, *args, **kwargs):
    """Read an image into a numpy array given the path of a H5 file.

    Parameters
    ----------
    h5_path : string
        Path to the H5 file.

    Returns
    -------
    image : ndarray
        HWC formatted image.
    """
    read_h5_file = h5py.File(h5_path, 'r')
	
	image = np.array(read_h5_file.get('image'))

    return image


def gray_scale_image(image, *args, **kwargs):
    """Return the gray_scale image by taking the mean over channels.

    Parameters
    ----------
    image : ndarray
        Original image.

    Returns
    -------
    gray_scale : ndarray
        HW formmated gray scale image.
    """
    # TODO: Implement the method

    return gray_scale


def find_difference_of_gaussian_blur(image, k1, k2, *args, **kwargs):
    """Find the difference of two Gaussian blurs from an image.

    Parameters
    ----------
    image : ndarray
        Original image.

    k1 : scalar
        First standard deviation for Gaussian kernel.

    k2 : scalar
        Second standard deviation for Gaussian kernel.

    Returns
    -------
    res : ndarray
        Normalized difference of Gaussian blurs.
    """
    # TODO: Implement the method

    return res


def keep_top_percentile(image, percentile, *args, **kwargs):
    """Find the difference of two Gaussian blurs from an image.

    Parameters
    ----------
    image : ndarray
        Original image.

    percentile : scalar
        Top percentile pixels will be kept.

    Returns
    -------
    thresholded : ndarray
        Image with the high value pixles.
    """
    # TODO: Implement the method

    return thresholded
