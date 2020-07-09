"""Core functions."""

import os

from imageio import mimwrite
import nibabel as nb
import numpy as np
from matplotlib.cm import get_cmap

import dashQC_fmri.viz.config as cfg


def parse_filename(filepath):
    """Parse input file path into directory, basename and extension.

    Parameters
    ----------
    filepath: string
        Input name that will be parsed into directory, basename and extension.

    Returns
    -------
    dirname: str
        File directory.
    basename: str
        File name without directory and extension.
    ext: str
        File extension.

    """
    path = os.path.normpath(filepath)
    dirname = os.path.dirname(path)
    filename = path.split(os.sep)[-1]
    basename, ext = filename.split(os.extsep, 1)
    return dirname, basename, ext


def load_and_prepare_image(in_file):
    """Load and prepare image data.

    Parameters
    ----------
    in_file: str
        Input file (eg. /john/home/image.nii.gz)

    Returns
    -------
    out_img: numpy array
    """
    # Load NIfTI file
    data = nb.load(in_file).get_fdata(dtype=np.float32)

    # Pad data array with zeros to make the shape isometric
    dShape = data.shape
    maximum = np.max(dShape[0:3])

    a, b, c = dShape[0:3]
    x, y, z = (list(dShape[0:3]) - maximum) / -2

    if len(dShape) > 3:
        out_img = np.zeros([maximum, maximum, maximum, dShape[3]],
                           dtype=np.float32)
        out_img[int(x):a + int(x),
                int(y):b + int(y),
                int(z):c + int(z), :] = data

    else:
        out_img = np.zeros([maximum] * 3, dtype=np.float32)
        out_img[int(x):a + int(x),
                int(y):b + int(y),
                int(z):c + int(z)] = data

    out_img /= out_img.max()  # scale image values between 0-1
    out_img *= 255
    out_img = out_img.astype(np.uint8)

    return out_img, maximum, dShape


def create_mosaic(out_img, maximum, origShape, slices=cfg.slices,
                  slicesOrder=cfg.slicesOrder):
    """Create grayscale image.

    Parameters
    ----------
    out_img: numpy array
    maximum: int

    Returns
    -------
    new_img: numpy array

    """
    if len(origShape) > 3:
        if slices == cfg.slices:
            slices = [int(origShape[0]/2),
                      int(origShape[1]/2),
                      int(origShape[2]*2/3)]

        if slicesOrder == 'csa':
            new_img = np.array(
                [np.hstack((
                    np.hstack((
                        np.flip(out_img[slices[0], :, :, i], 1).T,
                        np.flip(out_img[:, slices[1], :, maximum - i - 1], 1).T)),
                    np.flip(out_img[:, :, slices[2], maximum - i - 1], 1).T))
                    for i in range(maximum)])
        elif slicesOrder == 'cas':
            new_img = np.array(
                [np.hstack((
                    np.hstack((
                        np.flip(out_img[slices[0], :, :, i], 1).T,
                        np.flip(out_img[:, :, slices[2], maximum - i - 1], 1).T)),
                    np.flip(out_img[:, slices[1], :, maximum - i - 1], 1).T))
                    for i in range(maximum)])
    else:
        new_img = np.array(
            [np.hstack((
                np.hstack((
                    np.flip(out_img[i, :, :], 1).T,
                    np.flip(out_img[:, maximum - i - 1, :], 1).T)),
                np.flip(out_img[:, :, maximum - i - 1], 1).T))
                for i in range(maximum)])

    return new_img


def write_gif_normal(in_file, out_filename, mode, fps, colormap, slices,
                     slicesOrder):
    """Procedure for writing grayscale image.

    Parameters
    ----------
    in_file: str
        Input file (eg. /john/home/image.nii.gz)
    out_filename
    fps: int
        Frames per second

    """
    # Load NIfTI and put it in right shape
    out_img, maximum, origShape = load_and_prepare_image(in_file)

    # Create output mosaic
    new_img = create_mosaic(out_img, maximum, origShape, slices=slices,
                            slicesOrder=slicesOrder)
    del out_img

    # Figure out extension
    ext = '.{}'.format(parse_filename(in_file)[2])

    if mode == 'pseudocolor':
        # Transform values according to the color map
        cmap = get_cmap(colormap)
        color_transformed = [cmap(new_img[i, ...]) for i in range(maximum)]
        new_img = np.delete(color_transformed, 3, 3)

        new_img /= new_img.max()  # scale image values between 0-1
        new_img *= 255
        new_img = new_img.astype(np.uint8)

    if not out_filename:
        out_filename = in_file.replace(ext, '.gif')

    mimwrite(out_filename, new_img, format='gif', fps=fps)
