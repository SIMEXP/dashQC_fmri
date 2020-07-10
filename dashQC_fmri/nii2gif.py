"""Main entry point."""

import os

from imageio import mimwrite
from matplotlib.cm import get_cmap
import nibabel as nb
import numpy as np
import warnings  # mainly for ignoring imageio warnings
warnings.filterwarnings("ignore")

DEFAULT_MODES = ['normal', 'pseudocolor']


def parse_filename(in_filepath):
    """
    Parse input file path into directory, basename and extension.

    Parameters
    ----------
    in_filepath: str
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
    path = os.path.normpath(in_filepath)
    dirname = os.path.dirname(path)
    filename = path.split(os.sep)[-1]
    basename, ext = filename.split(os.extsep, 1)
    return dirname, basename, ext


def load_and_prepare_image(in_file):
    """
    Load and prepare image data.

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
    maxAxis = np.max(dShape[0:3])

    a, b, c = dShape[0:3]
    x, y, z = (list(dShape[0:3]) - maxAxis) / -2

    if len(dShape) > 3:
        out_img = np.zeros([maxAxis, maxAxis, maxAxis, dShape[3]],
                           dtype=np.float32)
        out_img[int(x):a + int(x),
                int(y):b + int(y),
                int(z):c + int(z), :] = data

    else:
        out_img = np.zeros([maxAxis] * 3, dtype=np.float32)
        out_img[int(x):a + int(x),
                int(y):b + int(y),
                int(z):c + int(z)] = data

    out_img /= out_img.max()  # scale image values between 0-1
    out_img *= 255
    out_img = out_img.astype(np.uint8)

    return out_img, dShape


def create_mosaic(in_img, origShape, slices=None, slicesOrder='csa'):
    """
    Create grayscale image.

    Parameters
    ----------
    in_img: numpy array
        Can be a 3D image or a 4D image.
    origShape: list
        Shape of the original image.
    slices: list
        Slices for each axis to use (only for 4D volumes)
    slicesOrder: str
        Can be csa or cas (Cortical, Sagital, Axial)

    Returns
    -------
    new_img: numpy array

    """

    maxAxis = np.max(origShape[0:3])

    if len(origShape) > 3:
        if slices:
            slices = [int(origShape[0]/2),
                      int(origShape[1]/2),
                      int(origShape[2]*2/3)]

        if slicesOrder == 'csa':
            new_img = np.array(
                [np.hstack((
                    np.hstack((
                        np.flip(in_img[slices[0], :, :, i], 1).T,
                        np.flip(in_img[:, slices[1], :, maxAxis - i - 1], 1).T)),
                    np.flip(in_img[:, :, slices[2], maxAxis - i - 1], 1).T))
                    for i in range(maxAxis)])
        elif slicesOrder == 'cas':
            new_img = np.array(
                [np.hstack((
                    np.hstack((
                        np.flip(in_img[slices[0], :, :, i], 1).T,
                        np.flip(in_img[:, :, slices[2], maxAxis - i - 1], 1).T)),
                    np.flip(in_img[:, slices[1], :, maxAxis - i - 1], 1).T))
                    for i in range(maxAxis)])
    else:
        new_img = np.array(
            [np.hstack((
                np.hstack((
                    np.flip(in_img[i, :, :], 1).T,
                    np.flip(in_img[:, maxAxis - i - 1, :], 1).T)),
                np.flip(in_img[:, :, maxAxis - i - 1], 1).T))
                for i in range(maxAxis)])

    return new_img


def write_gif(in_file, out_filename, mode, fps, colormap, slices, slicesOrder):
    """
    Procedure for writing grayscale image.

    Parameters
    ----------
    in_file: str
        Input file (eg. /john/home/image.nii.gz)
    out_filename
        Out file name (*.gif)
    mode: str
        Either normal or pseudocolor rendering
    fps: int
        Frame per second of the gif file
    colormap: str
        CMAP used for pseudocolor (from matplotlib cmaps)
    slices: list
        Slices for each axis to use (only for 4D volumes)
    slicesOrder: str
        Could be either csa or cas (Cortical, Sagital, Axial)
    """
    # Load NIfTI and put it in right shape
    out_img, origShape = load_and_prepare_image(in_file)

    maxAxis = np.max(origShape[0:3])

    # Create output mosaic
    new_img = create_mosaic(out_img, origShape, slices=slices,
                            slicesOrder=slicesOrder)
    del out_img

    # Figure out extension
    ext = '.{}'.format(parse_filename(in_file)[2])

    if mode == 'pseudocolor':
        # Transform values according to the color map
        cmap = get_cmap(colormap)
        color_transformed = [cmap(new_img[i, ...]) for i in range(maxAxis)]
        new_img = np.delete(color_transformed, 3, 3)

        new_img /= new_img.max()  # scale image values between 0-1
        new_img *= 255
        new_img = new_img.astype(np.uint8)

    if not out_filename:
        out_filename = in_file.replace(ext, '.gif')

    mimwrite(out_filename, new_img, format='gif', fps=fps)


def create_gif(in_file, out_filename=None, mode='normal', fps=20,
               colormap='hot', slices=None, slicesOrder='csa'):
    """
    in_file: str
        Path of the file to be converted into GIF
    out_filename: str
        Path of the output GIF file with extension
        If not set, will be ${in_file}.gif
    mode: str
        Either normal or pseudocolor rendering
    fps: int
        Frame per second of the gif file
    colormap: str
        CMAP used for pseudocolor (from matplotlib cmaps)
    slices: list
        Slices for each axis to use (only for 4D volumes)
    slicesOrder: str
        Could be either csa or cas (Cortical, Sagital, Axial)
    """
    # Determine gif creation mode
    if mode.lower() in DEFAULT_MODES:
        if mode == 'normal':
            write_gif(in_file, out_filename, mode, fps, colormap, slices,
                      slicesOrder)
        elif mode == 'pseudocolor':
            write_gif(in_file, out_filename, mode, fps, colormap, slices,
                      slicesOrder)
    else:
        raise ValueError("Unrecognized mode.")
