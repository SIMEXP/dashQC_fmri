import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image as nii
from nilearn import plotting as nlp
from matplotlib.animation import FuncAnimation


def init():
    ax.clear()
    return (ax,)


def animate(i):
    nlp.plot_anat(nii.index_img(data_img, i), annotate=False,
                  draw_cross=False, black_bg=True,
                  cut_coords=(1, 1, 1), display_mode='ortho',
                  axes=ax, colorbar=False, cmap=cmap,
                  vmin=vmin, vmax=vmax)
    return (ax,)


def make_gif(in_path, out_path):
    global ax, vmin, vmax, cmap, data_img
    data_img = nib.load(in_path)
    if not len(data_img.shape) == 4:
        raise RuntimeError(f'I did not find a 4D nifti at {in_path}')
    vmin = np.percentile(data_img.get_fdata(), 1)
    vmax = np.percentile(data_img.get_fdata(), 99.9)
    frames = data_img.shape[-1]
    cmap = nlp.cm.black_red

    f_target = plt.figure(figsize=(3, 1))
    ax = f_target.add_axes([0, 0, 1, 1])
    
    anim = FuncAnimation(f_target, animate, init_func=init,
                         frames=frames, interval=200, blit=True)
    anim.save(out_path, writer='imagemagick', fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str,
                        help="path to some 4D nifti file")
    parser.add_argument("output_path", type=str,
                        help="path where gif should be created")
    args = parser.parse_args()
    main(args.in_path,
                       args.output_path)
