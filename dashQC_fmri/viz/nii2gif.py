"""Main entry point."""

import argparse
import viz.config as cfg
from viz import core
import warnings  # mainly for ignoring imageio warnings
warnings.filterwarnings("ignore")


def _build_args_parser():
    """Commandline interface."""
    parser = argparse.ArgumentParser()

    parser.add_argument('in_file',
                        help="Path to image. Multiple paths can be provided.")
    parser.add_argument('out_filename',
                        default=None,
                        help="Output GIF filename with extension.")

    parser.add_argument('--mode',
                        default=cfg.mode,
                        help="Gif creation mode. Available options are: "
                             "normal or pseudocolor.")

    parser.add_argument('--fps', type=int,
                        default=cfg.fps,
                        help="Frames per second.")

    parser.add_argument('--cmap',
                        default=cfg.cmap,
                        help="Color map. Used only in combination with "
                             "pseudocolor mode.")

    parser.add_argument('--sliceCor', type=int,
                        default=cfg.sliceCor,
                        help="Slice index for Coronal view.")

    parser.add_argument('--sliceSag', type=int,
                        default=cfg.sliceSag,
                        help="Slice index for Sagital view.")

    parser.add_argument('--sliceAx', type=int,
                        default=cfg.sliceAx,
                        help="Slice index for Axial view.")

    parser.add_argument('--slicesOrder',
                        default='csa',
                        help="Order of the three columns cortical, sagital, "
                             "axial.Can be csa, cas, asc, acs, sac or sca.")


def create_gif(in_file, out_filename, mode=cfg.mode, fps=cfg.fps,
               cmap=cfg.cmap, slices=cfg.slices, slicesOrder=cfg.slicesOrder):
    """
    in_file: str

    out_filename: str

    mode: str

    fps: int

    cmap: str

    slices: list

    slicesOrder: str

    """
    # Determine gif creation mode
    if mode.lower() in cfg.modes:
        if mode == 'normal':
            core.write_gif_normal(in_file, out_filename, fps, slices,
                                  slicesOrder)
        elif mode == 'pseudocolor':
            core.write_gif_pseudocolor(in_file, out_filename, fps, slices,
                                       slicesOrder)
    else:
        raise ValueError("Unrecognized mode.")


def main():
    parser = _build_args_parser()
    args = parser.parse_args()
    create_gif(args.in_file, args.out_filename, args.mode, args.fps, args.cmap,
               args.slices, args.slicesOrder)
