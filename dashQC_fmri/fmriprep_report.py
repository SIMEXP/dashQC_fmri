import re
import numpy as np
import pandas as pd
import pathlib as pal
import nibabel as nib
from nilearn import image as ni
from matplotlib import gridspec
from nilearn import plotting as nlp
from matplotlib import pyplot as plt


class DataInput(object):
    @staticmethod
    def _find_path(folder, pattern):
        search = list(folder.glob(pattern))
        if not len(search) == 1:
            raise FileNotFoundError(
                f'I found {len(search)} hits for {pattern} in {folder}. There needs to be exactly one match')
        return search[0]


class Subject(DataInput):
    def __init__(self, preproc_d, raw_d, subject_name, nl):
        self.nl = nl
        self.prep_d = pal.Path(preproc_d) / subject_name
        self.raw_d = pal.Path(raw_d) / subject_name
        self.subject_name = subject_name
        self.anat_d = self._find_path(self.prep_d, self.nl['anat_d'])
        self.func_d = self._find_path(self.prep_d, self.nl['func_d'])
        self.anat_mask_f = self._find_path(self.anat_d,
                                           self.nl['anat_mask_f'].format(self.subject_name))
        self.func_mask_f = self._find_path(self.func_d,
                                           self.nl['func_mask_f'].format(self.subject_name, '*'))
        self.func_raw_d = self._find_path(self.raw_d, self.nl['func_raw_d'])
        self.anat_f = self._find_path(self.anat_d,
                                      self.nl['anat_pre_f'].format(self.subject_name))
        self.func_f = self._find_path(self.func_d,
                                      self.nl['func_ref_pre_f'].format(self.subject_name, '*'))
        self.runs, self.run_names = self.find_runs()

    def find_runs(self):
        run_names = list()
        runs = list()

        run_cases = [re.search(r'(?<=run-)\d+', str(p)) for p in self.func_d.glob('*run-*')]
        run_ids = list(set(map(lambda m: int(m.group()), run_cases)))

        for run_id in run_ids:
            run_name = f'run-{run_id}'
            run = Run(self.func_d,
                      self.func_raw_d,
                      self.subject_name,
                      run_name, self.nl)
            run_names.append(run_name)
            runs.append(run)
        return runs, run_names


class Run(DataInput):
    def __init__(self, func_prep_d, func_raw_d, subject_name, run_name, nl):
        self.func_prep_d = pal.Path(func_prep_d)
        self.func_raw_d = pal.Path(func_raw_d)
        self.subject_name = subject_name
        self.run_name = run_name
        self.confounds_f = self._find_path(self.func_prep_d,
                                           nl['confound_f'].format(self.subject_name, self.run_name))
        self.func_prep_f = self._find_path(self.func_prep_d,
                                           nl['func_pre_f'].format(self.subject_name, self.run_name))
        self.func_raw_f = self._find_path(self.func_raw_d,
                                          nl['func_raw_f'].format(self.subject_name, self.run_name))
        self.func_ref_prep_f = self._find_path(self.func_prep_d,
                                               nl['func_ref_pre_f'].format(self.subject_name, self.run_name))
        self.func_ref_raw_f = self._find_path(self.func_raw_d,
                                              nl['func_ref_raw_f'].format(self.subject_name, self.run_name))
        self.func_mask_f = self._find_path(self.func_prep_d,
                                           nl['func_mask_f'].format(self.subject_name, self.run_name))


def reference_image(img_path, t_min=1, t_max=99.9):
    data_img = nib.load(str(img_path))

    if len(data_img.shape) > 3:
        median = np.median(data_img.get_data(), 3)
        data_img = nib.Nifti1Image(median, header=data_img.header, affine=data_img.affine)
    else:
        data_img = data_img

    vmin = np.percentile(data_img.get_data(), t_min)
    vmax = np.percentile(data_img.get_data(), t_max)

    return data_img, vmin, vmax


def make_reg_montage(data_in, overlay=None, cmap=nlp.cm.black_red):
    data_img, vmin, vmax = reference_image(data_in, t_min=1, t_max=100)

    # Anat plot with contours
    f_montage = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(3, 3, figure=f_montage, hspace=0)
    gs.update(bottom=0, left=0, right=1, top=1)
    ax_top = f_montage.add_subplot(gs[0, :])
    ax_mid = f_montage.add_subplot(gs[1, :])
    ax_bottom = f_montage.add_subplot(gs[2, :])
    d1 = nlp.plot_anat(data_img, annotate=False, draw_cross=False, black_bg=True,
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       cut_coords=(1, 1, 1), display_mode='ortho', axes=ax_top)

    d2 = nlp.plot_anat(data_img, annotate=False, draw_cross=False, black_bg=True,
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       cut_coords=(40, 30, 30), display_mode='ortho',
                       axes=ax_mid)

    d3 = nlp.plot_anat(data_img, annotate=False, draw_cross=False, black_bg=True,
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       cut_coords=(-40, -40, -30), display_mode='ortho',
                       axes=ax_bottom)
    if overlay is None:
        return f_montage
    else:
        d1.add_overlay(str(overlay), cmap=plt.cm.coolwarm_r, alpha=0.5)
        d2.add_overlay(str(overlay), cmap=plt.cm.coolwarm_r, alpha=0.5)
        d3.add_overlay(str(overlay), cmap=plt.cm.coolwarm_r, alpha=0.5)

        return f_montage


def motion_figure(img_path, x=3, y=1, cmap=nlp.cm.black_red):
    # Load the image
    data_img = nib.load(str(img_path))
    n_t = data_img.shape[3]

    vmin = np.percentile(data_img.get_data(), 1)
    vmax = np.percentile(data_img.get_data(), 99.9)

    # Make the figure
    f_motion = plt.figure(figsize=(x * n_t, y))
    gs = gridspec.GridSpec(1, n_t, figure=f_motion, hspace=0, wspace=0)
    gs.update(bottom=0, left=0, right=1, top=1)

    for i in np.arange(n_t, dtype=int):
        ax = f_motion.add_subplot(gs[i])
        nlp.plot_anat(ni.index_img(data_img, i), annotate=False,
                      draw_cross=False, black_bg=True,
                      cut_coords=(1, 1, 1), display_mode='ortho',
                      axes=ax, colorbar=False, cmap=cmap,
                      vmin=vmin, vmax=vmax)

    return f_motion


def target_figure(img_path, x=3, y=1, cmap=nlp.cm.black_red):
    data_img, vmin, vmax = reference_image(img_path, t_min=1, t_max=100)
    f_target = plt.figure(figsize=(x, y))
    ax = f_target.add_axes([0, 0, 1, 1])
    nlp.plot_anat(data_img, annotate=False, vmin=vmin, vmax=vmax, cmap=cmap,
                  draw_cross=False, black_bg=True, cut_coords=(1, 1, 1),
                  display_mode='ortho', axes=ax, colorbar=False)

    return f_target


def brain_overlap(img_test_p, img_ref_p):
    img_test, _, _ = reference_image(img_test_p)
    img_ref, _, _ = reference_image(img_ref_p)
    # Sample to reference image if not agree
    if not img_test.shape == img_ref.shape:
        img_test = ni.resample_img(img_test,
                                   target_affine=img_ref.affine,
                                   target_shape=img_ref.shape,
                                   interpolation='nearest')
    mask_test = img_test.get_data().astype(bool)
    mask_ref = img_ref.get_data().astype(bool)

    overlap = np.sum(mask_test & mask_ref) / np.sum(mask_ref)

    return float(overlap)


def brain_correlation(img_test_p, img_ref_p):
    img_test, _, _ = reference_image(img_test_p)
    img_ref, _, _ = reference_image(img_ref_p)
    # Sample to reference image if not agree
    if not img_test.shape == img_ref.shape:
        img_test = ni.resample_img(img_test,
                                   target_affine=img_ref.affine,
                                   target_shape=img_ref.shape,
                                   interpolation='nearest')
    data_test = img_test.get_data().flatten()
    data_ref = img_ref.get_data().flatten()
    correlation = np.corrcoef(data_test, data_ref)[0, 1]

    return float(correlation)


def report_run(run):
    conf = pd.read_csv(run.confounds_f, sep='\t')
    # First add motion parameters
    report = {f'{cat}_{axis}':list(conf[f'{cat}_{axis}'].values.astype(float))
              for cat in ['trans', 'rot']
              for axis in ['x', 'y', 'z']}

    report['n_vol_before'] = int(conf.shape[0])
    fd = conf.framewise_displacement.values # First FD is nan as per convention
    motion_mask = fd > 0.5
    # Make the scrubbing mask
    motion_ind = np.argwhere(motion_mask).flatten()
    scrubbed = np.zeros(report['n_vol_before'], dtype=int)
    for i in motion_ind:
        if i == 0:
            continue
        scrubbed[i - 1:i + 3] = 1
    report['mean_fd_before'] = float(np.nanmean(fd))
    report['mean_fd_after'] = float(np.nanmean(fd[~scrubbed.astype(bool)]))
    report['n_vol_after'] = int(np.sum(~scrubbed.astype(bool)))
    report['scrubbed'] = [float(scrub) for scrub in scrubbed]
    report['fd'] = [float(f) for f in fd]
    # Placeholder for current dashboard
    report['corr_run_ref'] = 1
    return report


def report_subject(sub):
    report = dict()
    report['ovlp_T1_stereo'] = brain_overlap(sub.anat_mask_f, sub.anat_mask_f) # replace with MNI
    report['corr_T1_stereo'] = brain_correlation(sub.anat_f, sub.anat_f) # replace with MNI
    report['ovlp_BOLD_T1'] = brain_overlap(sub.func_mask_f, sub.anat_mask_f) # replace with MNI
    report['corr_BOLD_T1'] = brain_correlation(sub.func_f, sub.anat_f) # replace with MNI
    report['run_names'] = sub.run_names
    report['runs'] = {run_name:report_run(sub.runs[run_id]) for run_id, run_name in enumerate(report['run_names'])}
    return report