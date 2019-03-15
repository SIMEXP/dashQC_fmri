import re
import json
import time
import shutil
import warnings
import argparse
import numpy as np
import pandas as pd
import pathlib as pal
import nibabel as nib
import subprocess as sb
from distutils import dir_util
from nilearn import image as ni
from matplotlib import gridspec
from nilearn import plotting as nlp
from matplotlib import pyplot as plt


class DataInput(object):
    @staticmethod
    def _set_path(folder, pattern, insert):
        return folder / pattern.format(*insert)

    @staticmethod
    def _find_path(folder, pattern):
        search = list(folder.glob(pattern))
        if not len(search) == 1:
            raise FileNotFoundError(f'I found {len(search)} hits for {pattern} in {folder}. '
                                    f'There needs to be exactly one match')
        return search[0]

    def check_outputs_done(self):
        return all([p.exists() for p in self.outputs])


class Subject(DataInput):
    def __init__(self, preproc_d, raw_d, subject_name, nl):
        self.nl = nl
        self.prep_d = pal.Path(preproc_d) / subject_name
        self.raw_d = self._find_path(raw_d, f'**/{subject_name}')
        self.subject_name = subject_name

        # Define paths
        self.anat_d = self._find_path(self.prep_d, self.nl['anat_d'])
        self.func_d = self._find_path(self.prep_d, self.nl['func_d'])
        self.fig_d = self._find_path(self.prep_d, self.nl['fig_d'])

        # Define inputs
        self.anat_mask_f = self._find_path(self.anat_d,
                                           self.nl['anat_mask_f'].format(self.subject_name))
        self.func_mask_f = self._find_path(self.func_d,
                                           self.nl['func_mask_f'].format(self.subject_name, '*'))
        self.func_raw_d = self._find_path(self.raw_d, self.nl['func_raw_d'])
        self.anat_f = self._find_path(self.anat_d,
                                      self.nl['anat_pre_f'].format(self.subject_name))
        self.anat_skull_f = self._find_path(self.anat_d,
                                            self.nl['anat_skull_pre_f'].format(self.subject_name))
        self.anat_transform_f = self._find_path(self.anat_d,
                                                self.nl['anat_to_stand_transform_f'].format(self.subject_name))
        self.func_f = self._find_path(self.func_d,
                                      self.nl['func_ref_pre_f'].format(self.subject_name, '*'))
        self.runs, self.run_names = self.find_runs()

        # Define outputs
        # Pasted subject names must be tuple for unpacking
        self.img_anat_skull_f = self._set_path(self.fig_d,
                                                     self.nl['img_anat_skull_pre_f'],
                                                     (self.subject_name,))
        self.fig_anat_reg_outline_f = self._set_path(self.fig_d,
                                                     self.nl['fig_anat_reg_outline_f'],
                                                     (self.subject_name,))
        self.fig_anat_reg_f = self._set_path(self.fig_d,
                                             self.nl['fig_anat_reg_f'],
                                             (self.subject_name,))
        self.fig_func_reg_f = self._set_path(self.fig_d,
                                             self.nl['fig_func_reg_f'],
                                             (self.subject_name,))
        self.report_f = self._set_path(self.fig_d,
                                       self.nl['report_f'],
                                       (self.subject_name,))
        self.outputs = [self.img_anat_skull_f, self.fig_anat_reg_outline_f,
                        self.fig_anat_reg_f, self.fig_func_reg_f,
                        self.report_f]

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
                      run_name, self.fig_d, self.nl)
            run_names.append(run_name)
            runs.append(run)
        return runs, run_names


class Run(DataInput):
    def __init__(self, func_prep_d, func_raw_d, subject_name, run_name, fig_d, nl):
        # Define paths
        self.func_prep_d = pal.Path(func_prep_d)
        self.func_raw_d = pal.Path(func_raw_d)
        self.fig_d = pal.Path(fig_d)
        self.subject_name = subject_name
        self.run_name = run_name
        # Define inputs
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
        # Define outputs
        self.fig_func_ref_raw_f = self._set_path(self.fig_d,
                                                 nl['fig_func_ref_raw_f'], (self.subject_name, self.run_name))

        self.fig_func_ref_f = self._set_path(self.fig_d,
                                             nl['fig_func_ref_f'], (self.subject_name, self.run_name))
        self.fig_mot_raw_f = self._set_path(self.fig_d,
                                            nl['fig_func_mot_raw_f'], (self.subject_name, self.run_name))
        self.fig_mot_f = self._set_path(self.fig_d,
                                        nl['fig_func_mot_f'], (self.subject_name, self.run_name))
        self.outputs = [self.fig_func_ref_raw_f, self.fig_func_ref_f, self.fig_mot_raw_f, self.fig_mot_f]


def get_name_lookup():
    # TODO: BOLD reference at run level
    # TODO: standard space explicit check
    # TODO: version specific lookups
    name_lookup = {'anat_d': 'anat',
                   'func_d': 'func',
                   'fig_d': 'figures',
                   'anat_mask_f': '{}_space-*_desc-brain_mask.nii.gz',
                   'func_mask_f': '{}_*_{}_space-*_desc-brain_mask.nii.gz',
                   'func_raw_d': 'func',
                   'anat_pre_f': '{}_*_desc-preproc_T1w.nii.gz',
                   'anat_skull_pre_f': '{}_desc-preproc_T1w.nii.gz',
                   'anat_to_stand_transform_f': '{}_from-T1w_to-*_mode-image_xfm.h5',
                   'func_ref_pre_f': '{}_*_{}_space-*_boldref.nii.gz',
                   'func_pre_f': '{}_*_{}_space-*_desc-preproc_bold.nii.gz',
                   'func_ref_raw_f': '{}_*_{}*.nii.gz',
                   'func_raw_f': '{}_*_{}*.nii.gz',
                   'confound_f': '{}_*_{}_desc-confounds_regressors.tsv',
                   'img_anat_skull_pre_f': '{}_anat_skull.nii.gz',
                   'fig_anat_reg_outline_f': '{}_anat_reg_outline.png',
                   'fig_anat_reg_f': '{}_anat_reg.png',
                   'fig_func_reg_f': '{}_func_reg.png',
                   'fig_func_ref_raw_f': '{}_{}_func_ref_raw.png',
                   'fig_func_ref_f': '{}_{}_func_ref.png',
                   'fig_func_mot_raw_f': '{}_{}_func_mot_raw.png',
                   'fig_func_mot_f': '{}_{}_func_mot.png',
                   'report_f': '{}_report.json'}
    return name_lookup


def get_report_lookup():
    report_lookup = {'fd': 'summary/js/fd.js',
                     'chartBOLD': 'summary/js/chartBOLD.js',
                     'chartBrain': 'summary/js/chartBrain.js',
                     'chartT1': 'summary/js/chartT1.js',
                     'filesIn': 'summary/js/filesIn.js',
                     'pipeSummary': 'summary/js/pipeSummary.js',
                     'listSubject': 'group/js/listSubject.js',
                     'listRun': 'group/js/listRun.js',
                     'dataMotion': 'motion/js/dataMotion_{}.js',
                     'fig_avg_func':'group/images/average_func_stereotaxic.png',
                     'fig_avg_mask_func': 'group/images/average_mask_func_stereotaxic.png',
                     'fig_mask_func': 'group/images/mask_func_group_stereotaxic.png',
                     'fig_avg_t1': 'group/images/average_t1_stereotaxic.png',
                     'fig_template': 'group/images/template_stereotaxic.png',
                     'fig_template_outline': 'group/images/template_stereotaxic_raw.png',
                     'fig_sub_func_reg': 'registration/images/{}_func.png',
                     'fig_sub_anat_reg': 'registration/images/{}_anat_raw.png',
                     'fig_sub_anat_reg_outline': 'registration/images/{}_anat.png',
                     'fig_run_ref_raw': 'motion/images/target_native_{}.png',
                     'fig_run_ref_prep': 'motion/images/target_stereo_{}.png',
                     'fig_run_mot_raw': 'motion/images/motion_native_{}.png',
                     'fig_run_mot_prep': 'motion/images/motion_stereo_{}.png',
                     'report_timestamp': 'registration/js/datasetID.js'
                     }
    return report_lookup


def get_template():
    # TODO accept other templates ?
    template_dir = (pal.Path(__file__) / '../data/images').resolve()
    template = {'T1': template_dir / 'MNI_ICBM152_T1_asym_09c.nii.gz',
                'mask': template_dir / 'MNI_ICBM152_09c_mask.nii.gz',
                'outline': template_dir / 'MNI_ICBM152_09c_outline.nii.gz'
                }
    if not all([template[key].exists() for key in template.keys()]):
        raise Exception(f'Could not find the builtin template images at {template_dir}: '
                        f'{[(key, template[key].exists()) for key in template.keys()]}')

    return template


def apply_transform(sub):
    command_result = sb.run(['antsApplyTransforms',
                             '-i', sub.anat_skull_f,
                             '-r', sub.anat_f,
                             '-o', sub.img_anat_skull_f,
                             '-t', sub.anat_transform_f],
                            stderr=sb.PIPE, stdout=sb.PIPE, universal_newlines=True)
    # Check if the command completed succesffuly
    if command_result.returncode == 0:
        return command_result.returncode
    else:
        # Something is bad
        raise ChildProcessError(f'Running antsApplyTransform on {sub.subject_name} failed:\n'
                                f'The command was:\n {" ".join([str(i) for i in command_result.args])}\n\n'
                                f'This resulted in the following error message:\n'
                                f'{command_result.stderr}')


def populate_report(report_p, clobber=False):
    # Copy the template into the report folder
    report_template_p = (pal.Path(__file__) / '../data/report').resolve()
    dir_util.copy_tree(str(report_template_p), str(report_p), verbose=0)
    # Create the directory tree for the files that are yet to be created
    tree_structure = [
        'assets/group/images',
        'assets/group/js',
        'assets/motion/images',
        'assets/motion/js',
        'assets/registration/images',
        'assets/registration/js',
        'assets/summary/js',
    ]

    for branch in tree_structure:
        branch_p = report_p / branch
        branch_p.mkdir(parents=True, exist_ok=clobber)


def reference_image(data_in, t_min=1, t_max=99.9):
    if issubclass(type(data_in), pal.Path) or type(data_in) == str:
        data_img = nib.load(str(data_in))
    else:
        data_img = data_in

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


def average_image(paths):
    # Average image from paths of images
    images = [nib.load(str(p)) for p in paths]
    sizes = set([img.shape for img in images])
    if not len(sizes)==1:
        raise Exception(f'images to be averaged have inconsitent sizes: {sizes}')
    size = list(sizes)[0]
    n_img = len(images)
    avg = np.zeros(size)
    for img in images:
        avg += img.get_data()
    avg /= n_img
    avg_img = nib.Nifti1Image(avg, affine=img.affine, header=img.header)

    return avg_img


def make_dict_str(var_name, items):
    dict_str = ',\n'.join([str({'id': iid+1,
                  'text': item
                 }) for iid, item in enumerate(items)])
    out_str = f'var {var_name} = [\n{dict_str}\n];'
    return out_str


def make_list_str(var_name, *kwargs):
    list_str = ',\n'.join(list(map(str, kwargs)))
    out_str = f'var {var_name} = [\n{list_str}\n];'
    return out_str


def make_run_str(run_report):
    boilerplate = 'selection: {\n  enabled: true\n},\nonclick: function(d) { selectTime(d.index);}'
    translation = ',\n'.join(['{}'.format([f'motion_t{axis}'] + run_report[f'trans_{axis}'])
                              for axis in ['x', 'y', 'z']])
    rotation = ',\n'.join(['{}'.format([f'motion_r{axis}'] + run_report[f'rot_{axis}'])
                           for axis in ['x', 'y', 'z']])
    fd = ['FD'] + list(map(str, run_report['fd']))
    fd_mask = ['scrub'] + list(map(str, run_report['scrubbed']))
    scrub = ',\n'.join(['{}'.format(var) for var in [fd, fd_mask]])

    part_1 = 'var tsl = {\n  columns: [\n' + translation + '],\n' + boilerplate + '\n};'
    part_2 = 'var rot = {\n  columns: [\n' + rotation + '],\n' + boilerplate + '\n};'
    part_3 = 'var fd = {\n  columns: [\n' + scrub + '],\n' + boilerplate + '\n};'

    out_str = '\n'.join([part_1, part_2, part_3])
    return out_str


def report_run(run):
    # TODO remove first nan from FD
    conf = pd.read_csv(run.confounds_f, sep='\t')
    # First add motion parameters
    report = {f'{cat}_{axis}': list(conf[f'{cat}_{axis}'].values.astype(float))
              for cat in ['trans', 'rot']
              for axis in ['x', 'y', 'z']}

    report['n_vol_before'] = int(conf.shape[0])
    fd = conf.framewise_displacement.values  # First FD is nan as per convention
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
    report['ovlp_T1_stereo'] = brain_overlap(sub.anat_mask_f, sub.anat_mask_f)  # replace with MNI
    report['corr_T1_stereo'] = brain_correlation(sub.anat_f, sub.anat_f)  # replace with MNI
    report['ovlp_BOLD_T1'] = brain_overlap(sub.func_mask_f, sub.anat_mask_f)  # replace with MNI
    report['corr_BOLD_T1'] = brain_correlation(sub.func_f, sub.anat_f)  # replace with MNI
    report['run_names'] = sub.run_names
    report['runs'] = {run_name: report_run(sub.runs[run_id]) for run_id, run_name in enumerate(report['run_names'])}
    return report


def find_available_subjects(prep_p, raw_p, subject_list_f=None):
    potential_subjects = [str(query.relative_to(prep_p))
                          for query in prep_p.glob('sub-*') if query.is_dir()]
    available_subjects = list()
    n_potential = len(potential_subjects)
    for sub_id, sub_name in enumerate(potential_subjects):
        try:
            sub = Subject(prep_p, raw_p, sub_name, get_name_lookup())
            available_subjects.append(sub_name)
        except FileNotFoundError:
            warnings.warn(f'Could not find all data for subject {sub_name} in {prep_p}. '
                          f'subject {sub_id} / {n_potential}')

    if subject_list_f is not None:
        with subject_list_f.open(mode='w') as f:
            f.writelines('\n'.join(available_subjects))
        return None
    else:
        return available_subjects


def process_subject(prep_p, raw_p, subject_name, clobber=True):
    sub = Subject(prep_p, raw_p, subject_name, get_name_lookup())
    temp = get_template()
    # Check if the outputs are already generated
    if sub.check_outputs_done() and not clobber:
        raise Exception(f'Some subject-level outputs for {sub.subject_name} are already done. '
                        'Force clobber if you want to overwrite them')
    if all([run.check_outputs_done() for run in sub.runs]) and not clobber:
        raise Exception(f'Some run-level outputs for {sub.subject_name} are already done. '
                        'Force clobber if you want to overwrite them')
    # Check specifically if the skull-anat file exists and run create it if not
    if not sub.img_anat_skull_f.exists():
        _ = apply_transform(sub)

    # Generate subject level outputs
    fig_anat_reg_outline = make_reg_montage(sub.img_anat_skull_f, cmap=plt.cm.Greys_r, overlay=temp['outline'])
    fig_anat_reg = make_reg_montage(sub.img_anat_skull_f, cmap=plt.cm.Greys_r)
    fig_func_reg = make_reg_montage(sub.func_f)
    report = report_subject(sub)
    # Store subject level outputs
    fig_anat_reg_outline.savefig(sub.fig_anat_reg_outline_f, dpi=300)
    fig_anat_reg.savefig(sub.fig_anat_reg_f, dpi=300)
    fig_func_reg.savefig(sub.fig_func_reg_f, dpi=300)
    with sub.report_f.open('w') as f:
        json.dump(report, f, indent=4, sort_keys=False)

    # Generate the run level outputs
    for run in sub.runs:
        fig_func_ref_raw = target_figure(run.func_ref_raw_f)
        fig_func_ref = target_figure(run.func_ref_prep_f)
        fig_mot_raw = motion_figure(run.func_raw_f)
        fig_mot = motion_figure(run.func_prep_f)

        fig_func_ref_raw.savefig(run.fig_func_ref_raw_f, dpi=100)
        fig_func_ref.savefig(run.fig_func_ref_f, dpi=100)
        fig_mot_raw.savefig(run.fig_mot_raw_f, dpi=100)
        fig_mot.savefig(run.fig_mot_f, dpi=100)


def generate_dashboard(prep_p, raw_p, report_p, clobber=True):
    try:
        report_p.mkdir(exist_ok=clobber)
    except FileExistsError as e:
        raise Exception(f'The report directory already exists. Set clobber=True if you want to overwrite.') from e

    populate_report(report_p, clobber=clobber)
    potential_subjects = [str(query.relative_to(prep_p))
                          for query in prep_p.glob('sub-*') if query.is_dir()]
    available_subjects = list()

    subjects = list()
    for sub_name in potential_subjects:
        try:
            sub = Subject(prep_p, raw_p, sub_name, get_name_lookup())
        except FileNotFoundError:
            warnings.warn(f'Could not find all data for subject {sub_name} in {prep_p}')
            continue
        if sub.check_outputs_done():
            available_subjects.append(sub_name)
            subjects.append(sub)
        else:
            warnings.warn(f'{sub_name} is not finished yet in {prep_p}')

    # Get the corresponding runs
    runs = [run for sub in subjects for run in sub.runs]
    available_runs = [f'{run.subject_name}_{run.run_name}' for run in runs]
    reports = {sub.subject_name: json.loads(sub.report_f.read_text())
               for sub in subjects}

    # Generate group level data
    t1_group_average = average_image([sub.anat_f for sub in subjects])
    func_group_average = average_image([sub.func_f for sub in subjects])
    func_group_mask_average = average_image([sub.func_mask_f for sub in subjects])
    # Threshold the average subject at the 95th percentile
    func_group_mask = nib.Nifti1Image((func_group_mask_average.get_data() > 0.95).astype(int),
                                      affine=func_group_mask_average.affine,
                                      header=func_group_mask_average.header)

    # Generate group level figures
    temp = get_template()
    fig_t1_group_average = make_reg_montage(t1_group_average,
                                            overlay=temp['outline'],
                                            cmap=plt.cm.Greys_r)
    fig_func_group_average = make_reg_montage(func_group_average)
    fig_func_group_mask_average = make_reg_montage(func_group_mask_average)
    fig_func_group_mask = make_reg_montage(func_group_mask)
    fig_template_outline = make_reg_montage(temp['T1'], overlay=temp['outline'],
                                            cmap=plt.cm.Greys_r)
    fig_template = make_reg_montage(temp['T1'], cmap=plt.cm.Greys_r)

    # Generate js data
    subject_list_str = make_dict_str('listSubject', available_subjects)
    run_list_str = make_dict_str('dataRun', available_runs)

    fd_part1 = make_list_str('dataFD',
                             ['Run'] + available_runs,
                             ['FD_before'] + [str(reports[run.subject_name]['runs'][run.run_name]['mean_fd_before'])
                                              for run in runs],
                             ['FD_after'] + [str(reports[run.subject_name]['runs'][run.run_name]['mean_fd_after'])
                                             for run in runs]
                             )
    fd_part2 = make_list_str('dataNbVol',
                             ['Run'] + available_runs,
                             ['vol_scrubbed'] + [str(reports[run.subject_name]['runs'][run.run_name]['n_vol_after'])
                                                 for run in runs],
                             ['vol_ok'] + [str(reports[run.subject_name]['runs'][run.run_name]['n_vol_before'])
                                           for run in runs]
                             )
    fd_str = '\n'.join((fd_part1, fd_part2))

    chart_brain_str = make_list_str('dataIntra',
                                    ['Run'] + available_runs,
                                    ['corr_target'] + [str(reports[run.subject_name]['runs'][run.run_name]['corr_run_ref'])
                                                       for run in runs]
                                    )

    bold_part1 = make_list_str('dataBOLD',
                               ['Subject'] + available_subjects,
                               ['corr_target'] + [str(reports[sub.subject_name]['corr_BOLD_T1'])
                                                  for sub in subjects]
                               )
    bold_part2 = make_list_str('dataBrain',
                               ['Subject'] + available_subjects,
                               ['overlap_brain'] + [str(reports[sub.subject_name]['ovlp_BOLD_T1'])
                                                    for sub in subjects]
                               )
    chart_bold_str = '\n'.join((bold_part1, bold_part2))

    anat_part1 = make_list_str('dataT1',
                               ['Subject'] + available_subjects,
                               ['corr_target'] + [str(reports[sub.subject_name]['corr_T1_stereo'])
                                                  for sub in subjects]
                               )
    anat_part2 = make_list_str('dataOverlapT1',
                               ['Subject'] + available_subjects,
                               ['overlap_brain'] + [str(reports[sub.subject_name]['ovlp_T1_stereo'])
                                                    for sub in subjects]
                               )
    chart_t1_str = '\n'.join((anat_part1, anat_part2))

    # Copy pregenerated images to the repository
    asset_p = report_p / 'assets'
    ol = get_report_lookup()
    for sub in subjects:
        sub_name = sub.subject_name
        shutil.copyfile(sub.fig_anat_reg_f, asset_p / ol['fig_sub_anat_reg'].format(sub_name))
        shutil.copyfile(sub.fig_anat_reg_outline_f, asset_p / ol['fig_sub_anat_reg_outline'].format(sub_name))
        shutil.copyfile(sub.fig_func_reg_f, asset_p / ol['fig_sub_func_reg'].format(sub_name))

        for run in sub.runs:
            run_name = f'{sub_name }_{run.run_name}'
            shutil.copyfile(run.fig_func_ref_f, asset_p / ol['fig_run_ref_prep'].format(run_name))
            shutil.copyfile(run.fig_func_ref_raw_f, asset_p / ol['fig_run_ref_raw'].format(run_name))
            shutil.copyfile(run.fig_mot_f, asset_p / ol['fig_run_mot_prep'].format(run_name))
            shutil.copyfile(run.fig_mot_raw_f, asset_p / ol['fig_run_mot_raw'].format(run_name))

    # Save js data
    for sub in subjects:
        for run in sub.runs:
            run_str = make_run_str(reports[run.subject_name]['runs'][run.run_name])
            run_name = f'{run.subject_name}_{run.run_name}'
            with open(asset_p / ol['dataMotion'].format(run_name), 'w') as f:
                f.write(run_str)

    with open(asset_p / ol['listSubject'], 'w') as f:
        f.write(subject_list_str)

    with open(asset_p / ol['listRun'], 'w') as f:
        f.write(run_list_str)

    with open(asset_p / ol['chartBOLD'], 'w') as f:
        f.write(chart_bold_str)

    with open(asset_p / ol['chartBrain'], 'w') as f:
        f.write(chart_brain_str)

    with open(asset_p / ol['chartT1'], 'w') as f:
        f.write(chart_t1_str)

    with open(asset_p / ol['fd'], 'w') as f:
        f.write(fd_str)

    # Save figure data
    fig_func_group_average.savefig(str(asset_p / ol['fig_avg_func']), dpi=300)
    fig_func_group_mask.savefig(str(asset_p / ol['fig_mask_func']), dpi=300)
    fig_func_group_mask_average.savefig(str(asset_p / ol['fig_avg_mask_func']), dpi=300)
    fig_t1_group_average.savefig(str(asset_p / ol['fig_avg_t1']), dpi=300)
    fig_template.savefig(str(asset_p / ol['fig_template']), dpi=300)
    fig_template_outline.savefig(str(asset_p / ol['fig_template_outline']), dpi=300)

    # Create timestamp data for the dashboard
    with open(asset_p / ol['report_timestamp'], 'w') as f:
        data_id = {'data': time.strftime("%Y-%m-%d-%H:%M:%S"),
                   'timestamp': time.time() }
        f.write(f'var datasetID = {json.dumps(data_id)};')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_dir", type=str,
                        help="bids conform directory of the fmriprep outputs")
    parser.add_argument("raw_dir", type=str,
                        help="bids conform directory of the raw fmriprep input")
    parser.add_argument("mode", type=str, choices=['detect', 'subject', 'dashboard'],
                        help="Choose an operating mode.")
    parser.add_argument("output_path", type=str,
                        help="Select the path where outputs should be generated. "
                             "Depending on the mode, this should be either a directory name or a path to a text file.")
    parser.add_argument("-s", "--subject", type=str,
                        help="Specify the subject you want to run on. Only works when mode = subject.")
    parser.add_argument("-c", "--clobber", type=bool, default=False,
                        help="If set to 'True', existing outputs will be overwritten. Default is 'False'.")
    args = parser.parse_args()
    if args.mode == 'detect':
        find_available_subjects(pal.Path(args.preproc_dir),
                                pal.Path(args.raw_dir),
                                pal.Path(args.output_path))
    if args.mode == 'subject':
        # Check if ANTs is setup correctly
        if shutil.which('antsApplyTransforms') is None:
            raise EnvironmentError('ANTs does not seem to be correctly configured on your system. '
                                   'Please make sure that ANTs is installed and that you can run '
                                   '"antsApplyTransforms" from inside your command line.')
        process_subject(pal.Path(args.preproc_dir),
                        pal.Path(args.raw_dir),
                        args.subject,
                        clobber=args.clobber)
    if args.mode == 'dashboard':
        generate_dashboard(pal.Path(args.preproc_dir),
                           pal.Path(args.raw_dir),
                           pal.Path(args.output_path))
