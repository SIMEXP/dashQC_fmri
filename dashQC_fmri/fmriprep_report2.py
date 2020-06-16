import re
import json
import time
import tqdm
import shutil
import warnings
import argparse
import numpy as np
import pandas as pd
import pathlib as pal
import nibabel as nib
import multiprocessing
import subprocess as sb
from distutils import dir_util
from nilearn import image as ni
from matplotlib import gridspec
from nilearn import plotting as nlp
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


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
        median = np.median(data_img.get_fdata(), 3)
        data_img = nib.Nifti1Image(median, header=data_img.header, affine=data_img.affine)
    else:
        data_img = data_img

    vmin = np.percentile(data_img.get_fdata(), t_min)
    vmax = np.percentile(data_img.get_fdata(), t_max)

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


def motion_figure(img_path, x=3, y=1, cmap=nlp.cm.black_red, crop=False):
    # Load the image
    data_img = nib.load(str(img_path))
    if crop:
        n_t = 50
    else:
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
    mask_test = img_test.get_fdata().astype(bool)
    mask_ref = img_ref.get_fdata().astype(bool)

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
    data_test = img_test.get_fdata().flatten()
    data_ref = img_ref.get_fdata().flatten()
    correlation = np.corrcoef(data_test, data_ref)[0, 1]

    return float(correlation)


def average_image(paths):
    # Average image from paths of images
    images = [nib.load(str(p)) for p in paths]
    sizes = set([img.shape for img in images])
    if not len(sizes)==1:
        print(f'I got these paths: {paths}')
        for img in images:
            print(f'{img.get_filename()} has shape {img.shape}')
        raise Exception(f'images to be averaged have inconsitent sizes: {sizes}')
    size = list(sizes)[0]
    n_img = len(images)
    avg = np.zeros(size)
    for img in images:
        avg += img.get_fdata()
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
    conf = pd.read_csv(run['confound'], sep='\t')
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


def report_subject(sub, temp):
    report = dict()
    report['ovlp_T1_stereo'] = brain_overlap(temp['T1'], sub['t1'])  # replace with MNI
    report['corr_T1_stereo'] = brain_correlation(temp['T1'], sub['t1'])  # replace with MNI
    # Compute the average boldref
    runs = sub['runs']
    boldref_avg = average_image([runs[run]['boldref'] for run in runs.keys()])

    report['ovlp_BOLD_T1'] = brain_overlap(
        boldref_avg, temp['mask'])  # replace with MNI
    report['corr_BOLD_T1'] = brain_correlation(
        boldref_avg, temp['mask'])  # replace with MNI
    report['run_names'] = runs.keys()
    report['runs'] = {run_name: report_run(runs[run_name]) for run_name in report['run_names']}
    return report


def process_subject(sub_d, asset_p, sub_name, clobber=True):
    temp = get_template()
    runs = sub_d['runs']
    ol = get_report_lookup()
    
    # Go through the outputs and see if they already exist
    if not (asset_p /
    ol['fig_sub_anat_reg_outline'].format(sub_name)).is_file() or clobber:
        fig_anat_reg_outline = make_reg_montage(
            sub_d['t1'], cmap=plt.cm.Greys_r, overlay=temp['outline'])
        fig_anat_reg_outline.savefig(asset_p /
                                     ol['fig_sub_anat_reg_outline'].format(sub_name), dpi=300)
    if not (asset_p /
    ol['fig_sub_anat_reg'].format(sub_name)).is_file() or clobber:
        fig_anat_reg = make_reg_montage(sub_d['t1'], cmap=plt.cm.Greys_r)
        fig_anat_reg.savefig(asset_p /
                             ol['fig_sub_anat_reg'].format(sub_name), dpi=300)
    if not (asset_p /
    ol['fig_sub_func_reg'].format(sub_name)).is_file() or clobber:
        # Get the average boldref of the underlying runs
        avg_boldref = average_image([runs[run]['boldref'] for run in runs.keys()])
        fig_func_reg = make_reg_montage(avg_boldref)
        fig_func_reg.savefig(asset_p /
                             ol['fig_sub_func_reg'].format(sub_name), dpi=300)

    # Generate the run level outputs
    for run_name in runs.keys() or clobber:
        run_d = runs[run_name]
        if not (asset_p /
        ol['fig_run_ref_prep'].format(run_name)).is_file():
            fig_func_ref = target_figure(run_d['boldref'])
            fig_func_ref.savefig(asset_p /
                                 ol['fig_run_ref_prep'].format(run_name), dpi=100)
            fig_func_ref.savefig(asset_p /
                                 ol['fig_run_ref_raw'].format(run_name), dpi=100)
            fig_func_ref.savefig(asset_p /
                                 ol['fig_run_mot_prep'].format(run_name), dpi=100)
            fig_func_ref.savefig(asset_p /
                                 ol['fig_run_mot_raw'].format(run_name), dpi=100)


def get_run_name(path_str): 
    search_str = r'^(?P<sub>sub-\d+)_*?(?P<session>ses-[a-zA-Z0-9]+)?_*(?P<task>task-[a-zA-Z0-9]+)_*(?P<run>run-\d+)?'
    search = re.search(search_str, path_str)
    if search is None:
        warnings.warn(
            f'Could not find valid BIDS data at {path_str}. I will (probably) die now.')
    search_dict = search.groupdict()
    return '_'.join([search_dict[key]
                     for key in ['sub', 'session', 'task', 'run']
                     if search_dict[key] is not None])


def generate_dashboard(prep_p, report_p, clobber=True, 
                       n_cpu=multiprocessing.cpu_count()-2):
    if n_cpu < 1:
        n_cpu = 1
    elif n_cpu > multiprocessing.cpu_count():
        
        warnings.warn(
            f'You requested {n_cpu} cores but this sytem only has {multiprocessing.cpu_count()}. CPU_count is set to 1!')
        n_cpu = 1
        
    try:
        report_p.mkdir(exist_ok=clobber)
    except FileExistsError as e:
        raise Exception(f'The report directory already exists. Set clobber=True if you want to overwrite.') from e

    templates = {
        'mask': '*_desc-brain_mask.nii.gz',
        'preproc': '*_desc-preproc_bold.nii.gz',
        'boldref': '*_boldref.nii.gz',
        'confound': '*_desc-confounds_regressors.tsv'
        }

    populate_report(report_p, clobber=clobber)
    
    # In this version, a subject can also be a session when multi-session data is used.
    subjects = {}
    runs = {}
    for root_query in prep_p.glob('sub-*'):
        if root_query.is_dir() and (root_query / 'anat').is_dir():
            if (root_query / 'func').is_dir():
                sub_query = [root_query]
                # This is a single-session (no-session) subject
            elif len(list(root_query.glob('ses-*'))) > 0:
                sub_query = root_query.glob('ses-*')
            else:
                warnings.warn(
                    f'{root_query.name} has no data at {root_query.resolve()}')
                continue

            for ses_q in sub_query:
                if root_query.name == ses_q.name:
                    sub_name = root_query.name
                else:
                    sub_name = f'{root_query.name}_{ses_q.name}'

                if (ses_q / 'func').is_dir():
                    sub_data = {
                        'anat_d': root_query / 'anat',
                        'func_d': ses_q / 'func'
                    }

                    try:
                        sub_data['mask'] = list(sub_data['anat_d'].glob(
                            f'{root_query.name}_space{templates["mask"]}'))[0]
                        sub_data['t1'] = list(sub_data['anat_d'].glob(
                            f'{root_query.name}_space*_desc-preproc_T1w.nii.gz'))[0]
                    except IndexError:
                        warnings.warn(
                        f'{root_query.name}_{ses_q.name} is missing some anatomical data @ {sub_data["anat_d"]}')
                        continue
                    # Look for runs inside this subject
                    run_names = list(set([get_run_name(str(p.name))
                                          for p in sub_data['func_d'].glob('*task-*')]))
                    if len(run_names)==0:
                        warnings.warn(
                            f'{ses_q.name} has no functional data at {sub_data["func_d"]}')
                        continue

                    run_d = {}
                    for run_name in run_names:
                        run_data = {'sub_name': sub_name}
                        try:
                            for fname, fglob in templates.items():
                                run_search = list(
                                    sub_data['func_d'].glob(f'{run_name}{fglob}'))
                                if len(run_search) == 1:
                                    run_data[fname] = run_search[0]
                                else:
                                    # Either more or less than one, skip the whole run
                                    err = (
                                        f'{root_query.name}: run {run_name} is missing {fname} @ {sub_data["func_d"]}')
                                    raise FileNotFoundError
                        # Break only the outer loop by passing the excepting the error.
                        except FileNotFoundError:
                            warnings.warn(err)
                            continue
                        run_d[run_name] = run_data
                        runs[run_name] = run_data
                    sub_data['runs'] = run_d
                    subjects[sub_name] = sub_data
    #TODO: make sure there are any subjects found here and error out if not!

    subject_list = list(subjects.keys())
    run_list = list(runs.keys())

    # Make the json reports
    print('START JSON reports')
    temp = get_template()
    # Empty scrubbing masks can raise warnings here and clutter STDOUT
    # We catch them and ignore them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reports = {sub_name: report_subject(subjects[sub_name], temp) for sub_name in tqdm.tqdm(subject_list)}
    print('DONE JSON reports')

    # Generate group level data
    print('START Group level images')
    t1_group_average = average_image(
        [subjects[sub]['t1'] for sub in subjects.keys()])
    func_group_average = average_image(
        [runs[run]['boldref'] for run in runs.keys()])
    func_group_mask_average = average_image(
        [runs[run]['mask'] for run in runs.keys()])
    # Threshold the average mask at the 95th percentile to make a group mask
    func_group_mask = nib.Nifti1Image((func_group_mask_average.get_fdata() > 0.95).astype(int),
                                      affine=func_group_mask_average.affine,
                                      header=func_group_mask_average.header)
    print('DONE Group level images')

    # Generate group level figures
    print('START Group level montages')
    fig_t1_group_average = make_reg_montage(t1_group_average,
                                            overlay=temp['outline'],
                                            cmap=plt.cm.Greys_r)
    fig_func_group_average = make_reg_montage(func_group_average)
    fig_func_group_mask_average = make_reg_montage(func_group_mask_average)
    fig_func_group_mask = make_reg_montage(func_group_mask)
    fig_template_outline = make_reg_montage(temp['T1'], overlay=temp['outline'],
                                            cmap=plt.cm.Greys_r)
    fig_template = make_reg_montage(temp['T1'], cmap=plt.cm.Greys_r)
    print('DONE Group level montages')

    # Generate js data
    subject_list_str = make_dict_str('listSubject', subject_list)
    run_list_str = make_dict_str('dataRun', run_list)

    fd_part1 = make_list_str('dataFD',
                             ['Run'] + run_list,
                             ['FD_before'] + [str(reports[runs[run_name]['sub_name']]['runs'][run_name]['mean_fd_before'])
                                              for run_name in run_list],
                             ['FD_after'] + [str(reports[runs[run_name]['sub_name']]['runs'][run_name]['mean_fd_after'])
                                             for run_name in run_list]
                             )
    fd_part2 = make_list_str('dataNbVol',
                             ['Run'] + run_list,
                             ['vol_scrubbed'] + [str(reports[runs[run_name]['sub_name']]['runs'][run_name]['n_vol_after'])
                                                 for run_name in run_list],
                             ['vol_ok'] + [str(reports[runs[run_name]['sub_name']]['runs'][run_name]['n_vol_before'])
                                           for run_name in run_list]
                             )
    fd_str = '\n'.join((fd_part1, fd_part2))

    chart_brain_str = make_list_str('dataIntra',
                                    ['Run'] + run_list,
                                    ['corr_target'] + [str(reports[runs[run_name]['sub_name']]['runs'][run_name]['corr_run_ref'])
                                                       for run_name in run_list]
                                    )

    bold_part1 = make_list_str('dataBOLD',
                               ['Subject'] + subject_list,
                               ['corr_target'] + [str(reports[sub]['corr_BOLD_T1'])
                                                  for sub in subject_list]
                               )
    bold_part2 = make_list_str('dataBrain',
                               ['Subject'] + subject_list,
                               ['overlap_brain'] + [str(reports[sub]['ovlp_BOLD_T1'])
                                                    for sub in subject_list]
                               )
    chart_bold_str = '\n'.join((bold_part1, bold_part2))

    anat_part1 = make_list_str('dataT1',
                               ['Subject'] + subject_list,
                               ['corr_target'] + [str(reports[sub]['corr_T1_stereo'])
                                                  for sub in subject_list]
                               )
    anat_part2 = make_list_str('dataOverlapT1',
                               ['Subject'] + subject_list,
                               ['overlap_brain'] + [str(reports[sub]['ovlp_T1_stereo'])
                                                    for sub in subject_list]
                               )
    chart_t1_str = '\n'.join((anat_part1, anat_part2))

    # Generate subject level outputs
    print(f'START PARALLEL processing data to repository @ {report_p}')
    asset_p = report_p / 'assets'
    # Opening a lot of pyplot figures raises a memory warning. We ignore that. 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Parallel(n_jobs=n_cpu) as parallel:
            results = parallel(delayed(process_subject)(subjects[sub_name], asset_p, sub_name, clobber=clobber)
                            for sub_name in subject_list)
    print('Subject processing DONE')

    # Copy pregenerated strings to the repository
    ol = get_report_lookup()
    for sub_name in subject_list:
        sub_d = subjects[sub_name]
        for run_name in sub_d['runs'].keys():
            run_d = sub_d['runs'][run_name]

            # JSON reports too
            run_str = make_run_str(
                reports[sub_name]['runs'][run_name])
            with open(asset_p / ol['dataMotion'].format(run_name), 'w') as f:
                f.write(run_str)

    # TMP solution, legacy pipeline information.
    filesIn_str = '''
                function buildFilesIn (evt) {
                  switch(evt.params.data.id) {
                    case "1":
                      var filesIn = {}

                 break
                };
                return filesIn 
                }
    '''

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

    with open(asset_p / ol['filesIn'], 'w') as f:
        f.write(filesIn_str)

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
    print(f'DONE moving data to repository @ {report_p}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_dir", type=str,
                        help="bids conform directory of the fmriprep outputs")
    parser.add_argument("output_path", type=str,
                        help="Select the path where report should be generated. "
                             "Depending on the mode, this should be either a directory name or a path to a text file.")
    parser.add_argument("-c", "--clobber", type=bool, default=False,
                        help="If set to 'True', existing outputs will be overwritten. Default is 'False'.")
    parser.add_argument("-n", "--ncpu", type=int, default=2,
                        help="Define the number of CPUs to be used.")
    args = parser.parse_args()
    generate_dashboard(pal.Path(args.preproc_dir),
                        pal.Path(args.output_path),
                        args.clobber,
                        args.ncpu)
