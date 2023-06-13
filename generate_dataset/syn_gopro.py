import os
import cv2
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm
import esim_py
import random
import shutil
# local modules
from tools.event_packagers import hdf5_packager_multiscale


config = {
    'Cp_init': 0.1,
    'Cn_init': 0.1,
    'refractory_period': 1e-4,
    'log_eps': 1e-3,
    'use_log':True,
    'CT_range': [0.2, 0.5],
    'max_CT': 0.5,
    'min_CT': 0.2,
    'mu': 1,
    'sigma': 0.1,

    'fps': 240,
}


def write_img(img: np.ndarray, idx: int, imgs_dir: str):
    assert os.path.isdir(imgs_dir)
    path = os.path.join(imgs_dir, "%05d.png" % idx)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, img)


def write_timestamps(timestamps: list, timestamps_filename: str):
    with open(timestamps_filename, 'w') as t_file:
        t_file.writelines([str(t) + '\n' for t in timestamps])


def write_config(config_path):
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value} \n')


def prepare_output_dir(src_dir: str, dest_dir: str):
    # Copy directory structure.
    def ignore_files(directory, files):
        return [f for f in files if os.path.isfile(os.path.join(directory, f))]
    shutil.copytree(src_dir, dest_dir, ignore=ignore_files)


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_path', default='/path/to/data')
    parser.add_argument('--path_to_h5', default='/path/to/output')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    flags = get_flags()

    root_data_path = flags.root_data_path
    path_to_h5 = flags.path_to_h5

    if not os.path.exists(path_to_h5):
        os.makedirs(path_to_h5, exist_ok=False)

    data_dirs = sorted([os.path.join(root_data_path, dir) for dir in os.listdir(root_data_path)])
    resolution = None
    fps = config['fps']

    esim = esim_py.EventSimulator(config['Cp_init'], 
                                  config['Cn_init'], 
                                  config['refractory_period'], 
                                  config['log_eps'], 
                                  config['use_log'])

    CT = []
    for data_dir in tqdm(data_dirs):
        print(f'\n processing {data_dir}')
        path_to_rgb = os.path.join(data_dir, 'rgb')
        path_to_mono = os.path.join(data_dir, 'mono')
        path_timestamps = os.path.join(data_dir, 'timestamps.txt') 

        prex = os.path.splitext(os.listdir(path_to_rgb)[0])[-1]
        rgb_imgs = sorted(glob(os.path.join(path_to_rgb, '*' + prex)))
        basename = os.path.basename(data_dir)
        h5_filename = basename + '.h5'
        ep = hdf5_packager_multiscale(os.path.join(path_to_h5, h5_filename))

        # images writing
        for idx, img_path in enumerate(rgb_imgs):
            ori_img = cv2.imread(img_path, 1)
            if ori_img is None and idx == 0:
                print('Images is None! Donot write images!')
                break
            if resolution is None and idx == 0:
                resolution = ori_img.shape[:-1]
            ep.package_image('ori', ori_img, idx/fps, idx)

        # simulate events
        print('Events simulating and writing!')
        Cp = random.uniform(config['CT_range'][0], config['CT_range'][1])
        Cn = random.gauss(config['mu'], config['sigma']) * Cp
        Cp = min(max(Cp, config['min_CT']), config['max_CT'])
        Cn = min(max(Cn, config['min_CT']), config['max_CT'])
        msg = f'{data_dir}:Cp={Cp}, Cn={Cn}'
        CT.append(msg)
        print(f'{msg}')
        esim.setParameters(Cp, Cn, config['refractory_period'], config['log_eps'], config['use_log'])
        events = esim.generateFromFolder(path_to_mono, path_timestamps) # x y t p
        ep.package_events('ori', events[:, 0], events[:, 1], events[:, 2], events[:, 3])
        ep.add_event_indices()
        ep.add_data(resolution)
        resolution = None

    path_to_config = os.path.join(path_to_h5, 'config')
    os.makedirs(path_to_config)
    write_config(os.path.join(path_to_config, 'config.txt'))
    write_timestamps(CT, os.path.join(path_to_config, 'ct.txt'))
    print('all {} files are done!'.format(len(data_dirs)))

