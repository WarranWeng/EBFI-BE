import torch
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
import cv2
# Local modules
from tools.event_packagers import *


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', type=str, default='/path/to/data')
    parser.add_argument('--path_to_h5', type=str, default='/path/to/output')
    args = parser.parse_args()
    return args


def main():
    flags = get_flags()

    path_to_data = flags.path_to_data
    path_to_h5 = flags.path_to_h5
    os.makedirs(path_to_h5, exist_ok=True)

    sequences = [os.path.join(path_to_data, dir) for dir in os.listdir(path_to_data)]
    print(f'all sequences: {sequences}')

    for sequence in sequences:
        print(f'Processing sequence: {sequence}')
        events_file = os.path.join(sequence, 'events', 'events.npz')
        imgs_dir = sorted(glob(os.path.join(sequence, 'frames', '*.png')))
        timestamps = pd.read_csv(os.path.join(sequence, 'frame_time.txt'), header=None).values.flatten().tolist()
        num_imgs = len(imgs_dir)

        h5_dir = os.path.join(path_to_h5, f'{os.path.basename(sequence)}.h5')
        ep = hdf5_packager_multiscale(h5_dir)

        print('Adding events')
        events = np.load(events_file)['data']
        x, y, t, p = events['x'].astype(np.int16), events['y'].astype(np.int16), events['timestamp'].astype(np.float64), events['polarity'].astype(np.int8)
        p[p==0] = -1
        t = t / 1e6 # microsecs to seconds
        ep.package_events('ori', x, y, t, p)

        print('Adding images')
        for idx in range(num_imgs):
            img = cv2.imread(imgs_dir[idx], 1)
            timestamp = int(timestamps[idx].split(' ')[1]) / 1e6 # microsecs to seconds
            resolution = img.shape[0:2]
            ep.package_image('ori', img, timestamp, idx)

        ep.add_event_indices()
        ep.add_data(resolution)

    print('all {} files are done!'.format(len(sequences)))

if __name__ == '__main__':
    main()