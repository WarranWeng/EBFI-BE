import os
import random
import argparse
import glob


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--valid_data_path', default=None)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--valid_num', type=int, default=None)
    parser.add_argument('--portion', type=float, default=None)
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_txt_name', type=str, default='train.txt')
    parser.add_argument('--valid_txt_name', type=str, default='valid.txt')
    flags = parser.parse_args()

    return flags


def write_txt(path: str, data: list):
    with open(path, 'w') as f:
        f.writelines([str(i) + '\n' for i in data])


if __name__ == '__main__':
    flags = get_flags()

    data_path = flags.data_path
    valid_data_path = flags.valid_data_path
    num = flags.num
    valid_num = flags.valid_num
    portion = flags.portion
    mode = flags.mode
    train_txt_name = flags.train_txt_name
    valid_txt_name = flags.valid_txt_name
    seed = flags.seed

    assert os.path.exists(data_path)
    data_paths = sorted(glob.glob(os.path.join(data_path, '*.h5')))
    data_len = len(data_paths)
    if valid_data_path is not None:
        assert os.path.exists(valid_data_path)
        valid_data_paths = sorted(glob.glob(os.path.join(valid_data_path, '*.h5')))
        valid_data_len = len(valid_data_paths)

    if mode == 0:
        if num is None:
            num = data_len 
        assert num > 0 and num <= data_len, f'num must be set > 0 and < data len {data_len}, but got {num}'
        random.seed(seed)
        train_candidates = sorted(random.sample(data_paths, num))
        write_txt(f'datalist/{train_txt_name}', train_candidates)

        print(f'Sample {num} training items from {data_path}')

    elif mode == 1:
        assert num is not None
        assert valid_num is not None
        assert valid_num > 0 and valid_num < data_len
        assert num > 0 and num < data_len
        assert (valid_num + num) > 0 and (valid_num + num) <= data_len
        
        random.seed(seed)
        train_candidates = random.sample(data_paths, num)
        left_candidates = sorted(list(set(data_paths) - set(train_candidates)))
        random.seed(seed)
        valid_candidates = sorted(random.sample(left_candidates, valid_num))

        write_txt(f'datalist/{train_txt_name}', train_candidates)
        write_txt(f'datalist/{valid_txt_name}', valid_candidates)

        print(f'Sample {num} training items from {data_path}')
        print(f'Sample {valid_num} validating items from {data_path}')

    elif mode == 2:
        assert portion is not None
        train_num = int(data_len * portion)
        random.seed(seed)
        train_candidates = random.sample(data_paths, train_num)
        valid_candidates = sorted(list(set(data_paths) - set(train_candidates)))

        write_txt(f'datalist/{train_txt_name}', train_candidates)
        write_txt(f'datalist/{valid_txt_name}', valid_candidates)

        print(f'Sample {train_num} training items from {data_path}')
        print(f'Sample {data_len - train_num} validating items from {data_path}')

    elif mode == 3:
        assert valid_data_path is not None
        assert valid_num is not None
        assert num is not None

        random.seed(seed)
        train_candidates = sorted(random.sample(data_paths, num))
        random.seed(seed)
        valid_candidates = sorted(random.sample(valid_data_paths, valid_num))

        write_txt(f'datalist/{train_txt_name}', train_candidates)
        write_txt(f'datalist/{valid_txt_name}', valid_candidates)

        print(f'Sample {num} training items from {data_path}')
        print(f'Sample {valid_num} validating items from {valid_data_path}')
    
    else:
        raise Exception(f'Invalid mode {mode}')
