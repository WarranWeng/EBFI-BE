import os
from numpy import mean
import torch
import argparse
from tqdm import tqdm
import cv2
import pandas as pd
from collections import defaultdict
import time
from thop import profile
from numpy import inf
import copy
# local modules
from myutils.utils import *
from models.Ours.model_singleframe import EVFIAutoEx
from dataloader.h5dataset import *
from dataloader.h5dataloader import *
from loss import *
from dataloader.encodings import *
from myutils.vis_events.matplotlib_plot_events import *


def init_seeds(seed=0, cuda_deterministic=True):
    print(f'seed:{seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.backends.cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def infer_body(dataloader_config, data_path, models, real_blur,
                    img_path, logger: Logger_yaml,
                                device, vis: event_visualisation, metrics):
    lpips = metrics['lpips']
    ssim = metrics['ssim']
    psnr = metrics['psnr']
    mse = metrics['mse']
    l1 = metrics['l1']
    model = models['model']

    # build dataset
    logger.log_dict(dataloader_config, 'eval_datasetloader_config')
    if real_blur:
        dataloader = InferenceHDF5DataLoaderRealData(data_path, dataloader_config)
    else:
        dataloader = InferenceHDF5DataLoader(data_path, dataloader_config)
    DeblurPretrain = dataloader_config['dataset'].get('DeblurPretrain', False)
    NumFramePerPeriod = dataloader_config['dataset']['NumFramePerPeriod']
    NumPeriodPerLoad = dataloader_config['dataset']['NumPeriodPerLoad']

    BlurryPath = os.path.join(img_path, 'blurry_frame')
    EventPath = os.path.join(img_path, 'event')
    GTPath = os.path.join(img_path, 'gt_frame')
    os.makedirs(BlurryPath, exist_ok=False)
    os.makedirs(EventPath, exist_ok=False)
    os.makedirs(GTPath, exist_ok=False)
    if model is not None:
        RestoredPath = os.path.join(img_path, 'restored_frame')
        os.makedirs(RestoredPath, exist_ok=False)

    metric_step = {
        'psnr': []
    }
    metric_track = MetricTracker([
                                  'mse', 'psnr', 'ssim', 'lpips',
                                  ])

    iL = -1
    iF = -1
    for index, inputs_seq in enumerate(tqdm(dataloader, total=len(dataloader))):
        # if i > 1000:
        #     break
        if not real_blur:
            SeqLatentF = inputs_seq['SeqLatentF'].transpose(0, 1).to(device) # LxBxNumPxNumFx3xHxW 
        SeqBlurryF = inputs_seq['SeqBlurryF'].transpose(0, 1).to(device) # LxBxNumPx3xHxW
        SeqHREv = inputs_seq['SeqHREv'].transpose(0, 1).to(device) # LxBxTBx2xHxW
        RelativeLatentTs = inputs_seq['RelativeLatentTs'].transpose(0, 1).to(device) # LxBxNumPx(NumP*NumF)
        SeqExposureDuty = inputs_seq['SeqExposureDuty'].transpose(0, 1).to(device) # LxBxNumPx1

        if not real_blur:
            L, B, NumP, NumF, C, H, W = SeqLatentF.size() # NumP = 1
        else:
            L, B, NumP, NumF = RelativeLatentTs.size()
        for idxL in range(L):
            iL += 1

            if not real_blur:
                LatentFMul = SeqLatentF[idxL].view(B, NumP*NumF, -1, H, W) # BxNumFx3xHxW
            BlurryF = SeqBlurryF[idxL].squeeze(1) # Bx3xHxW
            HREv = SeqHREv[idxL] # BxTBx2xHxW
            TsMul = RelativeLatentTs[idxL].squeeze(1) # BxNumF
            ExposureDuty = SeqExposureDuty[idxL].squeeze(1) # Bx1
 
            NumI = TsMul.size(-1) # NumF
            for i in range(NumI):
                iF += 1

                if not real_blur:
                    LatentF = LatentFMul[:, i] # Bx3xHxW
                Ts = TsMul[:, [i]] # Bx1

                pred_sharp = model(
                                Frame=BlurryF.contiguous(), # Bx3xHxW
                                Event=HREv.contiguous(), # BxTBx2xHxW
                                T=Ts.contiguous(), # Bx1
                                GTEx=ExposureDuty.contiguous(), # Bx1
                            )[-1]

                if not real_blur:
                    psnr_value = psnr(pred_sharp, LatentF)
                    mse_value = mse(pred_sharp, LatentF)
                    ssim_value = ssim(pred_sharp, LatentF)
                    lpips_value = lpips(pred_sharp, LatentF)
                    metric_step['psnr'].append(psnr_value.item())
                    metric_track.update('mse', mse_value.item())
                    metric_track.update('psnr', psnr_value.item())
                    metric_track.update('ssim', ssim_value.item())
                    metric_track.update('lpips', lpips_value.item())

                # images save
                if not real_blur:
                    vis.plot_frame((LatentF[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'), is_save=True, 
                                            path=os.path.join(GTPath, '{:09d}_{}.png'.format(iF, iL)))
                vis.plot_frame((pred_sharp[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'), is_save=True, 
                                        path=os.path.join(RestoredPath, '{:09d}_{}.png'.format(iF, iL)))
            vis.plot_frame((BlurryF[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'), is_save=True, 
                                    path=os.path.join(BlurryPath, f'{iL:09d}.png'))
            for idx in range(HREv.size(1)):
                vis.plot_event_cnt(HREv[0, idx].cpu().numpy().transpose(1, 2, 0), is_save=True, 
                                    path=os.path.join(EventPath, '{}_TB{:09d}.png'.format(iL, idx)), 
                                    color_scheme="blue_red", is_black_background=False, is_norm=True)

    result = metric_track.result()
    result_step = metric_step
    out = {
        'result': result,
        'result_step': result_step,
    }
    logger.log_dict(result, 'evaluation results')
    logger.log_dict(result_step, 'evaluation step results')

    return out


def load_model(model_path, device):
    if model_path is not None:
        assert os.path.isfile(model_path)
        cpt = torch.load(model_path, map_location=device)
        print(f'Load model from: {model_path}...')

        # build model
        config = cpt['config']
        model = eval(config['model']['name'])(**config['model']['args'])
        model.load_state_dict(cpt['model']['states'])
        model.to(device)
        model.eval()
        print(model)
    else:
        model = None

    return model


def process(data: list):
    minL = inf
    for item in data:
        if minL > len(item):
            minL = len(item)

    print(f'min L: {minL}')

    outList = []
    for i in range(minL):
        tempList = []
        for item in data:
            tempList.append(item[i])
        outList.append(float(mean(tempList)))

    return outList


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    
    parser.add_argument('--data_list', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_path', type=str, required=True)
    
    parser.add_argument('--scale', type=int, default=None)
    parser.add_argument('--ori_scale', type=str, default=None)
    parser.add_argument('--time_bins', type=int, default=None)
    parser.add_argument('--interp_num', type=int, default=None)
    parser.add_argument('--num_frame_per_period', type=int, default=None)
    parser.add_argument('--num_frame_per_blurry', type=int, default=None)
    parser.add_argument('--num_period_per_seq', type=int, default=None)
    parser.add_argument('--sliding_window_seq', type=int, default=None)
    parser.add_argument('--num_period_per_load', type=int, default=None)
    parser.add_argument('--sliding_window_load', type=int, default=None)
    parser.add_argument('--exposure_method', type=str, default=None)
    parser.add_argument('--exposure_time', type=str, default=None)
    parser.add_argument('--deblur_pretrain', default=False, action='store_true')
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--noise_enabled', default=True, action='store_false') # false for real-world data
    parser.add_argument('--center_crop_size', type=int, nargs='+', default=None)
    parser.add_argument('--real_blur', default=False, action='store_true')
    # parser.add_argument('--need_gt_frame', default=False, action='store_true')

    return parser.parse_args()


@torch.no_grad()
def main():
    SCALE = 4
    ORI_SCALE = 'down4'
    TIME_BINS = 1
    NumFramePerPeriod = 16
    NumFramePerBlurry = 9
    NumPeriodPerSeq = 2
    SlidingWindowSeq = 2
    NumPeriodPerLoad = 2
    SlidingWindowLoad = 2
    ExposureMethod = 'Fixed'
    ExposureTime = None
    DeblurPretrain = False

    dataloader_config = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': False,
        'dataset': {
            'scale': SCALE,
            'ori_scale': ORI_SCALE,
            'time_bins': TIME_BINS,
            'interp_num': 16,
            'NumFramePerPeriod': NumFramePerPeriod,
            'NumFramePerBlurry': NumFramePerBlurry,
            'NumPeriodPerSeq': NumPeriodPerSeq,
            'SlidingWindowSeq': SlidingWindowSeq,
            'NumPeriodPerLoad': NumPeriodPerLoad,
            'SlidingWindowLoad': SlidingWindowLoad,
            'ExposureMethod': ExposureMethod,
            'ExposureTime': ExposureTime,
            'DeblurPretrain': DeblurPretrain,
            'data_augment': {
                'enabled': True,
                'augment': ['RandomCrop', 'CenterCrop', "HorizontalFlip", "VertivcalFlip", 'Noise', 'HotPixel'],
                'random_crop': {
                    'enabled': False,
                    'size': [128, 128],
                },
                'center_crop': {
                    'enabled': False,
                    'size': [128, 128],
                },
                'flip': {
                    'enabled': False,
                    'horizontal_prob': 0.5,
                    'vertical_prob': 0.5,
                },
                'noise': {
                    'enabled': True,
                    'noise_std': 1.0,
                    'noise_fraction': 0.05,
                },
                'hot_pixel': {
                    'enabled': True,
                    'hot_pixel_std': 2.0,
                    'hot_pixel_fraction': 0.001,
                },
            },
        },
    }

    flags = get_flags()    

    model_path = flags.model_path

    data_list = flags.data_list
    device = torch.device(flags.device)
    output_path = flags.output_path
    os.makedirs(output_path, exist_ok=True)

    scale = flags.scale
    ori_scale = flags.ori_scale
    time_bins = flags.time_bins
    interp_num = flags.interp_num
    num_frame_per_period = flags.num_frame_per_period
    num_frame_per_blurry = flags.num_frame_per_blurry
    num_period_per_seq = flags.num_period_per_seq
    sliding_window_seq = flags.sliding_window_seq
    num_period_per_load = flags.num_period_per_load
    sliding_window_load = flags.sliding_window_load
    exposure_method = flags.exposure_method
    exposure_time = flags.exposure_time
    deblur_pretrain = flags.deblur_pretrain
    noise_std = flags.noise_std
    noise_enabled = flags.noise_enabled
    center_crop_size = flags.center_crop_size
    real_blur = flags.real_blur

    if scale is not None:
        dataloader_config['dataset']['scale'] = scale
    if ori_scale is not None:
        dataloader_config['dataset']['ori_scale'] = ori_scale
    if time_bins is not None:
        dataloader_config['dataset']['time_bins'] = time_bins
    if interp_num is not None:
        dataloader_config['dataset']['interp_num'] = interp_num
    if num_frame_per_period is not None:
        dataloader_config['dataset']['NumFramePerPeriod'] = num_frame_per_period
    if num_frame_per_blurry is not None:
        dataloader_config['dataset']['NumFramePerBlurry'] = num_frame_per_blurry
    if num_period_per_seq is not None:
        dataloader_config['dataset']['NumPeriodPerSeq'] = num_period_per_seq
    if sliding_window_seq is not None:
        dataloader_config['dataset']['SlidingWindowSeq'] = sliding_window_seq
    if num_period_per_load is not None:
        dataloader_config['dataset']['NumPeriodPerLoad'] = num_period_per_load
    if sliding_window_load is not None:
        dataloader_config['dataset']['SlidingWindowLoad'] = sliding_window_load
    if exposure_method is not None:
        dataloader_config['dataset']['ExposureMethod'] = exposure_method
    if exposure_time is not None:
        dataloader_config['dataset']['ExposureTime'] = exposure_time
    if deblur_pretrain is not None:
        dataloader_config['dataset']['DeblurPretrain'] = deblur_pretrain
    if noise_std is not None:
        dataloader_config['dataset']['data_augment']['noise'].update({'enabled': True, 'noise_std': noise_std, 'noise_fraction': 0.05})
    if noise_enabled is not None:
        dataloader_config['dataset']['data_augment']['noise']['enabled'] = noise_enabled
        dataloader_config['dataset']['data_augment']['hot_pixel']['enabled'] = noise_enabled
    if center_crop_size is not None:
        dataloader_config['dataset']['data_augment']['center_crop'].update({'enabled': True, 'size': center_crop_size})
    
    print(dataloader_config)

    vis = event_visualisation()
    metrics = {
        'lpips': perceptual_loss(net='alex'),
        'ssim': ssim_loss(),
        'psnr': psnr_loss(),
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
    }

    model = load_model(model_path, device)
    models = {
        'model': model,
    }
    model_paths = [
        model_path, 
                        ]
    data_list = pd.read_csv(data_list, header=None).values.flatten().tolist()
    logger_all = Logger_yaml(os.path.join(output_path, 'inference_all.yml'))
    logger_all.log_info(f'inference {model_paths} \n on {data_list}')
    logger_all_step = Logger_yaml(os.path.join(output_path, 'inference_all_step.yml'))
    logger_all_step.log_info(f'inference {model_paths} \n on {data_list}')
    results = []
    for data_path in tqdm(data_list):
        print(f'processing {data_path}')
        data_name = os.path.basename(data_path)
        root_path = os.path.join(output_path, data_name)
        img_path = os.path.join(root_path, 'img')
        os.makedirs(root_path, exist_ok=False)
        os.makedirs(img_path, exist_ok=False) 
        logger = Logger_yaml(os.path.join(root_path, 'inference.yml'))
        logger.log_info(f'inference {model_paths} on {data_path}')
        args = {
            'dataloader_config': dataloader_config,
            'data_path': data_path,
            'models': models,
            'img_path': img_path,
            'logger': logger,
            'device': device,
            'vis': vis,
            'metrics': metrics,
            'real_blur':real_blur,
        }
        result = infer_body(**args)
        result['data_name'] = data_name
        results.append(result)

    results_dict = defaultdict(dict)
    results_mean = defaultdict(list)
    results_dict_step = defaultdict(dict)
    results_mean_step = defaultdict(list)
    for entry in results:
        data_name = entry.pop('data_name')
        for k, v in entry['result'].items():
            results_dict[k][data_name] = v
            results_mean[k].append(v)
        for k, v in entry['result_step'].items():
            results_dict_step[k][data_name] = v
            results_mean_step[k].append(v)
    for k, v in results_mean.items():
        results_mean[k] = float(mean(v))
    for k, v in results_mean_step.items():
        results_mean_step[k] = process(v)
    logger_all.log_dict(dict(results_dict), 'breakdown results for each data')
    logger_all.log_dict(dict(results_mean), 'mean results for the whole data')
    logger_all_step.log_dict(dict(results_dict_step), 'breakdown results for each data')
    logger_all_step.log_dict(dict(results_mean_step), 'mean results for the whole data (based on min length)')


if __name__ == '__main__':
    init_seeds(seed=123)
    main()


