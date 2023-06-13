import torch
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.optim import *
import torch.nn.functional as f
from torch.nn.parallel import DistributedDataParallel as ddp
import collections
from torch.optim.lr_scheduler import *
from numpy import inf
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
from contextlib import nullcontext
# local modules
from config.parser import YAMLParser
from dataloader.h5dataloader import HDF5DataLoader
from dataloader.h5dataloader import HDF5DataLoaderFast
from loss import *
from models.Ours.model_singleframe import EVFIAutoEx
from myutils.utils import *
from logger import *
from myutils.timers import Timer
from myutils.vis_events.matplotlib_plot_events import *
from dataloader.encodings import *


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


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        raise Exception('Only support DDP')

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    dist_url = 'env://'
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    dist.barrier()
    setup_for_distributed(rank == 0)

    return gpu


class Trainer:
    def __init__(self, args):
        # config parser
        self.config_parser = args['config_parser']
        # dataloader
        self.train_dataloader = args['train_dataloader']
        self.valid_dataloader = args['valid_dataloader']
        # models
        self.model = args['model']
        # loss fts
        self.loss = args['loss']
        # optimizers
        self.optimizer = args['optimizer']
        # lr scheduler
        self.lr_scheduler = args['lr_scheduler']
        # metadata
        self.logger = args['logger']
        self.device = args['device']

        self.monitor = self.config_parser['trainer'].get('monitor', 'off')
        self.checkpoint_dir = self.config_parser.save_dir
        # self.inp_sensor_resolution = self.train_dataloader.dataset.datasets[0].inp_sensor_resolution
        # self.gt_sensor_resolution = self.train_dataloader.dataset.datasets[0].gt_sensor_resolution
        self.do_validation = self.config_parser['trainer']['do_validation'] and self.valid_dataloader is not None
        self.NumFramePerPeriod = self.config_parser['train_dataloader']['dataset']['NumFramePerPeriod']
        self.NumPeriodPerLoad = self.config_parser['train_dataloader']['dataset']['NumPeriodPerLoad']
        self.DeblurPretrain = self.config_parser['train_dataloader']['dataset'].get('DeblurPretrain', False)
        self.LoadPretrainEX = self.config_parser['model']['args'].get('LoadPretrainEX', False)
        self.DetailEnabled = self.config_parser['model']['args'].get('DetailEnabled', False)
        self.UseGTEx = self.config_parser['model']['args'].get('UseGTEx', False)
        self.FixEx = self.config_parser['model']['args'].get('FixEx', False)

        # training mode setting
        is_epoch_based_train = self.config_parser['trainer']['epoch_based_train']['enabled']
        is_iteration_based_train = self.config_parser['trainer']['iteration_based_train']['enabled']
        if (is_epoch_based_train and is_iteration_based_train) or \
            (not is_epoch_based_train and not is_iteration_based_train):
            raise Exception('Please set correct training mode in the configuration file!')
        elif is_epoch_based_train:
            # metadata for epoch-based training
            if dist.get_rank() == 0:
                self.logger.info('Apply epoch-based training...')
            self.training_mode = 'epoch_based_train'
            self.epochs = self.config_parser['trainer']['epoch_based_train']['epochs']
            self.start_epoch = 1
            self.len_epoch = len(self.train_dataloader)
            self.save_period = self.config_parser['trainer']['epoch_based_train']['save_period']
            self.train_log_step = max(len(self.train_dataloader) \
                                     // self.config_parser['trainer']['epoch_based_train']['train_log_step'], 1)
            self.valid_log_step = max(len(self.valid_dataloader) \
                                     // self.config_parser['trainer']['epoch_based_train']['valid_log_step'], 1)
            self.valid_step = self.config_parser['trainer']['epoch_based_train']['valid_step']
        elif is_iteration_based_train:
            # metadata for epoch-based training
            if dist.get_rank() == 0:
                self.logger.info('Apply iteration-based training...')
            self.training_mode = 'iteration_based_train'
            self.iterations = int(self.config_parser['trainer']['iteration_based_train']['iterations'])
            self.len_epoch = len(self.train_dataloader)
            self.save_period = self.config_parser['trainer']['iteration_based_train']['save_period']
            self.train_log_step = self.config_parser['trainer']['iteration_based_train']['train_log_step']
            self.valid_log_step = self.config_parser['trainer']['iteration_based_train']['valid_log_step']
            self.valid_step = self.config_parser['trainer']['iteration_based_train']['valid_step']
            self.lr_change_rate = self.config_parser['trainer']['iteration_based_train']['lr_change_rate']

        # visualization tool
        self.vis = event_visualisation()

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.config_parser['trainer'].get('early_stop', inf)

        # setup visualization writer instance      
        if dist.get_rank() == 0:          
            self.writer = TensorboardWriter(self.config_parser.log_dir, self.logger, self.config_parser['trainer']['tensorboard'])
        else:
            self.writer = None

        # setup metric tracker
        train_mt_keys = ['train_loss']
        valid_mt_keys = ['valid_loss']
        self.train_metrics = MetricTracker(train_mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(valid_mt_keys, writer=self.writer)

        # resume checkpoint
        if self.config_parser.args.resume is not None:
            self._resume_checkpoint()
        
        if not self.UseGTEx and not self.FixEx:
            if self.LoadPretrainEX:
                if dist.get_rank() == 0:
                        self.logger.info('Load pretrained ExposureDecision!')
                self.model.module.LoadExposureDecision()

    def train(self):
        """
        Full training logic
        """
        if self.training_mode == 'epoch_based_train':
            self.epoch_based_training()
        elif self.training_mode == 'iteration_based_train':
            self.iteration_based_training()
        else:
            raise Exception('Incorrect training config!')

    def iteration_based_training(self):
        """
        Iteration-based training logic
        """
        valid_stamp = 1
        epoch = 0
        self.train_iter_idx = 0
        self.valid_iter_idx = 0
        accu_step = self.config_parser['trainer']['accu_step']
        accu_count = 0
        stop_training = False
        complete_training = False
        # lamda = 0.01
        self.not_improved_count = 0

        self.model.train()
        self.train_metrics.reset()

        while True:
            if stop_training or complete_training:
                break
            self.train_dataloader.sampler.set_epoch(epoch)

            for idx_epoch, inputs_seq in enumerate(self.train_dataloader):
                if stop_training or complete_training:
                    break
                self.optimizer.zero_grad()

                SeqLatentF = inputs_seq['SeqLatentF'].transpose(0, 1).to(self.device) # LxBxNumPxNumFx3xHxW 
                SeqBlurryF = inputs_seq['SeqBlurryF'].transpose(0, 1).to(self.device) # LxBxNumPx3xHxW
                # SeqNeighborF = inputs_seq['SeqNeighborF'].transpose(0, 1).to(self.device) # LxBxNumPxNumFx2x3xHxW
                SeqHREv = inputs_seq['SeqHREv'].transpose(0, 1).to(self.device) # LxBxTBx2xHxW
                RelativeLatentTs = inputs_seq['RelativeLatentTs'].transpose(0, 1).to(self.device) # LxBxNumPx(NumP*NumF)
                SeqExposureDuty = inputs_seq['SeqExposureDuty'].transpose(0, 1).to(self.device) # LxBxNumPx1
                
                L, B, NumP, NumF, C, H, W = SeqLatentF.size() # NumP = 1
                for idxL in range(L):
                    if stop_training or complete_training:
                        break
                    LatentFMul = SeqLatentF[idxL].view(B, NumP*NumF, -1, H, W) # BxNumFx3xHxW 
                    BlurryF = SeqBlurryF[idxL].squeeze(1) # Bx3xHxW
                    # NeighborFMul = SeqNeighborF[idxL].view(B, NumP*NumF, 2, 3, H, W).squeeze(1) # BxNumFx2x3xHxW
                    HREv = SeqHREv[idxL] # BxTBx2xHxW
                    TsMul = RelativeLatentTs[idxL].squeeze(1) # BxNumF
                    ExposureDuty = SeqExposureDuty[idxL].squeeze(1) # Bx1

                    NumI = TsMul.size(-1) # NumP*NumF
                    for i in range(NumI):
                        LatentF = LatentFMul[:, i] # Bx3xHxW
                        # NeighborF = NeighborFMul[:, i] # Bx2x3xHxW
                        Ts = TsMul[:, [i]] # Bx1

                        with self.model.no_sync():
                            SharpPre, Sharp = self.model(
                                Frame=BlurryF.contiguous(), # Bx3xHxW
                                Event=HREv.contiguous(), # BxTBx2xHxW
                                T=Ts.contiguous(), # Bx1
                                GTEx=ExposureDuty.contiguous(), # Bx1
                            )

                            if self.DetailEnabled:

                                if self.train_iter_idx < 10e3:
                                    Loss = ( 0.1* (self.loss['Lap'](Sharp, LatentF) + self.loss['census'](Sharp, LatentF)) + \
                                            (self.loss['Lap'](SharpPre, LatentF) + self.loss['census'](SharpPre, LatentF)) ) / accu_step
                                else:
                                    Loss = ( (self.loss['Lap'](Sharp, LatentF) + self.loss['census'](Sharp, LatentF)) + \
                                            0.1* (self.loss['Lap'](SharpPre, LatentF) + self.loss['census'](SharpPre, LatentF)) ) / accu_step

                            else:
                                Loss = (self.loss['Lap'](Sharp, LatentF) + self.loss['census'](Sharp, LatentF)) / accu_step
                                
                            Loss.backward()

                            loss = Loss
                        accu_count += 1
    
                        if accu_count % accu_step == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # reduce losses over all GPUs for logging purposes
                            ReducedLoss = reduce_tensor(loss)
                            # setup log info
                            if dist.get_rank() == 0:  
                                learning_rate = self.lr_scheduler.get_last_lr()[0]
                                self.writer.set_step(self.train_iter_idx)
                                self.train_metrics.update('train_loss', ReducedLoss.item())
                                self.writer.writer.add_scalar(f'learning rate', learning_rate, global_step=self.train_iter_idx)
                                if self.train_iter_idx % self.train_log_step == 0:
                                    msg = 'Train Epoch: {} {} Iteration: {} {}'.format(epoch+1, self._progress(idx_epoch, self.train_dataloader, is_train=False), \
                                                                                       self.train_iter_idx, self._progress(self.train_iter_idx, self.train_dataloader, is_train=True))
                                    msg += ' {}: {:.4e}'.format('train_loss', ReducedLoss.item())
                                    msg += ' {}: {:.4e}'.format('learning rate', learning_rate)
                                    self.logger.info(msg)
                                # visualize
                                if self.config_parser['trainer']['vis']['enabled']:
                                    with torch.no_grad():
                                        train_vis_step = self.config_parser['trainer']['vis']['train_img_writer_num']
                                        if self.train_iter_idx % train_vis_step == 0:
                                            self.writer.writer.add_image('train_HR_events', 
                                                                         self.vis.plot_event_cnt(HREv.sum(1)[0].cpu().numpy().transpose(1, 2, 0), is_save=False, is_black_background=True), 
                                                                         global_step=self.train_iter_idx, dataformats='HWC')
                                            self.writer.writer.add_image('train_blurry_frame', 
                                                                         (BlurryF[0].cpu().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 
                                                                         global_step=self.train_iter_idx, dataformats='HWC')
                                            self.writer.writer.add_image('train_sharp_frame', 
                                                                         (Sharp[0].clamp(0, 1).cpu().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 
                                                                         global_step=self.train_iter_idx, dataformats='HWC')
                                            self.writer.writer.add_image('train_gt_frame', 
                                                                         (LatentF[0].cpu().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 
                                                                         global_step=self.train_iter_idx, dataformats='HWC')
                            # do validation
                            best = False
                            if self.do_validation:
                                if self.train_iter_idx % self.valid_step == 0 and self.train_iter_idx != 0:
                                    with torch.no_grad():
                                        val_log = self._valid(valid_stamp)
                                        if dist.get_rank() == 0:
                                            # plot stamp train & valid logs
                                            for key, value in val_log.items():
                                                self.writer.writer.add_scalar(f'stamp_{key}', value, global_step=valid_stamp)
                                            self.writer.writer.add_scalar(f'stamp_train_loss', ReducedLoss.item(), global_step=valid_stamp)
                                            log = {'Valid stamp': valid_stamp}
                                            log.update(val_log)
                                            for key, value in log.items():
                                                self.logger.info('    {:25s}: {}'.format(str(key), value))
                                        # evaluate model performance
                                        stop_training, best = self.eval_model_performance(val_log)
                                        if stop_training:
                                            break
                                        valid_stamp += 1
                                self.model.train()
                            # save model
                            if dist.get_rank() == 0:
                                if (self.train_iter_idx % self.save_period == 0 and self.train_iter_idx != 0) or best:
                                    self._save_checkpoint(self.train_iter_idx, save_best=best)
                            # change learning rate
                            if self.lr_scheduler is not None:
                                if self.train_iter_idx % self.lr_change_rate == 0 and self.train_iter_idx != 0 \
                                                and self.lr_scheduler.get_last_lr()[0] >= self.config_parser['trainer']['lr_min']:
                                    self.lr_scheduler.step()
                            # logits for stopping training
                            if self.train_iter_idx + 1 == self.iterations:
                                if dist.get_rank() == 0:
                                    self.logger.info('Training completes!')
                                complete_training = True
                                break
                            
                            # sync all processes
                            dist.barrier()
                            self.train_iter_idx += 1

            epoch += 1

    def epoch_based_training(self):
        """
        Epoch-based training logic
        """
        self.not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs+1):
            self.train_dataloader.sampler.set_epoch(epoch)

            with Timer('Time of training one epoch', self.logger):
                epoch_result = self._train_epoch(epoch)

            # plot epoch average statics
            if dist.get_rank() == 0:
                for key, value in epoch_result.items():
                    self.writer.writer.add_scalar(f'epoch_{key}', value, global_step=epoch)
                # save log informations into log dict
                log = {'epoch': epoch}
                log.update(epoch_result)
                # print log informations to the screen
                for key, value in log.items():
                    self.logger.info('    {:25s}: {}'.format(str(key), value))

            # evaluate model performance
            stop_training, best = self.eval_model_performance(epoch_result)
            if stop_training:
                break

            # save model
            if dist.get_rank() == 0:
                if epoch % self.save_period == 0 or best:
                    self._save_checkpoint(epoch, save_best=best)

            # sync all processes
            dist.barrier()

        # complete training
        if dist.get_rank() == 0:
            self.logger.info('Training completes!')

    def eval_model_performance(self, log):
        """
        Evaluate model performance according to configured metric
        log: log includes validation metric
        """
        if dist.get_rank() == 0:
            if self.monitor == 'off':
                self.logger.info('Please set the correct metric to evaluate model!')
            else:
                self.logger.info(f'Evaluate current model using metric "{self.mnt_metric}", and save the current best model...')
                self.logger.info("Last best validation performance: {}".format(self.mnt_best))

        best = False
        is_KeyError = False
        stop_training = False

        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                           (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                is_KeyError = False
            except KeyError:
                if dist.get_rank() == 0:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Ignore this stamp where using this metric to monitor.".format(self.mnt_metric))
                is_KeyError = True
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
                best = True
            elif not is_KeyError:
                self.not_improved_count += 1

            if self.not_improved_count > self.early_stop:
                if dist.get_rank() == 0:
                    self.logger.info("Validation performance didn\'t improve for {} stamps. "
                                     "Training stops.".format(self.early_stop))
                    
                stop_training = True

        return stop_training, best

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        self.train_metrics.reset()
        for batch_idx, inputs in enumerate(self.train_dataloader):
            # if isinstance(self.model, ddp):
            #     self.model.module.reset_states()    
            # else:
            #     self.model.reset_states()

            self.optimizer.zero_grad()

            # lr forward pass
            inp = ...
            pred_logits, gt_logits, gt_pol, pred_sparse = self.model(inp)

            # loss and backward pass
            num_layers = len(pred_logits)
            bce_loss = 0
            for pred_logit, gt_logit in zip(pred_logits, gt_logits):
                curr_bce_loss = self.loss['bce'](pred_logit.F.squeeze(), gt_logit.type(pred_logit.dtype))
                bce_loss += curr_bce_loss 
            bce_loss = bce_loss / num_layers
            mse_loss = self.loss['mse'](pred_sparse.F.squeeze(), gt_pol)
            loss = bce_loss + mse_loss
            loss.backward()
            self.optimizer.step()

            # reduce losses over all GPUs for logging purposes
            reduced_bce_loss = reduce_tensor(bce_loss)
            reduced_mse_loss = reduce_tensor(mse_loss)
            reduced_loss = reduce_tensor(loss)

            # setup log info
            if dist.get_rank() == 0:  
                log_step = (epoch - 1) * self.len_epoch + batch_idx
                learning_rate = self.lr_scheduler.get_last_lr()[0]
                self.writer.set_step(log_step)
                self.train_metrics.update('train_bce_loss', reduced_bce_loss.item())
                self.train_metrics.update('train_mse_loss', reduced_mse_loss.item())
                self.train_metrics.update('train_loss', reduced_loss.item())
                self.writer.writer.add_scalar(f'learning rate', learning_rate, global_step=log_step)
                if batch_idx % self.train_log_step == 0:
                    msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.train_dataloader, is_train=True))
                    msg += ' {}: {:.4e}'.format('train_bce_loss', reduced_bce_loss.item())
                    msg += ' {}: {:.4e}'.format('train_mse_loss', reduced_mse_loss.item())
                    msg += ' {}: {:.4e}'.format('train_loss', reduced_loss.item())
                    msg += ' {}: {:.4e}'.format('learning rate', learning_rate)
                    self.logger.debug(msg)

                # visualize
                if self.config_parser['trainer']['vis']['enabled']:
                    with torch.no_grad():
                        train_vis_step = self.config_parser['trainer']['vis']['train_img_writer_num']
                        if batch_idx % train_vis_step == 0:
                            e_dict = ...
                            self.writer.writer.add_image('train_inp_events_cnt_white', 
                                                         self.vis.plot_event_cnt(inputs['inp_cnt'][0].numpy().transpose(1, 2, 0), is_save=False, is_black_background=False), 
                                                         global_step=log_step, dataformats='HWC')
                            self.writer.writer.add_image('train_inp_events_cnt_black', 
                                                         self.vis.plot_event_cnt(inputs['inp_cnt'][0].numpy().transpose(1, 2, 0), is_save=False, is_black_background=True), 
                                                         global_step=log_step, dataformats='HWC')
                            self.writer.writer.add_image('train_events_cnt_white', 
                                                         self.vis.plot_event_cnt(e_dict['e_cnt'][0].numpy().transpose(1, 2, 0), is_save=False, is_black_background=False), 
                                                         global_step=log_step, dataformats='HWC')
                            self.writer.writer.add_image('train_events_cnt_black', 
                                                         self.vis.plot_event_cnt(e_dict['e_cnt'][0].numpy().transpose(1, 2, 0), is_save=False, is_black_background=True), 
                                                         global_step=log_step, dataformats='HWC')
                            self.writer.writer.add_image('train_gt_events_cnt_white', 
                                                         self.vis.plot_event_cnt(inputs['gt_cnt'][0].numpy().transpose(1, 2, 0), is_save=False, is_black_background=False), 
                                                         global_step=log_step, dataformats='HWC')
                            self.writer.writer.add_image('train_gt_events_cnt_black', 
                                                         self.vis.plot_event_cnt(inputs['gt_cnt'][0].numpy().transpose(1, 2, 0), is_save=False, is_black_background=True), 
                                                         global_step=log_step, dataformats='HWC')
                            self.writer.writer.add_image('train_gt_frame', 
                                                         (inputs['gt_img'][0].squeeze(0).numpy() * 255).astype('uint8'), 
                                                         global_step=log_step, dataformats='HW')
                            plt.close('all')

            # Must clear cache at regular interval
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # sync all processes
            dist.barrier()

        # only main process has non-zero train_log
        train_log = self.train_metrics.result()

        # do validation
        if self.do_validation:
            if epoch % self.valid_step == 0:
                with torch.no_grad():
                    val_log = self._valid(epoch)
                    train_log.update(val_log)

        # change learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return train_log

    def _valid(self, stamp):
        """
        Validate after training an epoch or several iterations

        :param stamp: the timestamp for validation,
                      epoch-based training -> epoch; iteration-based training -> valid_stamp
        :return: A log that contains information about validation
        """
        self.logger.debug('validation')

        self.model.eval()

        self.valid_metrics.reset()
        for batch_idx, inputs_seq in enumerate(self.valid_dataloader):
            SeqLatentF = inputs_seq['SeqLatentF'].transpose(0, 1).to(self.device) # LxBxNumPxNumFx3xHxW 
            SeqBlurryF = inputs_seq['SeqBlurryF'].transpose(0, 1).to(self.device) # LxBxNumPx3xHxW
            SeqHREv = inputs_seq['SeqHREv'].transpose(0, 1).to(self.device) # LxBxTBx2xHxW
            RelativeLatentTs = inputs_seq['RelativeLatentTs'].transpose(0, 1).to(self.device) # LxBxNumPx(NumP*NumF)
            SeqExposureDuty = inputs_seq['SeqExposureDuty'].transpose(0, 1).to(self.device) # LxBxNumPx1
            
            L, B, NumP, NumF, C, H, W = SeqLatentF.size() # NumP = 1
            for idxL in range(L):
                LatentFMul = SeqLatentF[idxL].view(B, NumP*NumF, -1, H, W) # BxNumFx3xHxW 
                BlurryF = SeqBlurryF[idxL].squeeze(1) # Bx3xHxW
                HREv = SeqHREv[idxL] # BxTBx2xHxW
                TsMul = RelativeLatentTs[idxL].squeeze(1) # BxNumF
                ExposureDuty = SeqExposureDuty[idxL].squeeze(1) # Bx1

                loss = 0

                NumI = TsMul.size(-1) # NumP*NumF
                for i in range(NumI):
                    LatentF = LatentFMul[:, i] # Bx3xHxW
                    Ts = TsMul[:, [i]] # Bx1

                    SharpPre, Sharp = self.model(
                            Frame=BlurryF.contiguous(), # Bx3xHxW
                            Event=HREv.contiguous(), # BxTBx2xHxW
                            T=Ts.contiguous(), # Bx1
                            GTEx=ExposureDuty.contiguous(), # Bx1
                        )

                    # loss += (-self.loss['psnr'](Sharp, LatentF)) / NumI
                    loss += (self.loss['CB'](Sharp, LatentF)) / NumI

                # reduce losses over all GPUs for logging purposes
                ReducedLoss = reduce_tensor(loss)
                # setup log info
                if dist.get_rank() == 0:  
                    self.writer.set_step(self.valid_iter_idx, 'valid')
                    self.valid_metrics.update('valid_loss', ReducedLoss.item())
                    if self.valid_iter_idx % self.valid_log_step == 0:
                        msg = 'Valid timestamp: {} {}'.format(stamp, self._progress(batch_idx, self.valid_dataloader, is_train=False))
                        msg += ' {}: {:.4e}'.format('valid_loss', ReducedLoss.item())
                        self.logger.debug(msg)
                    # visualize
                    if self.config_parser['trainer']['vis']['enabled']:
                        with torch.no_grad():
                            valid_vis_step = self.config_parser['trainer']['vis']['valid_img_writer_num']
                            if self.valid_iter_idx % valid_vis_step == 0:
                                self.writer.writer.add_image('valid_HR_events', 
                                                             self.vis.plot_event_cnt(HREv.sum(1)[0].cpu().numpy().transpose(1, 2, 0), is_save=False, is_black_background=True), 
                                                             global_step=self.valid_iter_idx, dataformats='HWC')
                                self.writer.writer.add_image('valid_blurry_frame', 
                                                             (BlurryF[0].cpu().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 
                                                             global_step=self.valid_iter_idx, dataformats='HWC')
                                self.writer.writer.add_image('valid_sharp_frame', 
                                                             (Sharp[0].clamp(0, 1).cpu().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 
                                                             global_step=self.valid_iter_idx, dataformats='HWC')
                                self.writer.writer.add_image('valid_gt_frame', 
                                                             (LatentF[0].cpu().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 
                                                             global_step=self.valid_iter_idx, dataformats='HWC')
                self.valid_iter_idx += 1

        return self.valid_metrics.result()

    def _save_checkpoint(self, idx, save_best=False):
        """
        Saving checkpoints

        :param idx: epoch-based training -> epoch; iteration-based training -> iteration
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            # model and optimizer states:
            'model': {
                'name': self.config_parser['model']['name'],
                'states': self.model.module.state_dict()
            },
            'lr_scheduler': {
                'name': self.config_parser['lr_scheduler']['name'],
                'states': self.lr_scheduler.state_dict()
                },
            'optimizer': {
                'name': self.config_parser['optimizer']['name'],
                'states': self.optimizer.state_dict()
                },
            # config
            'config': self.config_parser.config
        }
        if self.training_mode == 'epoch_based_train':
            state['trainer'] = {
                'training_mode': self.training_mode,
                'epoch': idx,
                'monitor_best': self.mnt_best,
            }
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(idx))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            if save_best:
                best_path = str(self.checkpoint_dir / f'model_best_until_epoch{idx}.pth')
                torch.save(state, best_path)
                self.logger.info(f"Saving current best: model_best_until_epoch{idx}.pth ...")

        elif self.training_mode == 'iteration_based_train':
            state['trainer'] = {
                'training_mode': self.training_mode,
                'iteration': idx,
                'monitor_best': self.mnt_best,
            }
            filename = str(self.checkpoint_dir / 'checkpoint-iteration{}.pth'.format(idx))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            if save_best:
                best_path = str(self.checkpoint_dir / f'model_best_until_iteration{idx}.pth')
                torch.save(state, best_path)
                self.logger.info(f"Saving current best: model_best_until_iteration{idx}.pth ...")

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resumer = Resumer(self.config_parser.args.resume, self.logger, self.config_parser.config)
        trainer_states = resumer.resume_trainer('trainer')
        is_same_training_mode = trainer_states['training_mode'] == self.training_mode

        if not self.config_parser.args.reset and is_same_training_mode:
            resumer.resume_optimizer(self.optimizer, 'optimizer')
            resumer.resume_lr_scheduler(self.lr_scheduler, 'lr_scheduler')

            if self.training_mode == 'epoch_based_train':
                self.start_epoch = trainer_states['epoch'] + 1
                self.mnt_best = trainer_states['monitor_best']
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from epoch {}, \
                                        and use the previous best monitor metric".format(self.start_epoch))

            elif self.training_mode == 'iteration_based_train':
                start_iteration = trainer_states['iteration'] + 1
                self.mnt_best = trainer_states['monitor_best']
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from iteration {}, \
                                        and use the previous best monitor metric".format(start_iteration))

        else:
            if self.training_mode == 'epoch_based_train':
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from epoch 1, \
                                        and reset the previous best monitor metric")

            elif self.training_mode == 'iteration_based_train':
                if dist.get_rank() == 0:
                    self.logger.info("Checkpoint loaded. Resume training from iteration 1, \
                                        and reset the previous best monitor metric")

        resumer.resume_model(self.model, 'model')
        

    def _progress(self, idx, data_loader, is_train):
        base = '[{}/{} ({:.0f}%)]'
        current = idx

        if is_train:
            if self.training_mode == 'epoch_based_train':
                total = len(data_loader)
            elif self.training_mode == 'iteration_based_train':
                total = self.iterations
        else:
            total = len(data_loader)

        return base.format(current, total, 100.0 * current / total)


def main(config_parser):
    # init ddp
    local_rank = init_distributed_mode()

    # fix seed for each process
    seed = config_parser.args.seed
    rank = dist.get_rank()
    init_seeds(seed + rank)

    logger = config_parser.get_logger('train')
    config = config_parser.config
    device = torch.device(f'cuda:{local_rank}')

    # setup data_loader instances
    train_dataloader = HDF5DataLoader(config['train_dataloader'])
    valid_dataloader = HDF5DataLoader(config['valid_dataloader'])
    
    # build model architecture, then print to console
    # model
    model_withoutddp = eval(config['model']['name'])(**config['model']['args']).to(local_rank)
    model_withoutddp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_withoutddp).to(local_rank)
    # model_withoutddp = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_withoutddp).to(local_rank)
    if dist.get_rank() == 0:
        logger.info(model_withoutddp)
    model = ddp(model_withoutddp, device_ids=[local_rank], output_device=local_rank)

    # loss functions
    loss = {
            'mse': nn.MSELoss(),
            'L1': nn.L1Loss(),
            'psnr': psnr_loss(),
            'SmoothL1': nn.SmoothL1Loss(beta=0.01),
            'Lap': LaplacianLoss().cuda(),
            'GAN': Adversarial(PatchSize=config['train_dataloader']['dataset']['data_augment']['random_crop']['size'][0], gan_type='STGAN').cuda(),
            'census': Ternary(7),
            'lpips': perceptual_loss(gpu_ids=[local_rank]),
            'CB': CharbonnierLoss().to(device),
                }

    # optimizers
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = eval(config['optimizer']['name'])(trainable_params, **config['optimizer']['args'])

    # learning rate scheduler
    lr_scheduler = eval(config['lr_scheduler']['name'])(optimizer, **config['lr_scheduler']['args'])

    # sync all processes
    dist.barrier()

    # training loop
    args = {
        # config parser
        'config_parser': config_parser,
        # dataloader
        'train_dataloader': train_dataloader,
        'valid_dataloader': valid_dataloader,
        # models
        'model': model,
        # loss fts
        'loss': loss,
        # optimizers
        'optimizer': optimizer,
        # lr scheduler
        'lr_scheduler': lr_scheduler,
        # metadata
        'logger': logger,
        'device': device,
    }
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test YAMLParser')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-id', '--runid', default=None, type=str)
    args.add_argument('-seed', '--seed', default=123, type=int)
    args.add_argument('-r', '--resume', default=None, type=str)
    args.add_argument('--reset', default=False, action='store_true', help='if resume checkpoint, reset trainer states in the checkpoint')
    args.add_argument('--limited_memory', default=False, action='store_true',
                      help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".')

    if args.parse_args().limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='test;item1;body1'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config_parser = YAMLParser.from_args(args, options)
    main(config_parser)