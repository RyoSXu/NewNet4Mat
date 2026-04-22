import torch
import torch.nn as nn
from model.transformer import Transformer
from utils.builder import get_optimizer, get_lr_scheduler
from utils.metrics import MetricsRecorder
import utils.misc as utils
import time
import datetime
from pathlib import Path
import torch.cuda.amp as amp
import numpy as np
import os

class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.dos_minmax = self.params.get("dos_minmax", False)
        self.dos_zscore = self.params.get("dos_zscore", False)
        self.apply_log = self.params.get("apply_log", False)
        self.scale_factor = self.params.get("scale_factor", 1.0)
        self.logger = logger
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = 1000
        self.constants_len = self.params.get("constants_len", 0)
        self.begin_epoch = 0

        self.gscaler = amp.GradScaler(init_scale=1024, growth_interval=2000)

        sub_model = params.get('sub_model', {})
        for key in sub_model:
            if key == "transformer":
                self.model[key] = Transformer(**sub_model["transformer"])
            else:
                raise NotImplementedError('Invalid model type.')
            self.sub_model_name.append(key)

        self.optimizer = {}
        self.lr_scheduler = {}
        self.lr_scheduler_by_step = {}

        optimizer = params.get('optimizer', {})
        lr_scheduler = params.get('lr_scheduler', {})
        for key in self.sub_model_name:
            if key in optimizer:
                self.optimizer[key] = get_optimizer(self.model[key], optimizer[key])
            if key in lr_scheduler:
                self.lr_scheduler_by_step[key] = lr_scheduler[key].get('by_step', False)
                self.lr_scheduler[key] = get_lr_scheduler(self.optimizer[key], lr_scheduler[key])

        eval_metrics_list = params.get('metrics_list', [])
        if len(eval_metrics_list) > 0:
            self.eval_metrics = MetricsRecorder(eval_metrics_list)
        else:
            self.eval_metrics = None

        for key in self.model:
            self.model[key].eval()

    def to(self, device):
        self.device = device
        for key in self.model:
            self.model[key].to(device)
        for key in self.optimizer:
            for state in self.optimizer[key].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def data_preprocess(self, data):
        inp, pos, target, dos_mean, dos_std, dos_max, dos_min = data
        mask = (inp==0)
        inp = inp.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        pos = pos.to(self.device, non_blocking=True)
        dos_mean = dos_mean.to(self.device, non_blocking=True)
        dos_std = dos_std.to(self.device, non_blocking=True)
        dos_max = dos_max.to(self.device, non_blocking=True)
        dos_min = dos_min.to(self.device, non_blocking=True)
        mask = torch.tensor(mask.clone().detach(), dtype=torch.bool).to(self.device)
        return inp, pos, mask, target, dos_mean, dos_std, dos_max, dos_min

    def loss(self, predict, target):
        return torch.mean((predict - target) ** 2)

    def train_one_step(self, batch_data, step):
        inp, pos, mask, target, _, _, _, _ = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict = self.model[list(self.model.keys())[0]](inp, mask, pos)[0].squeeze(-1)
        else:
            raise NotImplementedError('Invalid model type.')

        loss = self.loss(predict, target)
        if len(self.optimizer) == 1:
            self.optimizer[list(self.optimizer.keys())[0]].zero_grad()
            loss.backward()
            self.optimizer[list(self.optimizer.keys())[0]].step()
        else:
            raise NotImplementedError('Invalid model type.')

        return {'loss': loss.item()}

    def test_one_step(self, batch_data, step=None, save_predict=False):
        inp, pos, mask, target, dos_mean, dos_std, dos_max, dos_min = self.data_preprocess(batch_data)
        if len(self.model) == 1:
            predict, attention = self.model[list(self.model.keys())[0]](inp, mask, pos)
        predict = predict.squeeze(-1)

        loss = self.loss(predict, target)

        data_dict = {'gt': target, 'pred': predict}
        metrics_loss = self.eval_metrics.evaluate_batch(data_dict)
        metrics_loss.update({'lp_loss': loss.item()})

        if self.dos_minmax:
            predict_n = predict * (dos_max - dos_min) + dos_min
            target_n = target * (dos_max - dos_min) + dos_min
        elif self.dos_zscore:
            predict_n = predict * dos_std + dos_mean
            target_n = target * dos_std + dos_mean
        else:
            predict_n = predict
            target_n = target

        if self.apply_log:
            predict_n = torch.exp(predict_n) - 1.0e-10
            target_n = torch.exp(target_n) - 1.0e-10

        if self.scale_factor != 1.0:
            predict_n = predict_n / self.scale_factor
            target_n = target_n / self.scale_factor

        predict_n[predict_n < 0] = 0
        MAE_ori = torch.mean(torch.abs(predict_n - target_n))
        MSE_ori = torch.mean((predict_n - target_n) ** 2)
        SS_res = torch.sum((predict - target) ** 2)
        SS_tot = torch.sum((target - torch.mean(target)) ** 2)
        R2 = 1 - (SS_res / SS_tot)
        metrics_loss.update({'MAE_ori': MAE_ori, 'MSE_ori': MSE_ori, 'lp_loss': loss.item(), 'R2': R2.item()})

        if save_predict:
            os.makedirs("dosdata", exist_ok=True)
            np.savetxt("dosdata/predict_%s.txt" % step, predict.squeeze(dim=0).cpu().numpy())
            np.savetxt("dosdata/target_%s.txt" % step, target.squeeze(dim=0).cpu().numpy())
            np.save("dosdata/attention_%s.npy" % step, attention.squeeze(dim=0).cpu().numpy())
            np.save("dosdata/input_%s.npy" % step, inp.squeeze(dim=0).cpu().numpy())

        return metrics_loss

    def train_one_epoch(self, train_data_loader, epoch, max_epoches):
        for key in self.lr_scheduler:
            if not self.lr_scheduler_by_step[key]:
                self.lr_scheduler[key].step(epoch)

        end_time = time.time()
        for key in self.optimizer:
            self.model[key].train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')
        max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'
        for step, batch in enumerate(train_data_loader):
            for key in self.lr_scheduler:
                if self.lr_scheduler_by_step[key]:
                    self.lr_scheduler[key].step(epoch * max_step + step)

            data_time.update(time.time() - end_time)
            loss = self.train_one_step(batch, step)
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            if (step + 1) % 100 == 0 or step + 1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches - epoch - 1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header, "lr: {lr}", "eta: {eta}", "time: {time}", "data: {data}", "memory: {memory:.0f}", "{meters}"]
                    ).format(
                        epoch=epoch + 1, max_epoches=max_epoches, step=step + 1, max_step=max_step,
                        lr=self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))

    def load_checkpoint(self, checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        checkpoint_model = checkpoint_dict['model']
        checkpoint_optimizer = checkpoint_dict['optimizer']
        checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
        for key in checkpoint_model:
            self.model[key].load_state_dict(checkpoint_model[key])
        for key in checkpoint_optimizer:
            self.optimizer[key].load_state_dict(checkpoint_optimizer[key])
        for key in checkpoint_lr_scheduler:
            self.lr_scheduler[key].load_state_dict(checkpoint_lr_scheduler[key])
        self.begin_epoch = checkpoint_dict['epoch']
        if 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if 'amp_scaler' in checkpoint_dict:
            self.gscaler.load_state_dict(checkpoint_dict['amp_scaler'])
        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(
            epoch=self.begin_epoch, metric_best=self.metric_best))

    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'):
        checkpoint_savedir = Path(checkpoint_savedir)
        checkpoint_path = checkpoint_savedir / (
            'checkpoint_best.pth' if save_type == 'save_best' else 'checkpoint_latest.pth'
        )
        is_ddp = utils.get_world_size() > 1
        model_states = {
            key: self.model[key].module.state_dict() if is_ddp else self.model[key].state_dict()
            for key in self.model
        }
        utils.save_on_master({
            'epoch': epoch + 1,
            'model': model_states,
            'optimizer': {key: self.optimizer[key].state_dict() for key in self.optimizer},
            'lr_scheduler': {key: self.lr_scheduler[key].state_dict() for key in self.lr_scheduler},
            'metric_best': self.metric_best,
            'amp_scaler': self.gscaler.state_dict(),
        }, checkpoint_path)

    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False

    def trainer(self, train_data_loader, test_data_loader, valid_data_loader, max_epoches, checkpoint_savedir=None, resume=False):
        for epoch in range(self.begin_epoch, max_epoches):
            train_data_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(train_data_loader, epoch, max_epoches)
            metric_logger = self.test(valid_data_loader, epoch)

            if checkpoint_savedir is not None:
                if self.whether_save_best(metric_logger):
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')

    @torch.no_grad()
    def test(self, test_data_loader, epoch, save_predict=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        for key in self.model:
            self.model[key].eval()

        for step, batch in enumerate(test_data_loader):
            loss = self.test_one_step(batch, save_predict=save_predict, step=step)
            metric_logger.update(**loss)

        self.logger.info('  '.join(
            [f'Epoch [{epoch + 1}](val stats)', "{meters}"]).format(meters=str(metric_logger)))

        return metric_logger
