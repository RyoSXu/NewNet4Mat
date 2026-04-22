import os
from torch.utils.data.distributed import DistributedSampler
from utils.misc import get_rank, get_world_size, dictToObj
from timm.scheduler import create_scheduler


class ConfigBuilder(object):
    def __init__(self, **params):
        super(ConfigBuilder, self).__init__()
        self.model_params = params.get('model', {})
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})
        self.trainer_params = params.get('trainer', {})

        self.logger = params.get('logger', None)
        self.dos_minmax = params.get('dos_minmax', False)
        self.dos_zscore = params.get('dos_zscore', False)
        self.apply_log = params.get('apply_log', False)
        self.scale_factor = params.get('scale_factor', 1.0)

    def get_model(self, model_params=None):
        from model.model import basemodel
        if model_params is None:
            model_params = self.model_params
        model_type = model_params.get('type', 'transformer')
        params = model_params.get('params', {})
        params['dos_minmax'] = self.dos_minmax
        params['dos_zscore'] = self.dos_zscore
        params['scale_factor'] = self.scale_factor
        params['apply_log'] = self.apply_log
        if model_type == 'transformer':
            model = basemodel(self.logger, **params)
        else:
            raise NotImplementedError('Invalid model type.')
        return model

    def get_dataset(self, dataset_params=None, split='train', dos_minmax=False, dos_zscore=False,
                    scale_factor=1.0, apply_log=False, smear=0, choice=[]):
        from datasets.dataset import Dos_Dataset
        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, None)
        if dataset_params is None:
            return None
        if isinstance(dataset_params, dict):
            dataset_type = str.lower(dataset_params.get('type', 'dos_dataset'))
            if dataset_type == 'dos_dataset':
                dataset = Dos_Dataset(
                    split=split, dos_minmax=dos_minmax, dos_zscore=dos_zscore,
                    scale_factor=scale_factor, apply_log=apply_log,
                    smear=smear, choice=choice, **dataset_params
                )
            else:
                raise NotImplementedError('Invalid dataset type: {}.'.format(dataset_type))
        else:
            raise AttributeError('Invalid dataset format.')
        return dataset

    def get_sampler(self, dataset, split='train'):
        shuffle = (split == 'train')
        rank = get_rank()
        num_gpus = get_world_size()
        return DistributedSampler(dataset, rank=rank, shuffle=shuffle, num_replicas=num_gpus, seed=0)

    def get_dataloader(self, dataset_params=None, split='train', smear=0, choice=[],
                       batch_size=None, dataloader_params=None,
                       dos_minmax=False, dos_zscore=False, scale_factor=1.0, apply_log=False):
        from torch.utils.data import DataLoader
        if batch_size is None:
            if split == 'train':
                batch_size = self.trainer_params.get('batch_size', 128)
            elif split == 'test':
                batch_size = self.trainer_params.get('test_batch_size', 1)
            else:
                batch_size = self.trainer_params.get('valid_batch_size', 1)
        if dataloader_params is None:
            dataloader_params = self.dataloader_params
        dataset = self.get_dataset(
            dataset_params, split=split, choice=choice,
            dos_minmax=dos_minmax, dos_zscore=dos_zscore,
            scale_factor=scale_factor, apply_log=apply_log
        )
        if dataset is None:
            return None
        sampler = self.get_sampler(dataset, split)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **dataloader_params)

    def get_max_epoch(self, trainer_params=None):
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('max_epoch', 40)


def get_optimizer(model, optimizer_params=None, resume=False, resume_lr=None):
    from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
    opt_type = optimizer_params.get('type', 'AdamW')
    params = optimizer_params.get('params', {})

    if resume:
        network_params = [{'params': model.parameters(), 'initial_lr': resume_lr}]
        params.update(lr=resume_lr)
    else:
        network_params = model.parameters()

    optimizers = {
        'SGD': SGD, 'ASGD': ASGD, 'Adagrad': Adagrad, 'Adamax': Adamax,
        'Adadelta': Adadelta, 'Adam': Adam, 'AdamW': AdamW, 'RMSprop': RMSprop,
    }
    if opt_type not in optimizers:
        raise NotImplementedError('Invalid optimizer type.')
    return optimizers[opt_type](network_params, **params)


def get_lr_scheduler(optimizer, lr_scheduler_params=None, resume=False, resume_epoch=None):
    scheduler_args = dictToObj(lr_scheduler_params)
    scheduler, _ = create_scheduler(scheduler_args, optimizer)
    return scheduler
