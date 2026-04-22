import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger
import copy
from utils.metrics import MetricsRecorder
from pymatgen.core import Structure
import glob
import numpy as np

#----------------------------------------------------------------------------

def subprocess_fn(args):
    # 设置随机种子
    utils.setup_seed(args.seed * args.world_size + args.rank)

    # 初始化日志记录器
    logger = get_logger("test", args.run_dir, utils.get_rank(), filename='test.log')

    # 将日志记录器添加到参数中
    args.cfg_params["logger"] = logger

    # 构建配置
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    # 构建模型
    logger.info('Building models ...')
    model = builder.get_model()

    # 加载模型检查点
    model_checkpoint = os.path.join(args.run_dir, 'checkpoint_best.pth')
    model.load_checkpoint(model_checkpoint)

    # 封装模型，支持分布式训练
    model_without_ddp = utils.DistributedParallel_Model(model, args.gpu)

    # 记录模型参数数量
    for key in model_without_ddp.model:
        params = [p for p in model_without_ddp.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])
        logger.info("params {key}: {params}".format(key=key, params=cnt_params))

    # 构建测试数据加载器
    logger.info('Building dataloaders ...')

    def pad_list(lst, target_length):
        return lst + [0] * max(0, target_length - len(lst))
    
    all_elements = []
    all_positions = []
    all_index = []
    for cif_file in glob.glob('./data/train4w2023/test_cif/*.cif'):
        structure = Structure.from_file(cif_file)
        index = cif_file[28:-4]
        elements = [126, 127] + list(structure.atomic_numbers)
        positions = [
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.angles[0],
            structure.lattice.angles[1],
            structure.lattice.angles[2],
        ] + [abs(x) for x in list(structure.frac_coords.flatten())]
        
        all_elements.append(pad_list(elements, 128))
        all_positions.append(pad_list(positions, 128 * 3))
        all_index.append(index)
    # 转换为 numpy 数组
    all_elements = np.array(all_elements)
    all_positions = np.array(all_positions)
    all_index = np.array(all_index)
    
    # 保存到文件
    np.save("./data/train4w2023/test_cif/elements_apet_test_cif.npy", all_elements)
    np.save("./data/train4w2023/test_cif/index.npy", all_index)
    np.save("./data/train4w2023/test_cif/positions_apet_test_cif.npy", all_positions)

    dataset_params = args.cfg_params['dataset']
    test_dataloader = builder.get_dataloader(dataset_params=dataset_params, split='test_cif', batch_size=args.batch_size)
    logger.info('Test dataloaders build complete')
    # 开始测试
    logger.info('begin testing ...')
    model_without_ddp.test(test_data_loader=test_dataloader, epoch=0, save_predict=True)
    
def main(args):
    # 初始化分布式训练环境（如果启用）
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.gpu = 0
        torch.cuda.set_device(args.gpu)

    # 设置运行目录
    run_dir = args.cfgdir
    print(run_dir)
    
    # 加载配置文件
    args.cfg = os.path.join(args.cfgdir, 'training_options.yaml')
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
    # 根据申请的cpu数来设置dataloader的线程数
    cfg_params['dataloader']['num_workers'] = args.per_cpus
    cfg_params['dataset']['test_cif'] = copy.deepcopy(cfg_params['dataset']['test'])
    cfg_params['dataset']['test_cif']['length'] = args.length


    # 判断是否使用常量数据
    dataset_vnames = cfg_params['dataset']['train'].get("vnames", None)
    if dataset_vnames is not None:
        constants_len = len(dataset_vnames.get('constants'))
    else:
        constants_len = 0
    cfg_params['model']['params']['constants_len'] = constants_len

    # 保存测试配置
    if args.rank == 0:
        with open(os.path.join(run_dir, 'test_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2)
            yaml.dump(cfg_params, f, indent=2)

    # 将配置参数和运行目录添加到args中
    args.cfg_params = cfg_params
    args.run_dir = run_dir

    if "relative_checkpoint_dir" in cfg_params:
        args.relative_checkpoint_dir = cfg_params['relative_checkpoint_dir']

    # 启动测试过程
    print('Launching processes...')
    subprocess_fn(args)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',           type = int,     default = 42,                                            help = 'seed')
    parser.add_argument('--cuda',           type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',     type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',       type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--batch_size',     type = int,     default = 256,                                           help = "batch size")
    parser.add_argument('--length',         type = int,     default = 64,                                           help = "predict len")
    parser.add_argument('--metric_list',    nargs = '+',                                                            help = 'metric list')
    parser.add_argument('--init_method',    type = str,     default = 'tcp://127.0.0.1:23456',                      help = 'multi process init method')
    parser.add_argument('--cfgdir',         type = str,     default = './output/transformer/world_size1-v2-newposition2-4wdataset-test-valid-2023',  help = 'Where to save the results')

    args = parser.parse_args()

    main(args)




    # metric list从args中获取，获取meanstd并传到valid中，额外参数怎么设置, 完善metric,是否使用常量数据的操作
