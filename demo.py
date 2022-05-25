from asyncio.log import logger
from calendar import EPOCH
from csv import writer
from random import shuffle
from sqlite3 import Timestamp
from threading import local
from time import time
from tracemalloc import start
from types import NoneType
from yaml import parse

from ContrastiveCrop.DDP_simsiam_ccrop import load_weights, parse_args
from ContrastiveCrop.builder.build import build_optimizer
from ContrastiveCrop.datasets.build import build_dataset, build_dataset_ccrop
from ContrastiveCrop.models.build import build_model


def main():
    args = parse_args()
    cfg = get_cfg(args)

    world_size = torch.cuda.device_count()
    print('GPUS on this node:', world_size)
    cfg.world_size = world_size

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.locatetime())
    log_file =os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    mp.spawn(mian_worker, nprocs=world_size,args=(world_size,cfg))

def main_worker(rank,world_size,cfg):
    print('==> Start rank:', rank)
    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl', init_method = f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank = rank)
    
    logger, writer = None，None
    if rank == 0:
        writer = SummaryWriter(log_dict=os.path.join(cfg.work_dir, 'tensorboard'))
        logger = builf_logger(cfg.world_dir,'pretrain')

    bsz_gpu = int(cfg.batch_size /cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    train_set = build_dataset_ccrop(cfg.data.train)
    len_ds = len(train_set)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,shuffle=True)
    train_loader = torch.utils.data.Dataloader(
        train_set,
        batch_size=bsz_gpu,
        num_worker=True,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    eval_train_test = build_dataset(cfg.data.eval_train)
    eval_train_sampler=torch.utils.data.distributed.DistributedSampler(eval_train_set, shuffle=False1)
    eval_train_loader=tirch.utils.data.Dataloader(
        eval_train_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=eval_train_sampler,
        drop_last=False
    )

    encoder = build_model(cfg.model)
    model = SimSiam(encoder, **cfg.simsiam)
    model=torch.nn.SynBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model=torch.nn.parallel.DistributedDataParallel(model, device_id=[cfg.local_rank])
    criterion=build_loss(cfg.loss).cuda()
    if cfg.fix_pred_lr:
        optim_params = [{'params':model.module.encoder.parameter(), 'fix_lr':False},
                        {'params':modek.module.predictor.parameters(), 'fix_lr':True}]
    else:
        optim_params = model.parameters()
    optimizer=build_optimizer(cfg.optimizerm optim_params)

    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, train_set, model, optimizer, resume=True)
    cudnn.benchmark=True

    print("==> Start training...")
    for epoch in range(start_epoch, cfg.epochs+1):
        train_sampler.set_epoch(epoch)
        adjust_lr_simsiam(cfg.lr_cfg, optimizer, epoch)

        train_set.use_box = epoch >= cfg.warmup_epochs + start_epoch

        train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer)

        if epoch >= cfg.warmup_epochs and epoch != cfg.epochs and epoch % cfg.loc_interval ==0:
            all_boxes = update_box(eval_train_loader, model.module.encoder, len_ds, logger,
                                   t=cfg.box_thresh)
            
            assert len(all_boxes) == len_ds
            train_set.boxes = all_boxes.cpu()

        if rank==0 and epoch % cfg.save_inerval == 0:
            model_path = os.path.join(cfg.work_dir, f'epch_{epoch}.pth')
            start_dict = {
                'optimizer_state';optimizer.state_dict(),
                'simsiam_state':model.state_dict(),
                'boxes':train_set.boxes,
                'epoch':epoch
            }
            torch.save(state_dict, model_path)

    if rank ==0:
        model_path=os.path.join(cdf.work_dir, 'last.pth')
        state_dict = {
            'optimizer_state':optimizer.state_dict(),
            'simsiam_state':model.state_dict(),
            'boxes':train_set.boxes,
            'epoch':cfg.epoches
        }
        torch.save(state_dict, model_path)
import datasets
from datasets.trasdfroms import build_transform
import torchvision

def build_dataset(cfg):
    args = cfg.copy()
    transforms = build)transforms(args.trans_dict)
    ds_dict=args.ds_dict
    ds_name=ds_dict.pop('type')
    ds_dict['transform']=trabsform
    if hasattr(torchvision.datasets, ds_name):
        ds = getattr(torchvision.datasets, ds_name)(**ds_dict)
    else:
        ds = datasets.__dict__[ds_name](**ds_dict)
    return ds
def build_dataset_ccrop(cfg):
    