import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
from utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--coef_pro', type=float, required=True, help='th coef od attention in block')
    parser.add_argument('--coef_2dgird', type=int, required=True, help='th coef 2d adpater grid size')
    parser.add_argument('--coef_3dgird', type=float, required=True, help='th coef 3d adpater grid size') 
    parser.add_argument('--maxmean_2d', action='store_true', default=False, help='whether the max or maxmean in 2d adapter')
    parser.add_argument('--num_view', type=int, default=6, help='') 
    parser.add_argument('--trans', type=float, required=True, help='th dim of 2d adpater attn')
    parser.add_argument('--attn2d_dim', type=int, required=True, help='th dim of 2d adpater attn')
    parser.add_argument('--patchknn', type=int, required=True, help='th knn size of patch embed')
    parser.add_argument('--lastdim', type=int, required=True, help='th lastdim of patch embed')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    parser.add_argument('--pretrained_path', type=str, required=True, help='')
    parser.add_argument('--mode', type=str, required=True, help='')
    parser.add_argument('--formal_dirs', type=str, required=True, help='')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1
    
    cfg.pretrained_path = args.pretrained_path
    cfg.mode = args.mode
    cfg.formal_dirs = args.formal_dirs

    cfg.model.encoder_args.attn2d_dim = args.attn2d_dim

    cfg.model.encoder_args.patchknn = args.patchknn
    cfg.model.encoder_args.num_view = args.num_view
    cfg.model.encoder_args.lastdim = args.lastdim
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        args.mode,
        cfg.exp_name,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    if args.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=args.pretrained_path)
        cfg.wandb.tags = [args.mode]
    else:  # resume from the existing ckpt and reuse the folder.
        #generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        generate_exp_directory(cfg, exp_name=args.formal_dirs, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name
    main = train

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main(0, cfg, profile=args.profile, args=args)
