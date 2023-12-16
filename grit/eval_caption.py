import os
import hydra
import random
import numpy as np
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from models.caption import Transformer
from models.caption.detector import build_detector

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.caption_engine import *


def main(gpu, config):
    # dist init
    torch.backends.cudnn.enabled = False
    dist.init_process_group('nccl', 'env://', rank=0, world_size=1)

    torch.manual_seed(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    # extract reg features + initial grid features
    detector = build_detector(config).to(device)

    model = Transformer(detector=detector, config=config)
    model.load_state_dict(torch.load(config.exp.checkpoint)['state_dict'], strict=False)
    model = model.to(device)

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)

    dataloaders, samplers = build_coco_dataloaders(config, mode='finetune', device=device)

    text_field = TextField(vocab_path=config.dataset.vocab_path)

    split = config.split
    print(f"Evaluating on split: {split}")
    scores = evaluate_metrics(
        model,
        optimizers=None,
        dataloader=dataloaders[f'{split}_dict'],
        text_field=text_field,
        epoch=-1,
        split=f'{split}',
        config=config,
        train_res=[],
        writer=None,
        best_cider=None,
        which='ft_sc',
        scheduler=None,
        log_and_save=False,
    )


@hydra.main(config_path="configs/caption", config_name="coco_config")
def run_main(config: DictConfig) -> None:
    mp.spawn(main, nprocs=1, args=(config,))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6688"
    run_main()