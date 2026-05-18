"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import time
import random
import numpy as np
import atexit
import copy
import datetime

import torch
import torch.nn as nn
import torch.distributed
import torch.backends.cudnn

from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data import DistributedSampler
# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader


def get_local_rank():
    return int(os.getenv('LOCAL_RANK', 0))


def _distributed_timeout():
    minutes = float(os.getenv('DEIM_DISTRIBUTED_TIMEOUT_MINUTES', '30'))
    return datetime.timedelta(minutes=minutes)


def setup_distributed(print_rank: int=0, print_method: str='builtin', seed: int=None, ):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv('RANK', -1))
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(
            backend=backend,
            init_method='env://',
            timeout=_distributed_timeout(),
        )
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        local_rank = int(os.getenv('LOCAL_RANK', rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.cuda.empty_cache()
        enabled_dist = True
        if get_rank() == print_rank:
            print(
                'Initialized distributed mode '
                f'(backend={backend}, rank={rank}, local_rank={local_rank}, world_size={WORLD_SIZE})...'
            )

    except Exception:
        enabled_dist = False
        print('Not init distributed mode.')

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)

    return enabled_dist


def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process
    """
    import builtins as __builtin__

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@atexit.register
def cleanup():
    """cleanup distributed environment
    """
    if is_dist_available_and_initialized():
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)



def warp_model(
    model: torch.nn.Module,
    sync_bn: bool=False,
    dist_mode: str='ddp',
    find_unused_parameters: bool=False,
    compile: bool=False,
    compile_mode: str='reduce-overhead',
    **kwargs
):
    if is_dist_available_and_initialized():
        local_rank = get_local_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if dist_mode == 'dp':
            model = DP(model, device_ids=[local_rank], output_device=local_rank)
        elif dist_mode == 'ddp':
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            raise AttributeError('')

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model

def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers)
    return loader



def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        local_keys = sorted(data.keys())
        gathered_keys = all_gather(local_keys)
        keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
        if not keys:
            return {}

        if data:
            zero = next(iter(data.values())).new_zeros(())
        elif torch.cuda.is_available():
            zero = torch.zeros((), device=torch.device('cuda', get_local_rank()))
        else:
            zero = torch.zeros(())

        values = [data.get(k, zero) for k in keys]

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def capture_rng_state():
    state = {
        'python': copy.deepcopy(random.getstate()),
        'numpy': copy.deepcopy(np.random.get_state()),
        'torch': torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state['cuda'] = [x.clone() for x in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    if not state:
        return

    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def capture_backend_state():
    return {
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
    }


def restore_backend_state(state):
    if not state:
        return

    if 'cudnn_benchmark' in state:
        torch.backends.cudnn.benchmark = state['cudnn_benchmark']
    if 'cudnn_deterministic' in state:
        torch.backends.cudnn.deterministic = state['cudnn_deterministic']


def seed_dataloader_worker(worker_id, base_seed, rank):
    worker_info = torch.utils.data.get_worker_info()
    epoch = -1
    if worker_info is not None and hasattr(worker_info.dataset, 'epoch'):
        epoch = worker_info.dataset.epoch
    epoch = max(int(epoch), 0)
    worker_seed = (int(base_seed) + int(rank) * 1000003 + epoch * 10007 + int(worker_id)) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader_generator(base_seed, rank):
    generator = torch.Generator()
    generator.manual_seed(int(base_seed) + int(rank))
    return generator


def sync_time():
    """sync_time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()



def setup_seed(seed: int, deterministic=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


# for torch.compile
def check_compile():
    import torch
    import warnings
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok

def is_compile(model):
    import torch._dynamo
    return type(model) in (torch._dynamo.OptimizedModule, )

def de_complie(model):
    return model._orig_mod if is_compile(model) else model
