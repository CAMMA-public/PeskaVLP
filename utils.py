import math
import os
import yaml
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import glob
import datetime
import warnings
from mmengine import DefaultScope
import random
from PIL import ImageFilter

def cumulative_sum(lst):
    cumulative_list = []
    sum = 0
    for num in lst:
        sum += num
        cumulative_list.append(sum)
    return cumulative_list


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class Gaussian(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def calc_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds.argmax(dim=1) == labels-1).sum().item()
    total = labels.numel()
    return correct / total

def distributed_calc_accuracy(
    preds: torch.Tensor, 
    labels: torch.Tensor,
    world_size: int,
    device: torch.device
) -> float:
    correct = (preds.argmax(dim=1) == labels).sum()
    total = torch.tensor(labels.numel(), device=device)

    # Reduce the correct and total counts to the main GPU
    dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)

    # Ensure the calculation is only done on the main GPU
    if dist.get_rank() == 0:
        accuracy = correct.item() / total.item()
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy
    return 0.0  # Return 0.0 for other GPUs

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_multistep_schedule(optimizer, milestones=[30,80], gamma=0.5):
    """ """
    scheduler = MultiStepLR(optimizer, milestones, gamma=0.5)
    return scheduler



def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt)) 
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def save_checkpoint_eval(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(state, os.path.join(checkpoint_dir, "eval_epoch{:0>4d}.pth.tar".format(epoch)))


def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, 'epoch*.pth.tar'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ''

def get_logfile(checkpoint_dir):
    log_name = os.path.join(checkpoint_dir, 'config.py')
    with open(log_name, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

def log(output, args):
    with open(os.path.join(args.work_dir,'log.txt'), "a") as f:
        f.write(output + '\n')

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmselfsup into the registries.
    Args:
        init_default_scope (bool): Whether initialize the mmselfsup default
            scope. When `init_default_scope=True`, the global default scope
            will be set to `mmselfsup`, and all registries will build modules
            from mmselfsup's registry node. To understand more about the
            registry, please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import codes.datasets  # noqa: F401,F403
    import codes.loss  # noqa: F401,F403
    # import codes.evaluation  # noqa: F401,F403
    import codes.models  # noqa: F401,F403
    # import codes.structures  # noqa: F401,F403
    # import codes.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmselfsup')
        if never_created:
            DefaultScope.get_instance('mmselfsup', scope_name='mmselfsup')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmselfsup':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmselfsup", '
                          '`register_all_modules` will force set the current'
                          'default scope to "mmselfsup". If this is not as '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmselfsup-{datetime.datetime.now()}'
            DefaultScope.get_instance(
                new_instance_name, scope_name='mmselfsup')