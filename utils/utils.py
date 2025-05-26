import os
import torch
import yaml, time
from collections import defaultdict
from diffusion.score_model import TensorProductScoreModel


def get_mar_diff_model(args):
    return TensorProductScoreModel(in_node_features=args.mar_diff_in_node_features, in_edge_features=args.mar_diff_in_edge_features,
                                   ns=args.mar_diff_ns, nv=args.mar_diff_nv, sigma_embed_dim=args.mar_diff_sigma_embed_dim,
                                   sigma_min=args.mar_diff_sigma_min, sigma_max=args.mar_diff_sigma_max,
                                   num_conv_layers=args.mar_diff_num_conv_layers,
                                   max_radius=args.mar_diff_max_radius, radius_embed_dim=args.mar_diff_radius_embed_dim,
                                   scale_by_sigma=args.mar_diff_scale_by_sigma,
                                   use_second_order_repr=args.mar_diff_use_second_order_repr,
                                   residual=not args.mar_diff_no_residual, batch_norm=not args.mar_diff_no_batch_norm, 
                                   plus_hidden_dim=args.mar_diff_plus_hidden_dim)


def get_optimizer_and_scheduler(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


class TimeProfiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.starts = {}
        self.curr = None

    def start(self, tag):
        self.starts[tag] = time.time()

    def end(self, tag):
        self.times[tag] += time.time() - self.starts[tag]
        del self.starts[tag]


import signal
from contextlib import contextmanager
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
