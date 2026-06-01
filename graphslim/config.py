'''Configuration'''
import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
import json
import logging

import click
from pprint import pformat
import graphslim
from graphslim.utils import seed_everything, f1_macro, accuracy, roc_auc


METRIC_ALIASES = {
    'accuracy': accuracy,
    'f1_macro': f1_macro,
    'roc_auc': roc_auc,
}


# Make default paths independent of the current working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_PATH = os.path.abspath(os.path.join(_THIS_DIR, '..', 'checkpoints'))
DEFAULT_DATA_PATH = os.path.abspath(os.path.join(_THIS_DIR, '..', 'data'))


def resolve_metric(metric_value):
    """Return metric callable and its canonical name."""
    if callable(metric_value):
        name = getattr(metric_value, '__name__', 'metric')
        key = name if name in METRIC_ALIASES else name
        return metric_value, key
    if isinstance(metric_value, str):
        key = metric_value.lower()
        if key not in METRIC_ALIASES:
            raise click.BadParameter(f"Unsupported metric '{metric_value}'")
        return METRIC_ALIASES[key], key
    raise click.BadParameter(f"Unsupported metric type: {type(metric_value).__name__}")


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        # Use pprint's pformat to print the dictionary in a pretty manner
        return pformat(self.__dict__, compact=True)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


def update_from_dict(obj, updates):
    for key, value in updates.items():
        # set higher priority from command line as we explore some factors
        if key in ['init'] and obj.init is not None:
            continue
        setattr(obj, key, value)


# fix setting here
def setting_config(args):
    representative_r = {
        'ele-fashion': 0.001,
        'books_nc': 0.00025,
        'nc_ar_cloth': 0.001,
        'nc_ar_sports': 0.001,
    }
    if args.reduction_rate == -1:
        args.reduction_rate = representative_r[args.dataset]
    if args.dataset in ['nc_ar_cloth', 'nc_ar_sports']:
        args.setting = 'trans'
    if args.dataset in ['ele-fashion', 'books_nc']:
        args.setting = 'ind'
    if not getattr(args, '_metric_overridden', False):
        if args.dataset in ['nc_ar_cloth', 'nc_ar_sports']:
            args.metric = roc_auc
            args.metric_name = 'roc_auc'
        else:
            args.metric = accuracy
            args.metric_name = 'accuracy'

    args.run_inter_eval = 3

    args.checkpoints = list(range(-1, args.epochs + 1, args.eval_interval))
    args.eval_epochs = 300
    return args


# recommend hyperparameters here
def method_config(args):
    if args.method in ['msgc']:
        args.batch_adj = 16
    elif args.method in ['sgdd']:
        args.ep_ratio = 0.5
        args.sinkhorn_iter = 10
        args.opt_scale = 1e-11
        args.beta = 0.5
    elif args.method in ['gcdm']:
        args.dis_metric = 'mse'
    # add temporary changes here
    # do not modify the config json

    return args


@click.command()
@click.option('--momentum', default=0.01, show_default=True)
@click.option('--weight_decay_hyper', default=5e-4, show_default=True)
@click.option('--metric', default='accuracy', type=click.Choice(tuple(METRIC_ALIASES.keys())), show_default=True)

@click.option('--dataset', '-D', default='ele-fashion', show_default=True)
@click.option('--gpu_id', '-G', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--setting', type=click.Choice(['trans', 'ind']), show_default=True,
              help='transductive or inductive setting')
@click.option('--split', default='fixed', show_default=True,
              help='only support public split now, do not change it')  # 'fixed', 'random', 'few'
@click.option('--run_reduction', default=3, show_default=True, help='repeat times of reduction')
@click.option('--run_eval', default=5, show_default=True, help='repeat times of final evaluations')
@click.option('--run_inter_eval', default=3, show_default=True, help='repeat times of intermediate evaluations')
@click.option('--eval_interval', default=20, show_default=True)
@click.option('--hidden', '-H', default=256, show_default=True)
@click.option('--eval_epochs', '--ee', default=300, show_default=True)
@click.option('--eval_model', '--em', default='GCN',
              type=click.Choice(
                  ['GCN', 'GAT', 'APPNP', 'Cheby', 'GraphSage']
              ), show_default=True)
@click.option('--final_eval_model', '--fem', default='GCN',
              type=click.Choice(
                  ['GCN', 'GAT', 'APPNP', 'Cheby', 'GraphSage']
              ), show_default=True)
@click.option('--condense_model', default='GCN',
              type=click.Choice(
                  ['GCN', 'GAT', 'APPNP', 'Cheby', 'GraphSage']
              ), show_default=True)
@click.option('--epochs', '-E', default=1000, show_default=True, help='number of reduction epochs')
@click.option('--lr', default=0.001, show_default=True)
@click.option('--weight_decay', '--wd', default=0.0, show_default=True)
@click.option('--pre_norm', default=True, show_default=True,
              help='pre-normalize features, forced true for arxiv, flickr and reddit')
@click.option('--outer_loop', default=5, show_default=True)
@click.option('--inner_loop', default=2, show_default=True)
@click.option('--reduction_rate', '-R', default=1.0, show_default=True,
              help='-1 means use representative reduction rate; reduction rate of training set, defined as (number of nodes in small graph)/(number of nodes in original graph)')
@click.option('--seed', '-S', default=1, help='Random seed', show_default=True)
@click.option('--nlayers', default=2, help='number of GNN layers of condensed model', show_default=True)
@click.option('--verbose', '-V', is_flag=True, show_default=True)
@click.option('--soft_label', default=0, show_default=True)
@click.option('--init', default='random', help='features initialization methods',
              type=click.Choice(
                  ['kcenter', 'herding', 'random']
              ), show_default=True)
@click.option('--method', '-M', default='gcond', 
              type=click.Choice(['gcond', 'msgc', 'sgdd', 'gcdm']),
              show_default=True)

@click.option('--eval_wd', '--ewd', default=0.0, show_default=True)
@click.option('--eval_loss', '--eloss', default='CE',
                type=click.Choice(
                  ['CE', 'KLD','MSE']
              ), show_default=True)
@click.option('--activation', default='relu', help='activation function when do NAS',
              type=click.Choice(
                  ['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leakyrelu', 'relu6', 'elu']
              ), show_default=True)
@click.option('--ptb_r', '-P', default=0.25, show_default=True, help='perturbation rate for corruptions')
@click.option('--agg', is_flag=True, show_default=True, help='use aggregation for coreset methods')
@click.option('--multi_label', is_flag=True, show_default=True, help='use aggregation for coreset methods')
@click.option('--dis_metric', default='ours', show_default=True,
              help='distance metric for all condensation methods,ours means metric used in GCond paper')
@click.option('--lr_adj', default=0.001, show_default=True)
@click.option('--lr_feat', default=0.001, show_default=True)
@click.option('--optim', default="Adam", show_default=True)
@click.option('--threshold', default=0.0, show_default=True, help='sparsificaiton threshold before evaluation')
@click.option('--dropout', default=0.0, show_default=True)
@click.option('--ntrans', default=1, show_default=True, help='number of transformations in SGC and APPNP')
@click.option('--with_bn', is_flag=True, show_default=True)
@click.option('--no_buff', is_flag=True, show_default=True,
              help='skip the buffer generation and use existing in geom,sfgc')
@click.option('--batch_adj', default=1, show_default=True, help='batch size for msgc')
# model specific args
@click.option('--alpha', default=0.1, help='for appnp', show_default=True)
@click.option('--save_path', '--sp', default=DEFAULT_SAVE_PATH, show_default=True, help='save path for synthetic graph')
@click.option('--data_path', '--dp', default=DEFAULT_DATA_PATH, show_default=True, help='data path for datasets')
@click.option('--eval_whole', '-W', is_flag=True, show_default=True, help='if run on whole graph')
@click.option('--with_structure', default=1, show_default=True, help='if synthesizing structure')

#==========SR=========#
@click.option('--SR', default=False, show_default=True)
@click.option('--d_text', default=512, show_default=True)
@click.option('--lambad', default=10000.0, show_default=True)
@click.option('--debug', default=False, show_default=True)
@click.option('--batch_num', default=256, show_default=True)
@click.option('--feat_name', default="clip", show_default=True)
@click.pass_context
def cli(ctx, **kwargs):
    metric_overridden = ctx.get_parameter_source('metric') == click.core.ParameterSource.COMMANDLINE
    args = dict2obj(kwargs)
    args.metric_name = args.metric
    args._metric_overridden = metric_overridden
    if args.gpu_id >= 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
        args.device = f'cuda:{args.gpu_id}'
    else:
        # if gpu_id=-1, use cpu
        args.device = 'cpu'
    path = args.save_path

    args = method_config(args)
    # setting_config has higher priority than methods_config
    args = setting_config(args)
    for key, value in ctx.params.items():
        if ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE:
            setattr(args, key, value)
    args.metric, args.metric_name = resolve_metric(args.metric)
    if hasattr(args, '_metric_overridden'):
        delattr(args, '_metric_overridden')
    if not os.path.exists(f'{path}/logs/{args.method}'):
        try:
            os.makedirs(f'{path}/logs/{args.method}')
        except:
            print(f'{path}/logs/{args.method} exists!')
    if not os.path.exists(f'{path}/logs/{args.method}/SR'):
        try:
            os.makedirs(f'{path}/logs/{args.method}/SR')
        except:
            print(f'{path}/logs/{args.method}/SR exists!') 
    
    if args.sr:
        logging.basicConfig(filename=f'{path}/logs/{args.method}/SR/{args.dataset}_{args.reduction_rate}.log',
                            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    else:
        logging.basicConfig(filename=f'{path}/logs/{args.method}/{args.dataset}_{args.reduction_rate}.log',
                            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args.logger = logging.getLogger(__name__)
    args.logger.addHandler(logging.StreamHandler())
    args.logger.info(args)
    return args


def get_args():
    return cli(standalone_mode=False)


if __name__ == '__main__':
    cli()
