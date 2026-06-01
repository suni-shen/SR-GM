import os
import sys

from graphslim.config import get_args
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.condensation import *
from graphslim.condensation import *
from graphslim.sparsification import *
from graphslim.utils import to_camel_case, seed_everything

if __name__ == '__main__':
    args = get_args()
    graph = get_dataset(args.dataset, args)
    seed_everything(args.seed)
    
    if args.method == 'gcond':
        agent = GCond(setting=args.setting, data=graph, args=args)
    elif args.method == 'sgdd':
        agent = SGDD(setting=args.setting, data=graph, args=args)
    elif args.method == 'msgc':
        agent = MSGC(setting=args.setting, data=graph, args=args)
    elif args.method == 'gcdm':
        agent = GCDM(setting=args.setting, data=graph, args=args)
        
    else:
        agent = eval(to_camel_case(args.method))(setting=args.setting, data=graph, args=args)
    reduced_graph = agent.reduce(graph, verbose=args.verbose)
    evaluator = Evaluator(args)
    res_mean, res_std = evaluator.evaluate(reduced_graph, model_type=args.final_eval_model)



