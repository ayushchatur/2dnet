import argparse

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=2, type=int, metavar='e',
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=2, type=int, metavar='b',
                        help='number of batch per gpu')
    parser.add_argument('--retrain', default=0, type=int, metavar='r',
                        help='retrain epochs')
    parser.add_argument('--amp', default="disable", type=str, metavar='m',
                        help='mixed precision')
    parser.add_argument('--out_dir', default=".", type=str, metavar='o',
                        help='default directory to output files')
    parser.add_argument('--num_w', default=1, type=int, metavar='w',
                        help='num of data loader workers')
    parser.add_argument('--new_load', default="false", type=str, metavar='p',
                        help='new data loader')
    parser.add_argument('--prune_amt', default=0.5, type=float, metavar='y',
                        help='prune amount ')
    parser.add_argument('--model', default="ddnet", type=str, metavar='z',
                        help='model type ')
    parser.add_argument('--port', default=9191, type=int, metavar='y',
                        help='port define')
    # options mag/l1_struc/random_unstru
    parser.add_argument('--prune_t', default="l1_stru", type=str, metavar='t',
                        help='pruning type')
    parser.add_argument('--gr_mode', default="none", type=str, metavar='t',
                        help='pruning type')
    parser.add_argument('--gr_backend', default="inductor", type=str, metavar='t',
                        help='pruning type')
    parser.add_argument('--wan', default=-1, type=int, metavar='w',
                        help='enable wandb configuration')
    parser.add_argument('--lr', default=0.0001, type=float, metavar='l',
                        help='enable wandb configuration')
    parser.add_argument('--dr', default=0.95, type=float, metavar='d',
                        help='enable wandb configuration')
    parser.add_argument('--distback', default="gloo", type=str, metavar='k',
                        help='enable wandb configuration')
    parser.add_argument('--enable_profile', default="false", type=str, metavar='x',
                        help='enable profiling ')
    parser.add_argument('--do_infer', default="true", type=str, metavar='g',
                        help='do inference after training')
    parser.add_argument('--enable_gr', default="false", type=str, metavar='g',
                        help='do inference after training')
    parser.add_argument('--schedtype', default="expo", type=str, metavar='z',
                        help='scheduler type')

    return parser