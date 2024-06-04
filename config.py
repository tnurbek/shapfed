import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='ShapFed Algorithm') 


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


arg = add_argument_group('Parameters')
arg.add_argument('--batch_size', type=int, default=64, help='# of images in each batch of data') 
arg.add_argument('--random_seed', type=int, default=1, help='Random seed') 

arg.add_argument('--data_dir', type=str, default='./data/mnist', help='Directory in which data is stored')

arg.add_argument('--save_name', type=str, default='custom_niidv1_lr001_diffdata_hier', help='Name of the model to save as')
arg.add_argument('--model_num', type=int, default=4, help='Number of models to train for CaPriDe')
arg.add_argument('--shapley', type=str2bool, default=True, help='Whether to use Shapley Computation')

arg.add_argument('--use_wandb', type=str2bool, default=True, help='Whether to use wandb for visualization') 
arg.add_argument('--use_tensorboard', type=str2bool, default=True, help='Whether to use tensorboard for visualization') 

arg.add_argument('--split', type=str, default='imbalanced', help='Data aplitting scenario')  # homogeneous, heterogeneous, imbalanced 
arg.add_argument('--alpha', type=float, default=1.0, help='Parameter of Dirichlet distribution') 
arg.add_argument('--rho', type=float, default=0.7, help='Major allocation probability') 
arg.add_argument('--alloc_n', type=int, default=1, help='The number of major allocation probs') 

arg.add_argument('--aggregation', type=int, default=1, help='Aggregation Method: 1, 2') 
arg.add_argument('--num_rounds', type=int, default=50, help='The number of communication rounds') 
arg.add_argument('--num_lepochs', type=int, default=5, help='The number of local epochs') 


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
