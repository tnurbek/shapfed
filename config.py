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
arg.add_argument('--batch_size', type=int, default=64, help='# of data points in each batch of data') 
arg.add_argument('--random_seed', type=int, default=1, help='random seed') 

arg.add_argument('--data_dir', type=str, default='./data/mnist', help='directory in which data is stored')

arg.add_argument('--save_name', type=str, default='custom_niidv1_lr001_diffdata_hier', help='name of the model to save as')
arg.add_argument('--model_num', type=int, default=4, help='number of participants in a afederated setting')
arg.add_argument('--shapley', type=str2bool, default=True, help='whether to use Shapley computation')

arg.add_argument('--use_wandb', type=str2bool, default=True, help='whether to use wandb for visualization') 
arg.add_argument('--use_tensorboard', type=str2bool, default=True, help='whether to use tensorboard for visualization') 

arg.add_argument('--split', type=str, default='imbalanced', help='data splitting scenario')  # homogeneous, heterogeneous, imbalanced 
arg.add_argument('--alpha', type=float, default=1.0, help='parameter of Dirichlet distribution') 
arg.add_argument('--rho', type=float, default=0.7, help='major allocation probability') 
arg.add_argument('--alloc_n', type=int, default=1, help='the number of major allocation probs') 

arg.add_argument('--aggregation', type=int, default=1, help='aggregation method: 1, 2') 
arg.add_argument('--num_rounds', type=int, default=50, help='the number of communication rounds') 
arg.add_argument('--num_lepochs', type=int, default=5, help='the number of local epochs') 


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
