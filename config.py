import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='ShapFedClass') 


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--num_classes', type=int, default=4, help='Number of classes to classify') 
data_arg.add_argument('--batch_size', type=int, default=64, help='# of images in each batch of data') 
data_arg.add_argument('--shuffle', type=str2bool, default=True, help='Whether to shuffle the train indices') 

# training params
train_arg = add_argument_group('Training Params') 
train_arg.add_argument('--is_train', type=str2bool, default=True, help='Whether to train or test the model') 
train_arg.add_argument('--momentum', type=float, default=0.9, help='Momentum value')
train_arg.add_argument('--epochs', type=int, default=1, help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate value')
train_arg.add_argument('--weight_decay', type=float, default=5e-4, help='value of weight dacay for regularization')
train_arg.add_argument('--nesterov', type=str2bool, default=True, help='Whether to use Nesterov momentum')
train_arg.add_argument('--lr_patience', type=int, default=10, help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=100, help='Number of epochs to wait before stopping train')
train_arg.add_argument('--gamma', type=float, default=0.1, help='value of learning rate decay') 

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True, help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=False, help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1, help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/mnist', help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/', help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False, help='Whether to resume training from checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10, help='How frequently to print training details')
misc_arg.add_argument('--save_name', type=str, default='custom_niidv1_lr001_diffdata_hier', help='Name of the model to save as')
misc_arg.add_argument('--model_num', type=int, default=4, help='Number of models to train for CaPriDe')
misc_arg.add_argument('--malignant_num', type=int, default=0, help='Number of poisonous models')
misc_arg.add_argument('--shapley', type=str2bool, default=True, help='Whether to use Shapley Computation')
misc_arg.add_argument('--total_lambda', type=int, default=1, help='Total sum of lambda coefficient')
misc_arg.add_argument('--intersection', type=float, default=0.0, help='Intersection of data points between data distributions of parties') 

misc_arg.add_argument('--use_wandb', type=str2bool, default=True, help='Whether to use wandb for visualization') 
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True, help='Whether to use tensorboard for visualization') 

data_arg.add_argument('--split', type=str, default='imbalanced', help='Data aplitting scenario')  # homogeneous, heterogeneous, imbalanced 
misc_arg.add_argument('--alpha', type=float, default=1.0, help='Parameter of Dirichlet distribution') 
misc_arg.add_argument('--rho', type=float, default=0.7, help='Major allocation probability') 
misc_arg.add_argument('--alloc_n', type=int, default=1, help='The number of major allocation probs') 

misc_arg.add_argument('--aggregation', type=int, default=1, help='Aggregation Method: 1, 2') 
misc_arg.add_argument('--num_rounds', type=int, default=50, help='The number of comm. rounds') 
misc_arg.add_argument('--num_lepochs', type=int, default=5, help='The number of local epochs') 


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed