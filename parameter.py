THOUSAND = 1000
MILLION = 1000000

import argparse

def create_args():
    # Arguments
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--model', type=str, default='gaussian', choices=['flow', 'gaussian'])
    # parser.add_argument('--input_dim', type=int, default=376)
    parser.add_argument('--position_dim', type=int, default=2)
    parser.add_argument('--encoded_position_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_tissues', type=int, default=5)
    parser.add_argument('--tissue_dim', type=int, default=32)
    parser.add_argument('--expression_dim', type=int, default=374)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--truncate_std', type=float, default=2.0)
    parser.add_argument('--latent_flow_depth', type=int, default=14)
    parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

    # Datasets and loaders
    # parser.add_argument('--dataset_path', type=str, default='/home/zihend1/Diffusion/diffusion-point-cloud/data/shapenet.hdf5')
    # parser.add_argument('--categories', type=str_list, default=['airplane'])
    # parser.add_argument('--scale_mode', type=str, default='shape_unit')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=64)

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--end_lr', type=float, default=1e-4)
    parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)
    parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)
    parser.add_argument('--train_epochs', type=int, default=100)

    # Training
    # parser.add_argument('--seed', type=int, default=2020)
    # parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    # parser.add_argument('--log_root', type=str, default='./logs_gen')
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--max_iters', type=int, default=float('inf'))
    # parser.add_argument('--val_freq', type=int, default=1000)
    # parser.add_argument('--test_freq', type=int, default=30*THOUSAND)
    # parser.add_argument('--test_size', type=int, default=400)
    # parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args(args=[])
    
    # # Filter out Jupyter's default arguments
    # known_args, unknown_args = parser.parse_known_args()
    # filtered_args = [arg for arg in sys.argv if arg in known_args]
    
    return args





