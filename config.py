import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="MARS")
    
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument('--resume_checkpoint', type=str, help='Path to the checkpoint file for resuming training')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--frames", type=int, default=5)
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--local-rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=1)

    parser.add_argument("--check_point", type=bool, default=False)
    parser.add_argument("--dir", type=str, default=None)
    return parser