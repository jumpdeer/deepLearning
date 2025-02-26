import argparse

import torch


def createParse():
    parser = argparse.ArgumentParser(description="training Hyperparameters")
    parser.add_argument('--alpha', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--weights', default='non_pretrained', type=str, help='non_pretrained, pretrained')
    parser.add_argument('--device', default='CPU', type=str, help='CPU, 0, 1, 2')
    parser.add_argument('--batch_size', default=20, type=int, help='the batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
    parser.add_argument('--epochs', default=5, type=int, help='number of training loops')
    parser.add_argument('--seed', default=3407, type=int, help='random seed')
    parser.add_argument('--train_path',default=None, type=str, help='train data path')

    args = parser.parse_args()
    return args

def main(args):

    device = torch.device(args.device)
    batch_size = args.batch_size

    print("using {} device.".format(device))



if __name__ == '__main__':
    args = createParse()

    main(args)