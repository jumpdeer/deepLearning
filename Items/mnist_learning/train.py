import torch
import torch.optim as optim
from model import Net


def main():
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.manual_seed(random_seed)
    network = Net()
    optimizer = optim.SGD(network.parameters(),lr=learning_rate,momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []


if __name__ == '__main__':
    main()
