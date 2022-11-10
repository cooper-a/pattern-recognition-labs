import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms
import json
from a4_utils import plot_results, train
import pathlib

ROOT_PATH = str(pathlib.Path(__file__).parent.resolve() / 'ex2_outputs') + "/"
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 64


class Net(nn.Module):
    """"
    Implement a MLP model
    """
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layers(x)
        return x


def main():
    pathlib.Path(ROOT_PATH).mkdir(parents=True, exist_ok=True)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

    mlp = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlp.to(device)

    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    results = train(mlp, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)

    plot_results(results, 'a4_ex2.png', ROOT_PATH, title='MLP')

    with open(ROOT_PATH + 'a4_ex2_results.json', 'w') as fp:
        json.dump(results, fp)

    # save the model
    torch.save(mlp.state_dict(), ROOT_PATH + 'mlp_model.pt')


if __name__ == '__main__':
    main()

