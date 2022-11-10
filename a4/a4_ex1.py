import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms
import json
from a4_utils import plot_results, train
import pathlib

ROOT_PATH = str(pathlib.Path(__file__).parent.resolve() / 'ex1_outputs') + "/"
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
BATCH_SIZE = 64


class Net(nn.Module):
    """"
    Implement the VGG-11 model
    """
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x


def main():
    pathlib.Path(ROOT_PATH).mkdir(parents=True, exist_ok=True)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

    vgg11 = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg11.to(device)

    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE, shuffle=True)

    # optimizer = torch.optim.SGD(vgg11.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # try adam optimizer
    optimizer = torch.optim.Adam(vgg11.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()
    results = train(vgg11, train_loader, test_loader, optimizer, criterion, device, NUM_EPOCHS)

    plot_results(results, 'a4_ex1.png', ROOT_PATH, title='VGG-11')

    with open(ROOT_PATH + 'a4_ex1_results.json', 'w') as fp:
        json.dump(results, fp)

    # save the model
    torch.save(vgg11.state_dict(), ROOT_PATH + 'model.pt')


if __name__ == '__main__':
    main()

