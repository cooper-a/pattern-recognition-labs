import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets

NUM_EPOCH = 10

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


def train(model, train_loader, optimizer, criterion, device):

    for epoch in range(NUM_EPOCH + 1):
        running_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch: %d, Batch: %d, Loss: %f" % (epoch, i, running_loss / 100))
                running_loss = 0.0




def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    # resize to 32x32
    mnist_trainset.data = np.array([np.array(x.resize((32, 32))) for x in mnist_trainset.data])
    mnist_testset.data = np.array([np.array(x.resize((32, 32))) for x in mnist_testset.data])

    vgg11 = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg11.to(device)

    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    optimizer = torch.optim.SGD(vgg11.parameters(), lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    train(vgg11, train_loader, optimizer, criterion, device)


if __name__ == '__main__':
    main()

