import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms
from a4_utils import train, collect_metrics, store_results
import pathlib

ROOT_PATH = str(pathlib.Path(__file__).parent.resolve() / 'ex1_outputs') + "/"
NUM_EPOCHS = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 128


class Net(nn.Module):
    """"
    Implement the VGG-11 model
    """
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
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


class SigmoidNet(nn.Module):
    """"
    Implement the VGG-11 model with the sigmoid activation instead of relu
    """
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x


class NoDropOutNet(nn.Module):
    """"
    Implement the VGG-11 model without dropout
    """
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x

def augment_exp(model, device):
    # horizontal flip Test
    transforms_horizontal_flip = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.RandomHorizontalFlip(p=1),
        torchvision.transforms.ToTensor()
    ])

    mnist_testset_horizontal_flip = datasets.MNIST(root='./data', train=False, download=True, transform=transforms_horizontal_flip)
    mnist_testloader_horizontal_flip = torch.utils.data.DataLoader(mnist_testset_horizontal_flip, batch_size=BATCH_SIZE, shuffle=True)
    print("Horizontal flip")
    collect_metrics(model, mnist_testloader_horizontal_flip, device)

    # vertical flip Test
    transforms_vertical_flip = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.RandomVerticalFlip(p=1),
        torchvision.transforms.ToTensor()
    ])

    mnist_testset_vertical_flip = datasets.MNIST(root='./data', train=False, download=True, transform=transforms_vertical_flip)
    mnist_testloader_vertical_flip = torch.utils.data.DataLoader(mnist_testset_vertical_flip, batch_size=BATCH_SIZE,
                                                                   shuffle=True)
    print("Vertical flip")
    collect_metrics(model, mnist_testloader_vertical_flip, device)

    # Show some images that are vertical flipped
    # vertical_flip = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((32, 32)),
    #     torchvision.transforms.RandomVerticalFlip(p=1),
    # ])
    # vertical_flipped = datasets.MNIST(root='./data', train=False, download=True, transform=vertical_flip)
    #
    # for i in range(10):
    #     print(vertical_flipped[i][1])
    #     im = vertical_flipped[i][0]
    #     plt.imshow(im)
    #     plt.gray()
    #     plt.show()

    # Show some images that are horizontal flipped
    # horizontal_flip = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((32, 32)),
    #     torchvision.transforms.RandomHorizontalFlip(p=1),
    # ])
    # horizontal_flipped = datasets.MNIST(root='./data', train=False, download=True, transform=horizontal_flip)
    #
    # for i in range(10):
    #   print(horizontal_flipped[i][1])
    #   im = horizontal_flipped[i][0]
    #   plt.imshow(im)
    #   plt.gray()
    #   plt.show()

    # Gaussian Blur Test
    transforms_gaussian_blur = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        torchvision.transforms.ToTensor()
    ])

    mnist_testset_gaussian_blur = datasets.MNIST(root='./data', train=False, download=True, transform=transforms_gaussian_blur)
    mnist_testloader_gaussian_blur = torch.utils.data.DataLoader(mnist_testset_gaussian_blur, batch_size=BATCH_SIZE,
                                                                 shuffle=True)
    print("Gaussian blur")
    collect_metrics(model, mnist_testloader_gaussian_blur, device)

    # Show some images that are gaussian blurred
    # gaussian_blur = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((32, 32)),
    #     torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)),
    # ])
    # gaussian_blurred = datasets.MNIST(root='./data', train=False, download=True, transform=gaussian_blur)
    #
    # regular = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((32, 32)),
    # ])
    # regular_loader = datasets.MNIST(root='./data', train=False, download=True, transform=regular)
    # for i in range(10):
    #   print(gaussian_blurred[i][1])
    #   im = gaussian_blurred[i][0]
    #   plt.imshow(im)
    #   plt.gray()
    #   plt.show()
    #   im = regular_loader[i][0]
    #   plt.imshow(im)
    #   plt.gray()
    #   plt.show()

def main():
    pathlib.Path(ROOT_PATH).mkdir(parents=True, exist_ok=True)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    vgg11 = Net()
    vgg11.to(device)
    optimizer = torch.optim.SGD(vgg11.parameters(), lr=0.1, momentum=0.9)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg1", results_train, ROOT_PATH)

    # run the augmentation tests
    augment_exp(vgg11, device)

    vgg11 = Net()
    vgg11.to(device)
    optimizer = torch.optim.SGD(vgg11.parameters(), lr=0.01, momentum=0.9)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg2", results_train, ROOT_PATH)

    vgg11 = Net()
    vgg11.to(device)
    optimizer = torch.optim.RMSprop(vgg11.parameters(), lr=0.1, momentum=0.9)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg3", results_train, ROOT_PATH)

    vgg11 = Net()
    vgg11.to(device)
    optimizer = torch.optim.Adam(vgg11.parameters(), lr=0.1)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg4", results_train, ROOT_PATH)

    vgg11 = Net()
    vgg11.to(device)
    optimizer = torch.optim.RMSprop(vgg11.parameters(), lr=0.001)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg5", results_train, ROOT_PATH)

    vgg11 = Net()
    vgg11.to(device)
    optimizer = torch.optim.Adam(vgg11.parameters(), lr=0.001)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg6", results_train, ROOT_PATH)

    vgg11 = SigmoidNet()
    vgg11.to(device)
    optimizer = torch.optim.SGD(vgg11.parameters(), lr=0.1, momentum=0.9)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg7", results_train, ROOT_PATH)

    vgg11 = NoDropOutNet()
    vgg11.to(device)
    optimizer = torch.optim.SGD(vgg11.parameters(), lr=0.1, momentum=0.9)
    results_train = train(vgg11, train_loader, test_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS)
    store_results("vgg8", results_train, ROOT_PATH)


if __name__ == '__main__':
    main()

