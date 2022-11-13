import torch
import tqdm as tqdm
import matplotlib.pyplot as plt
import time
import json

def plot_results(results, filename, path, title):
    plt.plot(results['train_accuracy'], '-o', label='train accuracy')
    plt.plot(results['test_accuracy'], '-o', label='test accuracy')
    plt.title(title + ' Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path + 'accuracy_' + filename)
    plt.show()


    plt.plot(results['train_loss'], '-o', label='train loss')
    plt.plot(results['test_loss'], '-o', label='test loss')
    plt.title(title + ' Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path + 'loss_' + filename)
    plt.show()
def run_metrics(model, loader, device, criterion=None):
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # compute the loss too
            if criterion is not None:
                loss = criterion(output, y)
                running_loss += loss.item()
        average_loss = running_loss / len(loader)
        return {'accuracy': correct / total, 'loss': average_loss}

def collect_metrics(model, test_loader, device, criterion=None):
    model.eval()
    metrics_test = run_metrics(model, test_loader, device, criterion=criterion)
    if criterion is not None:
        print(f"Test loss: {metrics_test['loss']:.3f}")
    print(f"Test accuracy: {metrics_test['accuracy'] * 100:.3f}%")
    return metrics_test

def store_results(title, results, path):
    plot_results(results, title + '.png', path, title='VGG-11')
    with open(title + '.json', 'w') as fp:
        json.dump(results, fp)

def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs):
    accuracies_train = []
    loss_train = []
    accuracies_test = []
    loss_test = []
    print('Training...')
    for epoch in range(num_epochs):
        time.sleep(0.01)
        model.train()
        running_loss = 0.0
        correct = 0
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader)):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            correct += (output.argmax(dim=1) == y).type(torch.float).sum().item()
            running_loss += loss.item()



        accuracy = correct / len(train_loader.dataset)
        loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} of {num_epochs} - Training loss: {loss:.3f}, train accuracy: {accuracy * 100:.3f}%")

        # test the model
        test_results = collect_metrics(model, test_loader, device, criterion)
        accuracies_train.append(accuracy)
        loss_train.append(loss)
        accuracies_test.append(test_results['accuracy'])
        loss_test.append(test_results['loss'])
        # prevent TQDM from messing up the output

    print("Finished Training")
    results = {'train_accuracy': accuracies_train, 'train_loss': loss_train, 'test_accuracy': accuracies_test, 'test_loss': loss_test}
    return results
