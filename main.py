import time

import torch
import torchvision
from torch import nn, optim, device
from torch.optim import lr_scheduler
from torchvision.transforms import transforms

from models import CNN, FullyConnected
from visualization import plot_loss

CLASSES = 10
INPUT_SIZE = 28 * 28
BATCH_SIZE = 50
EPOCHS = 20
LEARNING_RATE = 0.01
GAMMA = 0.01
DATA_SIZE = {'train': 10000,
             'test': 5000}


def train(model, loss_fun, optimizer, scheduler, epochs, data_loaders, dataset_size):
    since = time.time()

    for epoch in range(epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 10)

        ds_size = {'train': 0,
                   'test': 0}

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                ds_size[phase] += inputs.shape[0]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fun(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if ds_size[phase] >= DATA_SIZE[phase]:
                    break

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / DATA_SIZE[phase]
            epoch_acc = running_corrects.double() / DATA_SIZE[phase]

            model.losses[phase].append(epoch_loss)
            model.accuracies[phase].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

    time_elapsed = time.time() - since
    model.time = time_elapsed


def train_model(model, data_loaders, dataset_size):
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=GAMMA)

    print(f'\n\n----------------------- {model.__class__.__name__} -----------------------')
    train(model, loss_func, optimizer, None, EPOCHS, data_loaders, dataset_size)
    print(f'ACC: {model.accuracies["test"][-1]}\n\n')

    plot_loss(model.losses, EPOCHS, BATCH_SIZE, LEARNING_RATE, optimizer.__class__.__name__)
    plot_loss(model.accuracies, EPOCHS, BATCH_SIZE, LEARNING_RATE, optimizer.__class__.__name__, mode='accuracy')

    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST('./data/',
                                            train=True,
                                            download=True,
                                            transform=data_transforms)
    test_data = torchvision.datasets.MNIST('./data/',
                                           train=False,
                                           download=True,
                                           transform=data_transforms)

    data = {'train': train_data,
            'test': test_data}

    data_loaders = {x: torch.utils.data.DataLoader(data[x],
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)
                    for x in ['train', 'test']}

    dataset_size = {x: len(data[x]) for x in ['train', 'test']}
