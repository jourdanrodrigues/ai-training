from os import path

import torch
from matplotlib import pyplot
from numpy import transpose
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from app.net import Net


class Loader:
    answers = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    def __init__(self, *, data_path: str):
        self.net = Net()
        self.data_path = data_path
        self.model_path = path.join(self.data_path, "cifar_net.pth")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_train_loader(self, *, batch_size: int = 4) -> DataLoader:
        train_set = datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=self.transform)
        return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    def get_test_loader(self, *, batch_size: int = 4) -> DataLoader:
        test_set = datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=self.transform)
        return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    def save_image(self, images: Tensor) -> None:
        img = utils.make_grid(images) / 2 + 0.5  # unnormalize
        pyplot.imshow(transpose(img.numpy(), (1, 2, 0)))
        pyplot.savefig(path.join(self.data_path, "image.png"))

    def load_saved_model(self) -> None:
        self.net.load_state_dict(torch.load(self.model_path))

    def test_network(self) -> None:
        correct_guesses = {classname: 0 for classname in self.answers}
        total_guesses = {classname: 0 for classname in self.answers}

        with torch.no_grad():  # since we're not training, we don't need to calculate the gradients for our outputs
            for images, answers in self.get_test_loader(batch_size=2000):
                outputs = self.net(images)

                _, guesses = torch.max(outputs, 1)

                for answer, guess in zip(answers, guesses):
                    if answer == guess:
                        correct_guesses[self.answers[answer]] += 1
                    total_guesses[self.answers[answer]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_guesses.items():
            accuracy = 100 * float(correct_count) / total_guesses[classname]
            print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

    def perform_train(self, *, loops: int = 2) -> None:
        train_loader = self.get_train_loader()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        for loop in range(loops):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{loop + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        torch.save(self.net.state_dict(), self.model_path)
        print("Finished Training")
