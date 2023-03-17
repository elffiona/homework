import torch


class Model(torch.nn.Module):
    """
    Self implemented CNN
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        model of three layers
        """
        super(Model, self).__init__()

        # self.conv1 = torch.nn.Conv2d(
        #     num_channels, 32, kernel_size=3, stride=1, padding=1
        # )
        # self.bn1 = torch.nn.BatchNorm2d(32)
        # self.relu1 = torch.nn.ReLU(inplace=True)
        # self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.bn2 = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU(inplace=True)
        # self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn3 = torch.nn.BatchNorm2d(128)
        # self.relu3 = torch.nn.ReLU(inplace=True)
        # self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        # self.relu4 = torch.nn.ReLU(inplace=True)
        #
        # self.fc2 = torch.nn.Linear(256, num_classes)
        self.conv1 = torch.nn.Conv2d(num_channels, 16, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=5)
        # 6 * 6 * 16
        self.fc1 = torch.nn.Linear(576, 1000)
        self.fc2 = torch.nn.Linear(1000, num_classes)
        # Try initializing the weight
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.pool2(x)
        #
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)

        return x
