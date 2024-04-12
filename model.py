"""
Create model architecture
"""
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter:int):
        """
        Arguments:
             in_channels: number of channels for input image (ex: 3 channels for color images and 1 channel for gray images).
             out_channels: number of classes.
             filter: base number for fileters.
        """
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            self._block(in_channels = in_channels, out_channels = filter, kernel_size = (3,3), stride = 1, padding = 1),
            self._block(in_channels = filter, out_channels = filter*2, kernel_size = (3,3), stride = (2,2), padding = 1),
            nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2), padding = 0),
            self._block(in_channels = filter*2, out_channels = filter*4, kernel_size = (3,3), stride = (2,2), padding = 1),
            nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2), padding = 0),
            self._block(in_channels = filter*4, out_channels = filter*8, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2), padding = 0),
            self._block(in_channels = filter*8, out_channels = filter*16, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2), padding = 0),
            self._block(in_channels = filter*16, out_channels = filter*32, kernel_size = (2,2), stride = (1,1), padding = 1),
            nn.Flatten(),
            nn.Linear(in_features = 4*4*1024 , out_features = 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features = 4096, out_features=out_channels)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)




