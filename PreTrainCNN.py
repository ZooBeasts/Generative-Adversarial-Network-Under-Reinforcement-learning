
import torch.nn as nn

class PretrainedCNN(nn.Module):
    def __init__(self):
        super(PretrainedCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.flatten = nn.Flatten()
        self.fc_cnn = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 21),  # Output layer for 21-point regression
        )

    def forward(self, img):
        x = self.encoder(img)
        x = self.flatten(x)
        x = self.fc_cnn(x)
        return self.fc_combined(x)





