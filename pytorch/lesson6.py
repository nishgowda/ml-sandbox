import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
        def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 5)
                self.conv2 = nn.Conv2d(32, 64, 5)
                self.conv3 = nn.Conv2d(64, 128, 5)
                
                x = torch.randn(50,50).view(-1,1,50,50)
                self._to_linear = None
                self.convs(x)

                self.fc1 = nn.Linear(self._to_linear, 512)
                self.fc2 = nn.Linear(512, 2)

        def convs(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
                x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
                x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
                
                print(x[0].shape)
                if self._to_linear is None:
                        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
                return x
        
        def forward(self, x):
                x = self.convs(x)
                x = x.view(-1, self._to_linear)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

net = Net()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/225.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_siz:]








