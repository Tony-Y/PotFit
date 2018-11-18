import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

parser = argparse.ArgumentParser(description='Potentail Fitting')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--num_epochs', metavar='N', type=int, default=1000,
                    help='The number of epochs')
args = parser.parse_args()

device = None
if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)
    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0

class Net(nn.Module):
    def __init__(self, activation, num_hidden_units=100, num_layers=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, num_hidden_units, bias=False)
        self.fc2 = nn.ModuleList()
        for i in range(num_layers):
            self.fc2.append(nn.Linear(num_hidden_units, num_hidden_units, bias=False))
        self.fc3 = nn.Linear(num_hidden_units, 1)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        self.eval()
        y = self(x)
        x = x.cpu().numpy().flatten()
        y = y.cpu().detach().numpy().flatten()
        return [x, y]

# Load data
data = np.load('data.npy')
num_data = len(data[0])
print('data:', num_data)

# Grid points for the prediction
d = 1e-3
x_grid = torch.arange(0+d, 1.0, d).unsqueeze(1).to(device)

# Make a model
net = Net(activation=SSP(), num_hidden_units=500, num_layers=5).to(device)
print(net)

state = copy.deepcopy(net.state_dict())

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

lam1 = 1.0/25

pred_freq = 1000
num_batches = 10

best_loss = np.inf
for epoch in range(args.num_epochs):
    scheduler.step()
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    epoch_mse0 = 0.0
    epoch_mse1 = 0.0
    for batch in np.split(indices, num_batches):
        input = torch.tensor(data[0][batch], dtype=torch.float32, device=device).unsqueeze(1)
        target0 = torch.tensor(data[1][batch], dtype=torch.float32, device=device).unsqueeze(1)
        target1 = torch.tensor(data[2][batch], dtype=torch.float32, device=device).unsqueeze(1)

        net.eval()
        input.requires_grad = True

        output0 = net(input)
        output0.sum().backward(retain_graph=True, create_graph=True)
        output1 = input.grad
        input.requires_grad = False

        net.train()

        mse0 = criterion(output0, target0)
        mse1 = criterion(output1, target1)

        epoch_mse0 += mse0.item() * len(batch)
        epoch_mse1 += mse1.item() * len(batch)

        loss = mse0 + lam1 * mse1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_mse0 /= num_data
    epoch_mse1 /= num_data
    epoch_loss = epoch_mse0+lam1*epoch_mse1
    print('epoch', epoch,
          'lr', '{:.7f}'.format(optimizer.param_groups[0]['lr']),
          'mse0', '{:.5f}'.format(epoch_mse0),
          'mse1', '{:.5f}'.format(epoch_mse1),
          'loss', '{:.5f}'.format(epoch_loss))
    if ((epoch+1) % pred_freq) == 0:
        np.save('predictions_{}'.format(epoch+1), net.predict(x_grid))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        state = copy.deepcopy(net.state_dict())

print('Best score:', best_loss)
net.load_state_dict(state)
np.save('predictions_best', net.predict(x_grid))
