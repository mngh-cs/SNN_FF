# !pip install snntorch
# !pip install tonic

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen, surrogate
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import split
from torch.optim import Adam
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Grayscale
from torch.utils.data import DataLoader, Dataset, Subset
import os
import random
import numpy as np

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset

beta = 0.9
threshold = 1.0
num_steps = 10

spike_grad1 = surrogate.fast_sigmoid()  
spike_grad2 = surrogate.FastSigmoid.apply 
spike_grad3 = surrogate.fast_sigmoid(slope=75) 
spike_grad4 = surrogate.atan(alpha=2)

input_size=2312
output_size=500

torch.manual_seed(42)

class UnitLength(nn.Module):
    def forward(self, x):
        return F.normalize(x)


class LayerOutputs:
    def __init__(self, model, x):
        self.layers = iter(model)
        self.x = x

    def __iter__(self):
        return self

    def __next__(self):
        layer = next(self.layers)
        self.x = layer(self.x)
        return self.x

spk1_rec_global = []
fc1_global = []

class LeakyLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(LeakyLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.lif1 = snn.RLeaky(beta=beta, threshold=threshold, spike_grad=spike_grad4, linear_features=500, learn_threshold=True, learn_beta=True)
        self.layer_outputs = []
        self.num_steps = num_steps


    def forward(self, x):
        global spk1_rec_global
        spk1, mem1 = self.lif1.init_rleaky()

        spk1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x.select(dim=1, index=step))
            spk1, mem1 = self.lif1(cur1, spk1, mem1)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            mem_rec_global = mem1_rec
            spk1_rec_global = spk1_rec

        result = torch.stack(spk1_rec, dim=0).sum(dim=0)

        return result

class LeakyLayer2(nn.Module):

    def __init__(self, input_size, output_size):
        super(LeakyLayer2, self).__init__()
        self.fc2 = nn.Linear(input_size, output_size)
        self.lif2 = snn.RLeaky(beta=beta, threshold=threshold, spike_grad=spike_grad4, linear_features=500, learn_threshold=True, learn_beta=True)
        self.layer_outputs = []
        self.num_steps = num_steps

    def forward(self, x):
        global spk1_rec_global
        spk2, mem2 = self.lif2.init_rleaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur2 = self.fc2(spk1_rec_global[step])
            cur2 = F.normalize(cur2)
            spk2, mem2 = self.lif2(cur2, spk2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        result = torch.stack(spk2_rec, dim=0).sum(dim=0)

        return result

class Input_Layer(nn.Module):

    def __init__(self):
        super(Input_Layer, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        result = x.sum(dim=1)
        return result


def visualise_sample(x, title='', sample_index=0):
    img = x[sample_index].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.show()

#########################################################

save_path = './data/CIFAR10/baked/'
os.makedirs(save_path, exist_ok=True)
file_path = lambda x: os.path.join(save_path, x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sensor_size = tonic.datasets.NMNIST.sensor_size
print("sensor_size", sensor_size)


frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, n_time_bins=num_steps)
                                     ])

batch_size=4096

train_set = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, stabilize=True, train=True)
test_set = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, stabilize=True, train=False)

train_loader = DataLoader(train_set, batch_size=len(train_set),collate_fn=tonic.collation.PadTensors(), shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), collate_fn=tonic.collation.PadTensors(), shuffle=False)

train_x, train_y = next(iter(train_loader))
test_x, test_y = next(iter(test_loader))
test_x = test_x.to(device)
test_y = test_y.to(device)

train_x = train_x.view(train_x.size(0), train_x.size(1), -1)
test_x = test_x.view(test_x.size(0), test_x.size(1), -1)

def superimpose_label(x, y):
    x = x.clone()
    x[:, :, :10] = 0
    x[range(x.shape[0]), : , y] = x.max()
    return x

# %%
def goodness(h):
    return h.pow(2).mean(1)

@torch.no_grad()
def goodness_per_class(model, x):
    g_per_label = []
    for label in range(10):
        x_candidate = superimpose_label(x, label)
        g_candidate = sum(goodness(h) for h in LayerOutputs(model, x_candidate))
        g_per_label.append(g_candidate.unsqueeze(1)) # type: ignore
    return torch.cat(g_per_label, 1)

@torch.no_grad()
def predict(model, x):
    return goodness_per_class(model, x).argmax(1)

# %%
def make_examples(model, x, y_true, epsilon=1e-12):
    g = goodness_per_class(model, x)
    g[range(x.shape[0]), y_true] = 0
    y_hard = torch.multinomial(torch.sqrt(g) + epsilon, 1).squeeze(1)

    x_pos = superimpose_label(x, y_true)
    x_neg = superimpose_label(x, y_hard)
    return x_pos, x_neg

# %%
def swish_loss(h_pos, h_neg, alpha=6.0):
    g_pos, g_neg = goodness(h_pos), goodness(h_neg)
    Delta = g_pos - g_neg
    return F.silu(-alpha * Delta).mean()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
x_tr = train_x.to(device)
y_tr = train_y.to(device)
x_te = test_x.to(device)
y_te = test_y.to(device)

n_units = 500

model = nn.Sequential(
    nn.Sequential(LeakyLayer(2312, n_units)),

    nn.Sequential(LeakyLayer2(n_units, n_units)),
).to(device)

train_final_list = {'epoch':[], 'train_acc':[]}
test_final_list = {'epoch':[], 'test_acc':[]}

def save_list_to_txt(lst, file_path):
    with open(file_path, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

train_acc_path = '/content/train_acc.txt'
test_acc_path = '/content/test_acc.txt'


def print_evaluation(epoch=None):
    global model, x_tr, y_tr, test_x, test_y
    error_rate = lambda x, y: 1.0 - torch.mean((x == y).float()).item()
    prediction_error = lambda x, y: error_rate(predict(model, x), y)
    test_error = prediction_error(test_x, test_y)
    epoch_str = 'init' if epoch is None else f"{epoch:>4d}"
    print(f"[{epoch_str}] Test Acc: {(1-test_error)*100:>5.2f}%")


def test_rec(epoch=None):
    global model, x_tr, y_tr, test_x, test_y
    error_rate = lambda x, y: 1.0 - torch.mean((x == y).float()).item()
    prediction_error = lambda x, y: error_rate(predict(model, x), y)
    test_error = prediction_error(test_x, test_y)
    epoch_str = 'init' if epoch is None else f"{epoch:>4d}"

    test_final_list['epoch'].append(epoch)
    test_final_list['test_acc'].append(round(((1-test_error)*100),2))

torch.manual_seed(42)
loss_fn = swish_loss
learning_rate = 0.001
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 1 + (500)
batch_size = batch_size

# Train the model
print_evaluation()
for epoch in range(num_epochs):

    for x, y in zip(split(x_tr, batch_size), split(y_tr, batch_size)):

        x_pos, x_neg = make_examples(model, x, y)

        for layer in model:
            h_pos, h_neg = layer(x_pos), layer(x_neg)
            loss = loss_fn(h_pos, h_neg)

            if 75 > epoch >=50:
              learning_rate = 0.0005

            if 100 > epoch >=75:
              learning_rate = 0.0001

            if 125 > epoch >=100:
              learning_rate = 0.00005

            if 150 > epoch >=125:
              learning_rate = 0.000001

            if epoch >=150:
              learning_rate = 0.0000001

            optimiser.zero_grad()
            loss.backward()

            optimiser.step()
            with torch.no_grad():
                x_pos, x_neg = layer(x_pos), layer(x_neg)

    test_rec(epoch)

    if (epoch + 1) % 1 == 0:
        print_evaluation(epoch)


print('Max Test Acc:', max(test_final_list['test_acc']))
test_argmax = np.argmax((test_final_list['test_acc']))
print('Epoch of Max Test Acc:', test_argmax)

print('Max Train Acc:', (train_final_list['train_acc'])[test_argmax])