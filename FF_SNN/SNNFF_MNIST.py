#!pip install snntorch
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen, surrogate
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import os
from torch import split
from torch.optim import Adam

torch.manual_seed(42)

beta = 0.99
threshold = 1.5
num_steps = 10

num_inputs = 28*28
num_hidden = 500

spike_grad1 = surrogate.fast_sigmoid()  
spike_grad2 = surrogate.FastSigmoid.apply  
spike_grad3 = surrogate.fast_sigmoid(slope=75) 
spike_grad4 = surrogate.atan(alpha=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad1)
        self.layer_outputs = []
        self.num_steps = num_steps


    def forward(self, x):
        global spk1_rec_global
        mem1 = self.lif1.init_leaky()

        # Record the final layer
        spk1_rec = []
        mem1_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            mem_rec_global = mem1_rec
            spk1_rec_global = spk1_rec

        result = torch.stack(spk1_rec, dim=0).sum(dim=0)

        return result

class LeakyLayer2(nn.Module):

    def __init__(self, input_size, output_size):
        super(LeakyLayer2, self).__init__()
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad1)
        self.layer_outputs = []
        self.num_steps = num_steps

    def forward(self, x):
        global spk1_rec_global

        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            #cur2 = self.fc2(spk1_rec)
            cur2 = self.fc2(spk1_rec_global[step])
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        result = torch.stack(spk2_rec, dim=0).sum(dim=0)

        return result

def visualise_sample(x, title='', sample_index=0):
    img = x[sample_index].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.show()

batch_size=4096


save_path = './data/MNIST/baked/'
os.makedirs(save_path, exist_ok=True)
file_path = lambda x: os.path.join(save_path, x)

transform = Compose([
    ToTensor(),
    Lambda(lambda x: torch.flatten(x))])

train_set = MNIST('./data/', train=True, download=True, transform=transform)
test_set = MNIST('./data/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

test_x, test_y = next(iter(test_loader))

torch.save(test_x, file_path('test_x.pt'))
torch.save(test_y, file_path('test_y.pt'))

def superimpose_label(x, y):
    x = x.clone()
    x[:, :10] = 0
    x[range(x.shape[0]), y] = x.max()
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

x_te = torch.load('./data/MNIST/baked/test_x.pt', device)
y_te = torch.load('./data/MNIST/baked/test_y.pt', device)

n_units = num_hidden

model = nn.Sequential(
    nn.Sequential(LeakyLayer(784, n_units)),

    nn.Sequential(UnitLength(), LeakyLayer2(n_units, n_units)),
).to(device)

train_final_list = {'epoch':[], 'train_acc':[]}
test_final_list = {'epoch':[], 'test_acc':[]}

def print_batch_accuracy(data, targets, train=False):
    output, _ = model(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())


def print_evaluation(epoch=None):
    global model, x_tr, y_tr, x_te, y_te
    error_rate = lambda x, y: 1.0 - torch.mean((x == y).float()).item()
    prediction_error = lambda x, y: error_rate(predict(model, x), y)
    test_error = prediction_error(x_te, y_te)
    epoch_str = 'init' if epoch is None else f"{epoch:>4d}"
    print(f"[{epoch_str}] Test Acc: {(1-test_error)*100:>5.2f}%")

def test_rec(epoch=None):
    global model, x_tr, y_tr, x_te, y_te
    error_rate = lambda x, y: 1.0 - torch.mean((x == y).float()).item()
    prediction_error = lambda x, y: error_rate(predict(model, x), y)
    test_error = prediction_error(x_te, y_te)
    epoch_str = 'init' if epoch is None else f"{epoch:>4d}"

    test_final_list['epoch'].append(epoch)
    test_final_list['test_acc'].append(round(((1-test_error)*100),2))

    with open("test_acc.txt", "w") as f:
        for acc in test_final_list['test_acc']:
            f.write(str(acc))
            f.write("\n")

# %%
# Training parameters
torch.manual_seed(42)
loss_fn = swish_loss #hinton_loss
learning_rate = 0.0008
optimiser = Adam(model.parameters(), lr=learning_rate)
num_epochs = 1 + (300)

# %%
# Train the model
for epoch in range(num_epochs):
    iter_counter = 0

    for x, y in iter(train_loader):
        x = x.to(device)
        y = y.to(device)

        x_pos, x_neg = make_examples(model, x, y)

        # Train layers in turn
        for layer in model:
            h_pos, h_neg = layer(x_pos), layer(x_neg)
            loss = loss_fn(h_pos, h_neg)

            if 75 > epoch >= 50:
              learning_rate = 0.0003 

            if 100 > epoch >= 75:
              learning_rate = 0.00001  

            if epoch >=100:
              learning_rate = 0.000001  


            optimiser.zero_grad()
            loss.backward()

            optimiser.step()
            with torch.no_grad():
                x_pos, x_neg = layer(x_pos), layer(x_neg)

    test_rec(epoch)

    # Evaluate the model on the training and test set
    if (epoch + 1) % 5 == 1:
        print_evaluation(epoch)