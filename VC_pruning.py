import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from tqdm import tqdm
from tqdm.autonotebook import tqdm

from net.models import VGG
from net.quantization import apply_weight_sharing
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()

# Define which model to use
model = VGG(mask=True).to(device)

#모델 나중에
'''print(model)
util.print_model_parameters(model)
'''
# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()

losses = []
def train(epochs):
   print("TRAINING")
   for epoch in range(epochs):
      pbar = tqdm(enumerate(train_loader), desc="Loss: ", total=len(train_loader))
      total_loss = 0
    
      for batch_idx, (data, target) in pbar:
         data, target = data.to(device), target.to(device)
         optimizer.zero_grad()
         output= model(data)
         loss = criterion(output, target)
         loss.backward()
         
         for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

         optimizer.step()
         current_loss = loss.item()
         total_loss += current_loss
         pbar.set_description("Loss: {:.4f}".format(total_loss/(batch_idx+1)))
      losses.append(total_loss/len(train_loader))
      print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

   #torch.save(net, './save')

def test():
   model.eval()
   total_correct = 0
   total_images = 0
   confusion_matrix = np.zeros([10,10], int)
   with torch.no_grad():     
       for data, target in test_loader:
           data, target = data.to(device), target.to(device)
           outputs = model(data)
           _, predicted = torch.max(outputs.data, 1)
           total_images += target.size(0)
           total_correct += (predicted == target).sum().item()
           for i, l in enumerate(target):
               confusion_matrix[l.item(), predicted[i].item()] += 1 

   model_accuracy = total_correct / total_images * 100
   print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))

# Initial training
print("--- Initial training ---")
train(args.epochs)
accuracy = test()
util.log(args.log, f"initial_accuracy {accuracy}")
torch.save(model, f"saves/initial_model.ptmodel")
print("--- Before pruning ---")
util.print_nonzeros(model)

# Pruning
model.prune_by_std(args.sensitivity)
accuracy = test()
util.log(args.log, f"accuracy_after_pruning {accuracy}")
print("--- After pruning ---")
util.print_nonzeros(model)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(args.epochs)
torch.save(model, f"saves/model_after_retraining.ptmodel")
accuracy = test()
util.log(args.log, f"accuracy_after_retraining {accuracy}")
print("--- After Retraining ---")
util.print_nonzeros(model)
