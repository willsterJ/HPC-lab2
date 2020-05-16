from __future__ import print_function
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import datasets, transforms
from models.ResNet import *
import time
import os

# Training settings
parser = argparse.ArgumentParser(description='HPC lab2 example')
parser.add_argument('--title', type=str, default='NO TITLE',
                    help="name of this process")
parser.add_argument('--data', type=str, default='./data',
                    help="folder path where data is located.")
parser.add_argument('--cuda', type=int, default=0, choices={0, 1},
                    help="sets the usage of cuda gpu training.")
parser.add_argument('--workers', type=int, default=2,
                    help="sets number of workers used in DataLoader")
parser.add_argument('--optim', type=str, default='sgd', choices={"sgd", "nesterov", "adagrad", "adadelta", "adam"},
                    help="optimers include sgd,")
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train (default: 5)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--decay', type=float, default=5e-4,
                    help='weight decay')
args = parser.parse_args()
print(args)
print("Title: " + args.title)

if torch.cuda.is_available() and args.cuda == 0:  # exercise C5: CPU or GPU
    use_gpu = True
else:
    use_gpu = False

torch.manual_seed(1)

# Load and transform data
data_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # default is p=0.5
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])
if not os.path.isdir("./data"):
    os.mkdir("./data")
trainset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=data_transform)
testset = datasets.CIFAR10(root='./data/', train=False, download=True, transform=data_transform)

# exercise C3, C4: workers are set by argument
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.workers)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# select device, cpu or gpu
# device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

# select model
# net = ResNet18().to(device)
net = ResNet18()
#net = ResNet18NoBatchNorm()  # exercise C7: without batch norm
if use_gpu:
    net = net.cuda()

# if device == 'cuda':
if use_gpu:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Exercise C6: select optimizer and loss function
if args.optim == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optim == "nesterov":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
elif args.optim == "adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.decay)
elif args.optim == "adadelta":
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=args.decay)
elif args.optim == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
criterion = torch.nn.CrossEntropyLoss()


# training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    data_load_time = 0
    data_load_time_start = time.perf_counter()  # begin recording dataload time right before loop starts (i.e. data about to be loaded)
    forward_pass_time = 0
    backward_pass_time = 0
    run_time_start =time.perf_counter()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # exercise C2.1 for data load time
        data_load_time_end = time.perf_counter()
        data_load_time += (data_load_time_end - data_load_time_start)
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        t_start = time.perf_counter()  # exercise C2.2 for training time forward pass
        outputs = net(inputs)
        # loss = criterion(outputs, targets).to(device)
        loss = criterion(outputs, targets)
        if use_gpu:
            loss = loss.cuda()
        t_end = time.perf_counter()
        forward_pass_time += (t_end - t_start)

        t_start = time.perf_counter()  # exercise 3.2 for training time? backward pass
        loss.backward()
        optimizer.step()
        t_end = time.perf_counter()
        backward_pass_time += (t_end - t_start)

        # train_loss += loss.item()
        train_loss += loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        correct += (float)(predicted.eq(targets).sum().cpu().detach())

        if batch_idx % 10 == 0:
            # change loss.data to loss.data.cpu()[0] when running on pytorch 0.3
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}% ({}/{})'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.cpu()[0] / (batch_idx + 1), 100.*correct/total, correct, total))

        data_load_time_start = time.perf_counter()  # reset dataload timer for the next loop

    # after epoch prints
    training_accuracy = 100. * correct / len(train_loader.dataset)  # average accuracy
    print('\ttraining set: Average loss: {:.4f}, Accuracy: {:.2f}% ({}/{})'.format(
        train_loss.data.cpu()[0] / len(train_loader.dataset), training_accuracy, correct, len(train_loader.dataset)))

    # exercise C2.3 for total runtime
    run_time_end = time.perf_counter()
    print("\tdataload_time:%.8f forward_pass_time:%.8f backward_pass_time:%.8f training_time:%.8f total_runtime:%.8f" %
          (data_load_time, forward_pass_time, backward_pass_time, forward_pass_time+backward_pass_time, run_time_end-run_time_start))

    # question Q3: find training parameters and gradients
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total params: " + str(total_params))

    params = list(net.parameters())
    total_grads = 0
    for param in params:
        total_grads += param.grad.view(1, -1).squeeze().shape[0]
    print("total grads: %d" % total_grads)


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
