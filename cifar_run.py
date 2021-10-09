import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import torch.utils.data

#import torchvision.transforms as transforms
import transforms
import torchvision.datasets as datasets
import resnet
import os
import errno
from torch.utils.data.sampler import SubsetRandomSampler


parser = argparse.ArgumentParser(description='Cifar10 ResNet Compression')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--base_path', default='./result/', type = str, help = 'base path for saving result')
parser.add_argument('--model_path', default='test_run/', type = str, help = 'folder name for saving model')
parser.add_argument('--fine_tune_path', default='fine_tune/', type = str, help = 'folder name for saving fine tune model')

# Resnet Architecture
parser.add_argument('-depth', default=20, type=int, help='Model depth.')

# Random Erasing
parser.add_argument('-p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('-sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('-r1', default=0.3, type=float, help='aspect of erasing area')

# Training Setting
parser.add_argument('--only_fine_tune', default = 0, type = int, help = 'only fine tune')
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr_decay_time', default = [150, 225], type = int, nargs= '+', help = 'when to multiply lr by 0.1')
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')
parser.add_argument('--batch_test', default = 128, type = int, help = 'batch size for testing')

parser.add_argument('--temperature', default = 0.0001, type = float, help = 'temperature in SGHMC')


# Fine Tuning Setting
parser.add_argument('--nepoch_fine_tune', default = 10, type = int, help = 'total number of training epochs in fine tuning')
parser.add_argument('--lr_decay_time_fine_tune', default = [], type = int, nargs= '*', help = 'when to multiply lr by 0.1 in fine tuning')
parser.add_argument('--init_lr_fine_tune', default = 0.001, type = float, help = 'initial learning rate in fine tuning')
parser.add_argument('--momentum_fine_tune', default = 0.9, type = float, help = 'momentum in SGD in fine tuning')

# Prior Setting
parser.add_argument('--sigma0_init', default = 0.00004, type = float, help = 'sigma_0 in prior')
parser.add_argument('--sigma0_end', default = 0.000004, type = float, help = 'sigma_0 in prior')


parser.add_argument('--sigma1', default = 0.04, type = float, help = 'sigma_1 in prior')
parser.add_argument('--lambdan', default = 0.000000005, type = float, help = 'lambda_n in prior')

args = parser.parse_args()




def model_eval(net, data_loader, device, loss_func):
    net.eval()
    correct = 0
    total_loss = 0
    total_count = 0
    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_func(outputs, labels)
        prediction = outputs.data.max(1)[1]
        correct += prediction.eq(labels.data).sum().item()
        total_loss += loss.mul(images.shape[0]).item()
        total_count += images.shape[0]

    return  1.0 * correct / total_count, total_loss / total_count


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class SGHMC(torch.optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, temperature = 1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if temperature < 0.0:
            raise ValueError("Invalid temperature value: {}".format(temperature))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, temperature = temperature)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGHMC, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            temperature = group['temperature']
            lr = group['lr']

            alpha = 1 - momentum
            scale = np.sqrt(2.0*alpha*temperature/lr)

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.add_(torch.ones_like(buf).normal_().mul(scale))
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                else:
                    d_p = d_p.add(torch.ones_like(d_p).normal_().mul(scale))

                p.data.add_(-group['lr'], d_p)

        return loss



def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                          transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3)])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         normalize])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    np.random.seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_test, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_func = nn.CrossEntropyLoss().to(device)

    net = resnet.ResNet_sparse(args.depth, 10).to(device)

    NTrain = len(train_loader.dataset)

    lambda_n = args.lambdan
    prior_sigma_0_init = args.sigma0_init
    prior_sigma_0_anneal = args.sigma0_end

    prior_sigma_0 = prior_sigma_0_init


    prior_sigma_1 = args.sigma1
    momentum = args.momentum

    temperature = 0

    lr = args.init_lr

    c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
    c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

    optimizer = SGHMC(net.parameters(), lr=lr, momentum=momentum, weight_decay=0, temperature = temperature)



    PATH = args.base_path + args.model_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    num_epochs = args.nepoch
    train_accuracy_path = np.zeros(num_epochs)
    train_loss_path = np.zeros(num_epochs)

    test_accuracy_path = np.zeros(num_epochs)
    test_loss_path = np.zeros(num_epochs)
    sparsity_path = np.zeros(num_epochs)

    torch.manual_seed(args.seed)

    NTrain = len(train_loader.dataset)
    best_accuracy = 0
    for epoch in range(num_epochs):
        net.train()
        epoch_training_loss = 0.0
        total_count = 0
        accuracy = 0

        if epoch in args.lr_decay_time:
            for para in optimizer.param_groups:
                para['lr'] = para['lr'] / 10


        anneal_start = 150
        anneal_end = 200

        prior_anneal_end = 225

        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss = loss_func(output, target)

            optimizer.zero_grad()

            loss.backward()

            if epoch < anneal_start:
                anneal_lambda = 0

                temperature = 0
                for para in optimizer.param_groups:
                    para['temperature'] = 1.0 * temperature / NTrain
                    para['weight_decay'] = 5e-4

            elif epoch >= anneal_start and epoch < anneal_end:
                anneal_lambda = 1.0 * (epoch - anneal_start) / (anneal_end - anneal_start)
                temperature = 0.01
                for para in optimizer.param_groups:
                    para['temperature'] = 1.0 * temperature / NTrain
                    para['weight_decay'] = 0
            else:
                anneal_lambda = 1

                if epoch <= prior_anneal_end:
                    temperature = 0.01
                else:
                    temperature = 0.01 * 1.0 / (epoch - prior_anneal_end)
                for para in optimizer.param_groups:
                    para['temperature'] = 1.0 * temperature / NTrain
                    para['weight_decay'] = 0


            if epoch < anneal_end:
                prior_sigma_0 = prior_sigma_0_init
            if epoch >= anneal_end and epoch < prior_anneal_end:
                prior_sigma_0 = (epoch - anneal_end)*1.0/(prior_anneal_end - anneal_end) * prior_sigma_0_anneal + (prior_anneal_end - epoch)*1.0/(prior_anneal_end - anneal_end) * prior_sigma_0_init
            if epoch >= prior_anneal_end:
                prior_sigma_0 = prior_sigma_0_anneal

            c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
            c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
            threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                    0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

            with torch.no_grad():
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(NTrain)
                    para.grad.data -= anneal_lambda * prior_grad
            optimizer.step()

            epoch_training_loss += loss.mul(input.shape[0]).item()
            accuracy += output.data.argmax(1).eq(target.data).sum().item()
            total_count += input.shape[0]
            train_loss_path[epoch] = epoch_training_loss / total_count
            train_accuracy_path[epoch] = accuracy / total_count
        print("epoch: ", epoch, ", train loss: ", epoch_training_loss / total_count, "train accuracy: ",
              accuracy / total_count)

        # calculate test set accuracy
        with torch.no_grad():

            test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
            test_loss_path[epoch] = test_loss
            test_accuracy_path[epoch] = test_accuracy
            print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)

            total_num_para = 0
            non_zero_element = 0
            for name, para in net.named_parameters():
                total_num_para += para.numel()
                non_zero_element += (para.abs() > threshold).sum()
            print('sparsity:', non_zero_element.item() / total_num_para)
            sparsity_path[epoch] = non_zero_element.item() / total_num_para

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'best_model.pt')

            print('best accuracy:', best_accuracy)

        torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path, train_accuracy_path, test_loss_path, test_accuracy_path, sparsity_path], f)
    f.close()

    #-----------------fine tune-------------_#
    PATH = args.base_path + args.model_path
    net.load_state_dict(torch.load(PATH + 'model' + str(args.nepoch - 1) + '.pt'))
    test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
    print("test loss: ", test_loss, "test accuracy: ", test_accuracy)
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))
    user_mask = {}
    for name, para in net.named_parameters():
        user_mask[name] = para.abs() < threshold
    net.set_prune(user_mask)
    test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
    print("test loss: ", test_loss, "test accuracy: ", test_accuracy)

    total_num_para = 0
    non_zero_element = 0
    for name, para in net.named_parameters():
        total_num_para += para.numel()
        non_zero_element += (para != 0).sum()
    print('sparsity:', non_zero_element.item() / total_num_para)

    PATH = args.base_path + args.model_path + args.fine_tune_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr_fine_tune, momentum=args.momentum_fine_tune, weight_decay=0)


    num_epochs = args.nepoch_fine_tune
    train_accuracy_path_fine_tune = np.zeros(num_epochs)
    train_loss_path_fine_tune = np.zeros(num_epochs)

    test_accuracy_path_fine_tune = np.zeros(num_epochs)
    test_loss_path_fine_tune = np.zeros(num_epochs)

    sparsity_path_fine_tune = np.zeros(num_epochs)

    torch.manual_seed(args.seed)

    NTrain = len(train_loader.dataset)
    best_accuracy = 0

    for epoch in range(num_epochs):
        net.train()
        epoch_training_loss = 0.0
        total_count = 0
        accuracy = 0


        if epoch in args.lr_decay_time_fine_tune:
            for para in optimizer.param_groups:
                para['lr'] = para['lr']/10
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_training_loss += loss.mul(input.shape[0]).item()
            accuracy += output.data.argmax(1).eq(target.data).sum().item()
            total_count += input.shape[0]
            train_loss_path_fine_tune[epoch] = epoch_training_loss / total_count
            train_accuracy_path_fine_tune[epoch] = accuracy / total_count
        print("epoch: ", epoch, ", train loss: ", epoch_training_loss / total_count, "train accuracy: ",
              accuracy / total_count)

        # calculate training set accuracy
        with torch.no_grad():

            test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
            test_loss_path_fine_tune[epoch] = test_loss
            test_accuracy_path_fine_tune[epoch] = test_accuracy
            print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)

            total_num_para = 0
            non_zero_element = 0
            for name, para in net.named_parameters():
                total_num_para += para.numel()
                non_zero_element += (para.abs() > threshold).sum()
            print('sparsity:', non_zero_element.item() / total_num_para)
            sparsity_path_fine_tune[epoch] = non_zero_element.item() / total_num_para

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'best_model.pt')
            print('best accuracy:', best_accuracy)

        torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path_fine_tune, train_accuracy_path_fine_tune, test_loss_path_fine_tune,
                 test_accuracy_path_fine_tune, sparsity_path_fine_tune], f)
    f.close()


if __name__ == '__main__':
    main()