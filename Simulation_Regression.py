import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import argparse
import errno

import os

parser = argparse.ArgumentParser(description='Simulation Regression')

# Basic Setting
parser.add_argument('--data_index', default=1, type=int, help='set data index')
parser.add_argument('--activation', default='tanh', type=str, help='set activation function')
args = parser.parse_args()


class my_Net_tanh(torch.nn.Module):
    def __init__(self):
        super(my_Net_tanh, self).__init__()
        self.gamma = []
        self.fc1 = nn.Linear(2000, 10000)
        self.gamma.append(torch.ones(self.fc1.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc1.bias.shape, dtype=torch.float32))
        self.fc2 = nn.Linear(10000, 100)
        self.gamma.append(torch.ones(self.fc2.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc2.bias.shape, dtype=torch.float32))
        self.fc3 = nn.Linear(100, 10)
        self.gamma.append(torch.ones(self.fc3.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc3.bias.shape, dtype=torch.float32))
        self.fc4 = nn.Linear(10, 1)
        self.gamma.append(torch.ones(self.fc4.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc4.bias.shape, dtype=torch.float32))

    def to(self, *args, **kwargs):
        super(my_Net_tanh, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for index in range(self.gamma.__len__()):
            self.gamma[index] = self.gamma[index].to(device)

    def forward(self, x):
        for i, para in enumerate(self.parameters()):
            para.data.mul_(self.gamma[i])
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

    def mask(self, user_gamma, device):
        for i, para in enumerate(self.parameters()):
            if self.gamma[i].shape != user_gamma[i].shape:
                print('size doesn\'t match')
                return 0
        for i, para in enumerate(self.parameters()):
            self.gamma[i].data = torch.tensor(user_gamma[i], dtype=torch.float32).to(device)


class my_Net_relu(torch.nn.Module):
    def __init__(self):
        super(my_Net_relu, self).__init__()
        self.gamma = []
        self.fc1 = nn.Linear(2000, 10000)
        self.gamma.append(torch.ones(self.fc1.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc1.bias.shape, dtype=torch.float32))
        self.fc2 = nn.Linear(10000, 100)
        self.gamma.append(torch.ones(self.fc2.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc2.bias.shape, dtype=torch.float32))
        self.fc3 = nn.Linear(100, 10)
        self.gamma.append(torch.ones(self.fc3.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc3.bias.shape, dtype=torch.float32))
        self.fc4 = nn.Linear(10, 1)
        self.gamma.append(torch.ones(self.fc4.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc4.bias.shape, dtype=torch.float32))

    def to(self, *args, **kwargs):
        super(my_Net_relu, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for index in range(self.gamma.__len__()):
            self.gamma[index] = self.gamma[index].to(device)

    def forward(self, x):
        for i, para in enumerate(self.parameters()):
            para.data.mul_(self.gamma[i])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def mask(self, user_gamma, device):
        for i, para in enumerate(self.parameters()):
            if self.gamma[i].shape != user_gamma[i].shape:
                print('size doesn\'t match')
                return 0
        for i, para in enumerate(self.parameters()):
            self.gamma[i].data = torch.tensor(user_gamma[i], dtype=torch.float32).to(device)


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
    data_index = args.data_index
    subn = 500


    prior_sigma_0_init = 0.00005

    prior_sigma_0 = prior_sigma_0_init


    prior_sigma_0_anneal = 0.000001

    prior_sigma_1 = 0.01

    lambda_n = 0.0000001


    NTrain = 10000
    Nval = 1000
    NTest = 1000
    TotalP = 2000

    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    x_test_confidence = np.matrix(np.zeros([NTest, TotalP]))
    y_test_confidence = np.matrix(np.zeros([NTest, 1]))


    temp = np.matrix(pd.read_csv("./data/" + str(data_index) + "/x_train.csv"))
    x_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/" + str(data_index) + "/y_train.csv"))
    y_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/" + str(data_index) + "/x_val.csv"))
    x_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/" + str(data_index) + "/y_val.csv"))
    y_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/" + str(data_index) + "/x_test.csv"))
    x_test[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/" + str(data_index) + "/y_test.csv"))
    y_test[:, :] = temp[:, 1:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    temp = np.matrix(pd.read_csv("./data/" + str(1) + "/x_test.csv"))
    x_test_confidence[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/" + str(1) + "/y_test.csv"))
    y_test_confidence[:, :] = temp[:, 1:]

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)


    x_test_confidence = torch.FloatTensor(x_test_confidence).to(device)
    y_test_confidence = torch.FloatTensor(y_test_confidence).to(device)

    num_seed = 1


    num_selection_list = np.zeros([num_seed])
    num_selection_true_list = np.zeros([num_seed])
    train_loss_list = np.zeros([num_seed])
    val_loss_list = np.zeros([num_seed])
    test_loss_list = np.zeros([num_seed])

    for my_seed in range(num_seed):
        np.random.seed(data_index + my_seed)
        torch.manual_seed(data_index + my_seed)

        if args.activation == 'tanh':
            net = my_Net_tanh()
        elif args.activation == 'relu':
            net = my_Net_relu()
        else:
            print('unrecognized activation function')
            exit(0)

        net.to(device)
        loss_func = nn.MSELoss()


        sigma = torch.FloatTensor([1]).to(device)

        c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
        c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
        threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

        PATH = './result/' + args.activation + '/test_run/'

        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        show_information = 5000


        step_lr = 0.001
        step_momentum = 0.9

        temperature = 0

        optimization = SGHMC(net.parameters(), lr=step_lr, momentum=step_momentum, weight_decay=0, temperature=temperature)

        max_loop = 80001


        anneal_start = 5000

        anneal_end = 20000

        prior_anneal_end = 60000



        para_path = []
        para_gamma_path = []
        for para in net.parameters():
            para_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))
            para_gamma_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))

        train_loss_path = np.zeros([max_loop])
        val_loss_path = np.zeros([max_loop])
        test_loss_path = np.zeros([max_loop])

        confidence_interval = 100

        train_output_path = np.zeros([max_loop // confidence_interval + 1, NTrain])
        test_output_path = np.zeros([max_loop // confidence_interval + 1, NTest])

        for iter_index in range(max_loop):
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            if iter_index < anneal_start:
                anneal_lambda = 0
                temperature = 0
                for para in optimization.param_groups:
                    para['temperature'] = 1.0 * temperature / NTrain

            elif iter_index < anneal_end:
                anneal_lambda = iter_index * 1.0 / anneal_end
                temperature = 0.1
                for para in optimization.param_groups:
                    para['temperature'] = 1.0 * temperature / NTrain

            else:
                anneal_lambda = 1
                if iter_index <= prior_anneal_end:
                    temperature = 0.1
                else:
                    temperature = 0.1 * 1.0 / (iter_index - prior_anneal_end)
                for para in optimization.param_groups:
                    para['temperature'] = 1.0 * temperature / NTrain


            if iter_index < anneal_end:
                prior_sigma_0 = prior_sigma_0_init
            if iter_index >= anneal_end and iter_index < prior_anneal_end:
                prior_sigma_0 = (iter_index - anneal_end)*1.0/(prior_anneal_end - anneal_end) * prior_sigma_0_anneal + (prior_anneal_end - iter_index)*1.0/(prior_anneal_end - anneal_end) * prior_sigma_0_init
            if iter_index >= prior_anneal_end:
                prior_sigma_0 = prior_sigma_0_anneal

            c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
            c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
            threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                    0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

            net.zero_grad()
            output = net(x_train[subsample,])
            loss = loss_func(output, y_train[subsample,])

            train_loss_path[iter_index] = loss.cpu().data.numpy()

            loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))

            loss.backward()

            # prior gradient
            with torch.no_grad():
                for para in net.parameters():
                    temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                    prior_grad = temp.div(NTrain)
                    para.grad.data -= anneal_lambda * prior_grad


            optimization.step()

            with torch.no_grad():
                output = net(x_val)
                loss = loss_func(output, y_val)
                val_loss_path[iter_index] = loss.cpu().data.numpy()
                output = net(x_test)
                loss = loss_func(output, y_test)
                test_loss_path[iter_index] = loss.cpu().data.numpy()


            if iter_index % confidence_interval == 0:
                with torch.no_grad():
                    output = net(x_train)
                    train_output_path[iter_index//confidence_interval, :] = output.view(-1).cpu().data.numpy()

                    output = net(x_test_confidence)
                    test_output_path[iter_index//confidence_interval, :] = output.view(-1).cpu().data.numpy()

            if iter_index % show_information == 0:
                print('iteration:', iter_index)
                with torch.no_grad():

                    print("train loss:", train_loss_path[iter_index])

                    print("val loss:", val_loss_path[iter_index])

                    print("test loss:", test_loss_path[iter_index])

                    print('sigma:', sigma)

                    for i, para in enumerate(net.parameters()):
                        para_path[i][iter_index // show_information,] = para.cpu().data.numpy()
                        para_gamma_path[i][iter_index // show_information,] = (para.abs() > threshold).cpu().data.numpy()

                    print('number of 1:', np.sum(np.max(para_gamma_path[0][iter_index // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path[0][iter_index // show_information,], 0) > 0)[0:5]))

        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '_' + str(subn) + '_' + str(
            lambda_n) + '_' + str(prior_sigma_0) + '_' + str(prior_sigma_1) + '.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path, para_gamma_path, train_loss_path, val_loss_path, test_loss_path, train_output_path, test_output_path], f, protocol=4)
        f.close()

        num_selection_list[my_seed] = np.sum(np.max(para_gamma_path[0][-1,], 0) > 0)
        num_selection_true_list[my_seed] = np.sum((np.max(para_gamma_path[0][-1,], 0) > 0)[0:5])

        user_gamma = []
        for index in range(para_gamma_path.__len__()):
            user_gamma.append(para_gamma_path[index][-1,])

        with torch.no_grad():
            for i, para in enumerate(net.parameters()):
                para.data = torch.FloatTensor(para_path[i][-1,]).to(device)

        net.mask(user_gamma, device)

        fine_tune_loop = 40001


        para_path_fine_tune = []
        para_gamma_path_fine_tune = []

        for para in net.parameters():
            para_path_fine_tune.append(np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))
            para_gamma_path_fine_tune.append(np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))


        train_loss_path_fine_tune = np.zeros([fine_tune_loop ])
        val_loss_path_fine_tune = np.zeros([fine_tune_loop ])
        test_loss_path_fine_tune = np.zeros([fine_tune_loop ])


        train_output_path_fine_tune = np.zeros([fine_tune_loop // confidence_interval + 1, NTrain])
        test_output_path_fine_tune = np.zeros([fine_tune_loop // confidence_interval + 1, NTest])


        step_lr = 0.005
        step_momentum = 0.9
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr, momentum=step_momentum, weight_decay=0)



        for iter_index in range(fine_tune_loop):
            if subn == NTrain:
                subsample = range(NTrain)
            else:
                subsample = np.random.choice(range(NTrain), size=subn, replace=False)

            net.zero_grad()
            output = net(x_train[subsample,])
            loss = loss_func(output, y_train[subsample,])

            train_loss_path_fine_tune[iter_index] = loss.cpu().data.numpy()

            loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))

            loss.backward()


            optimization.step()

            with torch.no_grad():
                output = net(x_val)
                loss = loss_func(output, y_val)
                val_loss_path_fine_tune[iter_index] = loss.cpu().data.numpy()
                output = net(x_test)
                loss = loss_func(output, y_test)
                test_loss_path_fine_tune[iter_index] = loss.cpu().data.numpy()

            if iter_index % confidence_interval == 0:
                with torch.no_grad():
                    output = net(x_train)
                    train_output_path_fine_tune[iter_index//confidence_interval, :] = output.view(-1).cpu().data.numpy()
                    output = net(x_test_confidence)
                    test_output_path_fine_tune[iter_index//confidence_interval, :] = output.view(-1).cpu().data.numpy()



            if iter_index % show_information == 0:
                print('iteration:', iter_index)
                with torch.no_grad():
                    print("train loss:", train_loss_path_fine_tune[iter_index])
                    print("val loss:", val_loss_path_fine_tune[iter_index])
                    print("test loss:", test_loss_path_fine_tune[iter_index])
                    print('sigma:', sigma)

                    for i, para in enumerate(net.parameters()):
                        para_path_fine_tune[i][iter_index // show_information,] = para.cpu().data.numpy()
                        para_gamma_path_fine_tune[i][iter_index // show_information,] = (
                                    para.abs() > threshold).cpu().data.numpy()
                    print('number of 1:',
                          np.sum(np.max(para_gamma_path_fine_tune[0][iter_index // show_information,], 0) > 0))
                    print('number of true:',
                          np.sum((np.max(para_gamma_path_fine_tune[0][iter_index // show_information,], 0) > 0)[0:5]))

        import pickle

        filename = PATH + 'data_' + str(data_index) + "_simu_" + str(my_seed) + '_' + str(subn) + '_' + str(
            lambda_n) + '_' + str(
            prior_sigma_0) + '_' + str(prior_sigma_1) + '_fine_tune.txt'
        f = open(filename, 'wb')
        pickle.dump([para_path_fine_tune, para_gamma_path_fine_tune, train_loss_path_fine_tune, val_loss_path_fine_tune,
                     test_loss_path_fine_tune, train_output_path_fine_tune, test_output_path_fine_tune], f, protocol=4)
        f.close()

        output = net(x_train)
        loss = loss_func(output, y_train)
        print("Train Loss:", loss)
        train_loss_list[my_seed] = loss.cpu().data.numpy()

        output = net(x_val)
        loss = loss_func(output, y_val)
        print("Val Loss:", loss)
        val_loss_list[my_seed] = loss.cpu().data.numpy()

        output = net(x_test)
        loss = loss_func(output, y_test)
        print("Test Loss:", loss)
        test_loss_list[my_seed] = loss.cpu().data.numpy()

    import pickle

    filename = PATH + 'data_' + str(data_index) + '_result.txt'
    f = open(filename, 'wb')
    pickle.dump([num_selection_list,
                 num_selection_true_list, train_loss_list, val_loss_list, test_loss_list], f, protocol=4)
    f.close()


if __name__ == '__main__':
    main()