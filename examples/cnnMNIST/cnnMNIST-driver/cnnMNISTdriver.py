"""
This example is modified from the HpBandSter example (https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_pytorch_worker.py)
It implements a small CNN in PyTorch to train it on MNIST. 
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.
In this example implements a small CNN in Keras to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.
We'll optimise the following hyperparameters:
+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
+-------------------------+----------------+-----------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]       | only active if         |
|                         |                |                 | optimizer == SGD       |
+-------------------------+----------------+-----------------+------------------------+
| Number of conv layers   | integer        | [1,3]           | can only take integer  |
|                         |                |                 | values 1, 2, or 3      |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | logarithmically varied |
| the first conf layer    |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the second conf layer   |                |                 | of layers >= 2         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the third conf layer    |                |                 | of layers == 3         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in fully connected layer|                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
Please refer to the compute method below to see how those are defined using the
ConfigSpace package.
      
The network does not achieve stellar performance when a random configuration is samples,
but a few iterations should yield an accuracy of >90%. To speed up training, only
8192 images are used for training, 1024 for validation.
The purpose is not to achieve state of the art on MNIST, but to show how to use
PyTorch inside HpBandSter, and to demonstrate a more complicated search space.

"""

import numpy as np
import os, sys, re
import time

try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")



class MNISTConvNet(torch.nn.Module):
    def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, dropout_rate, num_fc_units, kernel_size):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=kernel_size)
        self.conv2 = None
        self.conv3 = None
        
        output_size = (28-kernel_size + 1)//2
        num_output_filters = num_filters_1
        
        if num_conv_layers > 1:
            self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=kernel_size)
            num_output_filters = num_filters_2
            output_size = (output_size - kernel_size + 1)//2

        if num_conv_layers > 2:
            self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=kernel_size)
            num_output_filters = num_filters_3
            output_size = (output_size - kernel_size + 1)//2
        
        self.dropout = nn.Dropout(p = dropout_rate)

        self.conv_output_size = num_output_filters*output_size*output_size

        self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, 10)
        

    def forward(self, x):
        
        # switched order of pooling and relu compared to the original example
        # to make it identical to the keras worker
        # seems to also give better accuracies
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        
        if not self.conv2 is None:
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        if not self.conv3 is None:
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = self.dropout(x)
        
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))



def execute(params, niter=1, budget=None, max_epoch=243, 
            ntrain=8192, nvalid=1024, batch_size=64):
    # Load the MNIST Data here
    device = torch.device('cpu')
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(ntrain))
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(ntrain, ntrain+nvalid))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)
    model = MNISTConvNet(num_conv_layers=params['num_conv_layers'],
                         num_filters_1=params['num_filters_1'],
                         num_filters_2=params['num_filters_2'] if params['num_conv_layers'] > 1 else None,
                         num_filters_3=params['num_filters_3'] if params['num_conv_layers'] > 2 else None,
                         dropout_rate=params['dropout_rate'],
                         num_fc_units=params['num_fc_units'],
                         kernel_size=3)
    
    criterion = torch.nn.CrossEntropyLoss()
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['sgd_momentum'])

    model.train()
    num_epoch = int(budget) if budget != None else max_epoch
    for epoch in range(num_epoch):
        loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
    def evaluate_accuracy(model, data_loader):
        model.eval()
        correct=0
        with torch.no_grad():
            for x, y in data_loader:
                output = model(x)
                #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
        #import pdb; pdb.set_trace()    
        accuracy = correct/len(data_loader.sampler)
        return(accuracy)

    train_accuracy = evaluate_accuracy(model, train_loader)
    validation_accuracy = evaluate_accuracy(model, validation_loader)
    test_accuracy = evaluate_accuracy(model, test_loader)
    

    print()
    print(f"ntrain={ntrain}, nvalid={nvalid}, num_epoch={num_epoch}" )
    print('params: ', params)
    print({'loss': 1-validation_accuracy, 
           'info': {    'test accuracy': test_accuracy,
                        'train accuracy': train_accuracy,
                        'validation accuracy': validation_accuracy,
                        'number of parameters': model.number_of_parameters()}})
    
    return 1-validation_accuracy


def cnnMNISTdriver(params, niter=1, JOBID:int=-1, budget=None, 
              max_epoch=243, ntrain=8192, nvalid=1024, batch_size=64):
    # global EXPDIR 
    # global ROOTDIR

    # MACHINE_NAME = os.environ['MACHINE_NAME']
    # TUNER_NAME = os.environ['TUNER_NAME']
    # EXPDIR = os.path.abspath(os.path.join("cnnMNIST-driver/exp", MACHINE_NAME + '/' + TUNER_NAME))
    # if (JOBID==-1):  # -1 is the default value if jobid is not set from command line
    #     JOBID = os.getpid()
    

    dtype = [("lr", float), ("optimizer", 'U10'), ("sgd_momentum", float), 
             ("num_conv_layers", int),("num_filters_1", int), 
             ("num_filters_2", int), ("num_filters_3", int),
             ("dropout_rate", float), ("num_fc_units", int)]
    params = np.array(params, dtype=dtype)
    
    res = []
    for param in params:
        res_cur = execute(param, niter=niter, budget=budget, max_epoch=max_epoch, 
                          ntrain=ntrain, nvalid=nvalid, batch_size=batch_size)
        res.append(res_cur)
    return res



if __name__ == "__main__":
    os.environ['MACHINE_NAME'] = 'cori'
    os.environ['TUNER_NAME'] = 'GPTune'
    # params = [(1e-3, "Adam", 0.5, 1, 4, 4, 4, 0.5, 8),\
    #           (1e-2, "SGD", 0.2, 2, 8, 8, 8, 0.3, 16)
    #           ]
    # params = [(1e-3, "Adam", 0.5, 1, 4, 4, 4, 0.5, 8)]
    params = [(0.001, 'Adam', 0.3512, 3, 18, 48, 26, 0.0585, 111)]
    res = cnnMNISTdriver(params, niter=1, budget=None, max_epoch=27, 
                          ntrain=2000, nvalid=2000)
    
    print(res)
