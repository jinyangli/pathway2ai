import os
import time
import argparse
import numpy as np

from dataclasses import dataclass

from sklearn.datasets import make_classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

use_my_dp = False

"""
ModelConfig contains the model and training parameters
with their default values
"""
@dataclass
class ModelConfig():
    n_layers : int = 4
    d_in : int = 1024
    n_classes: int = 32
    d_hidden : int = 256
    batch_size : int = 64
    train_size : int = 10000
    niters: int = 1000
    lr : float = 1e-1 #learning rate

def reduce_grad(grad):
    #print(f'rank={dist.get_rank()} backward: reduce gradient')
    dist.all_reduce(grad)
    return grad

class MyDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        #make all weights the same
        for n, p in self.model.named_parameters():
            #print(f'making {n} the same')
            dist.broadcast(p.data, src=0)
            p.register_hook(reduce_grad)
    def forward(self, data, target=None):
        return self.model.forward(data, target)

"""
check_parameter_closeness is a debugging function that checks with all workers have 
(almost) identical parameter weights
"""
def check_parameter_closeness(model):
    for n, p in model.named_parameters():
        if dist.get_rank() == 0:
            l = [torch.zeros_like(p.data) for _ in range(dist.get_world_size())]
        else:
            l = None
        dist.gather(p.data, l, dst=0)
        if dist.get_rank() == 0:
            for i in range(1, len(l)):
                if torch.allclose(l[0], l[i]) is False:
                    return False
    return True

"""
MLP is a simple classification model with n_layer linear layers.
The weights of the layers are: 
d_in x d_hidden, d_hidden x d_hidden, ..., d_hidden x n_classes
The activation function is F.tanh
"""
class MLP(nn.Module):

    def __init__(self, config, distributed=True):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        if config.n_layers == 1:
            l = nn.Linear(config.d_in, config.n_classes, bias=False)
            torch.nn.init.kaiming_uniform_(l.weight)
            self.layers.append(l)
        elif config.n_layers > 1:
            # TODO: create n_layers linear layers with the following weights:
            # d_in x d_hidden, d_hidden x d_hidden, ..., d_hidden x n_classes
            # do not use bias for now.
            # Hint: look at the body of the previous branch. The way you 
            # construct linear layers will be similar to that
            l = nn.Linear(config.d_in, config.d_hidden, bias=False)
            torch.nn.init.kaiming_uniform_(l.weight)
            self.layers.append(l)
            for i in range(config.n_layers-2):
                l = nn.Linear(config.d_hidden, config.d_hidden, bias=False)
                torch.nn.init.kaiming_uniform_(l.weight)
                self.layers.append(l)
            l = nn.Linear(config.d_hidden, config.n_classes, bias=False)
            torch.nn.init.kaiming_uniform_(l.weight)
            self.layers.append(l)
            
            
    def forward(self, data, targets=None):
         out = data
         i = 0
         for layer in self.layers:
             #TODO: fill in the body of this loop to perform computation on the forward path
             # first,do "out = layer(out)" to apply one linear layer to the input (output from previous layer)
             # then, apply non-linear activation function F.tanh to output
             # then repeat this process using the next layer
             # note that you should not apply activation on the output of the last linear layer
             i += 1
             out = layer(out)
             if i < len(self.layers):
                 out = F.tanh(out)
         loss = None
         if targets is not None:
             loss = F.cross_entropy(out, targets)
         return out, loss

"""
evaluate the mean loss of the model over a given dataset
"""
@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()

    #TODO: add your code here to calculate the average loss of 
    # of the model over the given dataset
    loader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        X, Y = batch
        X = torch.squeeze(X)
        Y = torch.squeeze(Y)
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

"""
gen_dataset uses scikit-learn's make_classification to 
generate artificial samples for classification
"""
def gen_dataset(config, rand_state):
    nX, nY = make_classification(n_samples=config.train_size, n_features=config.d_in, 
                    n_informative=config.d_in//2, n_classes=config.n_classes, n_clusters_per_class=10,
                    random_state=rand_state)
    X = torch.from_numpy(nX.astype(np.float32))
    Y = torch.from_numpy(nY).type(torch.LongTensor)
    return X, Y

"""
train performs the training loop
"""
def train(rank, world_size, config):
    if world_size > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.set_default_device('cpu')
    
    X, Y = gen_dataset(config, 0)
    
    #split the dataset so that 10% is test and the rest is training dataset
    split_idx = int(0.1*len(X))
    train_dataset = data.TensorDataset(X[:split_idx],Y[:split_idx])
    test_dataset = data.TensorDataset(X[-split_idx:],Y[-split_idx:])
    sampler = data.BatchSampler(data.SequentialSampler(train_dataset), batch_size=config.batch_size, drop_last=True)
    train_data_loader = data.DataLoader(dataset=train_dataset, sampler=sampler)
    test_data_loader = data.DataLoader(dataset=test_dataset)
    
    step = 0
    model = MLP(config)
    if world_size > 1:
        if use_my_dp:
            model = MyDP(model)
        else:
            model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=config.niters)

    model.train() # set the model to be in training mode
    while True:
        for i, batch in enumerate(train_data_loader):
            if (i % world_size) != rank:
                continue
            X, Y = batch
            X= torch.squeeze(X)
            Y= torch.squeeze(Y)

            t0 = time.time()
            _, loss = model(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()
                
            if step % 100 == 0:
                train_loss = evaluate(model, train_dataset, config.batch_size)
                test_loss = evaluate(model, train_dataset, config.batch_size)
                print(f"rank={rank} step {step} | train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | step time {(t1-t0)*1000:.2f}ms")
                check_parameter_closeness(model)

            scheduler.step()

            step += 1
            if step >= config.niters:
                return


#----------------------------------------------------------------------------

if __name__ == "__main__":

    dc = ModelConfig()

    parser = argparse.ArgumentParser(description="MLP training")
    parser.add_argument('--world-size', type=int, default=2, help="number of workers")
    parser.add_argument('--n-layers', type=int, default=dc.n_layers, help="number of layers")
    parser.add_argument('--dim-in', type=int, default=dc.d_in, help="input data dimension")
    parser.add_argument('--n-classes', type=int, default=dc.n_classes, help="output data dimension")
    parser.add_argument('--dim-hidden', type=int, default=dc.d_hidden, help="hidden dimension")
    parser.add_argument('--train-size', type=float, default=dc.train_size, help="learning rate")
    parser.add_argument('--batch-size', type=float, default=dc.batch_size, help="learning rate")
    parser.add_argument('--learning-rate', type=float, default=dc.lr, help="learning rate")
    parser.add_argument('--niters', type=int, default=dc.niters, help="number of iterations")
    parser.add_argument('--port', type=int, default=29000, help="base network port of workers")
    args = parser.parse_args()
    print(vars(args))


    config = ModelConfig(n_layers=args.n_layers, 
            d_in = args.dim_in, n_classes = args.n_classes, d_hidden=args.dim_hidden, 
            batch_size=args.batch_size, train_size = args.train_size, niters=args.niters, lr=args.learning_rate)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.port)
    mp.spawn(train, args=(args.world_size, config), nprocs=args.world_size, join=True)
   # train(config) 
