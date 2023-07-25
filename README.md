# Understanding parallel deep neural network training

This is a series of exercises to get your hands dirty on parallelizing the training of Deep Neural Networks Ideally, we want to parallelize across multiple GPU devices, possibly over over multiple machines.  However, GPUs are hard to get (esp if you want many of them), so we'll develop and test using your own laptop, on CPU only. The code and the concept is exactly the same, except that on your laptop, we won't be able to get any speedup from "parallelization".

## Exercise 1: Complete the basic training of a toy multi-layer perceptron

Download [mlp.py](https://github.com/jinyangli/pathway2ai/blob/master/mlp.py).  Alternatively, you can clone this git repository to get all the files using command `git clone git@github.com:jinyangli/pathway2ai.git`.

Read the code carefully. Complete the missing code block or function body that follow a comment starting with **TODO**

Run the code. `python mlp.py`. You should see something like

```
$ python mlp.py
{'n_layers': 4, 'dim_in': 1024, 'n_classes': 32, 'dim_hidden': 256, 'train_size': 10000, 'batch_size': 64, 'learning_rate': 0.1, 'niters': 1000, 'port': 29000}                                                                                          
step 0 | train_loss 3.7844 | test_loss 3.7780 | step time 332.70ms
step 100 | train_loss 0.7557 | test_loss 0.7426 | step time 12.70ms
step 200 | train_loss 0.2600 | test_loss 0.2585 | step time 33.89ms
step 300 | train_loss 0.2263 | test_loss 0.2236 | step time 23.02ms
step 400 | train_loss 0.2170 | test_loss 0.2156 | step time 11.63ms
step 500 | train_loss 0.2131 | test_loss 0.2196 | step time 12.49ms
step 600 | train_loss 0.2081 | test_loss 0.2081 | step time 110.72ms
step 700 | train_loss 0.2104 | test_loss 0.2062 | step time 34.14ms
step 800 | train_loss 0.2183 | test_loss 0.2131 | step time 11.67ms
step 900 | train_loss 0.2070 | test_loss 0.2232 | step time 11.66ms
```

## Exercise 2: Play with the built-in communication mechanisms, gather, allreduce etc.

Pytorch's linear algebra operations such as matrix multiplication use existing third-party fast libraries.  For example, when running on CPU, Pytorch typically uses Intel's [MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library) library.  When running on GPU, Pytorch uses NVIDIA CUDA libraries such as [cuDNN](https://developer.nvidia.com/cudnn) or [cuBLAS](https://developer.nvidia.com/cublas#:~:text=The%20cuBLAS%20library%20contains%20extensions,improvements%20and%20new%20GPU%20architectures.)  These libraries parallelize operations over a single GPU device or a single multicore-CPU-machine using multiple cores and vector instrincs.  However, what if we want to further parallelize across multiple GPU devices or multiple machines?  Doing so is beyond the capabilities of existing math libraries like MKL, cuDNN/cuBLAS.  In particular, when computation is cut into pieces and spread across different machines,  we need to explicitly communicate input, output or intermediate data between different machines.  In this exercise, we will get our hands dirty on how to parallelize computation across different machines.

Pytorch closely follows the parallel programming paradigm of [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) (message-passing interface).  Under this model, the program starts by launching multiple worker processes, each of which performs its execution from the same given entry point function.  Each worker process performs a piece of the computation, communicating their inputs/intermediate results/outputs using the so-called collective communication primitives, such as broadcast, scatter, gather, all\_reduce etc. You can read more about them in the Pytorch [documentation](https://pytorch.org/docs/stable/distributed.html), or the [tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html).

For this exercise, complete [dist.py](https://github.com/jinyangli/pathway2ai/blob/master/dist.py).

First, let's look at the following code segment in the main function:
```
os.environ["MASTER_ADDR"] = "localhost"                                                                                           
os.environ["MASTER_PORT"] = "9999"                                                                                                
mp.spawn(do_work, args=(2,), nprocs=2, join=True)       
```
The above code will spawn 2 processes (a process is an instance of a running program) whose starting point of execution is the `do_work` function. In order for these processes to communicate, they will need to listen on some local [port](https://en.wikipedia.org/wiki/Port_(computer_networking)), which we have specified to start from "9999".

Now, let's look at parts of the `do_work` function below:
```
def do_work(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f'My rank is {rank}, same as {dist.get_rank()}')
```
The entry point function's first argument is always `rank`, which is an integer identifier from 0...world\_size-1.  Thus, even though both worker processes start executing the same entry function `do_work`, they are passed different argument values for `rank`.
The rank of a process can also be queried using the library function `dist.get_rank()`. Hence, the overal results would print 
```
$ python dist.py
My rank is 1, same as 1
My rank is 0, same as 0
```
Next, let us do some simple exercise to parallelize matrix multiplication across multiple (aka `world_size`) workers.  Suppose the matrix multiplication operation under consideration is `C=A@B` where `A` is of shape `(m,k)`, B is of shape `(k,n)`, and C is of shape `(m,n)`.  We will implement two parallelization strategies, **partition-by-row** and **partition-by-reduction**.  

In **partition-by-row**, we cut matrix A by the row dimension and send one piece to each worker, and we duplicate matrix B on all workers.  Each worker computes `C_piece = A_piece @ B` , where `A_piece` is of shape `(m // world_size, k)` and `C_piece` is of shape `(m // world_size, n)`.  Each worker holds a piece of the result, which can be communicated to a single node, if necessary.  

In **partition-by-reduction**, we cut matrix A by the column dimension and cut matrix B by the row dimension . Each worker computes `C_i = A_piece @ B_piece`, where `A_piece` is of shape `(m, k//world_size)` and `B_piece` is of shape `(k//world_size, n)` and `C_i` is of shape `(m,n)`.  To compute the final result C, we then sum all the `C_i` matrixes together.

## Exercise 3: Turn the basic MLP training into distributed data parallel training using Pytorch's built-in DDP mechanism

In this exercise, we are going to modify the `mlp.py` program you wrote in Exercise 1 into one that performs distributed data parallelism training over multiple worker processes.

In single worker training, each iteration of the training loop works on a batch of data to perform the forward (to calculate the loss) and backward computation (to calculate the gradient). Then the model's weight/parameter is updated using the gradient. And the next iteration proceeds with the updated model weights.

Under the DP (data parallelism) parallelization strategy, each worker holds a complete copy of the model parameter, and we can view each batch of data as being cut into sub-batches/pieces. At each training iteration, each worker performs the forward and backward computation using its sub-batch.  In effect, each worker calculates the gradient based on the loss of its sub-batch.  To obtain the gradient of the overall batch of data, we must sum the gradients from all the sub-batch together at all workers using all_reduce.  

If you look carefully, you'll see the connection between DP and the parallel matrix multiplication we implemented in Exercise 2.  Here's the connection. In the forward path, DP is like parallelizing a matrix multiplication by partitiong the row dimension (aka the batch dimension) of its input data matrix (and duplicating the weight matrix). In the backward path, DP is like parallelizing a matrix multiplication by partitioning along the reduction/batch dimension of the error matrix (calculated from the previous layer) and the activation matrix (saved from the forward path).  

PyTorch has built-in mechanism for DP, which internally takes care of making weights identical initially and doing all_reduce on gradients etc. Read the [documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) on how to use the built-in DDP, and modify your `mlp.py` program accordingly. Your modification should be no more than several additional lines.  

There are several things to be careful about in your training using DDP. One, we'll create the same identical artificial input dataset on all workers, so there is no need to scatter a single input dataset from one to all workers.  This is done by having all workers call `gen_dataset(config, rand_state=0)` in their `train` function using the same random state (0).  However, different workers should work in different "batch" (aka sub-batch) of the same dataset during training.  Therefore, in your training loop, a worker should skip those batches that are to be processed by another worker. Suppose there are 2 workers, then worker-0 should process batch 0 and worker-1 should process batch 1 in the first iteration (and worker-0 should process batch 2 and worker-2 should process batch 3 in the second iteration and so on). Thus you should do something like the following:

```
for i, batch in enumerate(train_data_loader):
    if (i % world_size) != rank:
        continue
    ...
```

Another thing you should do is to add a debugging function to check that all workers have the same model parameter weights after each training iteration. You can have rank-0 node fetch the model weights from all workers and compare if they are identical using `torch.allclose` function.

```
def check_parameter_closeness(model):
    for n, p in model.named_parameters():
        #check that parameter p is the same across all workers
    return True
```

Insert `check_parameter_closeness` in your training loop to assure yourself that all workers start and/or end with the same model parameter weights at each iteration of the loop.

While DP is simple conceptually, there is a lot of cleverness in its actual implementation to make it fast.  You can read more about them [here](https://arxiv.org/pdf/2006.15704.pdf) and [here](https://sysnetome.com/papers/bytescheduler_sosp2019.pdf).

## Exercise 4: Implement our own distributed data parallel training

In this exercise, we are going to ignore PyTorch's built-in DDP and implement our own data parallel training.  You are going to make your implementation as easy to use as the built-in DDP. To do so, you will create a new wrapper class `MyDP`. 
Your class will ensure that all workers start with the same model weights by broadcasting rank-0 node's weights to overwrite those of all workers.  Your class will also register a hook function using PyTorch's [register_hook](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html) to perform all_reduce of the gradients on the backward path.

You can use the following skeleton:
```
class MyDP(nn.Module):
    def __init__(self, model):
        super().__init__()
	# 1. Add your code to broadcast rank-0 node's parameter weights to all workers 
        # so everyone starts with the same model weights
        # 2. register a hook on each model parameter to perform all_reduce of the gradients on the backward path
    def forward(self, data, target=None):
        return self.model.forward(data, target)


