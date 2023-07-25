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

Pytorch closely follows the parallel programming paradigm of [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) (message-passing interface).  Under this model, the program starts by launching a given number of worker processes,  each of which performs its execution from the same given entry point function.  

Look at the following code segment in the main function:
```
os.environ["MASTER_ADDR"] = "localhost"                                                                                           
os.environ["MASTER_PORT"] = "9999"                                                                                                
mp.spawn(do_work, args=(2,), nprocs=2, join=True)       
```
The above code will spawn 2 processes (a process is an instance of a running program) whose starting point of execution is the `do_work` function. In order for these processes to communicate, they will need to listen on some local [port](https://en.wikipedia.org/wiki/Port_(computer_networking\)), which we have specified to start from "9999".

Now, let's look at the `do_work` function below:
```
def do_work(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f'My rank is {dist.get_rank()}')
```
The entry point function's first argument is always `rank`, which is an integer identifier from 0...world\_size-1.  Thus, even though both worker processes start executing the same entry function `do_work`, they are passed different argument values for `rank`.
The rank of a process can also be queried using the library function `dist.get_rank()`. Hence, the overal results would print 
```
$ python dist.py
My rank is 1
My rank is 0
```

## Exercise 3: Turn the basic MLP training into distributed data parallel training using Pytorch's built-in DDP mechanism

## Exercise 4: Implement our own distributed data parallel training
