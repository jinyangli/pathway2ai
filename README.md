# Understanding parallel deep neural network training

This is a series of exercises to get your hands dirty on parallelizing the training of Deep Neural Networks Ideally, we want to parallelize across multiple GPU devices, possibly over over multiple machines.  However, GPUs are hard to get (esp if you want many of them), so we'll develop and test using your own laptop, on CPU only. The code and the concept is exactly the same, except that on your laptop, we won't be able to get any speedup from "parallelization".

## Exercise 1: Complete the basic training of a toy multi-layer perceptron

Download [mlp.py](https://github.com/jinyangli/pathway2ai/blob/master/mlp.py).  Alternatively, you can clone this git repository to get all the files using command `git clone git@github.com:jinyangli/pathway2ai.git`.

Read the code carefully. Complete the missing code block or function body that follow a comment starting with **TODO**

Run the code. `python mlp.py`. You should see something like

```
python mlp.py
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

## Exercise 2: Turn the basic MLP training into distributed data parallel training using Pytorch's built-in DDP mechanism

## Exercise 3: Play with the build-in communication mechanism, gather, allreduce etc.

## Exercise 4: Implement our own distributed data parallel training
