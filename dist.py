import os
import numpy as np
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

world_size = 2
# This program tries to parallelize matmul C = A @ B 
# where A is of shape (matmul_m,matmul_k) and 
# B is of shape (matmul_k, matmul_n)
matmul_m = 1024
matmul_n = 1024 
matmul_k = 1024

def par_matmul_by_row(A, B):
    # parallelize matmul C=A@B across world_size workers
    # each worker computes C_piece = A_piece @ B, where A_piece is a shard of A partitioned by row (dim=0)
    # The final C is obtained by concatenating all C_piece together
    #
    # Your parallelism algorithm performs the following 3 steps
    # Step1: broadcast matrix B to all workers (using dist.broadcast)
    # Hint: Note that B is None on non-rank-0 workers, so you should allocate space for B
    # using torch.empty(..) before invoking dist.broadcast
    #
    # Step2: scatter matrix A to all workers (using dist.scatter)
    # Hint: using torch.split to turn A into a list of pieces. Also note torch.split returns tuple type which you
    # have to convert to list type.
    #
    # Step 3: compute C_piece = A_piece @ B
    #
    # Step 4: gather all C_piece to the rank-0 node using dist.gather, concatenate them together and return
    print(f'TODO: complete me in par_matmul_by_row')
    assert False
    return None

def par_matmul_by_reduction(A, B):
    # parallelize matmul C=A@B across world_size workers
    # each worker computes C_i = A_piece @ B_piece, where A_piece is a shard of A partitioned by col (dim=1)
    # and B_piece is a shard of B partitioned by row (dim=0). C_i is of shape (matmul_m, matmul_n)
    # The final C is obtained by summing all C_i together
    #
    # Your parallelism algorithm performs the following 3 steps
    # Step1: scatter matrix A to all workers (using dist.scatter)
    # Step2: scatter matrix B to all workers (using dist.scatter)
    # Step 3: compute C_i = A_piece @ B_piece
    # Step 4: make all workers sum all C_i together using dist.all_reduce, and return
    print(f'TODO: complete me in par_matmul_by_reduction')
    assert False
    return None

def check_equality(X, Y):
    return torch.allclose(X,Y)

def do_work(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f'My rank is {rank}, same as {dist.get_rank()}')
   
    A = None
    B = None
    if rank == 0:
        A = torch.rand(matmul_m, matmul_k)
        B = torch.rand(matmul_k, matmul_n)
        C = A @ B # calculate result at one node for checking correctness
    C1 = par_matmul_by_row(A, B)
    if rank == 0:
        check_equality(C1, C)

    C2 = par_matmul_by_reduction(A, B)
    check_equality(C2, C)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP training")
    parser.add_argument('--world-size', type=int, default=2, help="number of workers")
    args = parser.parse_args()
    print(vars(args))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9999"
    mp.spawn(do_work, args=(args.world_size,), nprocs=args.world_size, join=True)
