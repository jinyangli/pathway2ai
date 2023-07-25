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
matmul_m = 4
matmul_n = 4
matmul_k = 4

def par_matmul_by_row(A, B):
    if dist.get_rank() != 0:
        B = torch.empty([matmul_k, matmul_n])
    dist.broadcast(B, src=0)

    A_chunks = None
    if dist.get_rank() == 0:
        A_chunks = list(torch.split(A, world_size, dim=0))
    A_piece = torch.empty([matmul_m//world_size, matmul_k])
    dist.scatter(A_piece, A_chunks, src=0)

    C_piece = A_piece @ B
   
    C_chunks = None
    if dist.get_rank() == 0:
        C = torch.empty([matmul_m, matmul_n])
        C_chunks = list(torch.split(C, world_size, dim=0))
    dist.gather(C_piece, C_chunks, dst=0)

    if dist.get_rank() == 0:
        return torch.cat(C_chunks, dim=0)
    return None

def par_matmul_by_reduction(A, B):
    A_chunks = None
    if dist.get_rank() == 0:
        A_chunks = list(torch.split(A, world_size, dim=1))
    A_piece = torch.empty([matmul_m, matmul_k//world_size])
    dist.scatter(A_piece, A_chunks, src=0)

    B_chunks = None
    if dist.get_rank() == 0:
        B_chunks = list(torch.split(B, world_size, dim=0))
    B_piece = torch.empty([matmul_k//world_size, matmul_n])
    dist.scatter(B_piece, B_chunks, src=0)

    C = A_piece @ B_piece

    dist.all_reduce(C) 
    return C

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
        print(f'par_matmul_by_row is correct...')

    C2 = par_matmul_by_reduction(A, B)
    if rank == 0:
        check_equality(C2, C)
        print(f'par_matmul_by_reduction is correct...')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP training")
    parser.add_argument('--world-size', type=int, default=2, help="number of workers")
    args = parser.parse_args()
    print(vars(args))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9999"
    mp.spawn(do_work, args=(args.world_size,), nprocs=args.world_size, join=True)
