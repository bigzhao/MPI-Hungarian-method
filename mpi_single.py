from mpi4py import MPI
import numpy as np
import os, operator, math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
from numba import jit


@jit(nopython=True)
def avg_normalized_happiness(pred, child_pref, gift_pref):
    
    # check if number of each gift exceeds n_gift_quantity
#     gift_counts = Counter(elem[1] for elem in pred)
#     for count in gift_counts.values():
#         assert count <= n_gift_quantity
                
#     # check if triplets have the same gift
    n_children = 1000000 
    n_gift_type = 1000 
    n_gift_quantity = 1000 
    n_gift_pref = 100 
    n_child_pref = 1000
    twins = math.ceil(0.04 * n_children / 2.) * 2   
    triplets = math.ceil(0.005 * n_children / 3.) * 3   
    ratio_gift_happiness = 2
    ratio_child_happiness = 2
    
    for t1 in np.arange(0,triplets,3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        # print(t1, triplet1, triplet2, triplet3)
        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1]
                
#     # check if twins have the same gift
    for t1 in np.arange(triplets,triplets+twins,2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        # print(t1)
        assert twin1[1] == twin2[1]

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)
    
#     for row in pred:
    for i in np.arange(len(pred)):
        child_id = pred[i, 0]
        gift_id = pred[i, 1]
        
        # check if child_id and gift_id exist
#         assert child_id < n_children
#         assert gift_id < n_gift_type
#         assert child_id >= 0 
#         assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness
        if (len(child_happiness) == 0):
            tmp_child_happiness = -1
        else:
            tmp_child_happiness = child_happiness[0]

        gift_happiness = ( n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness
        if (len(gift_happiness) == 0):
            tmp_gift_happiness = -1
        else:
            tmp_gift_happiness = gift_happiness[0]
            
        total_child_happiness += tmp_child_happiness    
        total_gift_happiness[gift_id] += tmp_gift_happiness    
    
#     print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) , \
#         ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))
#     return math.pow(total_child_happiness/(n_children * float(max_child_happiness)), 3) +\
#             math.pow(np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity), 3)
    return (total_child_happiness/(n_children * float(max_child_happiness))) ** 3 +\
            (np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))** 3
#     return np.power((total_child_happiness)/((n_children)*float(max_child_happiness)), 3) + \
#             np.power(np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity), 3)


# as the objective function is in cubic form - nonlinear, 
# adding child happiness and gift happiness will not work (like the original problem)
# the optimize block only takes child happiness to optimize first 
# then to verify each step if there is an improvement in the overall performance 



def optimize_block(child_block, current_gift_ids):
    gift_block = current_gift_ids[child_block]
    C = np.zeros((block_size, block_size))
    for i in range(block_size):
        c = child_block[i]
        for j in range(block_size):
            g = gift_ids[gift_block[j]]
            C[i, j] = child_happiness[c][g]
    row_ind, col_ind = linear_sum_assignment(C)
    return (child_block[row_ind], gift_block[col_ind])

# the block size may be twisted - considering the algorithm complexity in Hungarian method
# Be warned that the solving speed is not linearly aligned with the block size




def my_optimizer(subm, score_org, comm, rank,size, gift_data, child_data):
    answ_iter = np.zeros(len(child_data), dtype=np.int32)
    score_best = score_org
    subm_best = subm
    perm_len = 200
    block_len = 10
    count = 0
    iter_ = 1
    # np.random.seed(2018)
    while 1:
        # start_time = dt.datetime.now()

        # print('Current permutation step is: %d' %(iter_))
        child_blocks = np.split(np.random.permutation
                                (range(tts, n_children - children_rmd)), n_blocks)
        ## 广播
        child_blocks = comm.bcast(child_blocks[:size], root=0)

        # start = int(rank * block_len/ rank_size)
        # end = int(start + block_len/ rank_size) 
        child_block = child_blocks[rank]
        # for child_block in (child_blocks[start: end]):
#             start_time = dt.datetime.now()
        cids, gids = optimize_block(child_block, current_gift_ids=current_gift_ids)

        # send data
        if rank > 0:
            comm.send([cids, gids], dest=0, tag=123)
        # start_time_1 = dt.datetime.now()
        # recv data and update
        buf = []
        if rank == 0:
            current_gift_ids[cids] = gids
            buf = [[cids, gids],]
            for i in range(1, size):
                buf.append(comm.recv(source = i, tag=123))
                # current_gift_ids[temp[0]] = temp[1]
        buf = comm.bcast(buf, root=0)
        # end_time_1 = dt.datetime.now()
        # print("rank:{} bcast time for iter:{}: {}".format(rank, i, (end_time_1 - start_time_1).seconds ))

        for i in range(size):
            current_gift_ids[buf[i][0]] = buf[i][1]

        # current_gift_ids = comm.bcast(current_gift_ids, root=0)
        subm['GiftId'] = gift_ids[current_gift_ids]
        answ_iter[subm[['ChildId']]] = subm[['GiftId']]
        score_iter = avg_normalized_happiness(subm[['ChildId', 'GiftId']].values, gift_data, child_data)
        # print('Score achieved in current iteration: {:.10f}'.format(score_iter))

        if score_iter > score_best:
            subm_best['GiftId'] = gift_ids[current_gift_ids]
            score_best = score_iter
            count = 0
        else:
            count += 1

        if count > 3:
            print("not optimizing")
            return subm_best
        # current_gift_ids[cids] = gids
#             end_time = dt.datetime.now()
#             print('Time to optimize block in seconds: {:.2f}'.
#                   format((end_time-start_time).total_seconds()))
        ## need evaluation step for every block iteration 

        if rank == 0:
            subm_best[['ChildId', 'GiftId']].to_csv('improved_sub.csv', index=False)
            print('iteration:{} score achieved is: {:.10f}'.format(i , score_best))
        # end_time = dt.datetime.now()
        iter_ += 1
        # print("rank:{}optimiza time for iter:{}: {}".format(rank, i, (end_time - start_time).seconds ))
    return subm_best


# my_optimizer(comm, rank)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_size = comm.Get_size()

# print("Hello, i am the {}-th process".format(rank))

child_data = pd.read_csv('input/child_wishlist_v2.csv', 
                         header=None).drop(0, 1).values
gift_data = pd.read_csv('input/gift_goodkids_v2.csv', 
                        header=None).drop(0, 1).values

n_children = 1000000
n_gift_type = 1000 
n_gift_quantity = 1000
n_child_wish = 100
triplets = 5001
twins = 40000
tts = triplets + twins 

gift_happiness = (1. / (2 * n_gift_type)) * np.ones(
    shape=(n_gift_type, n_children), dtype = np.float32)

for g in range(n_gift_type):
    for i, c in enumerate(gift_data[g]):
        gift_happiness[g,c] = -2. * (n_gift_type - i)  

child_happiness = (1. / (2 * n_child_wish)) * np.ones(
    shape=(n_children, n_gift_type), dtype = np.float32)

for c in range(n_children):
    for i, g in enumerate(child_data[c]):
        child_happiness[c,g] = -2. * (n_child_wish - i) 

gift_ids = np.array([[g] * n_gift_quantity for g in range(n_gift_type)]).flatten()

initial_sub = 'baseline_res.csv'
subm = pd.read_csv(initial_sub)
subm['gift_rank'] = subm.groupby('GiftId').rank() - 1
subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']
subm['gift_id'] = subm['gift_id'].astype(np.int64)
current_gift_ids = subm['gift_id'].values

# wish = pd.read_csv('input/child_wishlist_v2.csv', 
#                    header=None).as_matrix()[:, 1:]
# gift_init = pd.read_csv('input/gift_goodkids_v2.csv', 
#                         header=None).as_matrix()[:, 1:]
score_org = avg_normalized_happiness(subm[['ChildId', 'GiftId']].values, gift_data, child_data)
# print(score_org)
# the block size may be twisted - considering the algorithm complexity in Hungarian method
# Be warned that the solving speed is not linearly aligned with the block size

block_size = 2000
n_blocks = int((n_children - tts) / block_size)
children_rmd = 1000000 - 45001 - n_blocks * block_size
# print('block size: {}, num blocks: {}, children reminder: {}'.
      # format(block_size, n_blocks, children_rmd))

# start_time = dt.datetime.now()
subm_best = my_optimizer(subm.copy(), score_org, comm, rank, rank_size, gift_data, child_data)
# end_time = dt.datetime.now()
# print("optimiza time: {}".format((end_time - start_time).seconds ))

if rank == 0:
    print("Nothing problem happen")
    subm_best[['ChildId', 'GiftId']].to_csv('improved_sub.csv', index=False)  
