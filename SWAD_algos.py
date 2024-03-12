#SWAD algorithms for finding start and end iterations
#Written by Brandon Weinhofer

import pandas as pd
import matplotlib.pyplot as plt

from SWAD_utility import oneNGreaterThan


#Proposed SWAD-S algorithm
#inputs: Validation loss vector, patience parameter N
#outputs: TS (starting iteration), TE (end iteration)
def Proposed_SWAD(val_loss, N = 3):
    
    #index of the min loss value for starting point
    min_index = val_loss.index(min(val_loss))
    TS = min_index
    TE = min_index + 1


    #Handle finding starting iteration
    if N < TS:
        while oneNGreaterThan(val_loss, N, TS, last=True) and TS != 0:
            TS = TS - 1
    else:
        TS = 0
    

    #Handle finding ending iteration
    if N < len(val_loss)-TE:
        while oneNGreaterThan(val_loss, N, TE, last=False) and TE != len(val_loss)-1:
            TE = TE + 1
    else:
        TE = len(val_loss)-1
    

    return TS, TE












