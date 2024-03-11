#SWAD algorithms for finding start and end iterations
#Written by Brandon Weinhofer

import pandas as pd
import matplotlib.pyplot as plt


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
        while oneOfLastNGreaterThan(val_loss, N, TS)
    else:
        TS = 0
    


    #Handle finding ending iteration
    if N < len(val_loss)-TE:
        pass
    else:
        TE = len(val_loss)-1
    


    return TS, TE

















#TESTING PORPOSED_SWAD FUNCTION
loss = pd.read_csv("loss.csv")
loss_vals = list(loss.iloc[:,1])



ts, te = Proposed_SWAD(loss_vals, 3)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(loss_vals, color='black')
ax1.axvline(x=ts, color='r')
ax1.axvline(x=te, color='b')
ax1.set(xlabel="Iteration", ylabel="Validation Loss")

ax2.plot(loss_vals, color='black')
ax2.axvline(x=ts, color='r')
ax2.axvline(x=te, color='b')
ax2.set(xlabel="Iteration", ylabel="Validation Loss")

plt.show()
