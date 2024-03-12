#utility functions for SWAD
#written by Brandon Weinhofer

#Helper function for new SWAD-S algorithm that returns true if one of the last
#N values in a list is greater than the currect value
def oneNGreaterThan(val_loss, N, curr_val, last):

    if last:
        if curr_val-N <= 0:
            return -1

        last_n_vals = val_loss[curr_val-N:curr_val]

    else:
        if len(val_loss)-curr_val-N <= 0:
            return -1
        
        last_n_vals = val_loss[curr_val+1:curr_val+N+1]


    last_vals_greater = [x for x in last_n_vals if x > val_loss[curr_val]]

    if len(last_vals_greater) > 0:
        return True
    else:
        return False

