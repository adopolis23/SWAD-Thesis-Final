import tensorflow as tf
import os
import gc

from utility import model_validation_loss
from SWAD_utility import AverageWeights





#checkpoint callback for "baseline" performance
#finds model weights from iteration with lowest validaion loss
class checkpoint(tf.keras.callbacks.Callback):

    def __init__(self, val_x, val_y):
        self.min_loss = float("inf")
        self.opt_weight = None
        self.val_x = val_x
        self.val_y = val_y

        #list to save loss curve
        self.loss_tracker = []

    def on_train_batch_end(self, epoch, logs=None):
        val_loss = model_validation_loss(self.model, self.val_x, self.val_y)
        self.loss_tracker.append(val_loss)

        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.opt_weight = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.opt_weight)




#callback for SWAD algorithm
class SWAD_callback(tf.keras.callbacks.Callback):

    def __init__(self, SWAD_Version, val_x, val_y):
        self.loss_tracker = []
        self.weights_saved = 0
        self.new_weights = list()
        self.SWAD_Version = SWAD_Version
        self.val_x = val_x
        self.val_y = val_y

        #make sure weights folder exists and if it does make sure it is empty
        if not os.path.isdir('Weights/'):
            os.mkdir("Weights")
        else:
            weights_folder = os.listdir("Weights")
            for file in weights_folder:
                os.remove("Weights/"+file)
        

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

    def on_train_batch_end(self, batch, logs=None):
        val_loss = model_validation_loss(self.model, self.val_x, self.val_y)
        self.loss_tracker.append(val_loss)
        self.model.save_weights("Weights/weights_" + str(self.weights_saved) + ".h5")
        self.weights_saved += 1

    def on_train_end(self, logs=None):
        ts, te = self.SWAD_Version(self.loss_tracker)
        self.new_weights = AverageWeights(self.model, ts, te, 200)

        #set model weights to new average
        if len(self.new_weights) > 0:
            print("\nSetting new model weights.\n")
            self.model.set_weights(self.new_weights)