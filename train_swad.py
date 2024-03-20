import tensorflow as tf
import numpy as np

from CXR_data_loader import load_CXR_data
from utility import setSeed
from Models.Resnet_18 import ResNet18

#hyperparameters
NS = 6
NE = 6
r = 1.2
N = 5

learning_rate = 0.0001
batch_size = 32
epochs = 20


#load in training data
train_x, train_y, val_x, val_y, test_seen_x, test_seen_y, test_unseen_x, test_unseen_y = load_CXR_data()


#define model 
model = ResNet18(2)
model.build(input_shape = (None,244,244,3))

#define optimizer
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
            optimizer=opt,
            metrics=['accuracy'])

#train the model
model.fit(x=np.array(train_x, np.float32),
            y=np.array(train_y, np.float32),
            validation_data=(val_x, val_y),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=[])


#Evaluate model on seen and unseen data
scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

scores_unseen = model.evaluate(test_unseen_x, test_unseen_y, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])