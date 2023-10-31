import os

import imageio.v2 as imageio
import tensorflow as tf
import numpy as np
new_model = tf.keras.models.load_model('NewDanceTypeTest_SaveModel/saved_model')

# Check its architecture
new_model.summary()
#loop over files
filepath = 'NewDanceTypeTest/Training'
list_dir = os.listdir(filepath)
print(list_dir)

for subdir, dirs, files in os.walk(filepath):
    print("We are in the " + subdir)
    for file in files:
        #print(os.path.join(subdir, file))
        # Read the file, but we only want 3 channels not 4
        img = imageio.imread(os.path.join(subdir, file), pilmode='RGB')
        # add the 4th dimension the model wants
        img.resize((191, 341,3))
        img = np.expand_dims(img, axis=0)
        # switch the 1st and 2nd slots I guess it was read differently with the image generator? Not sure why
        img = np.transpose(img, (0, 2, 1, 3))
        # Make the prediction
        prediction = new_model.predict([img])
        #prediction = new_model.predict_on_batch([img])
        print(prediction)
        theClassNumber = prediction.argmax(axis=-1)[0]
        print(list_dir[theClassNumber])

"""
#This was for reading a single file, now we want the entire folder
# Read the file, but we only want 3 channels not 4
#img = imageio.imread('NewDanceTypeTest/Lindy Hop/167.png', pilmode='RGB')
#update to the context of the loop
img = imageio.imread('NewDanceTypeTest/Lindy Hop/167.png', pilmode='RGB')
print(img.shape)
# add the 4th dimension the model wants
img = np.expand_dims(img, axis=0)
print(img.shape)
# switch the 1st and 2nd slots I guess it was read differently with the image generator? Not sure why
img = np.transpose(img,(0, 2,1, 3))
# Make the prediction
prediction = new_model.predict(img)
theClassNumber = prediction.argmax(axis=-1)[0]
print(list_dir[theClassNumber])

"""
