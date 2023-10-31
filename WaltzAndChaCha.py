import imageio as imageio
import numpy as np
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from keras.optimizers import RMSprop,SGD,Adam #, adam_v2
from keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.python.keras.callbacks import EarlyStopping

start_time = time.time()
#tf.compat.v1.set_random_seed(2019)


model = tf.keras.models.Sequential([
   # tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (180,180,3)) ,
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (341,191,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,
    tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),
    tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(550,activation="relu"),      #Adding the Hidden layer
    #tf.keras.layers.Dropout(0.1,seed = 2019), dropout layer
    tf.keras.layers.Dropout(0.3,seed = 2019), # Best so far is a 0.7 increase from 0.5 results are 0.53 to 0.57 val_acc out of 100 epochs
    tf.keras.layers.Dense(400,activation ="relu"),
    tf.keras.layers.Dropout(0.5,seed = 2019),
    tf.keras.layers.Dense(300,activation="relu"),
    tf.keras.layers.Dropout(0.5,seed = 2019),
    tf.keras.layers.Dense(200,activation ="relu"),
    tf.keras.layers.Dropout(0.2,seed = 2019),
#tf.keras.layers.Dense(3 ,activation = "softmax")   #Adding the Output Layer I think this is the number of classes
    tf.keras.layers.Dense(6 ,activation = "softmax")   #Adding the Output Layer saving this line so I can try some new ones
    #tf.keras.layers.Dense(8 ,activation = "tanh")   #Adding the Output Layer
])

model.summary()
#adam= adam_v2.Adam(learning_rate=0.001)
#adam= adam_v2.Adam(learning_rate=0.0000001)
adam= Adam(learning_rate=0.0001) #Best is lr of 0.0001 with val_acc of 0.53968

model.compile(optimizer='adam', loss=['mse','sparse_categorical_crossentropy'], metrics = ['acc'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])

# now that the model is set up, we need to import the images

#bs=30         #Setting batch size
bs=4096
#Best batch size so far is 128 val_acc 0.53
train_dir = "NewDanceTypeTest/Training/"   #Setting training directory
validation_dir = "NewDanceTypeTest/Testing/"   #Setting testing directory

# All images will be rescaled by 1./255.
#train_datagen = ImageDataGenerator( rescale = 1.0/255.0 )
train_datagen = ImageDataGenerator( rescale = 1.0/279.0, rotation_range = 179, shear_range = 0.1)
"""
Use this for reference for generating more data to test
datagen = ImageDataGenerator(rotation_range = 0,
                             width_shift_range = 0,
                             height_shift_range = 0,
                             rescale = None,
                             shear_range = 0,
                             zoom_range = 0,
                             horizontal_flip = False,
                             fill_mode = ‘nearest')
"""

#test_datagen  = ImageDataGenerator( rescale = 1.0/255.0 )
test_datagen = ImageDataGenerator( rescale = 1.0/279.0)

# Flow training images in batches of 20 using train_datagen generator
#Flow_from_directory function lets the classifier directly identify the labels from the name of the directories the image lies in
#train_generator=train_datagen.flow_from_directory(train_dir,batch_size=bs,class_mode='categorical',target_size=(180,180))
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=bs,
                                                    class_mode='categorical',
                                                    target_size=(341, 191),
                                                    shuffle = True,
                                                    seed= 1
                                                    )
# Flow validation images in batches of 20 using test_datagen generator
#validation_generator = test_datagen.flow_from_directory(validation_dir,
#                                                         batch_size=bs,
#                                                         class_mode  = 'categorical',
#                                                         target_size=(180,180))
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=bs,
                                                         class_mode  = 'categorical',
                                                         target_size=(341,191),
                                                         shuffle = True,
                                                         seed = 1
                                                        )

val_loss_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=160,
    min_delta=0.001,
    mode='min'
)

es = EarlyStopping(monitor='val_acc', mode='min', min_delta=1)

saveModel = tf.keras.callbacks.ModelCheckpoint(
    "NewDanceTypeTest_SaveModel/saved_model",
    monitor="val_acc",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    initial_value_threshold=None,
)
# Now to fit the model validation_steps=50 // bs
history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch= 4096// bs, epochs=1000, validation_steps=4096 // bs, verbose=1 ,callbacks=[saveModel])
# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
print("Finished training")
#print(history.history)
print("--- %s minutes ---" % ((time.time() - start_time)/60))

#model.save("NewDanceTypeTest_SaveModel/saved_model")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'g', label='Training acc' )
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()