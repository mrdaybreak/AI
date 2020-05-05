# 2.1. Suppose your input is a 100 by 100 gray image, and you use a convolutional layer with 50 filters that are each 5x5.How many parameters does this hidden layer have (including the bias parameters)?
print((5*5*1+1)*50)

# 2.2. What are "local invariant" and "parameter sharing" ?
print('local invariant is we can get the same value whatever where are the feature, parameter sharing is same kernel')

# 2.3. Why we use batch normalization ?
print('batch normalization is the input is forced to pull back to the normal distribution of the comparison standard '
      'with the mean of 0 and the variance of 1, (x-mean)/std')

# 2.4. What problem does dropout try to solve ?
print('over-fitting')

# 3.1 In the first session of the practical part, you will implement an image classification model using any deep learning libraries that you are familiar with, which means, except for tensorflow and keras, you can also use pytorch/caffe/... . The dataset used in this session is the cifar10 which contains 50000 color (RGB) images, each with size 32x32x3. All 50000 images are classified into ten categories.

from keras import models, layers, optimizers, datasets, utils, regularizers
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(x_test.shape)
# for i in range(1, 11):
#       plt.subplot(2, 5, i)
#       plt.imshow(x_train[i - 1])
#       plt.text(3, 10, str(y_train[i-1]))
#       plt.xticks([])
#       plt.yticks([])
# plt.show()
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
# cblr = ReduceLROnPlateau(verbose=1)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
# x_train = x_train/255
x_test = x_test/255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
x_train = train_datagen.flow(x_train, y_train, batch_size=256)
model.fit_generator(x_train, steps_per_epoch=200, epochs=100)
modelname = 'cifarfitmodel.h5'
model.save(modelname)
model = models.load_model('cifarfitmodel.h5')
scores = model.evaluate(x_test, y_test)
print(scores)
result = model.predict_classes(x_test[:50])
print(result)

# score = [0.4623522692680359, 0.8799999952316284]




