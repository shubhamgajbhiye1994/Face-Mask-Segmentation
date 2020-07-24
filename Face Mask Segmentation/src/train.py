import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import log, epsilon
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers
from data_preparation import data_prepare

MODEL = 'C:/Users/DarkWeb/Desktop/Face Mask Segmentation/models/'
INPUT_PATH ='C:/Users/DarkWeb/Desktop/Face Mask Segmentation/input/'
IMAGE_WIDTH, IMAGE_HEIGHT = 224 ,224
EPOCH = 100
BATCH_SIZE = 16

if __name__ == '__main__':

  # prepare data 
  x_train , y_train = data_prepare(INPUT_PATH,IMAGE_WIDTH,IMAGE_HEIGHT)

  # MOBILENET with UNET model
  def create_model(trainable=True):
    model = MobileNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), include_top=False, alpha=1.0, weights="imagenet")
    for layer in model.layers:
        layer.trainable = trainable

    # Add all the UNET layers here
    block1 = model.get_layer("conv_pw_5_relu").output
    block2 = model.get_layer("conv_pw_11_relu").output
    block3 = model.get_layer("conv_pw_13_relu").output
    block4 = model.get_layer("conv_pw_3_relu").output
    block5 = model.get_layer("conv_pw_1_relu").output
    block6 = model.get_layer("conv1_relu").output
   

    x = Concatenate()([UpSampling2D()(block3), block2])
    x = Concatenate()([UpSampling2D()(x), block1])
    x = Concatenate()([UpSampling2D()(x), block4])
    x = Concatenate()([UpSampling2D()(x), block5])
    x = Concatenate()([UpSampling2D()(x), UpSampling2D()(block5)])

    x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)
    x = Reshape((IMAGE_HEIGHT, IMAGE_WIDTH))(x)

    return Model(inputs=model.input, outputs=x)

  # metric 
  def dice_coefficient(y_true, y_pred):
      numerator = 2 * tf.reduce_sum(y_true * y_pred)
      denominator = tf.reduce_sum(y_true + y_pred)
      return numerator / (denominator + tf.keras.backend.epsilon())

  # loss
  def loss(y_true, y_pred):
      return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())


  checkpoint = ModelCheckpoint(os.path.join(MODEL,"model-{loss:.2f}.h5"), monitor="loss",verbose=1, save_best_only=True,save_weights_only=True, mode="min", period=1)
  stop = EarlyStopping(monitor="loss", patience=5, mode="min")
  reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="min")
  optimizer = optimizers.Adam(learning_rate = 0.001)

  # model creation and compile
  model = create_model(False)
  model.compile(loss=loss, optimizer=optimizer, metrics=[dice_coefficient])

  # train model
  result = model.fit(x=x_train,y=y_train,epochs=EPOCH,batch_size=BATCH_SIZE,callbacks=[checkpoint, reduce_lr, stop])

  # save model
  model.save(os.path.join(MODEL,'face_detection.h5'),custom_objects={'loss': loss,'dice_coefficient':dice_coefficient})

  print("Model trained and Saved...")