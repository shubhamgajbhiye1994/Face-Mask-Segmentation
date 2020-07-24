import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


MODEL = 'C:/Users/DarkWeb/Desktop/Face Mask Segmentation/models/'
INPUT_PATH ='C:/Users/DarkWeb/Desktop/Face Mask Segmentation/input/'
IMAGE_WIDTH, IMAGE_HEIGHT = 224 ,224

if __name__ == '__main__':

    # metric 
    def dice_coefficient(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return numerator / (denominator + tf.keras.backend.epsilon())

    # loss
    def loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())

    # load model
    model = load_model(os.path.join(MODEL,'face_detection.h5'),custom_objects={'loss': loss,'dice_coefficient':dice_coefficient})

    # load data
    x_test = np.load(os.path.join(INPUT_PATH,'X_test.npy'),allow_pickle=True)
    y_test = np.load(os.path.join(INPUT_PATH,'y_test.npy'),allow_pickle=True)

    try:
        sample_gen = np.random.randint(len(x_test),size=1)
        sample_test = x_test[sample_gen]
        pred_mask = cv2.resize(model.predict(np.array([sample_test]))[0],(IMAGE_WIDTH,IMAGE_HEIGHT))
        plt.imshow(sample_test)
        plt.imshow(pred_mask,cmap='gray',alpha=0.45)

    except:
        print("Value Not found")

    

    
