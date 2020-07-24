import cv2
import os
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input


def data_prepare(INPUT_PATH,IMAGE_WIDTH,IMAGE_HEIGHT):
    # load data
    data = np.load(os.path.join(INPUT_PATH,"images.npy"),allow_pickle=True)

    # prepare mask and input for model training
    masks = np.zeros((int(data.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH))
    X_train = np.zeros((int(data.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    for index in range(data.shape[0]):
        img = data[index][0]
        img = cv2.resize(img, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        try:
            img = img[:, :, :3]
        except:
            continue
        X_train[index] = preprocess_input(np.array(img, dtype=np.float32))
        for i in data[index][1]:
            x1 = int(i["points"][0]['x'] * IMAGE_WIDTH)
            x2 = int(i["points"][1]['x'] * IMAGE_WIDTH)
            y1 = int(i["points"][0]['y'] * IMAGE_HEIGHT)
            y2 = int(i["points"][1]['y'] * IMAGE_HEIGHT)
            masks[index][y1:y2, x1:x2] = 1

    # make train n test data
    X_test=np.zeros((9,224,224,3))
    mask=np.zeros((9,224,224))
    x_train=np.zeros((400,224,224,3))
    x_mask=np.zeros((400,224,224))
    index=[]
    j=0
    k=0

    for i in range(X_train.shape[0]):
        if i>=400:
            X_test[j]=X_train[i]
            mask[j]=masks[i]
            index.append(i)
            j+=1
        else:
            x_train[k]=X_train[i]
            x_mask[k]=masks[i]
            k+=1

    # save data 
    np.save(os.path.join(INPUT_PATH,'X_test.npy'),X_test,allow_pickle=True)
    np.save(os.path.join(INPUT_PATH,'y_test.npy'),mask,allow_pickle=True)

return x_train , x_mask
    
