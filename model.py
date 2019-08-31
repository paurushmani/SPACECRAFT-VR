import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def get_labels(run_no):
    file = open('run' + str(run_no) + '/picOutput/LABELS.txt')
    lines = file.read().splitlines()
    return lines
    # print(lines[0])


def parse_lbl(lbl):
    meta_data = lbl.split('LogTemp: RTMAT ')[1].split(' ')
    img_mat = meta_data[2]
    img_no = meta_data[1]
    cam_no = meta_data[0]
    return img_no, cam_no, img_mat


def load_img(lbl, extension='/picOutput/im_1230'):
    return cv2.imread(extension + lbl[0] + '_' + lbl[1] + '.png')


def load_all(run_no, mode=0):
    labels = get_labels(run_no)
    imgs = []
    lbls = []
    for lbl in labels:
        one_hot = [0, 0, 0, 0, 0, 0]
        meta = parse_lbl(lbl)
        img = load_img((meta[1], meta[0]), 'run' + str(run_no) + '/picOutput/im_1230')
        if img is None:
            continue
        imgs.append(np.array(img))
        one_hot[int(meta[2])] = 1
        lbls.append(one_hot)
    if mode:
        np.save('numpy_saved', [imgs, lbls])

    return imgs, lbls


imgs, lbls = load_all(9)
imgs2, lbls2 = load_all(11)

X=imgs+imgs2
Y=lbls+lbls2
Y=np.array(Y)
X= np.array(X).reshape(-1, 80, 80, 3)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

inputShape=(80,80,3)
chanDim=-1

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim)) # Normalizing the data across all axises
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25)) # Reduces overfitting of the data

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same")) # Adding new layers
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same")) # Adding new layers
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # Flattening and making data good for compilation
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# use a sigmoid activation for multi-label classification
model.add(Dense(6)) # 6 is the number of categories
model.add(Activation('sigmoid'))
opt = Adam(lr=0.01, decay=(0.01 /100))
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics = ['accuracy'])
lbls=np.array(lbls)
model.fit(X_train, Y_train, batch_size=10, epochs=5,verbose=1)
Y_pred = model.predict(X_test)

correct=0
wrong=0
for i in range(len(Y_pred)):
    if(np.argmax(Y_pred[i])==np.argmax(Y_test[i])):
        correct+=1
    else:
        wrong+=1
print()
print()
print()
print()
print()
print(correct,wrong)


