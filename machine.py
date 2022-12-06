import os
import random
import numpy as np
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.models import load_model
from keras.utils import np_utils
def data_x_y_preprocess(datapath):
    img_row,img_col = 28,28
    datapath=datapath
    data_x=np.zeros((28,28)).reshape(1,28,28)
    pictureCount=0
    data_y=[]
    num_class=10
    for root,dirs,files in os.walk(datapath):
        for f in files:
            label=int(root.split("\\")[2])
            data_y.append(label)
            fullpath=os.path.join(root,f)
            img=Image.open(fullpath)
            img=(np.array(img)/255).reshape(1,28,28)
            data_x=np.vstack((data_x,img))
            pictureCount+=1
    data_x=np.delete(data_x,[0],0)
    data_x=data_x.reshape(pictureCount,img_row,img_col,1)
    data_y=np_utils.to_categorical(data_y,num_class)
    return data_x,data_y

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10,activation='softmax'))

train_x, train_y = data_x_y_preprocess("train_image")
test_x, test_y = data_x_y_preprocess("test_image")

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
train_history=model.fit(train_x,train_y,
                       batch_size=32,
                       epochs=150,
                       verbose=1,
                       validation_split=0.1)
score=model.evaluate(test_x,test_y,verbose=0)
print("Test loss:",score[0])
print("Test accuracy:",score[1])
