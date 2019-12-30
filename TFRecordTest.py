from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import json


import os
import numpy as np


batch_size = 2
epochs = 4
IMG_HEIGHT = 150
IMG_WIDTH = 150



PATH = os.path.join('D:/work/PycharmProjects/TfAndroid/data/', 'yes_and_no')


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'YES')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'NO')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'YES')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'NO')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val


print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)




# 训练集
# 对训练图像应用了重新缩放，45度旋转，宽度偏移，高度偏移，水平翻转和缩放增强。
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

# 验证集

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')




# 创建模型

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])



# 编译模型

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# 模型总结
model.summary()


# 模型保存格式定义

model_class_dir='D:/work/PycharmProjects/TfAndroid/data/'
class_indices = train_data_gen.class_indices
class_json = {}
for eachClass in class_indices:
    class_json[str(class_indices[eachClass])] = eachClass

with open(os.path.join(model_class_dir, "model_class.json"), "w+") as json_file:
    json.dump(class_json, json_file, indent=4, separators=(",", " : "),ensure_ascii=True)
    json_file.close()
print("JSON Mapping for the model classes saved to ", os.path.join(model_class_dir, "model_class.json"))



model_name = 'model_ex-{epoch:03d}_acc-{val_accuracy:03f}.h5'

trained_model_dir='D:/work/PycharmProjects/TfAndroid/data/'
model_path = os.path.join(trained_model_dir, model_name)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
             filepath=model_path,
             monitor='val_accuracy',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            mode='max',
            period=1)


def lr_schedule(epoch):
    # Learning Rate Schedule

    lr =1e-3
    total_epochs =epochs
    check_1 = int(total_epochs * 0.9)
    check_2 = int(total_epochs * 0.8)
    check_3 = int(total_epochs * 0.6)
    check_4 = int(total_epochs * 0.4)

    if epoch > check_1:
        lr *= 1e-4
    elif epoch > check_2:
        lr *= 1e-3
    elif epoch > check_3:
        lr *= 1e-2
    elif epoch > check_4:
        lr *= 1e-1

    return lr



#lr_scheduler =tf.keras.callbacks.LearningRateScheduler(lr_schedule)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)


num_train = len(train_data_gen.filenames)
num_test = len(val_data_gen.filenames)

print(num_train,num_test)

# 模型训练
# 使用fit_generator方法ImageDataGenerator来训练网络。

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(num_train / batch_size),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(num_test / batch_size),
    callbacks=[checkpoint,lr_scheduler]
)



