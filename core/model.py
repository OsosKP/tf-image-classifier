import os
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Layer, concatenate, GlobalAveragePooling2D, Activation, Softmax
from tensorflow.keras.utils import normalize
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Accuracy, AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import preprocessing

data_dir = preprocessing.root_data_dir
test_path = preprocessing.test_path
train_path = preprocessing.train_path
val_path = preprocessing.validation_path
image_shape = (300, 300, 3)

loss_param = 'binary_crossentropy'
optimizer_param = 'adam'
stop_monitor = 'val_auc'
stop_mode = 'max'
stop_patience = 2
batch_size = 64

image_gen = ImageDataGenerator(fill_mode='nearest')
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
image_gen.flow_from_directory(val_path)


class fire_module(Layer):

    def __init__(self, squeeze_size=16, expand_size=64):
        super(fire_module, self).__init__()
        self.squeeze = Conv2D(filters=squeeze_size, kernel_size=(1, 1),
                              padding='valid', activation='relu', name="sq1x1")
        self.exp1_1 = Conv2D(filters=expand_size, kernel_size=(1, 1),
                             padding='valid', activation='relu', name="exp1x1")
        self.exp3_3 = Conv2D(filters=expand_size, kernel_size=(3, 3),
                             padding='same', activation='relu', name="exp3x3")

    def call(self, inpt):
        squeezed_value = self.squeeze(inpt)
        exp1_1_value = self.exp1_1(squeezed_value)
        exp3_3_value = self.exp3_3(squeezed_value)
        return concatenate([exp1_1_value, exp3_3_value], axis=-1, name='concat')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'squeeze': self.squeeze,
            'exp1_1': self.exp1_1,
            'exp3_3': self.exp3_3
        })
        return config


model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3),
                 input_shape=image_shape, activation='relu'))

model.add(fire_module(6, 8))
model.add(MaxPool2D(pool_size=(3, 3)))

model.add(fire_module(12, 16))
model.add(MaxPool2D(pool_size=(3, 3)))

model.add(fire_module(18, 24))
model.add(MaxPool2D(pool_size=(3, 3)))

model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid'))
model.add(GlobalAveragePooling2D())

model.compile(loss=loss_param, optimizer=optimizer_param,
              metrics=[AUC()])


early_stop = EarlyStopping(
    monitor=stop_monitor, mode=stop_mode, patience=stop_patience)

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               shuffle=False)  # Don't want to shuffle test data and lose labels

results = model.fit(train_image_gen, epochs=20,
                    validation_data=test_image_gen,
                    callbacks=[early_stop])

losses = model.history.history
losses['loss'] = np.asarray(losses['loss'])
losses['val_loss'] = np.asarray(losses['val_loss'])
final_number_of_epochs = len(losses['loss'])
min_loss = losses['loss'].min()
mean_loss = losses['loss'].mean()
final_loss = losses['loss'][-1]
min_val_loss = losses['val_loss'].min()
mean_val_loss = losses['val_loss'].mean()
final_val_loss = losses['val_loss'][-1]


def get_model_summary():
    output = []
    model.summary(print_fn=lambda line: output.append(line))
    return str(output).strip('[]')


summary = get_model_summary()

record = {
    'Epochs': final_number_of_epochs,
    'Batch_Size': batch_size,
    'Loss_Func': loss_param,
    'Optimizer': optimizer_param,
    'Early_Stop_Monitor': stop_monitor,
    'Early_Stop_Patience': stop_patience,
    'Min_Loss': min_loss,
    'Mean_Loss': mean_loss,
    'Final_Loss': final_loss,
    'Min_Val_Loss': min_val_loss,
    'Mean_Val_Loss': mean_val_loss,
    'Final_Val_Loss': final_val_loss,
    'Model': summary
}

new_data = pd.DataFrame(record, index=[0])

if os.path.exists('results.csv'):
    df_records = pd.read_csv('results.csv')
    df_records = df_records.append(new_data)
else:
    df_records = pd.DataFrame(new_data)

df_records.to_csv('results.csv', float_format='%g')

model.save('image_classifier_auc.h5')
