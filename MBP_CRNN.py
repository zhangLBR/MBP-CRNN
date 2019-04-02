from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling3D, Flatten, ConvLSTM2D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
# x,y are modified datasets for machine learning model training and validation
# batch_size are defined by research need
batch_size = 1
x = np.array(x)
y = np.array(y)
print (x.shape)
print (y.shape)
kf = KFold(n_splits=5,shuffle=True, random_state=5)
k = 0
for train_index, test_index in kf.split(x):
        train_x,test_x = x[train_index], x[test_index]
        train_y,test_y = y[train_index], y[test_index]
        X_train = train_x.reshape(train_x.shape[0],batch_size, train_x.shape[1], train_x.shape[2],train_x.shape[3])
        X_test = test_x.reshape(test_x.shape[0],batch_size, test_x.shape[1], test_x.shape[2],test_x.shape[3])
        model = Sequential()
        model.add(ConvLSTM2D(32, kernel_size=(3,3), activation='relu', padding='same', batch_input_shape=(None,batch_size,34,9,24),return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1,2,2), padding='same'))
        model.add(Dropout(0.2))
        model.add(ConvLSTM2D(64, kernel_size=(3,3), activation='relu', padding='same',return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1,2,2), padding='same'))
        model.add(Dropout(0.2))
        model.add(ConvLSTM2D(128, kernel_size=(3,3), activation='relu',padding='same',return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1,2,2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(625, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])
        model.fit(X_train, train_y, batch_size=100, epochs=1000, verbose=0)
        result = model.predict(X_test)
        para = model.evaluate(X_test,test_y, verbose=0)[1]
        k= k + 1
        with open("training_result","a") as f:
                f.write(str("prediction=")+str(result)+'\n')
                f.write(str("evaluation=")+str(para)+'\n')
                f.write(str("step")+str(k)+str("finished")+'\n')
        model.save("mhc_training_model")
