from keras.layers import Flatten as flat, Activation as act , Dense, Dropout as drop, Conv2D as conv, MaxPooling2D as maxp, AveragePooling2D as avgp, Input, concatenate as concat
from keras.models import Model as M
from utils import *
import keras

def module(entry, size, act = 'tanh'):
    size = int(size)
    
    one1 = conv(1,size, activation = act, strides = 1)(entry)
    one2 = maxp(pool_size = (size,size), strides = 1)(entry)
    one3 = avgp(pool_size = (size,size), strides = 1)(entry)
    
    return concat([one1, one2, one3])

def modele():
    entry = Input(shape=(32, 32, 5))
    entry = module(entry, 7)
    entry = module(entry, 7)
    entry = module(entry, 5)
    entry = module(entry, 5)
    

    entry = module(entry, 3, act = 'relu')
    entry = module(entry, 3, act = 'relu')
    entry = module(entry, 3, act = 'relu')
    entry = module(entry, 3, act = 'relu')
    
    entry = module(entry, 2, act = 'relu')
    entry = module(entry, 2, act = 'relu')
    entry = module(entry, 2, act = 'relu')
    #entry = module(entry, 2, act = 'relu')
    #entry = module(entry, 2, act = 'relu')
    
    print(entry.shape)

modele()
"""
model = M (inputs = In, outputs = tensor)
opt = keras.optimizers.SGD(lr=0.0001, momentum = 0.9, decay = 0.0000001)
#keras.optimizers.Adam()
model.compile(loss=rmse_loss_keras, optimizer=opt, metrics=['accuracy'])
print('compilation ok ...')

# ------------------ Execution et validation -----------------------------------------

batch_size = 128
ep = 20
dir_img = '/home/peta/ouissa/images/*.npy'

X_Train, Y_Train, X_Test, Y_Test, X_Valid, Y_Valid = get_train_test_valid_data_galaxy1(dir_img,test_size=0.4, valid_size=0.5, clas='GALAXY')

#stopearly = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=16, verbose=1, mode='min')
history = model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=ep, batch_size=batch_size, verbose=1)#, callbacks=stopearly)
dire = '/home/etud/ouissa/stage/redshit_estimation/new/stage/modeles'
save_all_model(model, dire+'model_modele1.json')
#plot_history(history);
# ------------------------- Test ---------------------------------

predict = model.predict(X_Test, batch_size=batch_size).reshape(-1)
result = compute_metrics(Y_Test, predict, 'Redshift')
print(result)
"""
