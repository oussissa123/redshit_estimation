from keras.layers import Add, Conv2D, BatchNormalization as batch, Input, AveragePooling2D as avp, Flatten as F, Dense as D
from keras.models import Model as M
from keras.optimizers import Adam
import utils as u

def get_basic_residual_block(x, size):    
    y = Conv2D(size, (3,3), activation="relu", padding = "same")(x)
    y = batch()(y)
    y = Conv2D(size, (3,3), activation="relu", padding = "same")(y)
    y = batch()(y)
    z = Conv2D(size, (1,1), activation="relu", padding = "same")(y)
    y = Add()([y,z])
    return y
def get_basic_residual_model(input_shape = Input(shape = (32,32,5)), layer_number = 32, k = 4, start_size = 32):
    first = input_shape
    tab = []
    l_n = layer_number // k
    l_r = layer_number % k
    for i in range(k):
        if i == k-1:
            tab.append(l_n+l_r)
        else:
            tab.append(l_n)
    size = start_size
    
    for i in range(k):
        for j in range(tab[i]):
            first = get_basic_residual_block(first, start_size*(2**i))
    S = input_shape.shape
    val = (int(S[1]), int(S[2]))
    first = avp(pool_size=val)(first)
    first = F()(first)
    
    model = u.model_nn2(first, 64 , 5, dropout=0, batch_normalization=True, activation=None)
    
    model = M(inputs = input_shape, outputs = model)
    model.compile(loss=u.rmse_loss_keras, optimizer=Adam(), metrics=['accuracy'])
    
    return model
