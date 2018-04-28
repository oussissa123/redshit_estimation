from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input, concatenate, Flatten
from utils import *
from keras.models import Model

def get_inception_conv_layer1(input_img):

    tower_1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_img)
    tower_1 = Conv2D(36, (3, 3), activation='relu', padding='same')(tower_1)

    tower_2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_img)
    tower_2 = Conv2D(36, (2, 2), activation='relu', padding='same')(tower_2)
    
    tower_3 = Conv2D(36, (1, 1), activation='relu', padding='same')(input_img)
    tower_4 = Conv2D(36, (2, 2), activation='relu', padding='same')(input_img)
    tower_5 = Conv2D(36, (3, 3), activation='relu', padding='same')(input_img)
    
    tow = concatenate([tower_3, tower_4], axis=1)
    tow = concatenate([tow, tower_5], axis=1)
    output = concatenate([tower_1, tower_2], axis=1)
    f = concatenate([tow, output], axis=1)
    
    return f

def get_inception_conv_layer2(input_img):
    tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(input_img)
    return tower_1
    #tower_2 = AveragePooling2D((2, 2), strides=(1, 1))(input_img)
    #return concatenate([tower_1, tower_2], axis=1)
    
def get_inception_conv_layer3(input_img): 
        tower_1 = Conv2D(32, (2, 2), activation='relu', padding='same')(input_img)
        #tower_2 = Conv2D(36, (3, 3), activation='relu', padding='same')(input_img)
        tower_3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_img)
        return concatenate([tower_1, tower_3], axis=1)
    
def get_inception_conv_layer4(input_img):
    tower_1 = MaxPooling2D((2, 2), strides=(2, 2))(input_img)
    return tower_1
    #tower_2 = AveragePooling2D((2, 2), strides=(2, 2))(input_img)
    #return concatenate([tower_1, tower_2], axis=1)        
        
        
def get_final_model():
    shape = (32,32, 5)
    input_img = Input(shape=shape)
    first = get_inception_conv_layer1(input_img)
    second = get_inception_conv_layer2(first)
    #third = get_inception_conv_layer3(second)
    #fourth = get_inception_conv_layer4(third)
    
    size = second.shape
    print(size)
    
    FullyConnect = Flatten()(second)
    FullyConnect = model_nn2(FullyConnect, 128, 5, dropout=0.3, batch_normalization=True, activation='relu', neurons_decay=0, starting_power=1, l2=10**-5)
    
    model = Model(inputs = input_img, outputs = FullyConnect)
    model.compile(loss=rmse_loss_keras, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print(model.output_shape)
    print('Okay ...')
    return model            