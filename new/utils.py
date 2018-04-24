import urllib.request as u
import bz2
import os
import pandas as pd
from astropy.io import fits
import numpy as np
import keras
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from IPython.display import SVG
import pylab as pl

bands = ['u', 'g', 'r', 'i', 'z']

def form_url(rerun, run, camcol, field):
    return ['https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/{}/{}/{}/frame-{}-{:06d}-{}-{:04d}.fits.bz2'.format(
        rerun, run, camcol, b, run, camcol, field) for b in ['u', 'g', 'r', 'i', 'z']]
def form_url_one(rerun, run, camcol, field, band):
    return 'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/{}/{}/{}/frame-{}-{:06d}-{}-{:04d}.fits.bz2'.format(
        rerun, run, camcol, band, run, camcol, field)

def dowload_as_fitbz2(url):
    result = u.urlopen(url)
    return result.read()


def dowload_as_fit(url):
    return bz2.decompress(dowload_as_fitbz2(url))

def save(file, data, type_save='wb'):
    saveFile = open(file,type_save)
    saveFile.write(data)
    saveFile.close()

def fit_dowload(url, file):
    save(file, dowload_as_fit(url))
        
def preprocessing(url):
    x = dowload_as_fit(url)
    return x
def dowload_as_fitbz2_store_in_file(url, file):
    save(file, dowload_as_fitbz2(url))
    
def dowload_as_fit_store_in_file(url, file):
    save(file, dowload_as_fit(url))
    
def comp_dowload(rerun, run, camcol, field, file):
    saveFile = open(file,'wb')
    saveFile.write(dowload_as_fitbz2(form_url(rerun, run, camcol, field)))
    saveFile.close()

"""
def fit_dowload(rerun, run, camcol, field, file):
    values = ['u', 'g', 'r', 'i', 'z']
    pos =0;
    for url in form_url(rerun, run, camcol, field):
        saveFile = open(file+'-'+values[pos]+'.fits','wb')
        saveFile.write(dowload_as_fit(url))
        saveFile.close()
        pos = 1 + pos

def fit_dowload(rerun, run, camcol, field, directory):
    values = ['u', 'g', 'r', 'i', 'z']
    pos =0;
    for url in form_url(rerun, run, camcol, field):
        saveFile = open(directory+'/'+rerun+'-'+run+'-'+camcol+'-'+field+'-'+values[pos]+'.fits','wb')
        saveFile.write(dowload_as_fit(url))
        saveFile.close()
        pos = 1 + pos """ 
          
        
def getName(rerun, run, camcol, field, filtre):
    #return (str(rerun)+'-'+str(run)+'-'+str(camcol)+'-'+str(field)+'-'+filtre+'.fits')
    return '{}-{:06d}-{}-{:04d}-{}.fits'.format(rerun, run, camcol, field, filtre)    
#def getName2(rerun, run, camcol, field):
 #   names = [[rerun, run, camcol, field, filtre] for filtre in ['u', 'g', 'r', 'i', 'z']]
 #   return names

def getName3(rerun, run, camcol, field):
    names = [getName(rerun, run, camcol, field, filtre) for filtre in ['u', 'g', 'r', 'i', 'z']]
    return names

def createDirectory(rep):
    if not os.path.isdir(rep):
        os.makedirs(rep)
"""
def allurlinseries(file):
    objects = pd.read_csv(file, usecols=['rerun','run','camcol','field'])
    return objects[['rerun','run','camcol','field']].apply(lambda x: form_url(*x), axis=1).sum(), objects[['rerun','run','camcol','field']].apply(lambda x: getName3(*x), axis=1).sum()

def allurlinseries1(file):
    objects = pd.read_csv(file, usecols=['rerun','run','camcol','field', 'objid'])
    return objects[['rerun','run','camcol','field']].apply(lambda x: form_url(*x), axis=1).sum(), objects[['rerun','run','camcol','field']].apply(lambda x: getName3(*x), axis=1).sum(), objects['objid'] 

w = 16 # Window size (images of the size 2w*2w)
for index, values in objects.iterrows(): 
    band_arrays = []
    rowc = int(round(values[[x for x in objects.columns if x.startswith('rowc_')]].mean()))
    colc = int(round(values[[x for x in objects.columns if x.startswith('colc_')]].mean()))
    for i in range(5):
        filename = get_file_name(form_url(*values[['rerun', 'run', 'camcol', 'field']].astype('int'))[i])
        file = fits.open(os.path.join('../data/images/', filename))
        im = file[0].data
        band_arrays.append(np.log(1 + -im.min() + 
                                  im[max(0, rowc-w):min(rowc+w, im.shape[0]),
                                     max(0,colc-w):min(colc+w, im.shape[1])]))
        file.close()
        del im
    object_tensor = np.stack(band_arrays, axis=2)
    del band_arrays
    np.save(os.path.join('../data/object_images/', str(int(values['objid']))), object_tensor)
    del object_tensor
    
  """
"""

def split_by_center(file_in_fit_format):
    w = 16
    file = fits.open(file_in_fit_format)
    im = file[0].data
    rowc = (im.shape[0]//2) 
    colc = (im.shape[1]//2)
    img = np.log(1-im.min() + im[rowc-w:rowc+w,colc-w:colc+w])
    return img

def split_normal(file_in_fit_format):
    return split_by_center(file_in_fit_format)

def get_final_fimage(directory, rerun, run, camcol, field):
    final = []
    file_names = getName3(rerun, run, camcol, field);
    try:
        for u in file_names:
            final.append(split_by_center(u))
    except:
        return []
    return np.stack(final, axis = 2)

def dowload_split_save(url, file):
    fit_dowload(url, file);
    return split_normal(file)




def dowload_final(file_in_csv_format, dir_all, dir_normales):
    #./data/csvs/galaxies/1.csv'
    urls, names, objid = allurlinseries1(str(file_in_csv_format));             
    temp = []
    pos = 1;
    i = 0;
    for v in range(len(urls)):
        #./data/images/galaxies1/all
        temp = dowload_split_save(urls[v], (str(dir_all)+'/{}').format(names[v]))
        if (pos == 5):
            #/data/images/galaxies1/normales
            np.save((str(dir_normales)+'/{}').format( objid[i]), np.stack(temp,axis=2))
            temp = []
            i = i + 1
        print(v)

"""
        
def save_preprocessing_images(data_path, dir_save, w, verbose = 0):
    #w = 16 # Window size (images of the size 2w*2w)
    data_path = str(data_path); dir_save = str(dir_save); w = int(w);
    objects = pd.read_csv(data_path)
    ob = 1
    for index, values in objects.iterrows(): 
        band_arrays = []
        rowc = int(round(values[[x for x in objects.columns if x.startswith('rowc_')]].mean()))
        colc = int(round(values[[x for x in objects.columns if x.startswith('colc_')]].mean()))
        all_urls_bands = form_url(*values[['rerun', 'run', 'camcol', 'field']].astype('int'));
        if (verbose == 1):
            print("Leading of object number : ", ob)
            ob = ob+1;
        for i in range(5):
            if (verbose == 1):
                print('------| leading of band ',bands[i])
            filename = 'file.fits'##get_file_name(all_urls_bands[i])
            fit_dowload(all_urls_bands[i], filename)
            file = fits.open(filename)
            im = file[0].data
            band_arrays.append(np.log(1 + -im.min() + im[max(0, rowc-w):min(rowc+w, im.shape[0]), max(0,colc-w):min(colc+w,        im.shape[1])]))
            file.close()
            del im
        object_tensor = np.stack(band_arrays, axis=2)
        del band_arrays
        # dir_save = '../data/images/galaxies1/all/'
        np.save(os.path.join(dir_save, str(int(values['objid']))), object_tensor)
        del object_tensor
    print('------------------------ dowloading ok ------------------------')
    
def rmse_loss_keras(y_true, y_pred):
    diff = keras.backend.square((y_pred - y_true) / (keras.backend.abs(y_true) + 1))
    return keras.backend.sqrt(keras.backend.mean(diff))

def model_nn(model, input_dim , n_hidden_layers, dropout=0, batch_normalization=False, activation='relu', neurons_decay=0,starting_power=1,l2=0, compile_model=True, trainable=True):
    assert dropout >= 0 and dropout < 1
    assert batch_normalization in {True, False}
    #model = keras.models.Sequential()
    for layer in range(n_hidden_layers):
        #print(layer)
        n_units = 2**(int(np.log2(input_dim)) + starting_power - layer*neurons_decay)
        if n_units < 8:
            n_units = 8
        if layer == 0:
            model.add(Dense(units=n_units, name='Dense_' + str(layer + 1), 
                            kernel_regularizer=keras.regularizers.l2(l2)))
        else:
            model.add(Dense(units=n_units, name='Dense_' + str(layer + 1), 
                            kernel_regularizer=keras.regularizers.l2(l2)))
        if batch_normalization:
            model.add(BatchNormalization(name='BatchNormalization_' + str(layer + 1)))
        model.add(Activation('relu', name='Activation_' + str(layer + 1)))
        if dropout > 0:
            model.add(Dropout(dropout, name='Dropout_' + str(layer + 1)))
            
    model.add(Dense(units=1, name='Dense_' + str(n_hidden_layers+1), 
    				kernel_regularizer=keras.regularizers.l2(l2)))
    model.trainable = trainable
    if compile_model:
        model.compile(loss=rmse_loss_keras, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

def save_model(model, file):
    model_json = model.to_json()
    with open(file, "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

def load_model(file):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return model_from_json(loaded_model_json)

def load_data(file_csv, dir_img):
    perte = 0
    data = pd.read_csv(file_csv, usecols=['objid','redshift']);
    O = data['objid']
    R = data['redshift']
    X = [] 
    Y = []
    all_ = ''
    for v in range(len(O)):
        try:
            suffix = 'npy'
            path = os.path.join(dir_img, str(int(O[v]))+'.'+suffix)
            image = np.load(path)
            assert image.shape == (32,32,5)
            X.append(image)
            Y.append(R[v])
        except :
            perte = perte + 1
            print('Un objet de moins')
            all_ = all_ + str(int(O[v])) +'\n'
    del O
    del R
    print('-------------- Data loading ok -----------')
    print('     perte = ', perte)
    print(all_)
    print('     number of images: ', len(X), ' class size :', len(Y))
    print(' -------- End -------')
    X = np.array(X)
    Y = np.array(Y)
    return X, Y   
    
def get_train_test_valid_data_galaxy(file_csv, dir_img, test_size = 0.21, valid_size = 0.4):
    X, Y = load_data(file_csv, dir_img)
    X_Train, X_Test, Y_Train, Y_Test =  train_test_split(X,Y, test_size = test_size)
    del X
    del Y
    X_Test, X_Valid, Y_Test, Y_Valid = train_test_split(X_Test,Y_Test, test_size = valid_size)
    return X_Train, Y_Train, X_Test, Y_Test, X_Valid, Y_Valid
    

def plot_model(model):
    return SVG(keras.utils.vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))    

def compute_metrics(y_true, y_pred, clf_name):
    result = pd.Series()
    delta_znorm = (y_pred - y_true)/(1 + y_true)
    result.loc['RMSE_znorm'] = np.sqrt(np.mean((delta_znorm)**2))
    val = np.std(delta_znorm)
    result.loc['bias_znorm'] = val
    result.loc['std_znorm'] = np.std(delta_znorm)
    result.loc['RMSE'] = np.sqrt(np.mean((y_pred - y_true)**2))
    result.loc['|znorm| > 0.15 (%)'] = 100*np.sum(np.abs(delta_znorm) > 0.15)/y_true.shape[0]
    result.loc['|znorm| > 3std (%)'] = 100*np.sum(np.abs(delta_znorm) > 3*val)/y_true.shape[0]
    result.name = clf_name
    return result 
    
def plot_history(history, ymax=None):
    pl.figure(figsize=(15, 7))
    pl.plot(history.history['loss'])
    pl.plot(history.history['val_loss'])
    pl.title('model loss')
    pl.ylabel('loss')
    pl.xlabel('epoch')
    if ymax is not None:
        pl.ylim(ymax=ymax)
    pl.legend(['train', 'validation'], loc='upper left')    
    pl.show();

def plot_history_ac(history, ymax=None):
    pl.plot(history.history['acc'])
    pl.plot(history.history['val_acc'])
    pl.title('model accuracy')
    pl.ylabel('accuracy')
    pl.xlabel('epoch')
    pl.legend(['train', 'test'], loc='upper left')
    pl.show()
    
    
def split_csv_data(file, result_dir, size_for_each=50):
    file_ = open(file, 'r');
    titles = file_.readline();
    data = titles
    compte = 1
    nbr_split = 1
    for line in file_.readlines():
        data = data + line
        if (compte==size_for_each):
            compte = 1
            save(result_dir+'/'+str(nbr_split)+'.csv', data, type_save='w')
            data = titles
            nbr_split = nbr_split + 1
        else:
            compte = compte + 1
    if data != titles:
        save(result_dir+'/'+str(nbr_split)+'.csv', data, type_save='w')
    file_.close();
    print('------------- spliting ok -----------------')
    
    