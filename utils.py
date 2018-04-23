import urllib.request as u
import bz2
import os
import pandas as pd

def form_url(rerun, run, camcol, field):
    return ['https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/{}/{}/{}/frame-{}-{:06d}-{}-{:04d}.fits.bz2'.format(
        rerun, run, camcol, b, run, camcol, field) for b in ['u', 'g', 'r', 'i', 'z']]

def dowload_as_fitbz2(url):
    result = u.urlopen(url)
    return result.read()


def dowload_as_fit(url):
    return bz2.decompress(dowload_as_fitbz2(url))
    

def fit_dowload(url, file):
        saveFile = open(file,'wb')
        saveFile.write(dowload_as_fit(url))
        saveFile.close()
   
    
    
def dowload_as_fitbz2_store_in_file(url, file):
    saveFile = open(file,'wb')
    saveFile.write(dowload_as_fitbz2(url))
    saveFile.close()
    
def dowload_as_fit_store_in_file(url, file):
    saveFile = open(file,'wb')
    saveFile.write(dowload_as_fit(url))
    saveFile.close()
    

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

def allurlinseries(file):
    objects = pd.read_csv(file, usecols=['rerun','run','camcol','field'])
    return objects[['rerun','run','camcol','field']].apply(lambda x: form_url(*x), axis=1).sum(), objects[['rerun','run','camcol','field']].apply(lambda x: getName3(*x), axis=1).sum()



"""
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

