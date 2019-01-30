import pandas as pd
import nibabel as nib
import numpy as np
import glob
import json
import random
import os
import sys
from scipy import stats
from numpy import copy

from keras.losses import categorical_crossentropy
from keras.models import model_from_json, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import Adam
from keras.utils import np_utils


def get_coord_random(dim,tile_width,nb_tiles):
    return list(np.random.random_integers(0,dim-tile_width,nb_tiles))

def consolidate_seg(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 5 indicate a fold in the gray matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    d={2:0,4:0,5:0,6:0,7:0,8:0,3:1}
    newArray = copy(seg)
    for k, v in d.items(): newArray[seg==k] = v
    return newArray

def segment(tile,seg,white):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it a high value of 10
    d={2:white,4:white,5:white,6:white,7:white,8:white}
    newArray = copy(tile)
    for k, v in d.items(): newArray[seg==k] = v
    return newArray

def normalize(tile):
    m=float(np.mean(tile))
    st=float(np.std(tile))
    if st > 0:
        norm = (tile - m) / float(st)
    else:
        norm = tile - m
    return norm

def get_channel(img):
    ch_ret=-1
    num_ch_labeled=0
    for ch in range(img.shape[2]):
        if len(np.unique(img[:,:,ch]))>1:
            ch_ret=ch
            num_ch_labeled+=1
    return ch_ret,num_ch_labeled

def rgb_2_lum(img):
    # the rgb channel is located at axis=2 for the data
    img=0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    return img

def flip_img(tile,seg):
    # we only flip along horizontal, located on second axis
    # tile is (2560,2560)
    # seg is (2560,2560)
    f1=1 # int(2*np.random.randint(0,2)-1)
    f2=int(2*np.random.randint(0,2)-1)
    return tile[::f1,::f2],seg[::f1,::f2]


def gen_tiles_random(img_fn,segment_fn,nb_tiles=1,tile_width=2560):
    data = nib.load(img_fn).get_data()
    white= int(stats.mode(data, axis=None)[0])
    shape = data.shape
    seg_data = nib.load(segment_fn).get_data()

    data=rgb_2_lum(data)

    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(segment_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(segment_fn))
    seg_data=seg_data[:,:,ch]
    data=segment(data,seg_data,white)
    seg_data=consolidate_seg(seg_data)
    data=normalize(data)

    coord_x=get_coord_random(shape[0],tile_width,nb_tiles)
    coord_y=get_coord_random(shape[1],tile_width,nb_tiles)
    tiles = np.zeros([nb_tiles]+[tile_width]*2+[1])
    # (1,2560,2560)
    seg = np.zeros([nb_tiles]+[tile_width]*2)
    tidx=0
    for tidx in range(nb_tiles):
        x = coord_x[tidx]
        y = coord_y[tidx]

        seg_tile=np.squeeze(seg_data[x:x+tile_width,y:y+tile_width])
        tile=data[x:x+tile_width,y:y+tile_width]

        tile,seg_tile=flip_img(tile,seg_tile)

        tiles[tidx]=tile
        seg[tidx]=seg_tile

    return tiles,seg


def jaccard_index(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def save_epochs(path,epochs_per_set,set_nb,epochs_running):
    # after each epoch completes and save as text file
    fn=os.path.join(path,'epochs.out')
    open_as='a'
    if not os.path.isfile(fn):
        open_as='w'
    with open(fn, open_as) as outfile:
        outfile.write('{}\n'.format(epochs_running+epochs_per_set))

def save_history(path,performance,set_nb):
    # track performance (accuracy and loss) for training and validation sets
    # after each epoch completes and save as .json string
    json_string=json.dumps(performance.history)
    fn=os.path.join(path,'history.set{0:03d}.performance.json'.format(set_nb))
    with open(fn, 'a') as outfile:
        json.dump(json_string, outfile)

def get_set_nb(path,epochs_per_set):
    fn=os.path.join(path,'epochs.out')
    if not os.path.isfile(fn):
        set_nb=0
        epochs_running=0
    else:
        with open(fn) as f:
            for i,epochs_running in enumerate(f):
                pass
        set_nb=i+1
        epochs_running=int(epochs_running.strip())
            # ep_list=[int(i.strip()) for i in f]
        # set_nb=int(ep_list[-1]/epochs_per_set)
        if set_nb>1000:
            sys.exit('\n>>>ENOUGH<<< We have reached sufficient number of sets\n')
    if not os.path.isdir(os.path.join(path,'set{0:03d}'.format(set_nb))):
        os.makedirs(os.path.join(path,'set{0:03d}'.format(set_nb)))
    return set_nb,epochs_running



def get_new_model(model_version,bn_bias,verbose=False):
    # dimension 2560x2560
    fn = "/home/rpizarro/histo/model/model.unet.v{0}_{1}.json".format(model_version,bn_bias)
    print('Loading model with architecture from : {}'.format(fn))
    with open(fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    # model.compile(optimizer=Adam(lr=1e-5), loss=weighted_categorical_crossentropy_fcn_loss, sample_weight_mode="temporal", metrics=[jaccard_index])
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[jaccard_index])
    if verbose:
        print(model.summary())
    return model

def get_model(path,model_version,bn_bias,verbose=False):
    list_of_files = glob.glob(os.path.join(path,'model*FINAL.h5'))
    if list_of_files:
        # print(list_of_files)
        model_fn = max(list_of_files, key=os.path.getctime)
        print('Loading model : {}'.format(model_fn))
        model = load_model(model_fn)
        if verbose:
            print(model.summary())
    else:
        print('We did not find any models.  Getting a new one!')
        model = get_new_model(model_version,bn_bias,verbose=verbose) 
    return model


def get_input_output_size(model_version=101):
    if '105' in model_version:
        input_size=(2430,2430,1)
        output_size=(2430,2430,2)
    elif '107' in model_version:
        input_size=(2500,2500,1)
        output_size=(2500,2500,2)
    else:
        input_size=(2560,2560,1)
        output_size=(2560,2560,2)
    return input_size,output_size

def runNN(train_df,valid_df,model_version,bn_bias,epochs_per_set):

    input_size,output_size = get_input_output_size(model_version)

    # number of tiles per step
    nb_step=1 #20
    # epochs_per_set=10
    steps_per_epoch=50

    weights_dir = os.path.dirname("/home/rpizarro/histo/weights/xplor/{}/v{}/".format(bn_bias,model_version))
    set_nb,epochs_running=get_set_nb(weights_dir,epochs_per_set)
    print('This is set {} : epochs previously completed {} : epochs in this set {}'.format(set_nb,epochs_running,epochs_per_set))

    model = get_model(weights_dir,model_version,bn_bias,verbose=True)

    # track performance (dice coefficient loss) on train and validation datasets
    performance = History()
    set_path=os.path.join(weights_dir,'set{0:03d}'.format(set_nb),'model.v{0}.set{1:03d}.'.format(model_version,set_nb)+'{epoch:04d}.valJacIdx{val_jaccard_index:0.3f}.h5')
    checkpointer=ModelCheckpoint(set_path, monitor='val_loss', verbose=0, save_best_only=False, mode='min', period=1)

    # fit the model using the data generator defined below
    model.fit_generator(fileGenerator(train_df,nb_step=nb_step,verbose=False,input_size=input_size,output_size=output_size), steps_per_epoch=steps_per_epoch, epochs=epochs_per_set, verbose=1,
            validation_data=fileGenerator(valid_df,nb_step=1,verbose=False,input_size=input_size,output_size=output_size),validation_steps=1,callbacks=[performance,checkpointer])

    # save the weights at the end of epochs
    model_FINAL_fn = os.path.join(weights_dir,'model.v{0}.set{1:03d}.epochs{2:04d}.FINAL.h5'.format(model_version,set_nb,epochs_running+epochs_per_set))
    model.save(model_FINAL_fn,overwrite=True)
    # save the performance (accuracy and loss) history
    save_history(weights_dir,performance,set_nb)
    save_epochs(weights_dir,epochs_per_set,set_nb,epochs_running)


def fileGenerator(df,nb_step=1,verbose=True,input_size=(2560,2560,1),output_size=(2560,2560,2)):
    X = np.zeros((nb_step,) + input_size )
    # (1,2560,2560,2)
    Y = np.zeros((nb_step,) + output_size )
    n = 0
    while True:
        while n < nb_step:
            try:
                i = np.random.randint(0,df.shape[0])
                slice_fn = df['slice_fn'][i]
                segment_fn = df['segment_fn'][i]
                if verbose:
                    print("{} : {}".format(slice_fn,segment_fn))
                tile_width = input_size[0]
                tiles,seg = gen_tiles_random(slice_fn,segment_fn,nb_step,tile_width)
                nb_slices = tiles.shape[0]
                seg = np.reshape(np_utils.to_categorical(seg,output_size[-1]),output_size)
                seg = seg.reshape((nb_slices,)+output_size)
                X[:nb_slices] = tiles
                Y[:nb_slices] = seg
                n += nb_slices
            except Exception as e:
                print(str(e))
                pass
        if X.size:
            yield X,Y
        else:
            print("X is empty!!!")
            continue


# input to : python ribbon.train_unet.model.py 101 bn_bias 100
model_version = sys.argv[1]
bn_bias = sys.argv[2]
epochs_per_set = int(sys.argv[3])

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

csv_dir = '/home/rpizarro/histo/XValidFns/rm311_128slices'
train_fn = os.path.join(csv_dir,'train.csv')
train_df = pd.read_csv(train_fn)

valid_fn = os.path.join(csv_dir,'valid.csv')
valid_df = pd.read_csv(valid_fn)

runNN(train_df,valid_df,model_version,bn_bias,epochs_per_set)


