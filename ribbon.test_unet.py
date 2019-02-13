import nibabel as nib
import numpy as np
import pandas as pd
# import cv2 as cv
import glob
import json
import os,sys
import tensorflow as tf
from scipy import ndimage,stats
from numpy import copy
import difflib
import random
from subprocess import check_call

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import model_from_json, load_model
from keras import backend as K
from keras.utils import np_utils

def get_coord_random(dim,tile_width,nb_tiles):
    # nx is the number of tiles in the x-direction to cover the edge
    nx=int(np.ceil(float(dim[0])/tile_width))
    # ny is the number of tiles in the y-direction to cover the edge
    ny=int(np.ceil(float(dim[1])/tile_width))
    gap=0 
    if nx>1:
        gap = (tile_width*nx-dim[0])/(nx-1)
    # uniformly sample along one dimension to cover the edge
    uni_x = [int(np.floor(i*(tile_width-gap))) for i in range(nx)]
    uni_x[-1]=dim[0]-tile_width
    edge_x=[0]*ny+[dim[0]-tile_width]*ny+uni_x*2
    rand_x=list(np.random.random_integers(0,dim[0]-tile_width,nb_tiles))
    # x=edge_x+x

    gap=0
    if ny>1:
        gap = (tile_width*ny-dim[1])/(ny-1)
    # uniformly sample along one dimension to cover the edge
    uni_y = [int(np.floor(i*(tile_width-gap))) for i in range(ny)]
    uni_y[-1]=dim[1]-tile_width
    edge_y=uni_y*2+[0]*nx+[dim[1]-tile_width]*nx
    rand_y=list(np.random.random_integers(0,dim[1]-tile_width,nb_tiles))
    # y=edge_y+y

    return [(a,b) for a in edge_x for b in edge_y]+list(zip(rand_x,rand_y))
    # return zip(x,y)


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

def segment(tile,seg,hull,white):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it a high value of 10
    d={2:white,4:white,5:white,6:white,7:white,8:white}
    tmpArray = copy(tile)
    for k, v in d.items(): tmpArray[seg==k] = v
    # return tmpArray

    # bring HULL back in
    d={0:white}
    newArray = copy(tmpArray)
    for k, v in d.items(): newArray[hull==k] = v
    return newArray


def swap_labels(tile,a,b):
    # we wish to swap elements in order to change the color in itksnap
    # this function swaps a for b
    for m,row in enumerate(tile):
        for n,elem in enumerate(row):
            if elem in [a]:
                tile[m][n]=b
    return tile

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

def zero_pad(data,tile_width):
    shape=data.shape
    shape_min=(tile_width,tile_width)
    shape_target=tuple([max(i,j) for i,j in zip(shape,shape_min)])
    data_pad=np.zeros(shape_target)+255
    data_pad[:data.shape[0],:data.shape[1]]=data
    return data_pad


def gen_tiles(save_dir,img_fn,seg_fn,hull_fn,tile_width,slice_nb,nb_tiles):
    data = nib.load(img_fn).get_data()
    white= int(stats.mode(data, axis=None)[0])
    seg_data = nib.load(segment_fn).get_data()
    hull_data = nib.load(hull_fn).get_data()

    data=np.squeeze(rgb_2_lum(data))
    shape = data.shape
    print(shape)

    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(segment_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(segment_fn))
        ch=1
    seg_data=np.squeeze(seg_data[:,:,ch])
    hull_data=np.squeeze(hull_data[:,:,1])
    data=segment(data,seg_data,hull_data,white)
    seg_data=consolidate_seg(seg_data)
    fn='{0}_6x_concat_6x_whole.luminance.nii'.format(slice_nb)
    save_to_nii(np.reshape(data,shape+(1,1,)),save_dir,fn)
    # prepare for input
    data=normalize(data)

    coord=get_coord_random(shape,tile_width,nb_tiles)
    coord=sorted(list(set(coord)))
    print(coord)
    nb_tiles=len(coord)

    # tiles should have dimension (20-30,560,560,3)
    tiles = np.zeros([nb_tiles]+[tile_width]*2+[1])
    seg = np.zeros([nb_tiles]+[tile_width]*2)
    tidx=0
    for x,y in coord:
        print((tidx,x,y))
        seg[tidx]=seg_data[x:x+tile_width,y:y+tile_width]
        tiles[tidx,:,:,0]=data[x:x+tile_width,y:y+tile_width]
        print(np.unique(seg[tidx]))
        print(np.amin(tiles[tidx,:,:,0]),np.amax(tiles[tidx,:,:,0]))
        tidx+=1
    return tiles,seg,coord,shape


def avg_tile(slice_avg,single_tile,x,y,tile_width):
    slice_sum=slice_avg[0]
    slice_sum[x:x+tile_width,y:y+tile_width]+=single_tile

    slice_count=slice_avg[1]
    slice_count[x:x+tile_width,y:y+tile_width]+=1
    return slice_sum,slice_count


def retile(tiles,coord,slice_shape,tile_width):
    # slice_shape is rgb shape with a 3 at the end
    nb_tiles=tiles.shape[2]
    print(slice_shape)
    print(tiles.shape)
    print(coord)
    # typical size: (25,2666,2760)
    slice_sum=np.zeros(slice_shape)
    slice_count=np.zeros(slice_shape)
    slice_avg=[slice_sum,slice_count]
    # tabulate the elements here, we will do a final mode at the end
    slice = np.zeros(slice_shape)
    tidx=0
    for x,y in coord:
        single_tile=tiles[:,:,tidx,0]
        slice_avg=avg_tile(slice_avg,single_tile,x,y,tile_width)
        tidx+=1
    print('slice : ( {},{} )'.format(np.amin(slice_avg[0]),np.amax(slice_avg[0])))
    print('slice : {}'.format(np.unique(slice_avg[1])))
    slice=np.true_divide(slice_avg[0],slice_avg[1])
    # flip slice to make equivalent to original dataset
    # slice=slice[::-1,::-1]
    slice=np.reshape(slice,slice.shape+(1,1,))
    return slice



def jaccard_index(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def calc_jac_idx(data,str_arg='None'):
    data=data.astype(np.int64)
    try:
        y = np.bincount(data)
    except Exception as e:
        print("Warning: {}".format(e))
        print(data)
        print(str_arg)
        return 0.0
    ii = np.nonzero(y)[0]
    # tn = true negative, means the two raters agree it is not gray matter
    # fn = false negative, means jeeda marked it as not gray matter and deepthi marked as gray matter
    # fp = false positive, means jeeda marked it as gray matter and deepthi marked as not gray matter
    # tp = true positive,  means the two raters agree it is gray matter
    ii_str=['tn','fn','fp','tp']
    tn,fn,fp,tp = y
    print(zip(ii_str,y[ii]))
    return 1.0*tp/(tp+fp+fn)

def calc_dc(data,str_arg='None'):
    data=data.astype(np.int64)
    try:
        y = np.bincount(data)
    except Exception as e:
        print("Warning: {}".format(e))
        print(data)
        print(str_arg)
        return 0.0
    ii = np.nonzero(y)[0]
    # tn = true negative, means the two raters agree it is not gray matter
    # fn = false negative, means jeeda marked it as not gray matter and deepthi marked as gray matter
    # fp = false positive, means jeeda marked it as gray matter and deepthi marked as not gray matter
    # tp = true positive,  means the two raters agree it is gray matter
    ii_str=['tn','fn','fp','tp']
    tn,fn,fp,tp = y
    print(zip(ii_str,y[ii]))
    return 2.0*tp/(2*tp+fp+fn)

def save_to_nii(data,out_dir,fn):
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=os.path.join(out_dir,fn)
    print(path)
    if not os.path.isfile(path+'.gz'):
        nib.save(img,path)
        check_call(['gzip', path])
    else:
        print('File {} already exists'.format(path))

def get_weight(path):
    list_of_files = glob.glob(os.path.join(path,'weights*FINAL.h5'))
    if list_of_files:
        # print(list_of_files)
        return max(list_of_files, key=os.path.getctime)
    else:
        return 0

def get_model(path,verbose=False):
    print(path)
    list_of_files = glob.glob(os.path.join(path,'model.*FINAL.h5'))
    if list_of_files:
        # print(list_of_files)
        model_fn = max(list_of_files, key=os.path.getctime)
        print('Loading model : {}'.format(model_fn))
        model = load_model(model_fn)
        if verbose:
            print(model.summary())
    else:
        print('We did not find any models.  Try again!')
        model = 0
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


def testNN(model_version,slice_fn,segment_fn,hull_fn,nb_tiles_in,verbose=False):

    input_size,output_size = get_input_output_size(model_version)

    # nb_step is number of tiles per step
    batch_size=32
    tile_width=input_size[0]

    # directory to save files
    save_dir=os.path.dirname('/home/rpizarro/histo/prediction/NN_arch/v{}/'.format(model_version))

    weights_dir = os.path.dirname("/home/rpizarro/histo/weights/NN_arch/v{}/".format(model_version))
    model = get_model(weights_dir, verbose=True)

    slice_nb=os.path.basename(slice_fn)[:4]
    if verbose:
        print("{} : {}".format(slice_fn,segment_fn))
    # tiles_rgb are for viewing, tiles are normalized used for predicting
    # coord identifies the location of the tile,seg
    # slice_shape specifies the shape of the image
    nb_tiles=nb_tiles_in
    tiles_norm,seg,coord,slice_shape=gen_tiles(save_dir,slice_fn,segment_fn,hull_fn,tile_width,slice_nb,nb_tiles)
    nb_tiles=len(coord)

    seg=np.reshape(np_utils.to_categorical(seg,output_size[-1]),(nb_tiles,)+output_size)

    output_size_tiles=output_size[:-1]+(nb_tiles,)+(2,)
    input_size_tiles=output_size[:-1]+(nb_tiles,)+(1,)

    y_true_tiles=np.zeros(output_size_tiles)
    y_prob_tiles=np.zeros(output_size_tiles)
    y_pred_tiles=np.zeros(output_size_tiles)
    jac_idx_val=np.zeros(nb_tiles)

    for n in range(nb_tiles):
        X_test=np.reshape(tiles_norm[n],(1,)+input_size)
        y_true=np.reshape(seg[n],(1,)+output_size)
        y_prob = model.predict(X_test,batch_size=batch_size,verbose=1)
        y_pred = np.around(y_prob)
        print('y_prob : ( {} , {} )'.format(np.amin(y_prob),np.amax(y_prob)))
        print('y_pred : {}'.format(np.unique(y_pred)))
        print('y_true : {}'.format(np.unique(y_true)))
        y_pred=y_pred.astype(dtype=y_true.dtype)

        # getting the jaccard index
        data12 = y_true[0]+2*y_pred[0]
        print(data12.shape)
        # dc_man = calc_dc(data12.flatten())
        jac_idx_man = calc_jac_idx(data12.flatten())
        print('This is jaccard index: {0:0.4f}'.format(jac_idx_man))
        jac_idx_val[n]=float(jac_idx_man)

        y_true_tiles[:,:,n,:]=y_true[0]
        y_prob_tiles[:,:,n,:]=y_prob[0]
        y_pred_tiles[:,:,n,:]=y_pred[0]

    print('{} : {}'.format(slice_nb,nb_tiles))

    Y_pred_slice=retile(np.expand_dims(y_pred_tiles[:,:,:,0],axis=3),coord,slice_shape,tile_width)
    Y_pred_slice=np.around(Y_pred_slice)
    fn='{0}.prediction.000.{1:04d}tiled.nii'.format(slice_nb,nb_tiles)
    save_to_nii(Y_pred_slice,save_dir,fn)

    Y_prob_slice=retile(np.expand_dims(y_prob_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
    Y_prob_slice=100*Y_prob_slice
    fn='{0}.probability.001.{1:04d}tiled.nii'.format(slice_nb,nb_tiles)
    save_to_nii(Y_prob_slice,save_dir,fn)

    Y_pred_slice=retile(np.expand_dims(y_pred_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
    Y_pred_slice=np.around(Y_pred_slice)
    fn='{0}.prediction.001.{1:04d}tiled.nii'.format(slice_nb,nb_tiles)
    save_to_nii(Y_pred_slice,save_dir,fn)

    Y_true_slice=retile(np.expand_dims(y_true_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
    Y_true_slice=np.around(Y_true_slice)
    fn='{0}.truesegment.{1:04d}tiled.nii'.format(slice_nb,nb_tiles)
    save_to_nii(Y_true_slice,save_dir,fn)

    data12_slice = Y_true_slice+2*Y_pred_slice
    jac_idx_slice = calc_jac_idx(data12_slice.flatten(),'Jacard index for true segmentation and predicion')
    print('Jaccard index ( avg tiles | slice ) : ( {0:0.3f} | {1:0.3f} )'.format(np.mean(jac_idx_val),jac_idx_slice))

    bins = np.linspace(0, 1, nb_tiles/2)
    plt.hist(jac_idx_val,bins)
    plt.title("Jaccard Index Distribution")
    plt.xlabel("Jaccard Index")
    plt.ylabel("Frequency")
    fn='{0}.jac_idx_distribution.{1:04d}tiled.jac_idx{2:0.3f}.png'.format(slice_nb,nb_tiles,np.mean(jac_idx_val))
    plt.savefig(os.path.join(save_dir,fn))
    plt.close()

    Y_out_slice=Y_true_slice+3*Y_pred_slice
    fn='{0}.segmented.{1:04d}tiled.jac_idx{2:0.3f}.nii'.format(slice_nb,nb_tiles,np.mean(jac_idx_val))
    save_to_nii(Y_out_slice,save_dir,fn)



def grab_files(path,end):
    return glob.glob(os.path.join(path,end))

def get_hull_fn(hull_fn_paths,slice_fn):
    slice_nb = os.path.basename(slice_fn)[:4]
    hull_fn_bases = [os.path.basename(f) for f in hull_fn_paths]
    hull_fn = difflib.get_close_matches(os.path.basename(slice_fn),hull_fn_bases)[0]
    if slice_nb in hull_fn:
        return os.path.join(os.path.dirname(hull_fn_paths[0]),hull_fn)
    else:
        print('No file matching : {}'.format(slice_fn))
        return False




# input to : python ribbon.train_unet.model.py 107(_drop) 120
model_version = sys.argv[1]
nb_tiles = int(sys.argv[2])

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

csv_dir = '/home/rpizarro/histo/XValidFns/rm311_128slices'
test_fn = os.path.join(csv_dir,'test.csv')
test_df = pd.read_csv(test_fn)

hull_data_path= '/home/rpizarro/histo/data/rm311_128requad_test_hull/'
hull_fns = grab_files(hull_data_path,"*.single_hull.nii.gz")

print('\n==Testing NN UNET ==\n')
for index,row in test_df.iterrows():
    slice_fn = row['slice_fn']
    segment_fn = row['segment_fn']
    hull_fn = get_hull_fn(hull_fns,slice_fn)
    if hull_fn:
        print('{} : {} : {}'.format(slice_fn,segment_fn,hull_fn))
        testNN(model_version,slice_fn,segment_fn,hull_fn,nb_tiles,verbose=True)



