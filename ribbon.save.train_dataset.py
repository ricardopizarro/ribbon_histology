import numpy as np
import pandas as pd
import glob
import random
import os
import sys
import difflib

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))

def split_cross_valid(slices_fn,segments_fn,train,valid,test):
    train_files =[] 
    validation_files=[]
    test_files=[]
    for n,segment_fn in enumerate(segments_fn):
        slice_fn=difflib.get_close_matches(segment_fn,slices_fn)[0]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        slice_nb = os.path.basename(segment_fn)[:4]
        if slice_nb in valid:
            validation_files.append((slice_fn,segment_fn))
        elif slice_nb in test:
            test_files.append((slice_fn,segment_fn))
        elif slice_nb in train:
            train_files.append((slice_fn,segment_fn))
        else:
            print('{} is not in any subset!'.format(segment_fn))
    return train_files,validation_files,test_files


def set_to_df(paths,df=[]):
    cols=['slice_fn','segment_fn']
    df = pd.DataFrame(columns=cols)
    for slice_fn,segment_fn in paths:
        # print(slice_fn,segment_fn)
        if df['slice_fn'].isin([slice_fn]).any():
            print('path entry exists double check duplicate...')
            # idx = l.index(max(l))
            # df.loc[df.path == p, cols[idx]] = df.loc[df.path == p, cols[idx]] + l[idx]
        else:
            row = pd.DataFrame([[slice_fn]+[segment_fn]], columns=cols)
            df = df.append(row, ignore_index=True)
    return df



# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

data_path = "/home/rpizarro/histo/data/rm311_128requad/"
if not os.access(data_path, os.R_OK):
    print('Cannot read any of the files in {}'.format(data_path))
    sys.exit()

slices=glob.glob(data_path+"*")
slices=[os.path.basename(s)[:4] for s in slices]
slices=sorted(list(set(slices)))
# print(slices)

itest=list(np.arange(0,len(slices),9))
ivalid=list(np.arange(1,len(slices),9))
itrain=[x for x in list(np.arange(len(slices))) if x not in itest+ivalid]

test=[s for i,s in enumerate(slices) if i in itest]
valid=[s for i,s in enumerate(slices) if i in ivalid]
train=[s for i,s in enumerate(slices) if i in itrain]

print(test)
print(train)
print(valid)
slices_fn = grab_files(data_path,"*.slice.nii.gz")
segments_fn = grab_files(data_path,"*segmented.nii.gz")

train_files,validation_files,test_files = split_cross_valid(slices_fn,segments_fn,train,valid,test)

save_dir = '/home/rpizarro/histo/XValidFns/rm311_128slices'
print('Saving the list of XValidFns in : {}'.format(save_dir))

test_df = set_to_df(test_files,df=[])
print(test_df.head())
test_fn = os.path.join(save_dir,'test.csv')
test_df.to_csv(test_fn)

train_df = set_to_df(train_files,df=[])
print(train_df.head())
train_fn = os.path.join(save_dir,'train.csv')
train_df.to_csv(train_fn)

valid_df = set_to_df(validation_files,df=[])
print(valid_df.head())
valid_fn = os.path.join(save_dir,'valid.csv')
valid_df.to_csv(valid_fn)

# random.shuffle(train_files)
# random.shuffle(validation_files)



