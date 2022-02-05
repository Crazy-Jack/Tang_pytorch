# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:20:47 2020

@author: pku
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
import h5py

# %%


Pics = h5py.File('./Pics.mat')
imgs = np.array(Pics['Pics'])
tnum = imgs.shape[0]
print(tnum)
train_x = np.reshape(np.transpose(imgs, (0, 2, 1)), (tnum, 1, 50, 50))
np.save('train_x.npy', train_x)

valPics = h5py.File('./valPics.mat')
valimgs = np.array(valPics['valPics'])
vnum = valimgs.shape[0]
print(vnum)
val_x = np.reshape(np.transpose(valimgs, (0, 2, 1)), (vnum, 1, 50, 50))
np.save('val_x.npy', val_x)

# %%
Rsp = h5py.File('./Rsp.mat')
data = np.array(Rsp['Rsp'])
np.save('Rsp.npy', data)

testRsp = h5py.File('./valRsp.mat')
test = np.array(testRsp['valRsp'])
np.save('valRsp.npy', test)
