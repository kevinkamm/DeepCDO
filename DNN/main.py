#!/usr/bin/python3
"""
The file "main.py" is part of the project "CDSSpreadDNN".
@author: kevin.
created on 25.11.2022: 10:30.
"""
import time as timer
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MCCDS import MCCDSSpread
from CDSnet_SubClassed import DeepCDSSpread
from IniNet_SubClassed import DeepInitialDatum

"Tensorflow settings"
# tf.config.set_visible_devices([], 'GPU')  # uncomment, if CPU only is desired
# print(tf.config.experimental.get_synchronous_execution())
# print(tf.config.experimental.list_physical_devices())
# print(tf.config.threading.get_inter_op_parallelism_threads())
# print(tf.config.threading.get_intra_op_parallelism_threads())

'Seeds for reproducibility'
np.random.seed(1234)
tf.random.set_seed(1234)

'Convenience parameters'
forceTrainingCDS = True
forceTrainingIni = True

'CDS parameters'
annuity = .25
T = 5.
LGD = .6

'Parameters for Monte-Carlo CDS spread dataset'
total = 2 ** 17  # = 65536*2
batch = 2 ** 8  # = 256, batch size per process
samples = int(total / batch)  # = 256
M = 10 ** 5  # Monte-Carlo paths
saveDirMC = '../Data/TrainingData'  # relative path only

cdsDate = '26_09_22'
r=.015 # interest rate for 26_09_22
# cdsDate = '05_12_22'
# r=.026 # interest rate for 05_12_22
sRange = (.01, .5)  # sigma \in [sRange[0],sRange[1]]
rhoRange = (.01, .99)  # rho \in [rhoRange[0],rhoRange[1]]
x0Range = (0, 6)  # x0 \in [x0Range[0],x0Range[1]]
dtypeNP = np.float32

'Load or Create Dataset'  # creating a new dataset can take a while
print('Generating or loading dataset')
mcCDS = MCCDSSpread(M, T, annuity, LGD, r=r, sRange=sRange, rhoRange=rhoRange, x0Range=x0Range,
                    dtype=dtypeNP)
dsStr= mcCDS.generateFileName()

tic = timer.time()
data = mcCDS.sampleParameters(samples, batch, saveDir=saveDirMC)
ctimeMC = timer.time() - tic
print(f'Elapsed time for sampling CDS spreads {ctimeMC} s.\n')

dataNP = data.to_numpy(dtype=dtypeNP)  # each row contains: 'r','sigma','rho','x0','spread','prot','pay'

'CDS Interpolation Network: Training Parameters'
batch_size = 128  # 2^i for max performance on CUDA device
epochs = 30
dtypeTF = tf.float32
saveDirDNN = '../Model/CDSNet'  # relative path only
saveDirFig = '../Figures/CDSNet'  # relative path only

fileDir = str((Path().resolve() / Path(saveDirDNN+ '\\' + dsStr)).resolve())
file = fileDir + '\\' + 'CDSNet'
Path(fileDir).mkdir(parents=True, exist_ok=True)
figDir = str((Path().resolve() / Path(saveDirFig+ '\\' + dsStr)).resolve())
Path(figDir).mkdir(parents=True, exist_ok=True)
# log_dir = "../logs/CDSNet/"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False)
if not Path(file).is_dir() or forceTrainingCDS:
    print('Train CDS model')
    dataset = tf.data.Dataset.from_tensor_slices(dataNP).batch(batch_size).shuffle(total)
    # dataset = tf.data.Dataset.from_tensor_slices(dataNP).batch(batch_size)

    tic = timer.time()
    DCS = DeepCDSSpread()
    ctime = timer.time() - tic
    print(f'Elapsed time for initializing {ctime} s')

    DCS.compile(optimizer=Adam(1e-4), loss='mae', metrics=['mae'], jit_compile=True)

    tic = timer.time()
    # histDCS=DCS.fit(dataset, epochs=epochs, batch_size=batch_size,
    #                 callbacks=[tensorboard_callback])
    histDCS = DCS.fit(dataset, epochs=epochs, batch_size=batch_size)
    ctime = timer.time() - tic
    print(f'Elapsed time for training {ctime} s')

    print('Save model')
    DCS.save(file)
    DCS.exportToMatlab(fileName=dsStr)

    print('Plot history')
    pd.DataFrame(histDCS.history).plot(figsize=(8, 5))
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.yscale('log')
    plt.show()
    plt.savefig(figDir + '\\' + 'Losses.pdf',format='pdf')
else:
    print('Load CDS model')
    DCS = tf.keras.models.load_model(file)

DCS.trainable = False
DCS.compile(jit_compile=True)  # essential, otherwise the other model will see the variables as trainable

"Example outputs for validation"
sigma = .18
rho = .22
beta = (r - sigma ** 2 / 2) / sigma

print('Model validation: ')
for x0 in [1., 2., 3., 4., 5., 6.]:
    print(f'Parameters: r={r}, sigma={sigma}, rho={rho}, x0={x0}:')
    inputs = tf.reshape(tf.constant([beta, rho, x0], dtype=tf.float32), (1, -1))
    tic = timer.time()
    cDNN = DCS(inputs)  # DCS.predict() is much slower for single inputs
    ctimeDNN = timer.time() - tic

    tic = timer.time()
    cMC, nomMC, denomMC = mcCDS.CDSspread(r, beta, rho, x0)  # first call will sample the Brownian motions, so slower
    ctimeMC = timer.time() - tic
    print(f'Elapsed time: MC={ctimeMC} s, DNN={ctimeDNN} s')
    print(f'cMC[0]=\t\t{cMC[0]},\t mean={np.mean(cMC)}')
    print(f'cDNN[0]=\t{cDNN[0]},\t mean={np.mean(cDNN)}')

'Initial datum DNN'
df = pd.read_csv(r'..\Data\iTRAXX\iTRAXX_CDS_'+cdsDate+'.csv', sep=';')
cMarket = df.iloc[:, 2].to_numpy() / 10000  # bps to decimal
cMarket[::-1].sort()
K = cMarket.shape[0]
batch_size = 128
epochs = 40
samples = 100
saveDirDNN = '../Model/IniNet'  # relative path only
saveDirFig = '../Figures/IniNet'  # relative path only

fileDir = str((Path().resolve() / Path(saveDirDNN+ '\\' + dsStr+'_'+cdsDate)).resolve())
file = fileDir + '\\' + 'IniNet'
Path(fileDir).mkdir(parents=True, exist_ok=True)
figDir = str((Path().resolve() / Path(saveDirFig+ '\\' + dsStr+'_'+cdsDate)).resolve())
Path(figDir).mkdir(parents=True, exist_ok=True)
# log_dir = "../logs/IniNet/"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False)
if not Path(file).is_dir() or forceTrainingIni:
    # print('Train Ini model') # build dataset beforehand
    #
    # 'Dataset'
    # beta1 = (r - (sRange[0] ** 2) / 2) / sRange[0]
    # beta2 = (r - (sRange[1] ** 2) / 2) / sRange[1]
    # betaRange = (min(beta1,beta2),max(beta1,beta2))
    # betaSet = np.random.uniform(low=betaRange[0], high=betaRange[1], size=(batch_size*samples,1)).astype(dtypeNP)
    # rhoSet = np.random.uniform(low=rhoRange[0], high=rhoRange[1], size=(batch_size*samples,1)).astype(dtypeNP)
    # params = np.concatenate([betaSet,rhoSet],axis=1)
    #
    # iniDataset = tf.data.Dataset.from_tensor_slices(params).batch(batch_size)
    #
    # tic = timer.time()
    # DIni = DeepInitialDatum(K, DCS, cMarket)
    # ctime = timer.time() - tic
    # print(f'Elapsed time for initializing of Ini net {ctime} s')
    #
    # DIni.compile(optimizer=Adam(1e-4), loss='mape', metrics=['mape'], jit_compile=False, run_eagerly=False)
    #
    # # with tf.device('/gpu:0'):
    # tic = timer.time()
    # # histIni=DIni.fit(iniDataset, epochs=epochs,
    # #                 callbacks=[tensorboard_callback])
    # histIni = DIni.fit(iniDataset, epochs=epochs)
    # ctime = timer.time() - tic
    # print(f'Elapsed time for training Ini net {ctime} s')

    print('Train Ini model') # dataset from generator

    'Dataset'
    beta1 = (r - (sRange[0] ** 2) / 2) / sRange[0]
    beta2 = (r - (sRange[1] ** 2) / 2) / sRange[1]
    betaRange = (min(beta1,beta2),max(beta1,beta2))
    def gen():
        while True:
            beta = tf.random.uniform(shape=(batch_size,1),minval=betaRange[0],maxval=betaRange[1],dtype=dtypeTF)
            rho = tf.random.uniform(shape=(batch_size,1),minval=rhoRange[0],maxval=rhoRange[1],dtype=dtypeTF)
            yield tf.concat([beta,rho],axis=1)

    iniDataset = tf.data.Dataset.from_generator(gen,
                                                output_signature=(tf.TensorSpec(shape=(batch_size,2)))
                                                )


    tic = timer.time()
    DIni = DeepInitialDatum(K, DCS, cMarket)
    ctime = timer.time() - tic
    print(f'Elapsed time for initializing of Ini net {ctime} s')

    DIni.compile(optimizer=Adam(1e-4), loss='mape', metrics=['mape'], jit_compile=False, run_eagerly=False)

    # with tf.device('/gpu:0'):
    tic = timer.time()
    # histIni = DIni.fit(iniDataset,epochs=epochs,steps_per_epoch=samples,
    #                     callbacks=[tensorboard_callback])
    histIni = DIni.fit(iniDataset, epochs=epochs, steps_per_epoch=samples)
    ctime = timer.time() - tic
    print(f'Elapsed time for training Ini net {ctime} s')


    print('Save model')
    DIni.save(file)
    DIni.exportToMatlab(fileName=dsStr+'_'+cdsDate)

    print('Plot history')
    pd.DataFrame(histIni.history).plot(figsize=(8, 5))
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.yscale('log')
    plt.show()
    plt.savefig(figDir + '\\' + 'Losses.pdf',format='pdf')
else:
    print('Load Ini model')
    DIni = tf.keras.models.load_model(file)

DIni.trainable = False
DIni.compile(jit_compile=False, run_eagerly=False)

print('Model validation: ')
print(f'Parameters: r={r}, sigma={sigma}, rho={rho}')
inputs = tf.reshape(tf.constant([beta, rho], dtype=tf.float32), (1, -1))
tic = timer.time()
x0DNN = DIni(inputs)
ctimeDNN = timer.time() - tic

inputs = tf.concat(
    [beta * tf.ones(shape=(K, 1)), rho * tf.ones(shape=(K, 1)), tf.reshape(x0DNN, (-1, 1))],
    axis=1)
cDNN = DCS(inputs).numpy()

tic = timer.time()
cMC, nomMC, denomMC = mcCDS.CDSspread(r,beta * np.ones((K, 1, 1), dtype=dtypeNP),
                                      rho * np.ones((K, 1, 1), dtype=dtypeNP),
                                      x0DNN.numpy().reshape(-1, 1, 1))  # first call will sample the Brownian motions
ctimeMC = timer.time() - tic
print(f'Elapsed time: MC={ctimeMC} s, Ini+DNN={ctimeDNN} s')

with np.printoptions(threshold=np.inf):
    print('x0 DNN:')
    print(x0DNN)
    print('c DNN:')
    print(cDNN)
    print('c Market:')
    print(cMarket)
    print('c MC:')
    print(cMC.ravel())
    print(f'Mean absolute error cDNN cMarket {np.mean(np.abs(cDNN - cMarket))}:')
    print(f'Mean relative error cDNN cMarket {np.mean(np.abs((cDNN - cMarket) / cMarket))}:')
    print(f'Mean absolute error cMC cMarket {np.mean(np.abs(cMC.ravel() - cMarket))}:')
    print(f'Mean relative error cMC cMarket {np.mean(np.abs((cMC.ravel() - cMarket) / cMarket))}:')
