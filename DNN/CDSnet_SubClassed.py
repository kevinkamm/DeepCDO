#!/usr/bin/python3
"""
The file "main.py" is part of the project "CDSSpreadDNN".
@author: kevin.
created on 25.11.2022: 10:30.
"""
import numpy as np
from numba import jit
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import BatchNormalization, Dense, Reshape, Activation, Conv2D, Flatten, Conv1D, Concatenate
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import time as timer
from tqdm import tqdm, trange

from typing import List, Tuple
from numpy.typing import DTypeLike
from pathlib import Path
from MCCDS import MCCDSSpread


class DeepCDSSpread(Model):
    def __init__(self):
        super().__init__()

        'Make network with 4 distinct inputs'
        # use batch_size = None for variable batch size
        # Beta = Input(shape=[1], batch_size=None, name='sigma')
        # Rho = Input(shape=[1], batch_size=None, name='rho')
        # X0 = Input(shape=[1], batch_size=None, name='x0')

        'Merge inputs'
        # self.C1 = Concatenate(axis=1)
        self.R1 = Reshape((3, 1))

        'Convolutional layers for local parameter features'
        self.C1D1 = Conv1D(16, 4, activation='relu', padding='same', data_format='channels_last')
        self.C1D2 = Conv1D(32, 16, activation='relu', padding='same', data_format='channels_last')
        self.C1D3 = Conv1D(64, 32, activation='relu', padding='same', data_format='channels_last')
        self.C1D4 = Conv1D(128, 64, activation='relu', padding='same', data_format='channels_last')
        self.F1 = Flatten()

        'Dense down-sampling'
        self.D1 = Dense(4 * 64, activation='relu')
        self.D2 = Dense(2 * 64, activation='relu')
        self.D3 = Dense(64, activation='relu')

        'Out>1 and taking the average has proven to be better'
        self.Out = Dense(8)

        # self.cds_model = Model(inputs=[Beta, Rho, X0], outputs=[x])
        # print(self.cds_model.summary())

    # @tf.function
    def _cds_post_net_op(self, c):
        return tf.reduce_mean(tf.abs(c), axis=1)

    # @tf.function
    def _cds_pre_net_op(self, inputs):
        return inputs

    def call(self,inputs,training=False):
        # x = self.C1(self._cds_pre_net_op(inputs))
        # x = self.R1(x)
        x = self.R1(self._cds_pre_net_op(inputs))
        x = self.C1D1(x)
        x = self.C1D2(x)
        x = self.C1D3(x)
        x = self.C1D4(x)
        x = self.F1(x)
        x = self.D1(x)
        x = self.D2(x)
        x = self.D3(x)
        return self._cds_post_net_op(self.Out(x))

    # @tf.function(jit_compile=True)
    def train_step(self, data):

        # This loop uses a penalty to ensure monotonicity in x0
        # with tf.GradientTape(persistent=True) as tape:
        #     tape.watch(x0)
        #     c = self._cds_post_net_op(self.cds_model(self._cds_pre_net_op(r,sigma,rho,x0),training=True))
        #     grads = tape.gradient(c, x0)
        #     loss = lfunc(tf.sqrt(cMC),tf.sqrt(c))+tf.reduce_mean(tf.maximum(0.,grads))

        beta = tf.reshape(data[:, 0],(-1,1))
        rho = tf.reshape(data[:, 1],(-1,1))
        x0 = tf.reshape(data[:, 2],(-1,1))
        cMC = tf.reshape(data[:, 3],(-1,1))
        inputs=tf.concat([beta, rho, x0],axis=1)
        with tf.GradientTape() as tape:
            c = self(inputs, training=True)
            loss = self.compiled_loss(tf.sqrt(cMC), tf.sqrt(c), regularization_losses=self.losses)  # sqrt makes small and big values closer

        var_list = self.trainable_variables
        gradients = tape.gradient(loss, var_list)
        self.optimizer.apply_gradients(zip(gradients, var_list))

        self.compiled_metrics.update_state(cMC, c)

        return {m.name: m.result() for m in self.metrics}

    def exportToMatlab(self,
                       saveDir:str = '../Model4Matlab/CDSNet',
                       fileName:str = 'temp'):
        fileDir = str((Path().resolve() / Path(saveDir + '\\' + fileName)).resolve())
        file = fileDir + '\\' + 'CDSNet'
        Path(fileDir).mkdir(parents=True, exist_ok=True)
        P=Input(shape=[3,1],batch_size=None,name='P')
        x = self.C1D1(P)
        x = self.C1D2(x)
        x = self.C1D3(x)
        x = self.C1D4(x)
        x = self.F1(x)
        x = self.D1(x)
        x = self.D2(x)
        x = self.D3(x)
        out=self.Out(x)
        matlabModel = Model(inputs=[P],outputs=[out])
        matlabModel.save(file)

if __name__ == "__main__":
    import time as timer

    # tf.config.set_visible_devices([], 'GPU')
    # print(tf.config.experimental.get_synchronous_execution())
    # print(tf.config.experimental.list_physical_devices())
    # print(tf.config.threading.get_inter_op_parallelism_threads())
    # print(tf.config.threading.get_intra_op_parallelism_threads())
    np.random.seed(1234)
    tf.random.set_seed(1234)

    'Parameters for Dataset'
    total = 2 ** 16  # = 65536
    batch = 2 ** 8  # = 256
    samples = int(total / batch)  # = 256

    annuity = .25
    T = 5.
    M = 10 ** 5
    LGD = .6

    r = .1  # fixed interest rate
    sRange = (.08, .4)  # sigma \in [sRange[0],sRange[1]]
    rhoRange = (.01, .99)  # rho \in [rhoRange[0],rhoRange[1]]
    x0Range = (0, 6)  # x0 \in [x0Range[0],x0Range[1]]
    dtypeNP = np.float32

    'Load or Create Dataset'  # creating a new dataset can take a while
    print('Generating or loading dataset')
    mcCDS = MCCDSSpread(M, T, annuity, LGD, r=r, sRange=sRange, rhoRange=rhoRange, x0Range=x0Range,
                        dtype=dtypeNP)
    dsStr = mcCDS.generateFileName()

    saveDir = '../Data/TrainingData'  # relative path only
    tic = timer.time()
    data = mcCDS.sampleParameters(samples, batch, saveDir=saveDir)
    ctimeMC = timer.time() - tic
    print(f'Elapsed time for sampling CDS spreads {ctimeMC} s.\n')

    dataNP = data.to_numpy(dtype=dtypeNP)  # each row contains: 'beta','rho','x0','spread','prot','pay'

    'DNN Training Parameters'
    batch_size = 128  # 2^i for max performance on CUDA device
    epochs = 30
    dtypeTF = tf.float32

    # dataset = tf.data.Dataset.from_tensor_slices(dataNP).batch(batch_size).shuffle(total)
    dataset = tf.data.Dataset.from_tensor_slices(dataNP).batch(batch_size)

    tic = timer.time()
    DCS = DeepCDSSpread()
    ctime = timer.time() - tic
    print(f'Elapsed time for initializing {ctime} s')

    DCS.compile(optimizer='adam',loss='mae',metrics=['mae'],jit_compile=True)

    tic = timer.time()
    DCS.fit(dataset,epochs=epochs,batch_size=batch_size)
    ctime = timer.time() - tic
    print(f'Elapsed time for training {ctime} s')

    r = .1
    sigma = .18
    rho = .22
    beta = (r-sigma**2/2)/sigma

    for x0 in [1., 2., 3., 4., 5., 6.]:
        print(f'Parameters: r={r}, sigma={sigma}, rho={rho}, x0={x0}:')
        inputs=tf.reshape(tf.constant([beta, rho, x0],dtype=tf.float32),(1,-1))
        tic = timer.time()
        cDNN = DCS(inputs)  # DCS.predict() is much slower for single inputs
        ctimeDNN = timer.time() - tic

        tic = timer.time()
        cMC, nomMC, denomMC = mcCDS.CDSspread(r, beta, rho,
                                              x0)  # first call will sample the Brownian motions, so slower
        ctimeMC = timer.time() - tic
        print(f'Elapsed time: MC={ctimeMC} s, DNN={ctimeDNN} s')
        print(f'cMC[0]=\t\t{cMC[0]},\t mean={np.mean(cMC)}')
        print(f'cDNN[0]=\t{cDNN[0]},\t mean={np.mean(cDNN)}')

    # df = pd.read_csv(r'..\Data\iTRAXX\iTRAXX_CDS_26_09_22.csv', sep=';')
    # cMarket = df.iloc[:, 2].to_numpy()/10000
    # cMarket[::-1].sort()
    # cMarket=cMarket[0]
