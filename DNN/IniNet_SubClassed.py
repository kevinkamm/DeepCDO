#!/usr/bin/python3
"""
The file "main.py" is part of the project "CDSSpreadDNN".
@author: kevin.
created on 25.11.2022: 10:30.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import BatchNormalization, Dense, Reshape, Activation, Conv2D, Flatten, Conv1D, Concatenate
from tensorflow.keras.optimizers import Adam

from pathlib import Path

from MCCDS import MCCDSSpread
from CDSnet_SubClassed import DeepCDSSpread

class DeepInitialDatum(Model):
    def __init__(self,
                 K: int,
                 cds_model: Model,
                 cMarket):
        super().__init__()

        self.cds_model = cds_model  # trained CDS DNN
        self.K = K
        self.cMarket = tf.reshape(tf.constant(cMarket),(1,-1))

        # 'Dense net has some problems, sometimes some entries of x0 are huge, e.g., x0=[1,1.1,...,1.9,20,2.0,2.1,...]'
        # self.D1 = Dense(2 * 64, activation='relu')
        # self.D2 = Dense(2 * K, activation='relu')
        # self.D3 = Dense(K**2, activation='relu')
        # self.D4 = Dense(2 * K, activation='relu')
        # self.Out = Dense(K)


        'Convolutional layers for local parameter features'
        self.R1 = Reshape((2, 1))
        self.C1D1 = Conv1D(16, 2, activation='relu', padding='same', data_format='channels_last')
        self.C1D2 = Conv1D(32, 16, activation='relu', padding='same', data_format='channels_last')
        self.C1D3 = Conv1D(64, 32, activation='relu', padding='same', data_format='channels_last')
        self.C1D4 = Conv1D(128, 64, activation='relu', padding='same', data_format='channels_last')
        self.F1 = Flatten()
        self.D1 = Dense(4 * K, activation='relu')
        self.D2 = Dense(2 * K, activation='relu')
        self.Out = Dense(K)

        # print(self.summary())
    def _ini_post_net_op(self, x0):
        return tf.abs(x0)

    def _ini_pre_net_op(self, inputs):
        return tf.reshape(inputs,(-1,2,1))

    # def call(self,inputs,training=False):
    #     x=self.D1(self._ini_pre_net_op(inputs))
    #     x=self.D2(x)
    #     x = self.D3(x)
    #     x = self.D4(x)
    #     return self._ini_post_net_op(self.Out(x))
    def call(self,inputs,training=False):
        x = self.R1(self._ini_pre_net_op(inputs))
        x = self.C1D1(x)
        x = self.C1D2(x)
        x = self.C1D3(x)
        x = self.C1D4(x)
        x = self.F1(x)
        x = self.D1(x)
        x = self.D2(x)
        return self._ini_post_net_op(self.Out(x))

    def train_step(self, data):
        beta = data[:, 0]
        rho = data[:, 1]
        betaVec = tf.reshape(tf.reshape(beta,(-1,1))* tf.ones((1,self.K)),(-1,1))
        rhoVec = tf.reshape(tf.reshape(rho,(-1,1))* tf.ones((1,self.K)),(-1,1))

        with tf.GradientTape() as tape:
            x0 = self(data, training=True)
            x0Vec = tf.reshape(x0,(-1,1))
            cdsInputs = tf.concat([betaVec, rhoVec,x0Vec],axis=1)
            cDNN = self.cds_model(cdsInputs , training=False)
            cDNN = tf.reshape(cDNN,(-1,self.K))
            loss = self.compiled_loss(tf.sqrt(self.cMarket),tf.sqrt(cDNN), regularization_losses=self.losses)
            # loss = self.compiled_loss(self.cMarket, cDNN)

        var_list = self.trainable_variables
        gradients = tape.gradient(loss, var_list)
        self.optimizer.apply_gradients(zip(gradients, var_list))
        self.compiled_metrics.update_state(self.cMarket, cDNN)
        return {m.name: m.result() for m in self.metrics}
        # return {'loss':loss/self.K,'x0':tf.reduce_mean(x0)}

    # def mergeBatchMetrics(self):
    #     # numMetrics = len(self.batchMetrics[0])
    #     keys = list(self.batchMetrics[0].keys())
    #     mergedDict = {x:[] for x in keys}
    #     for key in keys:
    #         l=[d[key] for d in self.batchMetrics]
    #         mergedDict[key]=l
    #     return mergedDict

    def exportToMatlab(self,
                       saveDir:str = '../Model4Matlab/IniNet',
                       fileName:str = 'temp'):
        fileDir = str((Path().resolve() / Path(saveDir + '\\' + fileName)).resolve())
        file = fileDir + '\\' + 'IniNet'
        Path(fileDir).mkdir(parents=True, exist_ok=True)
        P=Input(shape=[2,1],batch_size=None,name='P')
        x = self.C1D1(P)
        x = self.C1D2(x)
        x = self.C1D3(x)
        x = self.C1D4(x)
        x = self.F1(x)
        x = self.D1(x)
        x = self.D2(x)
        out = self.Out(x)
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
    sRange = (.01, .99)  # sigma \in [sRange[0],sRange[1]]
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

    dataNP = data.to_numpy(dtype=dtypeNP)  # each row contains: 'r','sigma','rho','x0','spread','prot','pay'

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

    DCS.compile(optimizer=Adam(1e-4),loss='mae',metrics=['mae'],jit_compile=True)

    tic = timer.time()
    DCS.fit(dataset,epochs=epochs,batch_size=batch_size)
    ctime = timer.time() - tic
    print(f'Elapsed time for training {ctime} s')

    DCS.trainable = False
    DCS.compile(jit_compile=True)  # essential, otherwise the other model will see the variables as trainable

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

    'Initial datum DNN'
    df = pd.read_csv(r'..\Data\iTRAXX\iTRAXX_CDS_26_09_22.csv', sep=';')
    cMarket = df.iloc[:, 2].to_numpy()/10000  # bps to decimal
    cMarket[::-1].sort()
    K = cMarket.shape[0]
    batch_size = 32
    epochs = 1
    samples = 4000

    # def gen():
    #     while True:
    #         r = tf.random.uniform(shape=(batch_size,1),minval=rRange[0],maxval=rRange[1],dtype=dtypeTF)
    #         sigma = tf.random.uniform(shape=(batch_size,1),minval=sRange[0],maxval=sRange[1],dtype=dtypeTF)
    #         rho = tf.random.uniform(shape=(batch_size,1),minval=rhoRange[0],maxval=rhoRange[1],dtype=dtypeTF)
    #         yield tf.concat([r,sigma,rho],axis=1)
    #
    # iniDataset = tf.data.Dataset.from_generator(gen,
    #                                             output_signature=(tf.TensorSpec(shape=(batch_size,3)))
    #                                             )

    beta1 = (r - (sRange[0] ** 2) / 2) / sRange[0]
    beta2 = (r - (sRange[1] ** 2) / 2) / sRange[1]
    betaRange = (min(beta1, beta2), max(beta1, beta2))
    betaSet = np.random.uniform(low=betaRange[0], high=betaRange[1], size=(batch_size*samples,1)).astype(dtypeNP)
    rhoSet = np.random.uniform(low=rhoRange[0], high=rhoRange[1], size=(batch_size*samples,1)).astype(dtypeNP)
    params = np.concatenate([betaSet,rhoSet],axis=1)

    iniDataset = tf.data.Dataset.from_tensor_slices(params).batch(batch_size)

    tic = timer.time()
    DIni = DeepInitialDatum(K,DCS,cMarket)
    ctime = timer.time() - tic
    print(f'Elapsed time for initializing of Ini net {ctime} s')

    DIni.compile(optimizer=Adam(1e-4), loss='mape', metrics=['mape'], jit_compile=False,run_eagerly=False)

    # with tf.device('/gpu:0'):
    tic = timer.time()
    # DIni.fit(iniDataset,epochs=epochs,steps_per_epoch=samples)
    DIni.fit(iniDataset, epochs=epochs)
    ctime = timer.time() - tic
    print(f'Elapsed time for training Ini net {ctime} s')

    print(f'Parameters: r={r}, sigma={sigma}, rho={rho}')
    inputs = tf.reshape(tf.constant([beta, rho], dtype=tf.float32), (1, -1))
    tic = timer.time()
    x0DNN = DIni(inputs)
    ctimeDNN = timer.time() - tic

    inputs = tf.concat([beta*tf.ones(shape=(K,1)), rho*tf.ones(shape=(K,1)),tf.reshape(x0DNN,(-1,1))], axis=1)
    cDNN=DCS(inputs).numpy()

    tic = timer.time()
    cMC, nomMC, denomMC = mcCDS.CDSspread(beta*np.ones((K,1,1),dtype=dtypeNP), rho*np.ones((K,1,1),dtype=dtypeNP), x0DNN.numpy().reshape(-1,1,1))  # first call will sample the Brownian motions
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
        print(f'Mean absolute error cDNN cMarket {np.mean(np.abs(cDNN-cMarket))}:')
        print(f'Mean relative error cDNN cMarket {np.mean(np.abs((cDNN - cMarket)/cMarket))}:')
        print(f'Mean absolute error cMC cMarket {np.mean(np.abs(cMC.ravel() - cMarket))}:')
        print(f'Mean relative error cMC cMarket {np.mean(np.abs((cMC.ravel() - cMarket) / cMarket))}:')
