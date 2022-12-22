#!/usr/bin/python3
"""
The file "MCCDS.py" is part of the project "CDSSpreadDNN".
@author: kevin.
created on 25.11.2022: 10:30.
"""

import numpy as np
from numba import jit
import pandas as pd
import tensorflow as tf

from pathlib import Path

import time as timer
from tqdm import tqdm, trange

from typing import List, Tuple, Optional
from numpy.typing import DTypeLike

from multiprocessing import Pool
# from multiprocessing.dummy import Pool
from psutil import cpu_count


@jit(nopython=True)
def mainSerial(Wt,Mt,t,batch,LGD,annuity,r,beta,rho,x0):
    N = Wt.shape[0]
    M = Wt.shape[1]
    payLeg = np.zeros(batch,dtype=Wt.dtype)
    protLeg = np.zeros(batch,dtype=Wt.dtype)
    bt = np.exp(-r * t)
    for iB in range(batch):
        for iM in range(M):
            df = 0.0
            nom = 0.0
            for iT in range(1,N):
                Xt = x0[iB] + beta[iB] * t[iT] + np.sqrt(1 - rho[iB]) * Wt[iT,iM] + np.sqrt(rho[iB]) * Mt[iT,iM]
                # df += bt[iT]
                if Xt <= 0.0 and iT<N-1:
                    nom=bt[iT+1]
                    break
                df += bt[iT]

            denom= df
            protLeg[iB]+= nom
            payLeg[iB]+=denom
    protLeg *= (LGD/M)
    payLeg *= (annuity/M)
    return protLeg/payLeg,protLeg,payLeg


class MCCDSSpread:
    def __init__(self,
                 M: int,
                 T: float,
                 annuity: float,
                 LGD: Optional[float] = 0.6,
                 r: float = 0.1,
                 sRange: Optional[Tuple[int]] = (.01, .99),
                 rhoRange: Optional[Tuple[int]] = (.01, .99),
                 x0Range: Optional[Tuple[int]] = (0, 6),
                 dtype: Optional[DTypeLike] = np.float32):

        self.M = M  # number of simulations for Brownian motions
        self.N = int(T / annuity) + 1  # number of time steps
        self.T = T
        self.LGD = LGD
        self.annuity = annuity
        self.dtype = dtype

        beta1 = (r - (sRange[0] ** 2) / 2) / sRange[0]
        beta2 = (r - (sRange[1] ** 2) / 2) / sRange[1]
        betaRange = (min(beta1, beta2), max(beta1, beta2))
        'Parameter ranges'
        self.r = r
        self.sRange = sRange # parameter range for sigma \in [sRange[0],sRange[1]]
        self.betaRange = betaRange # parameter range for beta
        self.rhoRange = rhoRange  # parameter range for rho \in [rhoRange[0],rhoRange[1]]
        self.x0Range = x0Range  # parameter range for x0 \in [x0Range[0],x0Range[1]]

        'Brownian motions for Monte-Carlo price'
        self.Wt = []
        self.Mt = []
        # self.simulateBM(M)
        self.t = np.reshape(np.linspace(0., T, self.N, endpoint=True, dtype=self.dtype), (1, -1, 1))

    def simulateBM(self, M):
        dt = self.T / (self.N - 1)
        Bt = np.concatenate([np.zeros((1, 1, M, 2), dtype=self.dtype),
                             np.sqrt(dt) * np.cumsum(np.random.randn(
                                 1, self.N - 1, M, 2).astype(self.dtype),
                                                     axis=1)],
                            1)
        self.Wt = Bt[:, :, :, 0]
        self.Mt = Bt[:, :, :, 1]

    def CDSspread(self, r, beta, rho, x0):
        if len(self.Wt) == 0:
            self.simulateBM(self.M)
        batch = np.size(rho)
        beta = np.array(beta, ndmin=1).reshape(-1)
        rho = np.array(rho, ndmin=1).reshape(-1)
        x0 = np.array(x0, ndmin=1).reshape(-1)
        return mainSerial(np.squeeze(self.Wt), np.squeeze(self.Mt), np.squeeze(self.t), batch, self.LGD, self.annuity, r,beta,rho,x0)

    def generateFileName(self):
        return f'r{self.r}_sigma{self.sRange}_rho{self.rhoRange}_x0{self.x0Range}_BM{self.M}_T{self.T}_N{self.N}'

    def _batchParameters(self, batch: int):
        beta = np.random.uniform(low=self.betaRange[0], high=self.betaRange[1], size=(batch, 1, 1)).astype(self.dtype)
        rho = np.random.uniform(low=self.rhoRange[0], high=self.rhoRange[1], size=(batch, 1, 1)).astype(self.dtype)
        x0 = np.random.uniform(low=self.x0Range[0], high=self.x0Range[1], size=(batch, 1, 1)).astype(self.dtype)
        cdsSpreads, protLeg, payLeg = mainSerial(np.squeeze(self.Wt), np.squeeze(self.Mt), np.squeeze(self.t), batch, self.LGD, self.annuity, self.r, np.squeeze(beta), np.squeeze(rho), np.squeeze(x0))

        return beta, rho, x0, cdsSpreads, protLeg, payLeg

    def sampleParameters(self, samples: int, batch: int, saveDir: Optional[str] = '../Data/TrainingData'):
        fileName = self.generateFileName()
        fileDir = str((Path().resolve() / Path(saveDir + '\\' + self.generateFileName())).resolve())
        file = fileDir + '\\' + fileName + '.csv'
        if Path(file).is_file():
            df = pd.read_csv(file,sep=';',index_col=0)
        else:
            self.simulateBM(self.M)

            betaList = []
            rhoList = []
            x0List = []
            cdsSpreadsList = []
            protLegList = []
            payLegList = []

            pool = Pool(cpu_count(logical=False))

            with pool as p:
                out = p.map(self._batchParameters, (batch * np.ones(samples)).astype(np.int64))

            # out=[]
            # for _ in range(samples):
            #     out.append(self._batchParameters(batch))

            for res in out:
                betaList.append(res[0])
                rhoList.append(res[1])
                x0List.append(res[2])
                cdsSpreadsList.append(res[3])
                protLegList.append(res[4])
                payLegList.append(res[5])

            beta = np.concatenate(betaList, axis=0)
            rho = np.concatenate(rhoList, axis=0)
            x0 = np.concatenate(x0List, axis=0)
            cdsSpreads = np.concatenate(cdsSpreadsList, axis=0)
            protLeg = np.concatenate(protLegList, axis=0)
            payLeg = np.concatenate(payLegList, axis=0)

            npDF = np.concatenate([beta.reshape(-1, 1),
                                   rho.reshape(-1, 1),
                                   x0.reshape(-1, 1),
                                   cdsSpreads.reshape(-1, 1),
                                   protLeg.reshape(-1, 1),
                                   payLeg.reshape(-1, 1)], axis=1)

            df = pd.DataFrame(npDF, index=None, columns=['beta', 'rho', 'x0', 'spread', 'prot', 'pay'],
                              dtype=self.dtype)
            Path(fileDir).mkdir(parents=True, exist_ok=True)

            df.to_csv(file, mode='w',sep=';')
        return df


if __name__ == "__main__":
    # Note pycharm is broken, multiprocessing not working in interactive python console
    import time as timer

    np.random.seed(1234)

    # multicore = 6, time = 114s for M=10^5, total=2^16, roughly 8GB RAM total
    total = 2 ** 17  # = 65536
    batch = 2 ** 8  # = 256
    samples = int(total / batch)  # = 256

    annuity = .25
    T = 5.
    M = 10 ** 5
    LGD = .6

    # cdsDate = '26_09_22'
    # r=.015 # interest rate for 26_09_22
    cdsDate = '05_12_22'
    r = .026  # interest rate for 05_12_22
    sRange = (.01, .5)
    rhoRange = (.01, .99)
    x0Range = (0, 6)
    dtype = np.float32

    'Make Dataset'
    mcCDS = MCCDSSpread(M, T, annuity, LGD, r=r, sRange=sRange, rhoRange=rhoRange, x0Range=x0Range,
                        dtype=dtype)

    saveDir = '../Data/TrainingData'
    tic = timer.time()
    data = mcCDS.sampleParameters(samples, batch, saveDir=saveDir)
    ctimeMC = timer.time() - tic
    print(f'Elapsed time for sampling CDS spreads {ctimeMC} s.\n')

    # r=.1
    # beta=np.array([.5],ndmin=1,dtype=np.float32)
    # rho=np.array([.5],ndmin=1,dtype=np.float32)
    # x0=np.array([3],ndmin=1,dtype=np.float32)
    # c1,_,_ = mcCDS.CDSspread(r, sigma, rho, x0)
    # print(f"c1={c1}")


    # dataNP = data.to_numpy(dtype=dtype)  # 'beta','rho','x0','spread','prot','pay'
