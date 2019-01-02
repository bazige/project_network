import numpy as np
import pandas as pd
import caffe
import os,sys,cv2
import math
import pdb
import shutill
import caffe.proto.caffe_pb2 as cp
import argparse

def BALG_Quantize(pInput, n, q):
    scale = 1 << q
    pOutput = np.zeros(shape=(n,1), dtype=np.float32)
    for i in range(n):
        wf = pInput[i]*scale
        if wf>0:
            wf += 0.5
        if wf<0:
            wf -= 0.5
      
        if wf >= 127:
            w8 = 127
        elif wf <= -128:
            w8 = -128
        else:
            w8 = int(wf)
      
        wf = float(w8) / float(scale)
        pOutput[i] = wf
    return pOutput

def dis_cosine_distance(a, b, n):
    AA = 0
    BB = 0
    AB = 0
    D = 0
    for i in range(n):
        AA += a[i] * a[i]
        BB += b[i] * b[i]
        AB += a[i] * b[i]
        
    if (AA == 0 or BB == 0):
        D = 0
    else:
        D = (float(AB * AB) / (AA * BB))
        
    return D
    
def find_best_quant_q(bottom_data_reshape, n):
    max0 = -1000
    dis = 0.0
    best_q = 0
    
    for q in range(16):
        pQuant_data = BALG_Quantize(bottom_data_reshape, n, q)
        dis = dis_cosine_distance(bottom_data_reshape, pQuant_data, n)
        
        if dis > max0:
            max0 = dis
            best_q = q
            
    return best_q
    
    
