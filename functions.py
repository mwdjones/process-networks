import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

#Generate probability distribution
def pdf(dat, bins = 15):
    #dat - data for which the pdf is to be derived
    #bins - number of histogram bins, bins are generally set to 15 given the results of
    #  optimization methods described in Ruddell & Kumar 2009
    hist, _ = np.histogram(dat,  bins = bins)
    n = len(dat)
    return hist/n

#Joint probability distribution
def jpdf(dats, bins = 15):
    #dats - columns of data for which the pdf is to be derived
    #bins - number of histogram bins, bins are generally set to 15 given the results of
    #  optimization methods described in Ruddell & Kumar 2009
    hist, _ = np.histogramdd(np.array(dats), bins = bins)
    return hist/hist.sum()

#Calculate Entropy
def entropy(dat):
    #dat - data for which the entropy is to be calculated
    probs = pdf(dat)
    h = -np.nansum([p*np.log2(p) for p in probs])
    return h

#Calculate Joint Entropy
def jentropy(dats):
    #dat - data for which the entropy is to be calculated
    probs = jpdf(dats)
    h = -np.nansum(np.nansum([p*np.log2(p) for p in probs]))
    return h

#Calculate Mutual Information
def mi(dats):
    #dats - two column data to compute mutual information
    return entropy(np.array(dats.iloc[:, 0])) + entropy(np.array(dats.iloc[:, 1])) - jentropy(np.array(dats))


#Calculate transfer entropy
def te(x, y, lag = 1):
    #x - x data to be used for the transfer entropy computation
    #y - y data to be used for the transfer entropy computation
    #lag - temporal lag that will be used to shift the X series

    #X_(t - \tau\delta_t)
    shiftedX = x[: len(x) - lag].reset_index(drop = True)
    #Y_(t - \delta_t)
    shiftedY = y[lag - 1 : len(x) - 1].reset_index(drop = True)
    #cut y
    cutY = y[lag:].reset_index(drop = True)
    
    #Check lengths are the same
    if((len(shiftedX) != len(shiftedY)) or(len(shiftedY) != len(cutY))):
        return "Lengths not equal, something went wrong"
    else:
        p1 = pd.DataFrame([shiftedX, shiftedY]).T
        p2 = pd.DataFrame([cutY, shiftedY]).T
        p3 = shiftedY
        p4 = pd.DataFrame([shiftedX, cutY, shiftedY]).T
        return jentropy(p1) + jentropy(p2) - entropy(p3) - jentropy(p4)
    
#Transfer Entropy with Significance Test
def te_test(x, y, lag = 1, n = 100, alpha = 0.05):
    #x - x data to be used for the transfer entropy computation
    #y - y data to be used for the transfer entropy computation
    #lag - temporal lag that will be used to shift the X series
    #n - number of MCMC iterations to run

    #testable te
    t = te(x, y, lag = lag)

    #randomly scramble data
    tss = []
    for i in range(0, n):
        #compute shuffled transfer entropy
        xss = x
        random.shuffle(xss)
        yss = y
        random.shuffle(yss)
        tss.append(te(xss, yss, lag = lag))

    #fit gaussian
    mean = np.mean(tss)
    sd = np.std(tss)

    #test
    if(alpha == 0.01):
        Tz = mean + 2.36*sd
    elif(alpha == 0.05):
        Tz = mean + 1.66*sd
    else:
        return "Only capable of computing 95% (alpha = 0.05) and 99% (alpha = 0.01) one tail distributions."
    
    return t, Tz, t > Tz

    