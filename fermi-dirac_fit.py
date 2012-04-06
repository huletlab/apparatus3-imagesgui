# To Do
#
# 1. See if larger size reporudces error
# - recieved error when looking at [125:375,125:375]
# - error seems to randomly appear
# 2. Get Gaus2d working
# 3. look up table for f1
# 4. check if parameters are physical ie BetaMu = [-6:10]


# Standard library imports
import os
import time
import pickle


from scipy import optimize
import pyfits

# Major library imports
import math
import numpy
from fermi_dirac import *
from matplotlib.pylab import *

import import_data

import warnings
warnings.simplefilter('error')

np_f1 = numpy.vectorize(f1)
np_f2 = numpy.vectorize(f2)




def genf1lookup(pre):
    exp = 10.0**pre
    lookuptable = {}
    for i in range(-6*int(exp),10*int(exp)):
        lookuptable[i/pre] = f1(i/pre)


    f= open("f1lookuptable", 'w')
    pickle.dump(lookuptable,f)
    f.close()

    return
    
def retrievef1table():
    f= open("f1lookuptable", 'r')
    f1lookuptable = pickle.load(f)
    f.close()
    return f1lookuptable





def fermi2d(n0, BetaMu, rg, ry, cg, cy, b, mg, my):
#def fermi2d(n0, BetaMu, rg, ry, cg, cy):
    # Returns the Fermi-Dirac 2D function
    return lambda y,g: fermi2dfun(y,g, n0, BetaMu, rg, ry, cg, cy, b, mg, my)
    

def fermi2dfun(y, g, n0, BetaMu, rg, ry, cg, cy, b, mg, my):
    
    try:
        if -6 < BetaMu < 10:
            
            BetaMu = round(BetaMu,pre)
            
            BetaMuFq = round(BetaMu - fq(BetaMu) * ( pow( (g-cg)/rg, 2) + pow( (y-cy)/ry,2)),pre)
            
            return n0/f1lookuptable[BetaMu] * f1lookuptable[BetaMuFq] + b + mg*g + my*y
            #f1( BetaMu - f0(BetaMu)/fm1(BetaMu) * ( pow( (g-cg)*magnif/rg, 2) + pow( (y-cy)*magnif/ry,2)))
        else:
            print "BetaMu=%f was outside the expected value" % (BetaMu)
    except:
        print "I failed when the parameters were n0=%f, BetaMu=%f, rg=%f, ry=%f, cg=%f, cy=%f, b=%f, mg=%f, my=%f, "\
        % (n0, BetaMu, rg, ry, cg, cy, b, mg, my)
        sys.exit()
    
def gaus2d(n0, rg, ry, cg, cy, b, mg, my):
    # Returns a 2D gaussian function function
    return lambda y,g:  n0*\
    exp(-1*( pow( (g-cg)/rg, 2) + pow( (y-cy)/ry,2))) + b + mg*g + my*y
    
def moments(data):
    """Returns (n0, BetaMu, rg, ry, cg, cy)
    as the starting paramters for a 2D Fermi fit
    
    n0, rg, rg, cg, cy are 
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    BetaMu = -1.0
    b  = 0.1 # offset of background plane
    mg = 0.1 # slope of background plane along g
    my = 0.1 # slope of background plane along y
    return height, BetaMu, width_y, width_x, y, x, b, mg, my
    
def crop(data):
    height, BetaMu, width_y, width_x, y, x, b, mg, my = moments(data)
    return data[(x-width_x/0.75):(x+width_x/0.75),(y-width_y/0.75):(y+width_y/0.75)]
    
def errorfunc(data):    
    return lambda p: ravel( numpy.vectorize(fermi2d(*p))(*indices(data.shape)) - data)
    
def fitfermi2d(data):
    params = moments(data)
    errorfunction = lambda p: ravel( numpy.vectorize(fermi2d(*p))(*indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p, numpy.vectorize(fermi2d(*p))(*indices(data.shape))
    
#def fitgaus2d(data):
#   params = del moments(data)[1]  #Was trying to get moments withouth the BetaMu parameter, because the gaussian fit doesn't care about the fugacity. 
#    errorfunction = lambda p: ravel( numpy.vectorize(gaus2d(*p))(*indices(data.shape)) - data)
#    p, success = optimize.leastsq(errorfunction, params)
#    return p, numpy.vectorize(fermi2d(*p))(*indices(data.shape))
    

if __name__ == '__main__':
    #Data matrix
    data = import_data.load('','6043')
    data = crop(data)
    print data.shape
    
    
    Xin,Yin = mgrid[0:512,0:512]
    
    #Generate and Retrive lookup table for f1
    
    pre = 3 #Decimal precision of lookup table

    genf1lookup(pre)
    f1lookuptable = retrievef1table()

    
    
    t0=time.time()
    
    #Fake result
    p=[2e6, -1., 50., 100. ,256.,256., 0.5, 0.01, 0.01]  #starting parameters for the fit
    # atom number,  BetaMu, radius along g, radius along y, center along g, center along y
    #fitdata = numpy.vectorize(fermi2d(*p))(Xin, Yin)
    
    errs=(errorfunc(data))(p)
    
    print '...Errorfunction evaluation time = %.2f seconds\n' % (time.time()-t0)
    
 
    
    # Comment the line below if you just want to plot the function with the starting parameters
    p, fitdata = fitfermi2d(data)
    #p, fitdata = fitgaus2d(data)
    
    print p
    matshow(   data, cmap=cm.gist_earth_r )
    matshow(fitdata, cmap=cm.gist_earth_r )
    show()
