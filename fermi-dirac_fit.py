# Standard library imports
import os
import time


from scipy import optimize
import pyfits

# Major library imports
import math
import numpy
from fermi_dirac import *
from matplotlib.pylab import *

from import_data import load, load_fits


np_f1 = numpy.vectorize(f1)
np_f2 = numpy.vectorize(f2)


def fermi2d(n0, BetaMu, rg, ry, cg, cy, b, mg, my):
#def fermi2d(n0, BetaMu, rg, ry, cg, cy):
    # Returns the Fermi-Dirac 2D function
    return lambda y,g:  n0/f1(BetaMu) *\
    f1( BetaMu - fq(BetaMu) * ( pow( (g-cg)/rg, 2) + pow( (y-cy)/ry,2))) + b + mg*g + my*y
    #f1( BetaMu - f0(BetaMu)/fm1(BetaMu) * ( pow( (g-cg)*magnif/rg, 2) + pow( (y-cy)*magnif/ry,2)))
    
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
    
def errorfunc(data):    
    return lambda p: ravel( numpy.vectorize(fermi2d(*p))(*indices(data.shape)) - data)
    
def fitfermi2d(data):
    params = moments(data)
    errorfunction = lambda p: ravel( numpy.vectorize(fermi2d(*p))(*indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p, numpy.vectorize(fermi2d(*p))(*indices(data.shape))
    

if __name__ == '__main__':
    data = load('data/app3/2011/1108/110825/','6043')
    p=[2e6, -1., 100., 100. ,256.,256., 0.5, 0.01, 0.01]
    Xin,Yin = mgrid[0:512,0:512]
    
    t0=time.time()
    
    errs=(errorfunc(data))(p)
    
    print '...Errorfunction evaluation time = %.2f seconds\n' % (time.time()-t0)
    
    fitdata = numpy.vectorize(fermi2d(*p))(Xin, Yin)
    #This line does the fit
    #p, fitdata = fitfermi2d(data)
    print p
    matshow(   data, cmap=cm.gist_earth_r )
    matshow(fitdata, cmap=cm.gist_earth_r )
    show()
