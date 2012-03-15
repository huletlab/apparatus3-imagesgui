import os
from ctypes import *
import math


if os.getenv('COMPUTERNAME') == 'APP3-ANALYSIS':
    gsldll = cdll.LoadLibrary('C:/Program Files (x86)/GnuWin32/bin/libgsl.dll')
else:
    gsldll = cdll.LoadLibrary('C:/Program Files/GnuWin32/bin/libgsl.dll')
    

gsldll.gsl_sf_fermi_dirac_1.restype = c_double
gsldll.gsl_sf_fermi_dirac_2.restype = c_double
gsldll.gsl_sf_fermi_dirac_3half.restype = c_double

def fq(x):
    # f function used to interpolate between the low and high temperature limits
    # for the size of the cloud
    return ( 1 + math.exp(x))/ math.exp(x) * math.log( 1+ math.exp(x))

def fm1(x):
    #print "Evaluating fm1(%.5f)" % x
    print gsldll.gsl_sf_fermi_dirac_m1( c_double(x))
    if x<-80.:
        return 0.0
    else:
        return gsldll.gsl_sf_fermi_dirac_1( c_double(x))

def f0(x):
    if x<-80.:
        return 0.0
    else:
        return gsldll.gsl_sf_fermi_dirac_1( c_double(x))
    
    
def f1(x):
    if x<-80.:
        return 0.0
    else:
        return gsldll.gsl_sf_fermi_dirac_1( c_double(x))
    
def f2(x):
    if x<-80.:
        return 0.0
    else:
        return gsldll.gsl_sf_fermi_dirac_2( c_double(x))
    
def f32(x):
    if x<-80.0:
        return 0.0
    else:
        return gsldll.gsl_sf_fermi_dirac_3half( c_double(x))
    

if __name__ == '__main__':
    print f1(-10.)
    print f2(-0.13388)
    print f32(-10.)
    print pow(1./f2(1.)/6., 1./3.)
    BetaM=9.96e-27/(1.38e-23*1e-6)
    BetaMu= 1.
    print BetaMu - BetaM/2. * 2.*3.14159*3800.
    #print f1( BetaMu - BetaM/2. * 2.*3.14159*3800. )