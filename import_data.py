# Standard library imports
import os

import scipy
import pyfits

# Major library imports
import numpy
from matplotlib.pylab import *


#~ # Check the OS and give correct path dependency
#~ if os.name == "posix":
    #~ #print os.getenv('COMPUTERNAME')
    #~ #Change this to the mount point for atomcool/lab. When using Linux.
    #~ atomcool_lab_path = '/home/ernie/atomcool_lab/'
#~ else:
    #~ #Change this to the map drive for atomcool/lab. When using Windows.
    #~ atomcool_lab_path = 'L:/'
    
atomcool_lab_path=''
    
def load(directory, shot):
    imgpath = atomcool_lab_path+directory+'column_'+shot+'.ascii'
    #print "\n******************"
    #print "    Inside load function: %s", imgpath
    a = numpy.loadtxt(imgpath)
    return a

def load_report(directory, shot):
    reportpath = atomcool_lab_path+directory+'report'+shot+'.INI'
    f = open(reportpath,'r')
    string = f.read()
    f.close()
    return string
    
def load_fits(dir,shot,type):
    fitspath = atomcool_lab_path+dir+shot+type+'.fits'
    hdulist = pyfits.open(fitspath)
    #print hdulist[0].data[0]
    #print "\n******************"
    #print "    Inside load_fits function: %s", fitspath
    return hdulist[0].data[0]

def load_fits_file(path):
    hdulist = pyfits.open(path)
    #print hdulist[0].data[0]
    return hdulist[0].data[0]
    
    

if __name__ == '__main__':
    atoms   = load_fits('data/app3/2011/1108/110825/','6043','atoms')
    noatoms = load_fits('data/app3/2011/1108/110825/','6043','noatoms')
    coldens = load('data/app3/2011/1108/110825/','6043')
    #p=[2e6, -1., 100., 100. ,256.,256.]
    #Xin,Yin = mgrid[0:512,0:512]
    #fitdata = numpy.vectorize(fermi2d(*p))(Xin, Yin)
    #p, fitdata = fitfermi2d(data)
    #print p
    matshow(   atoms, cmap=cm.gist_earth_r )
    matshow(noatoms, cmap=cm.gist_earth_r )
    matshow( coldens, cmap=cm.gist_earth_r )
    show()
