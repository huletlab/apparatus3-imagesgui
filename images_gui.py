# --- TO DO LIST:
# - Make center region tabbed and put different plots in different tabs 
#    --> Azimutha average data and results
#    --> 1D fits to integrated axial and radial distribution
# - Define analysis group that shows results of analysis
# - Implement real time analysis of the 1D cuts
# - Cross hair should only change on click 

# Standard library imports
from optparse import OptionParser
import sys
import os
import subprocess

import scipy
import pyfits
from scipy.ndimage.interpolation import rotate

# Major library imports
#from numpy import array, linspace, meshgrid, nanmin, nanmax,  pi, zeros
import numpy
from numpy import array

# Enthought library imports
from chaco.api import ArrayDataSource, ArrayPlotData, ColorBar, ContourLinePlot, \
                                 ColormappedScatterPlot, CMapImagePlot, \
                                 ContourPolyPlot, DataRange1D, VPlotContainer, \
                                 DataRange2D, GridMapper, GridDataSource, \
                                 HPlotContainer, ImageData, LinearMapper, \
                                 LinePlot, OverlayPlotContainer, Plot, PlotAxis
from chaco.default_colormaps import *
from traitsui.editors import FileEditor, DirectoryEditor
from enable.component_editor import ComponentEditor
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
from chaco.tools.api import LineInspector, PanTool, RangeSelection, \
                                   RangeSelectionOverlay, ZoomTool
from enable.api import Window, NativeScrollBar
from traits.api import *
from traitsui.api import View, Item, Group, HGroup, VGroup, HSplit, VSplit,Handler, CheckListEditor, EnumEditor, ListStrEditor,ArrayEditor, spring, ListEditor, ButtonEditor
from traitsui.menu import NoButtons
from traitsui.menu import ApplyButton
from traitsui.file_dialog import save_file,open_file
from chaco.api import Plot, ArrayPlotData
from enable.component_editor import ComponentEditor
from traits.api import Any, Array, Callable, CFloat, CInt, Enum, Event, Float, HasTraits, \
                             Int, Instance, Str, Trait, on_trait_change, File, Password, \
                             Bool, Directory, List, Property, DelegatesTo, Button
from traitsui.api import Group, Handler, HGroup, Item, View, HSplit, VSplit, ListStrEditor, TabularEditor, CustomEditor
from traitsui.menu import Action, CloseAction, Menu, \
                                     MenuBar, NoButtons, Separator
from traitsui.tabular_adapter import TabularAdapter

import import_data

import wx
import matplotlib
from mpl_figure_editor import MPLFigureEditor

sys.path.append('/lab/software/apparatus3/py')
import falsecolor
import copy
    
#-------------------------------------------------------------------------------
#  Function to filter column density file names
#-------------------------------------------------------------------------------
import re #regular expressions module
def iscol(x, namefilter):
        #return re.search("\wcolumn\w.ascii", x)
        return re.search(namefilter,x) and re.search(r'ascii',x)


#-------------------------------------------------------------------------------
#  Class to display list of available shots on the left pane
#-------------------------------------------------------------------------------
class SelectAdapter(TabularAdapter):
	""" This adapter class was borrowed from one of the traitsui examples
	"""
	# Titles and column names for each column of a table.
	# In this example, each table has only one column.
	columns = [ ('', 'myvalue') ]
	# Magically named trait which gives the display text of the column named 
	# 'myvalue'. This is done using a Traits Property and its getter:
	myvalue_text = Property

	# The getter for Property 'myvalue_text' simply takes the value of the 
	# corresponding item in the list being displayed in this table.
	# A more complicated example could format the item before displaying it.
	def _get_myvalue_text(self):
		return self.item



#-------------------------------------------------------------------------------
#  Remote fitting routine class
#-------------------------------------------------------------------------------

class RemoteFits ( HasTraits ): 
    """ This class determines whether the fits will be done locally or whether
    they will be done remotely via ssh.  If they are done via ssh it prompts
    the user for login and password
    """
    # Define traits
    remote    = Bool
    user_name = Str( "" ) 
    password  = Password
    # Define the group in which they will be displayed
    remote_group = Group(
                             Item('remote', resizable=True),   
                             Item('user_name', resizable=True),
                             Item('password', resizable=True),
                           )
    # Define the view
    view1 = View(remote_group, 
                 title = 'Set up fitting routine',
                 buttons = ['OK'],
                 width=300)

    def uname(self):
            return self.user_name
    def pwd(self):
            return self.password
        
#-------------------------------------------------------------------------------
#  Matplotlib figure
#-------------------------------------------------------------------------------
def MakePlot( parent, editor):
    fig = editor.object.figure
    panel = wx.Panel(parent,-1)
    canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg( panel, -1, fig)
    toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(canvas)
    toolbar.Realize()
    return panel

#-------------------------------------------------------------------------------
#  ImageGUI class 
#-------------------------------------------------------------------------------

class ImageGUI(HasTraits):
    
    # TO FIX : put here the last available shot
    #shot = File('L:\\data\\app3\\2011\\1108\\110823\\column_5200.ascii')
    #shot = File('/home/pmd/atomcool/lab/data/app3/2012/1203/120307/column_3195.ascii')

    #-- Get current data directory
    datadir = subprocess.Popen( 'gotodat', stdout=subprocess.PIPE).communicate()[0].strip()
    #print datadir
     
    #-- Shot traits
    #shotdir = Directory('/home/pmd/atomcool/lab/data/app3/2012/1203/120320/')
    shotdir = Directory(datadir)
    shots = List(Str)
    selectedshot = List(Str)
    namefilter = Str('column')

    #-- Report trait
    report = Str

    #-- Displayed analysis results
    number = Float
     
    #-- Column density plot container
    #***column_density = Instance(HPlotContainer)
    column_density = Instance( matplotlib.figure.Figure, ())
    #---- Plot components within this container
    #***imgplot     = Instance(CMapImagePlot)
    #***cross_plot  = Instance(Plot)
    #***cross_plot2 = Instance(Plot)
    #***colorbar    = Instance(ColorBar)
    #---- Plot data
    pd = Instance(ArrayPlotData)
    #---- Colorbar 
    num_levels = Int(15)
    colormap = Enum(color_map_name_dict.keys())

    
    replot = Button("replot")

    #-- Crosshair location
    xpos = Int(0.)
    ypos = Int(0.)
    angle = Int(0.)
    cursor_group = Group( Group(Item('xpos', show_label=True), 
				orientation='horizontal'),
			  Group(Item('ypos', show_label=True), 
				orientation='horizontal'),
			  Group(Item('angle', show_label=True), 
				orientation='horizontal'),
		          orientation='vertical', layout='normal',springy=True)

    
    #---------------------------------------------------------------------------
    # Traits View Definitions
    #---------------------------------------------------------------------------
    
    traits_view = View(
                    Group(
                      #Directory
                      Item( 'shotdir',style='simple', editor=DirectoryEditor(), width = 400, \
				      show_label=False, resizable=False ),
                           
                      #Bottom
                      HSplit(
		        #-- Pane for shot selection
        	        Group(
		          Item( 'namefilter', show_label=False,springy=False),		
                          Item( 'shots',show_label=False, width=180, height= 360, \
					editor = TabularEditor(selected='selectedshot',\
					editable=False,multi_select=True,\
					adapter=SelectAdapter()) ),
                          orientation='vertical',
		          layout='normal', ),

		        #-- Pane for column density plots
			Group(
			  Item('column_density',editor=MPLFigureEditor(), \
                                           show_label=False, width=600, height=500, \
                                           resizable=True ), 
			  Item('report',show_label=False, width=180, \
					springy=True, style='custom' ),
			  layout='tabbed', springy=True),

			#-- Pane for viewing controls
			Group(
                          Item('replot',show_label=False),
			  cursor_group,
                          orientation='vertical',
		          layout='normal', ),
                      ),
                      orientation='vertical',
                      layout='normal',
                    ),
                  width=1400, height=500, resizable=True)
    
    #-- Pop-up view when Plot->Edit is selcted from the menu
    plot_edit_view = View(
                    Group(Item('num_levels'),
                          Item('colormap')),
                          buttons=["OK","Cancel"])
                          
    
    #---------------------------------------------------------------------------
    # Private Traits
    #---------------------------------------------------------------------------

    #-- Represents the region where the data set is defined
    _image_index = Instance(GridDataSource) 

    #-- Represents the data that will be plotted on the grid
    _image_value = Instance(ImageData)

    #-- Represents the color map that will be used
    _cmap = Trait(jet, Callable)
    
    
    #---------------------------------------------------------------------------
    # Public View interface
    #---------------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
	#-- super is used to run the inherited __init__ method
	#-- this ensures that all the Traits machinery is properly setup
	#-- even though the __init__ method is overridden
        super(ImageGUI, self).__init__(*args, **kwargs)

	#-- after running the inherited __init__, a plot is created



    def create_plot(self):
        print "Creating Plot..."
 
        #print self.imgdata
        rotimgdata = rotate(self.imgdata,  self.angle, reshape=True) 


        #pngname, figuresize, axesdef = falsecolor.inspecpng([self.imgdata] , self.xpos, self.ypos,  self.minz, self.maxz,\
        #                          falsecolor.my_rainbow, "images_gui", 100, origin='upper', scale=0.5)

        #grid pixels per inch

        print type(rotimgdata.shape[1])
        if ( rotimgdata.shape[1] > rotimgdata.shape[0] ):
          landscape = True
          ratio = float(rotimgdata.shape[0]) / float(rotimgdata.shape[1])
        else:
          landscape = False
          ratio = float(rotimgdata.shape[1]) / float(rotimgdata.shape[0])

        #
        xstart = 0.06
        ystart = 0.08
        size = 0.6
        if landscape:
          xsize = size
          ysize = size * ratio
        else:
          xsize = size * ratio
          ysize = size
        axRect1 = [xstart, ystart, xsize, ysize]

        scale = 1.0
        gap = 0.05*scale
        #
        ystart = axRect1[1] + axRect1[3] + gap
        ysize  = 0.2 
        axROWRect1 = [xstart, ystart, xsize, ysize]

        #
        xstart = axRect1[0] + axRect1[2] + 1.4*gap
        xsize = 0.2
        ystart = axRect1[1]
        ysize = axRect1[3]
        axCOLRect1 = [xstart, ystart, xsize, ysize]
        

        #os.remove(pngname)
        for ax in self.column_density.get_axes():
          ax.cla() 
        self.column_density.clear()
       
        #figure = matplotlib.figure.Figure( figsize = (figuresize[0],figuresize[1]) )
        axes = []

        #for rect in axesdef:
        #  ax = self.column_density.add_axes( rect, frameon=True)
        #  axes.append(ax)

        ax = self.column_density.add_axes( axRect1, frameon=True)
        axes.append(ax)
        ax = self.column_density.add_axes( axROWRect1, sharex=axes[0])
        axes.append(ax)
        ax = self.column_density.add_axes( axCOLRect1, sharey=axes[0])
        axes.append(ax)

        axes[0].imshow( rotimgdata, cmap= falsecolor.my_rainbow, vmin=self.minz, vmax=self.maxz, origin='lower')
    
        row = self.ypos
        col = self.xpos 
        
        alphacross=0.4
        axes[0].axhline( row, linewidth=0.8, color='black', alpha=alphacross)
        axes[0].axvline( col, linewidth=0.8, color='black', alpha=alphacross)
 
        axes[1].set_xlim( 0, len( rotimgdata[ row, :])-1)
        axes[1].plot( rotimgdata[ row, :], color='blue')

        xarray = numpy.arange( len( rotimgdata[:, col] ))
        axes[2].set_ylim( 0, len(xarray)-1 )
        axes[2].plot( rotimgdata[ :, col], xarray, color='red')
        #axes[2].yaxis.set_ticklabels([]) 
        labels = axes[2].get_xticklabels()
        for label in labels:
          label.set_rotation(-90)
       
        #for ax in inspec_figure.get_axes():
        #  self.column_density.add_axes(ax)
        #  print ax
        

        #self.column_density.canvas.draw() 
        wx.CallAfter( self.column_density.canvas.draw)
         
        #axes = self.column_density.add_subplot(111)
        #axes.imshow( rotimgdata) 
        #wx.CallAfter( self.column_density.canvas.draw )
        #t = numpy.linspace(0, 2*numpy.pi, 200)
        #axes.plot(t, numpy.sin(t))
        
        return;

    def update(self):
	#print self.cursor.current_index
	#self.cursor.current_position = 100.,100.
        self.shots = self.populate_shot_list()
	print self.selectedshot    
        self.imgdata, self.report = self.load_imagedata()
        if self.imgdata is not None:
            self.minz = self.imgdata.min()
            self.maxz = self.imgdata.max()
            self.create_plot()

    def populate_shot_list(self):
        try:
            shot_list = os.listdir(self.shotdir)
	    fun = lambda x: iscol(x,self.namefilter)
            shot_list = filter( fun, shot_list)
	    shot_list = sorted(shot_list)
        except ValueError:
            print " *** Not a valid directory path ***"
        return shot_list

    def load_imagedata(self):
        try:
            directory = self.shotdir
	    if self.selectedshot == []:
		    filename = self.shots[0]
	    else:
		    filename = self.selectedshot[0]
            #shotnum = filename[filename.rindex('_')+1:filename.rindex('.ascii')]
	    shotnum = filename[:filename.index('_')]
        except ValueError:
            print " *** Not a valid path *** " 
            return None
        # Set data path
        # Prepare PlotData object
	print "Loading file #%s from %s" % (filename,directory)
        return import_data.load(directory,filename), import_data.load_report(directory,shotnum)


    #---------------------------------------------------------------------------
    # Event handlers
    #---------------------------------------------------------------------------
    
    def _selectedshot_changed(self):
	print self.selectedshot
        self.update()

    def _replot_fired(self):
        self.create_plot()

    def _shots_changed(self):
        self.shots = self.populate_shot_list()
	return

    def _namefilter_changed(self):
	self.shots = self.populate_shot_list()
	return

  
    def _xpos_changed(self):
        print "_xy_changed"
        print " xy = [%f,%f]" % (self.xpos, self.ypos)
    def _ypos_changed(self):
        print "_xy_changed"
        print " xy = [%f,%f]" % (self.xpos, self.ypos)

    def _metadata_changed(self):
	self._xy_changed()
	    
    def _colormap_changed(self):
        self._cmap = color_map_name_dict[self.colormap]
        if hasattr(self, "polyplot"):
            value_range = self.polyplot.color_mapper.range
            self.polyplot.color_mapper = self._cmap(value_range)
            value_range = self.cross_plot.color_mapper.range
            self.cross_plot.color_mapper = self._cmap(value_range)
            # FIXME: change when we decide how best to update plots using
            # the shared colormap in plot object
            self.cross_plot.plots["dot"][0].color_mapper = self._cmap(value_range)
            self.cross_plot2.plots["dot"][0].color_mapper = self._cmap(value_range)
            self.column_density.request_redraw()

    def _num_levels_changed(self):
        if self.num_levels > 3:
            self.polyplot.levels = self.num_levels
            self.lineplot.levels = self.num_levels



        
        #~ plotdata = ArrayPlotData(imagedata = load(dir,shotnum))
        #~ #plotdata = ArrayPlotData(imagedata = load_fits('data/app3/2011/1108/110823/','5200','atoms'))
        #~ # Create a plot and associate it with the PlotData
        #~ plot = Plot(plotdata)
        #~ plot.img_plot("imagedata",colormap=jet)
        #~ self.plot = plot  


class Controller(Handler):

    #---------------------------------------------------------------------------
    # State traits
    #---------------------------------------------------------------------------

    #model = Instance(Model)
    view = Instance(ImageGUI)

    #---------------------------------------------------------------------------
    # Handler interface
    #---------------------------------------------------------------------------

    def init(self, info):
        #self.model = info.object.model
        self.view = info.object.view
        #self.model.on_trait_change(self._model_changed, "model_changed")


    #---------------------------------------------------------------------------
    # Public Controller interface
    #---------------------------------------------------------------------------

    #def edit_model(self, ui_info):
    #    self.model.configure_traits()

    def edit_plot(self, ui_info):
        self.view.configure_traits(view="plot_edit_view")


    #---------------------------------------------------------------------------
    # Private Controller interface
    #---------------------------------------------------------------------------

    #def _model_changed(self):
    #    if self.view is not None:
    #        self.view.update(self.model)



class ModelView(HasTraits):

    #~ model = Instance(Model)
    view = Instance(ImageGUI)
    traits_view = View(Item('@view',
                            show_label=False),
                       menubar=MenuBar(Menu(Action(name="Edit Plot",
                                                   action="edit_plot"),
                                            CloseAction,
                                            name="File")),
                       handler = Controller,
		       title = "APPARATUS3 :: Analyze shots",
                       resizable=True)

    @on_trait_change('model, view')
    def update_view(self):
        #if self.model is not None and self.view is not None:
        if self.view is not None:
            self.view.update()
            

# Instances of model and view
#~ options_dict = {'colormap' : "jet"}
#~ model=Model(**options_dict)
#~ view=PlotUI(**options_dict)
#~ popup = ModelView(model=model, view=view)



def show_plot(**kwargs):
    #model = Model(**kwargs)
    view = ImageGUI(**kwargs)
    modelview=ModelView(view=view)
    modelview.configure_traits()



if __name__ == '__main__':
    # Setup ssh access to be able to perform fitting routines
    #demo =  RemoteFits()
    #demo.configure_traits()
    
    #print demo.uname()
    #print demo.pwd()

    # Set path to data - if running in linux it is 
    # recommended that the data is in a mounted file system

    show_plot(colormap='jet')
