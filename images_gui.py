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

import scipy
import pyfits

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
from enthought.traits.ui.editors import FileEditor, DirectoryEditor
from enable.component_editor import ComponentEditor
from chaco.tools.api import LineInspector, PanTool, RangeSelection, \
                                   RangeSelectionOverlay, ZoomTool
from enable.api import Window, NativeScrollBar
from traits.api import Any, Array, Callable, CFloat, CInt, Enum, Event, Float, HasTraits, \
                             Int, Instance, Str, Trait, on_trait_change, File, Password, \
                             Bool, Directory, List, Property
from traitsui.api import Group, Handler, HGroup, Item, View, HSplit, VSplit, ListStrEditor, TabularEditor
from traitsui.menu import Action, CloseAction, Menu, \
                                     MenuBar, NoButtons, Separator
from traitsui.tabular_adapter import TabularAdapter

import import_data


    
#-------------------------------------------------------------------------------
#  Function to filter column density file names
#-------------------------------------------------------------------------------
import re #regular expressions module
def iscol(x):
        #return re.search("\wcolumn\w.ascii", x)
        return re.search(r'column',x) and re.search(r'ascii',x)


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
#  ImageGUI class 
#-------------------------------------------------------------------------------

class ImageGUI(HasTraits):
    
    # TO FIX : put here the last available shot
    #shot = File('L:\\data\\app3\\2011\\1108\\110823\\column_5200.ascii')
    #shot = File('/home/pmd/atomcool/lab/data/app3/2012/1203/120307/column_3195.ascii')

    #-- Shot traits
    shotdir = Directory('/home/pmd/atomcool/lab/data/app3/2012/1203/120307/')
    shots = List(Str)
    selectedshot = List(Str)

    #-- Report trait
    report = Str

    #-- Displayed analysis results
    number = Float

    
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
                          Item( 'shots',show_label=False, width=180, \
					editor = TabularEditor(selected='selectedshot',\
					editable=False,multi_select=True,\
					adapter=SelectAdapter()) ),
			  Item('report',show_label=False, width=180, \
					springy=True, style='custom' ),
                          orientation='vertical',
		          layout='normal', ),

		        #-- Pane for column density plots
                        Item('container',editor=ComponentEditor(), \
                                         show_label=False, width=600, height=600, \
                                         resizable=True ), 

			#-- Pane for analysis results
			Group(
		          Item('number',show_label=False)
			  )
                      ),
                      orientation='vertical',
                      layout='normal',
                    ),
                  width=1400, height=600, resizable=True)
    
    #-- Pop-up view when Plot->Edit is selcted from the menu
    plot_edit_view = View(
                    Group(Item('num_levels'),
                          Item('colormap')),
                          buttons=["OK","Cancel"])
                          
    num_levels = Int(15)
    colormap = Enum(color_map_name_dict.keys())
    
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
        self.create_plot()



    def create_plot(self):

        #-- Create the index for the x an y axes and the range over
	#-- which they vary
        self._image_index = GridDataSource(array([]),
                                          array([]),
                                          sort_order=("ascending","ascending"))
        image_index_range = DataRange2D(self._image_index)
        
        self._image_index.on_trait_change(self._metadata_changed,
                                          "metadata_changed")


	#-- Create the image values and determine their range
        self._image_value = ImageData(data=array([]), value_depth=1)
        image_value_range = DataRange1D(self._image_value)

        
        # Create the image plot
        self.imgplot = CMapImagePlot( index=self._image_index,
                                      value=self._image_value,
                                      index_mapper=GridMapper(range=image_index_range),
                                      color_mapper=self._cmap(image_value_range),)
                                 

        # Create the contour plots
        #~ self.polyplot = ContourPolyPlot(index=self._image_index,
                                        #~ value=self._image_value,
                                        #~ index_mapper=GridMapper(range=
                                            #~ image_index_range),
                                        #~ color_mapper=\
                                            #~ self._cmap(image_value_range),
                                        #~ levels=self.num_levels)

        #~ self.lineplot = ContourLinePlot(index=self._image_index,
                                        #~ value=self._image_value,
                                        #~ index_mapper=GridMapper(range=
                                            #~ self.polyplot.index_mapper.range),
                                        #~ levels=self.num_levels)


        # Add a left axis to the plot
        left = PlotAxis(orientation='left',
                        title= "axial",
                        mapper=self.imgplot.index_mapper._ymapper,
                        component=self.imgplot)
        self.imgplot.overlays.append(left)

        # Add a bottom axis to the plot
        bottom = PlotAxis(orientation='bottom',
                          title= "radial",
                          mapper=self.imgplot.index_mapper._xmapper,
                          component=self.imgplot)
        self.imgplot.overlays.append(bottom)


        # Add some tools to the plot
        self.imgplot.tools.append(PanTool(self.imgplot,
                                            constrain_key="shift"))
        self.imgplot.overlays.append(ZoomTool(component=self.imgplot,
                                            tool_mode="box", always_on=False))
        self.imgplot.overlays.append(LineInspector(component=self.imgplot,
                                               axis='index_x',
                                               inspect_mode="indexed",
                                               write_metadata=True,
                                               is_listener=False,
                                               color="white"))
        self.imgplot.overlays.append(LineInspector(component=self.imgplot,
                                               axis='index_y',
                                               inspect_mode="indexed",
                                               write_metadata=True,
                                               color="white",
                                               is_listener=False))

        # Add the plot (or plots) to a container
        contour_container = OverlayPlotContainer(padding=20,
                                                 use_backbuffer=True,
                                                 unified_draw=True)
        contour_container.add(self.imgplot)
        #~ contour_container.add(self.polyplot)
        #~ contour_container.add(self.lineplot)


        # Create a colorbar
        cbar_index_mapper = LinearMapper(range=image_value_range)
        self.colorbar = ColorBar(index_mapper=cbar_index_mapper,
                                 plot=self.imgplot,
                                 padding_top=self.imgplot.padding_top,
                                 padding_bottom=self.imgplot.padding_bottom,
                                 padding_right=40,
                                 resizable='v',
                                 width=30)
                                 
                        
        # Create the two cross plots
        self.pd = ArrayPlotData(line_index = array([]),
                                line_value = array([]),
                                scatter_index = array([]),
                                scatter_value = array([]),
                                scatter_color = array([]))

        self.cross_plot = Plot(self.pd, resizable="h")
        self.cross_plot.height = 100
        self.cross_plot.padding = 20
        self.cross_plot.plot(("line_index", "line_value"),
                             line_style="dot")
        self.cross_plot.plot(("scatter_index","scatter_value","scatter_color"),
                             type="cmap_scatter",
                             name="dot",
                             color_mapper=self._cmap(image_value_range),
                             marker="circle",
                             marker_size=6)

        self.cross_plot.index_range = self.imgplot.index_range.x_range

        self.pd.set_data("line_index2", array([]))
        self.pd.set_data("line_value2", array([]))
        self.pd.set_data("scatter_index2", array([]))
        self.pd.set_data("scatter_value2", array([]))
        self.pd.set_data("scatter_color2", array([]))

        self.cross_plot2 = Plot(self.pd, width = 140, orientation="v", resizable="v", padding=20, padding_bottom=160)
        self.cross_plot2.plot(("line_index2", "line_value2"),
                             line_style="dot")
        self.cross_plot2.plot(("scatter_index2","scatter_value2","scatter_color2"),
                             type="cmap_scatter",
                             name="dot",
                             color_mapper=self._cmap(image_value_range),
                             marker="circle",
                             marker_size=8)

        self.cross_plot2.index_range = self.imgplot.index_range.y_range


        # Create a container and add components
        self.container = HPlotContainer(padding=40, fill_padding=True,
                                        bgcolor = "white", use_backbuffer=False)
        inner_cont = VPlotContainer(padding=0, use_backbuffer=True)
        inner_cont.add(self.cross_plot)
        inner_cont.add(contour_container)
        self.container.add(self.colorbar)
        self.container.add(inner_cont)
        self.container.add(self.cross_plot2)

	self.hscrollbar = NativeScrollBar(orientation = "vertical",
			                  position=[20,50],
					  bounds=[100,100],
					  bgcolor = "red",
					  color = "white",
					  hjustify = "center",
					  vjustify = "center"
					 )

	self.container.add( self.hscrollbar )

	 
        # print dir(self.container)


    def update(self):
        self.shots = self.populate_shot_list()
	print self.selectedshot    
        imgdata, self.report = self.load_imagedata()
        if imgdata is not None:
            self.minz = imgdata.min()
            self.maxz = imgdata.max()
            self.colorbar.index_mapper.range.low = self.minz
            self.colorbar.index_mapper.range.high = self.maxz
            xs=numpy.linspace(0,imgdata.shape[0],imgdata.shape[0]+1)
            ys=numpy.linspace(0,imgdata.shape[1],imgdata.shape[1]+1)
            #print xs
            #print ys
            self._image_index.set_data(xs,ys)
            self._image_value.data = imgdata
            self.pd.set_data("line_index", xs)
            self.pd.set_data("line_index2",ys)
            self.container.invalidate_draw()
            self.container.request_redraw()                        

    def populate_shot_list(self):
        try:
            shot_list = os.listdir(self.shotdir)
            shot_list = filter( iscol, shot_list)
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
            shotnum = filename[filename.rindex('_')+1:filename.rindex('.ascii')]
        except ValueError:
            print " *** Not a valid column density path *** " 
            return None
        # Set data path
        # Prepare PlotData object
	print "Loading shot #%s from %s" % (shotnum,directory)
        return import_data.load(directory,shotnum), import_data.load_report(directory,shotnum)


    #---------------------------------------------------------------------------
    # Event handlers
    #---------------------------------------------------------------------------
    
    def _selectedshot_changed(self):
	print self.selectedshot
        self.update()

    def _shots_changed(self):
        self.shots = self.populate_shot_list()
	return

    def _metadata_changed(self, old, new):
        """ This function takes out a cross section from the image data, based
        on the line inspector selections, and updates the line and scatter
        plots."""
        self.cross_plot.value_range.low = self.minz
        self.cross_plot.value_range.high = self.maxz
        self.cross_plot2.value_range.low = self.minz
        self.cross_plot2.value_range.high = self.maxz
        if self._image_index.metadata.has_key("selections"):
            x_ndx, y_ndx = self._image_index.metadata["selections"]
            if y_ndx and x_ndx:
                self.pd.set_data("line_value",
                                 self._image_value.data[y_ndx,:])
                self.pd.set_data("line_value2",
                                 self._image_value.data[:,x_ndx])
                xdata, ydata = self._image_index.get_data()
                xdata, ydata = xdata.get_data(), ydata.get_data()
                self.pd.set_data("scatter_index", array([xdata[x_ndx]]))
                self.pd.set_data("scatter_index2", array([ydata[y_ndx]]))
                self.pd.set_data("scatter_value",
                    array([self._image_value.data[y_ndx, x_ndx]]))
                self.pd.set_data("scatter_value2",
                    array([self._image_value.data[y_ndx, x_ndx]]))
                self.pd.set_data("scatter_color",
                    array([self._image_value.data[y_ndx, x_ndx]]))
                self.pd.set_data("scatter_color2",
                    array([self._image_value.data[y_ndx, x_ndx]]))
        else:
            self.pd.set_data("scatter_value", array([]))
            self.pd.set_data("scatter_value2", array([]))
            self.pd.set_data("line_value", array([]))
            self.pd.set_data("line_value2", array([]))

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
            self.container.request_redraw()

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
    demo =  RemoteFits()
    demo.configure_traits()
    
    print demo.uname()
    print demo.pwd()

    # Set path to data - if running in linux it is 
    # recommended that the data is in a mounted file system

    show_plot(colormap='jet')
