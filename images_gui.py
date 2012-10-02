

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
from traitsui.editors import FileEditor
from enable.component_editor import ComponentEditor
from chaco.tools.api import LineInspector, PanTool, RangeSelection, \
                                   RangeSelectionOverlay, ZoomTool
from enable.api import Window
from traits.api import Any, Array, Callable, CFloat, CInt, Enum, Event, Float, HasTraits, \
                             Int, Instance, Str, Trait, on_trait_change, File
from traitsui.api import Group, Handler, HGroup, Item, View, HSplit, VSplit
from traitsui.menu import Action, CloseAction, Menu, \
                                     MenuBar, NoButtons, Separator


from import_data import load, load_fits
    


class ImageGUI(HasTraits):
    
    # TO FIX : put here the last available shot
    shot = File('L:\\data\\app3\\2011\\1108\\110823\\column_5200.ascii')
    
    #---------------------------------------------------------------------------
    # Traits View Definitions
    #---------------------------------------------------------------------------
    
    traits_view = View(
                    HSplit(
                        Item('shot',style='custom',editor=FileEditor(filter=['*.ascii']),show_label=False, resizable=True, width=400),
                        Item('container', editor=ComponentEditor(), show_label=False, width=800, height=800)),
                        width=1200, height=800, resizable=True, title="APPARATUS 3 :: Analyze Images")
                        
    plot_edit_view = View(
                    Group(Item('num_levels'),
                          Item('colormap')),
                          buttons=["OK","Cancel"])
                          
    num_levels = Int(15)
    colormap = Enum(color_map_name_dict.keys())
    
    #---------------------------------------------------------------------------
    # Private Traits
    #---------------------------------------------------------------------------

    _image_index = Instance(GridDataSource)
    _image_value = Instance(ImageData)

    _cmap = Trait(jet, Callable)
    
    
    #---------------------------------------------------------------------------
    # Public View interface
    #---------------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        super(ImageGUI, self).__init__(*args, **kwargs)
        self.create_plot()



    def create_plot(self):

        # Create the mapper, etc
        self._image_index = GridDataSource(array([]),
                                          array([]),
                                          sort_order=("ascending","ascending"))
        image_index_range = DataRange2D(self._image_index)
        
        self._image_index.on_trait_change(self._metadata_changed,
                                          "metadata_changed")

        self._image_value = ImageData(data=array([]), value_depth=1)
        image_value_range = DataRange1D(self._image_value)

        
        # Create the image plot
        self.imgplot = CMapImagePlot(index=self._image_index,
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
        #~ self.polyplot.tools.append(PanTool(self.polyplot,
                                           #~ constrain_key="shift"))
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

        # Add these two plots to one container
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
                             marker_size=8)

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


    def update(self):
        imgdata = self.load_imagedata()
        if imgdata is not None:
            self.minz = imgdata.min()
            self.maxz = imgdata.max()
            self.colorbar.index_mapper.range.low = self.minz
            self.colorbar.index_mapper.range.high = self.maxz
            xs=numpy.linspace(0,imgdata.shape[0],imgdata.shape[0]+1)
            ys=numpy.linspace(0,imgdata.shape[1],imgdata.shape[1]+1)
            print xs
            print ys
            self._image_index.set_data(xs,ys)
            self._image_value.data = imgdata
            self.pd.set_data("line_index", xs)
            self.pd.set_data("line_index2",ys)
            self.container.invalidate_draw()
            self.container.request_redraw()                        
        
    def load_imagedata(self):
        try:
            dir = self.shot[self.shot.index(':\\')+2:self.shot.rindex('\\')+1]
            shotnum = self.shot[self.shot.rindex('_')+1:self.shot.rindex('.ascii')]
        except ValueError:
            print " *** Not a valid column density path *** " 
            return None
        # Set data path
        # Prepare PlotData object
        print dir
        print shotnum
        return load(dir,shotnum)


    #---------------------------------------------------------------------------
    # Event handlers
    #---------------------------------------------------------------------------
    


    def _shot_changed(self):
        self.update()

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
                       title = "Function Inspector",
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
    #demo.configure_traits()
    show_plot(colormap='jet')
