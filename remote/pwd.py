from traits.api import HasTraits, Str, Int, Password
from traitsui.api import Item, Group, View

#-------------------------------------------------------------------------------
#  GetPassword Class
#-------------------------------------------------------------------------------

class GetPassword ( HasTraits ): 
    """ This class prompts the user for login and password
    """

    # Define a trait for each of three variants
    user_name     = Str( "" ) 
    password         = Password


    # TextEditor display with secret typing capability (for Password traits):
    text_pass_group = Group( Group( Item('user_name', resizable=True),
                                           style='simple', 
                                           show_border=False), 
                                    Group(Item('password', resizable=True),
                                          style='custom', 
                                          show_border=False))



    # The view 
    view1 = View(text_pass_group, 
                 title = 'Password',
                 buttons = ['OK']) 

	
# Code to run demo:
demo =  GetPassword()
	
if __name__ == "__main__":
    demo.configure_traits()