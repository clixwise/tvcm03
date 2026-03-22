import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import datetime
import glob
from matplotlib.gridspec import GridSpec

# ****
# Class
# ****

# ----
# Figu
# ----
class FiguTran:
    
    def __init__(self):
        
        self.size = None
        self.titl = None
        self.hspa = None
        self.vspa = None
        self.axis_list = None
        self.dict = {}
        # 
        # fig.tight_layout() : UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
        self.fig = plt.figure()
        # https://gemini.google.com/app/f11fdb0ca1fc6a70
        # self.fig = plt.figure(layout="constrained")

class FiguTran1x1(FiguTran):
    def __init__(self):
        super().__init__()
        self.ax1 = None
        self.name = FiguTran1x1.__name__ 

    def upda(self):
        if self.size:
            self.fig.set_size_inches(*self.size)
        
        # GridSpec defined as 1 row, 1 column
        gs = self.fig.add_gridspec(
            1, 1,
            hspace=self.hspa if self.hspa is not None else 0,
            wspace=self.vspa if self.vspa is not None else 0 # Note : superfluous since only 1 column
        )
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.axis_list = [self.ax1]
        
        if self.titl:
            self.fig.suptitle(self.titl)
            
class FiguTran2x1(FiguTran):
    def __init__(self):
        super().__init__()
        self.ax1 = None
        self.ax2 = None
        self.name = FiguTran2x1.__name__ # this is the default name

    def upda(self):
        if self.size:
            self.fig.set_size_inches(*self.size)
        gs = self.fig.add_gridspec(
            2, 1,
            hspace=self.hspa if self.hspa is not None else 0,
            wspace=self.vspa if self.vspa is not None else 0 # Note : superfluous since only 1 column
        )
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.axis_list = [self.ax1, self.ax2]
        if self.titl:
            self.fig.suptitle(self.titl)
          
class FiguTran1x3(FiguTran):
    def __init__(self):
        super().__init__()
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.name = FiguTran1x3.__name__ # this is the default name

    def upda(self):
        if self.size:
            self.fig.set_size_inches(*self.size)
        gs = self.fig.add_gridspec(
            1, 3,
            hspace=self.hspa if self.hspa is not None else 0,
            wspace=self.vspa if self.vspa is not None else 0 # Note : superfluous since only 1 column
        )
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.axis_list = [self.ax1, self.ax2, self.ax3]
        if self.titl:
            self.fig.suptitle(self.titl)
                        
class FiguTran3x1(FiguTran):
    def __init__(self):
        super().__init__()
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.name = FiguTran3x1.__name__ # this is the default name

    def upda(self):
        if self.size:
            self.fig.set_size_inches(*self.size)
        gs = self.fig.add_gridspec(
            3, 1,
            hspace=self.hspa if self.hspa is not None else 0,
            wspace=self.vspa if self.vspa is not None else 0 # Note : superfluous since only 1 column
        )
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[2, 0])
        self.axis_list = [self.ax1, self.ax2, self.ax3]
        if self.titl:
            self.fig.suptitle(self.titl)
            
class FiguTran2x2(FiguTran):
    def __init__(self):
        super().__init__()
        self.ax00 = None
        self.ax01 = None
        self.ax10 = None
        self.ax11 = None
        self.name = FiguTran2x2.__name__ # this is the default name

    def upda(self):
        if self.size:
            self.fig.set_size_inches(*self.size)
        gs = self.fig.add_gridspec(
            2, 2,
            hspace=self.hspa if self.hspa is not None else 0,
            wspace=self.vspa if self.vspa is not None else 0 # Note : superfluous since only 1 column
        )
        self.ax00 = self.fig.add_subplot(gs[0, 0])
        self.ax01 = self.fig.add_subplot(gs[0, 1])
        self.ax10 = self.fig.add_subplot(gs[1, 0])
        self.ax11 = self.fig.add_subplot(gs[1, 1])
        self.axis_list = [self.ax00, self.ax01, self.ax10, self.ax11]
        if self.titl:
            self.fig.suptitle(self.titl)
            
def grid_hori_1x2():
    
    # 1. Setup the Figure
    fig = plt.figure(figsize=(10.5, 8))

    # 2. Define the GridSpec for the 2x2 layout with Internal Spacing
    # hspace controls the vertical gap (between top and bottom rows)
    # wspace controls the horizontal gap (between bottom left and bottom right columns)
    # Positioning "WITHIN"
    # Define the 2x2 GridSpec with spacing parameters
    # hspace=0.5 will create a large vertical gap between the top and bottom rows
    # wspace=0.2 will create a small horizontal gap between the two bottom columns
    gs = GridSpec(
        2, 2, 
        figure=fig, 
        hspace=0.25,  # <-- Adjusts vertical space (Top/Bottom)
        wspace=0.2   # <-- Adjusts horizontal space (Left/Right)
    )

    # 3. Top Subplot (Rectangle 2x1)
    # Spans row 0 and columns 0 and 1
    ax_top = fig.add_subplot(gs[0, :])
    #ax_top.set_title('Ax 1: Top Rectangle (2x1)')
    #ax_top.text(0.5, 0.5, "Top Plot", ha='center', va='center', fontsize=16)

    # 4. Bottom-Left Subplot (Square 1x1)
    # Occupies row 1, column 0
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    #ax_bottom_left.set_title('Ax 2: Bottom Left Square (1x1)')
    #ax_bottom_left.text(0.5, 0.5, "Bottom Left", ha='center', va='center', fontsize=16)

    # 5. Bottom-Right Subplot (Square 1x1)
    # Occupies row 1, column 1
    ax_bottom_right = fig.add_subplot(gs[1, 1])
    #ax_bottom_right.set_title('Ax 3: Bottom Right Square (1x1)')
    #ax_bottom_right.text(0.5, 0.5, "Bottom Right", ha='center', va='center', fontsize=16)

    # Exit
    # ----
    return fig

def grid_vert_2x1():
    
    # 1. Setup the Figure
    fig = plt.figure(figsize=(8, 8))

    # 2. Define the GridSpec for the 2x2 layout with Internal Spacing
    # hspace controls the vertical gap (between top and bottom rows)
    # wspace controls the horizontal gap (between bottom left and bottom right columns)
    # Positioning "WITHIN"
    # Define the 2x2 GridSpec with spacing parameters
    # hspace=0.5 will create a large vertical gap between the top and bottom rows
    # wspace=0.2 will create a small horizontal gap between the two bottom columns
    gs = GridSpec(
        2, 1, 
        figure=fig, 
        hspace=0.25,  # <-- Adjusts vertical space (Top/Bottom)
        wspace=0.2   # <-- Adjusts horizontal space (Left/Right)
    )

    # 4. Bottom-Left Subplot (Square 1x1)
    ax1 = fig.add_subplot(gs[0, 0])
    #ax_bottom_left.set_title('Ax 2: Bottom Left Square (1x1)')
    #ax_bottom_left.text(0.5, 0.5, "Bottom Left", ha='center', va='center', fontsize=16)

    # 5. Bottom-Right Subplot (Square 1x1)
    ax2 = fig.add_subplot(gs[1, 0])
    #ax_bottom_right.set_title('Ax 3: Bottom Right Square (1x1)')
    #ax_bottom_right.text(0.5, 0.5, "Bottom Right", ha='center', va='center', fontsize=16)

    # Exit
    # ----
    return fig

def f():
      

    # Reset all font sizes to default (Matplotlib 3.3+)
    # This is equivalent to setting all rcParams to their default values.
    # If you don't do this, you might only be overriding a few params.
    # If you want to strictly 'reset', you might restart the kernel or run:
    # plt.rcParams.update(plt.rcParamsDefault)

    # --- Set New Global Defaults ---
    if False:
        # For all text elements (titles, labels, ticks, legends)
        plt.rcParams.update({'font.size': 12}) # Example: set base font size to 12

        # For Axes labels (x and y labels)
        plt.rcParams.update({'axes.labelsize': 14})

        # For Axes titles (figure title, subplot titles)
        plt.rcParams.update({'axes.titlesize': 16})

        # For Tick labels (numbers/text next to the axis ticks)
        plt.rcParams.update({'xtick.labelsize': 10,
                            'ytick.labelsize': 10})

        # --- Example Plot ---
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title('Title Font Size 16')
        ax.set_xlabel('X-Label Font Size 14')
        ax.tick_params(axis='both', which='major') # Tick labels will be size 10
        plt.show()
    
    if False:
        # Set a global font size for the entire figure
        plt.rcParams['font.size'] = 14  # set global font size
        
        
        # reset to Matplotlib defaults
        plt.rcParams.update(plt.rcParamsDefault)


# ----
# Plot
# ---- 
class PlotTran:
                
    def __init__(self):
        pass

    def upda(self):
        raise NotImplementedError