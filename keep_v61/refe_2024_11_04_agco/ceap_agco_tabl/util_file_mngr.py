import os

from matplotlib import pyplot as plt

file_object = None

def set_file_objc(file):
    global file_object
    file_object = file

def get_file_objc():
    return file_object
        
def write(strg):
    get_file_objc().write("\n")
    get_file_objc().write(strg)
    
plot_object = None

def set_plot_objc(plot):
    global plot_object
    plot_object = plot

def get_plot_objc():
    return plot_object
        
def saveplot(what, suff):
    script_path = os.path.abspath(__file__)
    script_dire = os.path.dirname(script_path)
    file_path = os.path.join(script_dire, f'../plot/{suff} {what}.pdf')
    plt.savefig(file_path)