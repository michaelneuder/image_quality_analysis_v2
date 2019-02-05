import colorsys
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def getImageGridAxes(rows=1, columns=1, im_size=4):
    """
        provides set of axes for plotting images on.

        args:
            rows: number of rows
            columns: number of columns
            im_size: size of each image
        returns:
            dictionary of size rows*columns indexed by range(rows*columns)    
    """
    plt.figure(figsize = (columns*im_size, rows*im_size))
    gs1 = gridspec.GridSpec(rows, columns)
    gs1.update(wspace=0.03, hspace=0.03)

    ax_dict = {}
    for ii in range(rows*columns):
        ax_dict[ii] = plt.subplot(gs1[ii])
        ax_dict[ii].set_xticklabels([])
        ax_dict[ii].set_yticklabels([])
        ax_dict[ii].get_xaxis().set_visible(False)
        ax_dict[ii].get_yaxis().set_visible(False)
    
    return ax_dict

def rgbToBwImage(image):
    """
        converts a rgb image into black and white use hue, lightness, saturation

        args:
            image: a 3 dimensional array
        returns:
            a two dimensional array
    """
    image = np.asarray(image)
    if len(image.shape) != 3:
        raise ValueError('make sure image has shape [rows, cols, rgb]')
    rows, cols = image.shape[:2]
    for row in range(rows):
        for col in range(cols):
            image[row,col,:] = colorsys.rgb_to_hls(
                image[row,col,0],
                image[row,col,1],
                image[row,col,2]) 
    return image[:,:,1]

def zscore(im):
    return (im - im.mean()) / im.std()

def uniform(im):
    return np.asarray((im - im.min()) / (im.max()-im.min()))
