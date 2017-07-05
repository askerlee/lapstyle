# Generate image Laplacian of a given image, save it as a .png file for visualization

import numpy as np
import scipy.optimize, scipy.ndimage, scipy.misc
import lasagne
from lasagne.layers import Conv2DLayer, InputLayer
import sys
import pdb

def prepare_image(image):
    """Given an image loaded from disk, turn it into a representation compatible with the model.
    The format is (b,c,y,x) with batch=1 for a single image, channels=3 for RGB, and y,x matching
    the resolution.
    """
    image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)[::-1, :, :]
    image = image.astype(np.float32)
    return image[np.newaxis]
    
def finalize_image(image, scale):
    """Based on the output of the neural network, convert it into an image format that can be saved
    to disk -- shuffling dimensions as appropriate.
    """
    image = np.swapaxes(np.swapaxes(image[::-1], 0, 1), 1, 2)
    # ignore the sign, scale up the pixel value to make it more striking
    image = np.abs(image) * scale
    image = np.clip(image, 0, 255).astype('uint8')
    return image

infilename = sys.argv[1]
outfilename = sys.argv[2]
if len(sys.argv) > 3:
    scale = int( sys.argv[3] )
else:
    scale = 3
    
net = {}
# once a channel among 'R','G','B', 
# if all channels are fed to laplacian conv at the same time, 
# the output will be sum of the outputs from all input channels
net['img']     = InputLayer((None, 1, None, None))
net['laplacian'] = Conv2DLayer(net['img'], 1, 3, pad=1)
laplacian = np.array( [ [0,-1,0], [-1,4,-1], [0,-1,0] ], dtype=np.float32 )
W = np.zeros( (1, 1, 3, 3), dtype=np.float32 )
W[0,0] = laplacian
net['laplacian'].W.set_value(W)

orig_img = scipy.ndimage.imread(infilename, mode='RGB') 
img = prepare_image(orig_img)

output = []
for i in xrange(3):
    img_chan = img[:, [i], :, :]
    tensor_input = { net['img']: img_chan }
    out_chan = lasagne.layers.get_output( net['laplacian'], tensor_input )
    output.append( out_chan.eval()[0,0] )

output = np.array(output)
    
out_img = finalize_image(output, scale)
scipy.misc.toimage(out_img, cmin=0, cmax=255).save(outfilename)
