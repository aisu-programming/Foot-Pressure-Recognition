''' Libraries '''
from PIL import Image
import numpy as np


''' Execution '''
if __name__ == '__main__':
    
    image = np.array(Image.open('test/1/image_0001_adjusted_sharpen_contour.png'))
    print(image.shape)