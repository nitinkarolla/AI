from PIL import Image
import numpy as np
from Colorization.NeuralNetwork import NeuralNetwork

class Colorizer():

    def __init__(self, image_location='some_image.jpg'):
        self.im_size = (0, 0)
        self.image_loc = image_location
        self.image = None
        self.pix_object = None
        self.pixel_values = None

    def extract_pixels(self):
        # Open the image
        self.image = Image.open(self.image_loc)
        # Load the image - Pixel object
        self.pix_object = self.image.load()
        # Size of the image - Number of rows * Number of columns
        self.im_size = self.image.size
        # Extract all the pixel values - Start from left corner (Moving from Left to Right)
        self.pixel_values = list(self.image.getdata())

    def change_color(self):

        # To change the color
        # For every row
        for i in range(self.im_size[0]):
            # For every column
            for j in range(self.im_size[1]):
                # Change the value of every pixel to black
                self.pix_object[i, j] = (0, 0, 0)
        # Save the new image
        self.image.save('new_image.png')

if __name__ == '__main__':

    new_color = Colorizer()
    new_color.extract_pixels()
    new_color.change_color()
    print("Done")

    X = np.random.rand(4, 3)
    y = np.array([1, 0, 0, 1])
    NeuralNetwork(X,y)
    nn = NeuralNetwork(X, y)
    nn.weightsInitialisation()
    print(nn.feedForward([0.97794334, 0.03784321, 0.64579876]))