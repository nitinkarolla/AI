from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from NeuralNetwork import NeuralNetwork

# class Colorizer():

#     def __init__(self, image_location='some_image.jpg'):
#         self.im_size = (0, 0)
#         self.image_loc = image_location
#         self.image = None
#         self.pix_object = None
#         self.pixel_values = None

#     def extract_pixels(self):
#         # Open the image
#         self.image = Image.open(self.image_loc)
#         # Load the image - Pixel object
#         self.pix_object = self.image.load()
#         # Size of the image - Number of rows * Number of columns
#         self.im_size = self.image.size
#         # Extract all the pixel values - Start from left corner (Moving from Left to Right)
#         self.pixel_values = list(self.image.getdata())

#     def change_color(self):

#         # To change the color
#         # For every row
#         for i in range(self.im_size[0]):
#             # For every column
#             for j in range(self.im_size[1]):
#                 # Change the value of every pixel to black
#                 self.pix_object[i, j] = (0, 0, 0)
#         # Save the new image
#         self.image.save('new_image.png')

class ImageData():
    def __init__(self, f_s, directory):
        self.f_s = f_s
        self.directory = directory

    def pad_images(self, image):
        p = int((self.f_s - 1)/2)
        image = np.pad(array=image, pad_width=p, mode='constant', constant_values=0)
        return image

    def create_dataset(self, gray, red, green, blue):
        X = []
        y = []
        p_gray = self.pad_images(gray)
        p_red = self.pad_images(red)
        p_green = self.pad_images(green)
        p_blue = self.pad_images(blue)
        
        for i in range(0, len(p_gray)-(self.f_s-1)):
            for j in range(0, len(p_gray)-(self.f_s-1)):
                X.append(list(p_gray[i:i+self.f_s,j:j+self.f_s].flatten()))
                y.append([p_red[i:i+self.f_s,j:j+self.f_s].flatten()[int(self.f_s*self.f_s/2)], 
                        p_green[i:i+self.f_s,j:j+self.f_s].flatten()[int(self.f_s*self.f_s/2)], 
                        p_blue[i:i+self.f_s,j:j+self.f_s].flatten()[int(self.f_s*self.f_s/2)]])
        
        return X, y
        
    def get_images(self):
        exts = ["jpg", "jpeg", "png"]
        print("Opening directory {}".format(self.directory))
        for root, _, files in os.walk(self.directory):
            if root:
                X = []
                y = []
                file_name = []
                for f in files:
                    if f.split(".")[1] in exts:
                        image = cv2.imread(os.path.join(root, f))
                        image = cv2.resize(image, (20,20), interpolation = cv2.INTER_AREA)
                        # gray image
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        # red, green, blue components
                        red, green, blue = image[:,:,2], image[:,:,1], image[:,:,0]
                        m_X, m_y = self.create_dataset(gray, red, green, blue)
                    
                        X.append(m_X)
                        y.append(m_y)
                        file_name.append(f)

        return X, y, file_name

if __name__ == '__main__':
    # new_color = Colorizer()
    # new_color.extract_pixels()
    # new_color.change_color()
    # print("Done")

    # Getting images and returning X, y and file_name lists
    # Use indexes in X, y and file_name to get values for respective images
    image_data = ImageData(f_s=3, directory="./Images/")
    X, y, file_name = image_data.get_images()

    X = np.random.rand(4, 3)
    y = np.array([1, 0, 0, 1])
    NeuralNetwork(X,y)
    nn = NeuralNetwork(X, y)
    nn.weightsInitialisation()
    print(nn.feedForward([0.97794334, 0.03784321, 0.64579876]))