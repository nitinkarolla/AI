from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from NeuralNetwork2 import NeuralNetwork
from sklearn.preprocessing import StandardScaler
import pickle

class Colorizer():
    def __init__(self, red_list, blue_list, green_list, image_location, epochs, f_s, result_size=300):
        self.im_size = (0, 0)
        self.image_loc = image_location
        self.image = None
        self.pix_object = None
        self.pixel_values = None
        self.red = red_list
        self.blue = blue_list
        self.green = green_list
        self.result_img_size = result_size
        self.epochs = epochs
        self.f_s = f_s

    def extract_pixels(self):
        # Open the image
        self.image = Image.open(self.image_loc)
        # Load the image - Pixel object
        self.pix_object = self.image.load()
        # Size of the image - Number of rows * Number of columns
        self.im_size = self.image.size
        # Extract all the pixel values - Start from left corner (Moving from Left to Right)
        self.pixel_values = list(self.image.getdata())
        
    def create_image_from_array(self):
        w, h = self.result_img_size, self.result_img_size
        count = 0
        data = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(w):
            for j in range(h):
                data[i,j] = (int(self.red[count]), int(self.green[count]), int(self.blue[count]))
                count +=1
        
        img = Image.fromarray(data, 'RGB')
        img.save("./Results/" + "epochs_" + str(self.epochs) + "f_s_" + str(self.f_s) + "_" + self.image_loc.split("/")[-1])
        img.show()

class ImageData():
    def __init__(self, f_s, directory, image_size=300):
        self.f_s = f_s
        self.directory = directory
        self.img_size = image_size

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
        
    def get_images(self, directory = None, data_type=None):
        if data_type == "test":
            self.directory = directory

        exts = ["jpg", "jpeg", "png"]
        print("Opening directory {}".format(self.directory))
        for root, _, files in os.walk(self.directory):
            if root:
                X = []
                y = []
                file_name = []
                for f in files:
                    if f.split(".")[1] in exts:
                        print("Accessing ", f)
                        image = cv2.imread(os.path.join(root, f))
                        image = cv2.resize(image, (self.img_size,self.img_size), interpolation = cv2.INTER_AREA)
                        # gray image
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        # red, green, blue components
                        red, green, blue = image[:,:,2], image[:,:,1], image[:,:,0]
                        m_X, m_y = self.create_dataset(gray, red, green, blue)
                    
                        X.append(m_X)
                        y.append(m_y)
                        file_name.append(f)

        return X, y, file_name

    def align_data(self, X, y):
        data_X = []
        for sublist in X:
            for item in sublist:
                data_X.append(item)

        data_y_red = []
        data_y_green = []
        data_y_blue = []

        for sublist in y:
            for item in sublist:
                data_y_red.append(item[0])
                data_y_green.append(item[1])
                data_y_blue.append(item[2])
                
        return data_X, data_y_red, data_y_green, data_y_blue

    def bound_predictions(self, predictions_b, predictions_g, predictions_r):
        for i in range(len(predictions_g)):
            if predictions_b[i] < 0:
                predictions_b[i] = 0
            if predictions_b[i] > 255:
                predictions_b[i] = 255
            if predictions_g[i] < 0:
                predictions_g[i] = 0
            if predictions_g[i] > 255:
                predictions_g[i] = 255
            if predictions_r[i] < 0:
                predictions_r[i] = 0
            if predictions_r[i] > 255:
                predictions_r[i] = 255
        
            predictions_b[i] = int(predictions_b[i])
            predictions_g[i] = int(predictions_g[i])
            predictions_r[i] = int(predictions_r[i])
            
        return predictions_r, predictions_g, predictions_b


if __name__ == '__main__':
    # Getting images and returning X, y and file_name lists
    # Use indexes in X, y and file_name to get values for respective images
    f_s = 7
    image_data = ImageData(f_s=f_s, directory="./Images/", image_size=300)
    X, y, file_name = image_data.get_images()
    data_X, data_y_red, data_y_green, data_y_blue = image_data.align_data(X, y)
    
    scaler = StandardScaler()
    scaler.fit(data_X)

    X_train = scaler.transform(data_X)

    # # TRAINING
    # nn_r= NeuralNetwork(epochs = 10,
    #                     batch_size = 100,
    #                     num_hidden_layers = 3,
    #                     num_neurons_each_layer = [10, 20, 10],
    #                     learning_rate = 0.003)
    # nn_g= NeuralNetwork(epochs = 10,
    #                     batch_size = 100,
    #                     num_hidden_layers = 3,
    #                     num_neurons_each_layer = [10, 20, 10],
    #                     learning_rate = 0.003)
    # nn_b= NeuralNetwork(epochs = 10,
    #                     batch_size = 100,
    #                     num_hidden_layers = 3,
    #                     num_neurons_each_layer = [10, 20, 10],
    #                     learning_rate = 0.003)
    
    # print("Training for red model\n-------------------------")
    # nn_r.fit(X_train, data_y_red)
    # print("-------------------------\nTraining for green model\n-------------------------")
    # nn_g.fit(X_train, data_y_green)
    # print("-------------------------\nTraining for blue model\n-------------------------")
    # nn_b.fit(X_train, data_y_blue)
    
    filename_red = 'finalized_model_red.sav'
    # pickle.dump(nn_r, open(filename_red, 'wb'))
    filename_green = 'finalized_model_green.sav'
    # pickle.dump(nn_g, open(filename_green, 'wb'))
    filename_blue = 'finalized_model_blue.sav'
    # pickle.dump(nn_b, open(filename_blue, 'wb'))

    # TESTING
    directory = "./Images/test"
    test_image = './Images/test/scene5.jpeg'

    X_test, y_test, files = image_data.get_images(directory, "test")
    print(files)
    data_X_test, data_y_red_test, data_y_green_test, data_y_blue_test = image_data.align_data(X_test, y_test)
    data_X_test = scaler.transform(data_X_test)
    
    loaded_model_b = pickle.load(open(filename_blue, 'rb'))
    loaded_model_g = pickle.load(open(filename_green, 'rb'))
    loaded_model_r = pickle.load(open(filename_red, 'rb'))

    test_predictions_b = loaded_model_b.predict(data_X_test)
    test_predictions_g = loaded_model_g.predict(data_X_test)
    test_predictions_r = loaded_model_r.predict(data_X_test)
    
    new_color = Colorizer(red_list=test_predictions_r, 
                            blue_list=test_predictions_b, 
                            green_list=test_predictions_g, 
                            image_location=test_image, 
                            result_size=300,
                            epochs=loaded_model_b.epochs,
                            f_s=f_s
                            )
    new_color.extract_pixels()
    new_color.create_image_from_array()
    print("Done Colorization")
    
