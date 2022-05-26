
import numpy as np
from PIL import Image
import cv2

############################################

def relu(x):
    return np.maximum(0, x)

relu2deriv = lambda x: x >= 0

def set0_1(inp):

    return np.array(list(map(lambda x: x / 255, inp)))

def set255(inp):
    return np.array(list(map(lambda x: x * 255, inp))).astype(int)

############################################

image = cv2.imread("src/5.jpg")

image_for_learn = set0_1(image)

height = len(image_for_learn)
width = len(image_for_learn[0])

alpha = 0.00001
iteration = 10  # default: 3

weights_0_1 = np.random.random((3, 2))

weights_1_2 = np.random.random((3, 3))

res1 = np.empty(shape=(width, 3), dtype=float)
res2 = np.empty(shape=(height, width, 3), dtype=float)

err = 0

         # Train Network #

for j in range(iteration):
    
    for x in range(height):

        for y in range(width):

            in_x = x / (height + 1)
            in_y = y / (width + 1)
            
            out = image_for_learn[x][y] # has value pixel | Example: [0.123, 0.6, 0.23]

            res_layer0 = np.array([in_x, in_y])
            
            res_layer1 = relu(weights_0_1 @ res_layer0)

            res_layer2 = relu(weights_1_2 @ res_layer1)

            err += np.sum((res_layer2 - out) ** 2)

            layer_2_delta = (out - res_layer2) #delta pixel (right - now) | Example: [0.34, 0.234, 0.74]
            
            layer_1_delta = layer_2_delta @ (weights_1_2 * relu2deriv(res_layer1)) # weights_1_2.T

            ###############    TRAIN    !!!!!!

            weights_1_2 += alpha * res_layer1.T.dot(layer_2_delta)
            #weights_0_1 += alpha * res_layer0.dot(layer_1_delta) # many problem with it
            for i in range(3):
                weights_0_1[i] += alpha * layer_1_delta[i]
                
    # some needing information
    print("____________________________")
    print("Layer_delta1: ", layer_1_delta)
    print("Error: ", err / y)
    print("Weight: \n", weights_0_1)
    err = 0

print("obuchenije gotovo")

for x in range(height):

        res2[x] = res1

        for y in range(width):

            in_x = x / (height + 1)
            in_y = y / (width + 1)
            
            res_layer0 = np.array([in_x, in_y])
            
            res_layer1 = relu(weights_0_1 @ res_layer0)

            res_layer2 = relu(weights_1_2 @ res_layer1)

            res1[y] = res_layer2



############################################

result_image = set255(res2)


result_image = result_image.astype(np.uint8)

cv2.imshow('image window', result_image)

cv2.waitKey()

cv2.destroyAllWindows()

