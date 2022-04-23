
import numpy as np
from PIL import Image
import cv2
import cv2

############################################

def relu(x):
    return (x >= 0) * x

def relu2deriv(x):
    return x >= 1

def set0_1(inp):
    return np.array(list(map(lambda x: x / 255, inp)))

def set255(inp):
    return np.array(list(map(lambda x: x * 255, inp))).astype(int)

############################################


image = cv2.imread('src/hello.png')

image_for_learn = set0_1(image)

alpha = 0.01
iteration = 10 # default: 50
hiden_size = 2

weights_0_1 = np.random.random((3, 2))
print(weights_0_1)


res1 = np.empty(shape=(300, 3), dtype=float)
res2 = np.empty(shape=(168, 300, 3), dtype=float)

         # Train Network #

for j in range(iteration):
    
    for x in range(len(image_for_learn)):

        for y in range(len(image_for_learn[x])):

            in_x = (x + 1) / len(image_for_learn)
            in_y = (y + 1) / len(image_for_learn[x])
            inp = np.array([in_x, in_y])
            out = image_for_learn[x][y] # has value pixel | Example: [0.123, 0.6, 0.23]
            
            res = relu(np.dot(weights_0_1, inp))

            err = np.sum((res - out) ** 2)
            
            #TODO:
            #delta and back propogation algorithm

            layer_delta = alpha * (out - res) #delta pixel | Example: [0.34, 0.234, 0.74]

            for i in range(3):
                weights_0_1[i] += layer_delta[i]
    
    print("____________________________")
    print("Layer_delta: ", layer_delta)
    print("Error: ", err)
    print("Weight: \n", weights_0_1)




# generate image from already train network

for x in range(len(image_for_learn)):

        res2[x] = res1

        for y in range(len(image_for_learn[x])):

            in_x = x / len(image_for_learn)
            in_y = y / len(image_for_learn[x])
            inp = np.array([in_x + 1, in_y + 1])
            out = image_for_learn[x][y] # has value pixel | Example: [0.123, 0.6, 0.23]
            
            res = relu(np.dot(weights_0_1, inp))
            res1[y] = res



############################################

result_image = set255(res2)


result_image = result_image.astype(np.uint8)

cv2.imshow('image window', result_image)

cv2.waitKey()

cv2.destroyAllWindows()






