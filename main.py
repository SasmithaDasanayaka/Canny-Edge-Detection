# libraries only for reading and saving images
import numpy as np
import cv2
from pathlib import Path
import glob
import os
import math

def read_all_images():
    paths = []
    image_names = []
    for image_name in glob.glob("*.jpg")+glob.glob("*.jpeg"):
        path = os.path.join(Path().absolute(),image_name)
        paths.append(path)
        image_names.append(image_name.split('.'))

    raw_images = []
#     read numpy arrays
    for path in paths:
        image = cv2.imread(path,0)
        raw_images.append(image)
        
#     convert to python list 
    images = []
    for image in raw_images:
        image = image.tolist()
        images.append(image)
    return images,image_names

#save files
def write_image(image, image_name):
    name = image_name[0]+'_canny'+'.'+image_name[1]
    cv2.imwrite(name,np.asarray(image))
	
#wrap image to avoid dimensionality reduction
def wrap_image(images):
    for image in images:
        image.insert(0,image[-1]+[])
        image.append(image[1]+[])
        for k in range(len(image)):
            image[k].insert(0,image[k][-1])
            image[k].append(image[k][1])
    return images

#return gaussian kernel
def get_gaussian_filter(kernel_size = 5, sigma = 1.4):
    g_filter = [[0.0]*kernel_size for i in range(kernel_size)]

    height = kernel_size // 2
    width = kernel_size // 2

    for x in range(-height, height + 1):
        for y in range(-width, width + 1):
            x1 = 2 * math.pi * (sigma ** 2)
            x2 = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            g_filter[x + height][ y + width] = (1 / x1) * x2
    return g_filter

# gaussian filtering
def gaussian_filter_process(image, kernel):
    height = len(image)
    width = len(image[0])
    kernel_size = len(kernel)
    
    result = []   
    middle = kernel_size // 2
    for y in range(middle, (height-middle)):
        result_row = []
        for x in range(middle, (width-middle)):
            value = 0
            for a in range(kernel_size):
                for b in range(kernel_size):
                    tem_value = kernel[a][b]*image[y-middle+a][x-middle+b]
                    value += tem_value
            result_row.append(value)
        result.append(result_row)
    return result          

# sobel gradient finding
def sobel(image):
    kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    
    height = len(image)
    width = len(image[0])
    kernel_size = len(kernel_x)
    
    gx, gy = [], []
    middle = kernel_size // 2
    for y in range(middle, (height-middle)):
        gx_row = []
        gy_row = []
        for x in range(middle, (width-middle)):
            value_x = 0
            value_y = 0
            for a in range(kernel_size):
                for b in range(kernel_size):
                    tem_value_x = kernel_x[a][b]*image[y-middle+a][x-middle+b]
                    tem_value_y = kernel_y[a][b]*image[y-middle+a][x-middle+b]
                    value_x += tem_value_x
                    value_y += tem_value_y
                    
            gx_row.append(value_x)
            gy_row.append(value_y)

        gx.append(gx_row)
        gy.append(gy_row)

    return gx, gy

# calculate gradients and angles
def gradient_cal(gx, gy, epsilon = 0.0000000000001):
    new_grad = []
    new_theta = []
    for row in range (len(gx)):
        new_grad_row = []
        new_theta_row = []
        for col in range (len(gx[row])):
            grad = math.sqrt((gx[row][col]**2)+(gy[row][col]**2))
            theta = math.degrees(math.atan((gy[row][col]*1.0)/(gx[row][col]+epsilon)))
            new_grad_row.append(grad)
            new_theta_row.append(theta)
        new_grad.append(new_grad_row)
        new_theta.append(new_theta_row)
    return new_grad, new_theta

# calculate max gradient to get the thresholds
def get_max_grad(grad_image):
    maximum = 0
    for row in grad_image:
        if(max(row)>=maximum):
            maximum = max(row)
    return maximum
                       
# executable main func
def main():
    images, image_names = read_all_images()
    images = wrap_image(images)
    g_filter = get_gaussian_filter()
    index = 0
    for image in images:
        image_height = len(image)
        image_width = len(image[0])
        
        filtered_image = gaussian_filter_process(image, g_filter)
        gx, gy = sobel(filtered_image)
        new_grad, new_theta = gradient_cal(gx, gy)
        new_grad_max = get_max_grad(new_grad)
        
        low_th = new_grad_max * 0.1
        high_th = new_grad_max * 0.5
        
        final_output = [[0.0]*image_width for i in range(image_height)]
        
        # Looping through every pixel of the grayscale
        # image
        for i in range(1,image_height-1):
            for j in range(1,image_width-1):
                try:

                    l = 255
                    r = 255
                    
#                     get angle 
                    grad_ang = new_theta[i][j]
                    grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)

                    if (0 <= grad_ang < 22.5) or (157.5 <= grad_ang <= 180):
                        l = new_grad[i][j+1]
                        r = new_grad[i][j-1]
                        
                    elif (22.5 <= grad_ang < 67.5):
                        l = new_grad[i+1][j-1]
                        r = new_grad[i-1][j+1]
                        
                    elif (67.5 <= grad_ang < 112.5):
                        l = new_grad[i+1][j]
                        r = new_grad[i-1][j]
                        
                    elif (112.5 <= grad_ang < 157.5):
                        l = new_grad[i-1][j-1]
                        r = new_grad[i+1][j+1]

#                     non max suppressing step
                    if (new_grad[i][j] >= l) and (new_grad[i][j] >= r):
                        final_output[i][j] = new_grad[i][j]
                    else:
                        final_output[i][j] = 0
                        
                except IndexError as e:
                    pass
                
        weak_ids =  [[0.0]*image_width for i in range(image_height)]
        strong_ids = [[0.0]*image_width for i in range(image_height)]            
        ids = [[0.0]*image_width for i in range(image_height)]

        # double thresholding step
        for i in range(image_width):
            for j in range(image_height):
                try:
                    grad_mag = final_output[j][i]

                    if grad_mag<low_th:
                        final_output[j][i]= 0
                    elif high_th>grad_mag>= low_th:
                        ids[j][i]= 1
                    else:
                        ids[j][i]= 2
                except IndexError as e:
                    pass
        
#         save image
        write_image(final_output, image_names[index])
        
        index+=1

main()