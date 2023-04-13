import os
import cv2
import numpy as np
from skimage import io
# path of images
dir_path = r'C:\Users\Eswar' + '\'' + 's Dell\Downloads\Cancerous cell smears 2023\Cancerous cell smears'
# create two empty list to storea images and the labels
images = []
labels = {'1': 'cyl', '2': 'inter', '3': 'let', '4': 'mod', '5': 'para', '6': 'super', '7': 'svar'}
image_labels = []
# loop through all files in the directory
for fName in os.listdir(dir_path):
    # check if file is an image (we assume all BMP files are images)
    if fName.endswith('.BMP'):
        for key, value in labels.items():
            if value in fName:
                label = key
                break     
        # construct full file path
        f_path = os.path.join(dir_path, fName)
        # load image
        img = cv2.imread(f_path) 
        images.append(img)
        image_labels.append(label)



def calculate_histogram(image):
    # Initialize an array of zeros with 256 elements to represent the histogram
    histogram = np.zeros((256,), dtype=int)
    # Iterate over each pixel in the image and increment the corresponding histogram bin
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] += 1
    # Return the histogram array
    return histogram

def convert_to_single_color(image, color='red'):
    # Check if the specified color is valid
    if color not in ['red', 'green', 'blue']:
        raise ValueError("Invalid color specified. Must be 'red', 'green', or 'blue'.")
    # Create a grayscale image by extracting the specified color channel from the input image
    gray = np.zeros_like(image[:, :, 0])   
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if color == 'red':
                gray[i, j] = image[i, j, 0]
            elif color == 'green':
                gray[i, j] = image[i, j, 1]
            else:
                gray[i, j] = image[i, j, 2]
    # Return the grayscale image
    return gray

def histogram_equalization(image):
    # Compute the histogram of the input image
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    # Compute the cumulative distribution function of the histogram
    distr = hist.cumsum()
    # Normalize the cumulative distribution function to the range [0, 255]
    intensity = np.ma.masked_equal(distr * hist.max() / distr.max(),0)
    intensity = (intensity - intensity.min())*255/(intensity.max()-intensity.min())
    distr = np.ma.filled(intensity,0).astype('uint8')
    # Apply the histogram equalization transformation to the input image using the computed mapping
    return distr[image]


def apply_linear_filter(image, mask, weights):
    # Get the image's dimensions.
    h, w = image.shape[:2]
    # Make a blank output image.
    result = np.zeros_like(image)
    weights = np.reshape(weights, (mask, mask))
    # Each pixel in the image should receive the linear filter.
    for i in range(mask // 2, h - mask // 2):
        for j in range(mask // 2, w - mask // 2):
            # Remove the mask from the source image.
            maska = image[i - mask // 2:i + mask // 2 + 1, j - mask // 2:j + mask // 2 + 1]
            # Calculate the mask's weighted sum of the pixels.
            sum = np.sum(maska * weights)
            # Keep the outcome in the final image.
            result[i, j] = sum
    # Bring up the result image
    return result

def median_filter(image, mask_size):
    # Get the dimensions of the input image
    r, c = image.shape
    # Pad the input image with reflection of the border pixels
    pad = np.pad(image, mask_size // 2, mode='reflect')
    # Create an empty output image of the same shape as the input image
    filterI = np.zeros_like(image)   
    # Iterate over each pixel in the original image
    for i in range(r):
        for j in range(c):
            # Extract a mask_size x mask_size neighborhood centered on the pixel
            sur = pad[i:i+mask_size, j:j+mask_size].flatten()
            # Sort the neighborhood values and set the pixel value to the median
            filterI[i, j] = np.sort(sur)[len(sur) // 2]
    # Return the filtered image
    return filterI

def salt_and_pepper(img, strength):
    # Get the dimensions of the input image
    height, width = img.shape
    # Create a copy of the input image to add noise to
    noisy_img = img.copy()
    # Set approximately strength/2 fraction of the pixels to black (0)
    noisy_img[np.random.rand(height, width) < strength/2] = 0
    # Set approximately strength/2 fraction of the pixels to white (255)
    noisy_img[np.random.rand(height, width) > 1 - strength/2] = 255
    # Return the noisy image
    return noisy_img

def gaussian_noise(image, mean, standardDeviation):
    # Generate an array of the same shape as the input image by sampling from a normal distribution
    noise = np.random.normal(mean, standardDeviation, size=image.shape)
    # Add the noise to the input image and clip the resulting values to the range [0, 255]
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    # Return the noisy image
    return noisy_image

gray_images = []
histograms = []
equalized_histogram = []
equalized_histograms = []
i = 0
for img in images:
    i = i +1
    gray_img = convert_to_single_color(img)
    gray_images.append(gray_img)
    cv2.imwrite(r'C:\Users\Eswar' + '\'' + 's Dell\Downloads\Cancerous cell smears 2023'+'temp'+ str(i) +'.BMP', gray_img)
    histogram = calculate_histogram(gray_img)
    equalized_histogram = histogram_equalization(gray_img)
    histograms.append(histogram)
    equalized_histograms.append(equalized_histogram)
    print(f'Histogram for image {i}: {histogram}')

with open('histograms.txt', 'w') as f:
    for i, histogram in enumerate(histograms):
        f.write(f'Histogram for image {i+1}: {histogram}\n')

c_histograms = {}
avg_histograms = {}
for label in labels.keys():
    c_histograms[label] = []

for index, histogram in enumerate(histograms):
    label = image_labels[index]
    c_histograms[label].append(histogram)

for label, histograms in c_histograms.items():
    avg_histograms[label] = np.mean(histograms, axis=0)

with open('avg_histograms.txt', 'w') as f:
    for label, histogram in avg_histograms.items():
        f.write(f'Averaged histogram for class {labels[label]}: {histogram}\n')

with open('equal_histograms.txt', 'w') as f:
    for i, equalized_histogram in enumerate(equalized_histograms):
        f.write(f'Histogram for image {i+1}: {equalized_histogram}\n')


for img in images:
    gray_img = convert_to_single_color(img)
    gray_images.append(gray_img)

getSaltNoice = []
getSaltNoices = []
choice = float(input("enter your choice for salt_and_pepper strength "))
for gray_img in gray_images:
    getSaltNoice = salt_and_pepper(gray_img, strength= choice)
    getSaltNoices.append(getSaltNoice)

with open('salt_pepper_Noice.txt', 'w') as f:
    for i, getSaltNoice in enumerate(getSaltNoices):
        f.write(f'salt_pepper noice for image {i+1}: {getSaltNoice}\n')

getMean = float(input("enter your choice of mean value for gaussian_noise"))
getStd = float(input("enter your value of standard deviation for gaussian_noise "))

getGaussian = []
getGaussians = []
for gray_img in gray_images:
    getGaussian = gaussian_noise(gray_img, mean= getMean, standardDeviation= getStd)
    getGaussians.append(getGaussian)

with open('gaussian_noise.txt', 'w') as f:
    for i, getGaussian in enumerate(getGaussians):
        f.write(f'gaussian_noise noice for image {i+1}: {getGaussian}\n')

pic =  np.random.rand(100,100)
mask_size = int(input("enter the mask size for linear filter"))
weights = []

for x in range(mask_size*mask_size):
    weight = int(input("enter " + str(mask_size*mask_size) + " weights: "))
    weights.append(weight)

getLinearFilterImg = []
getLinearFilterImgs = []
for gray_img in gray_images:
    getLinearFilterImg = apply_linear_filter(gray_img, mask = mask_size, weights= weights)
    getLinearFilterImgs.append(getLinearFilterImg)

with open('apply_linear_filter.txt', 'w') as f:
    for i, getLinearFilterImg in enumerate(getLinearFilterImgs):
        f.write(f'apply_linear_filter filter for image {i+1}: {getLinearFilterImg}\n')


getMedianFilterImg = []
getMedianFilterImgs = []
mask_size = int(input("enter the mask size for median filter"))
for gray_img in gray_images:
    getMedianFilterImg = median_filter(gray_img, mask_size = mask_size)
    getMedianFilterImgs.append(getMedianFilterImg)

with open('median_filter.txt', 'w') as f:
    for i, getMedianFilterImg in enumerate(getMedianFilterImgs):
        f.write(f'median_filter filter for image {i+1}: {getMedianFilterImg}\n')

def sobel_edge(img):
    # Convert image to grayscale for shape greater than 3
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img
    # reducing noise by applying gaussian noise
    blur = gaussian_noise(gray_img, mean= 2, standardDeviation= 2)

    # finding sobel derivatives in x and y directions and finding magnitude
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # finding threshold magnitude to convert into binary image
    finalImage = np.zeros_like(magnitude)
    finalImage[(magnitude >= np.percentile(magnitude, 50)) & (magnitude <= np.percentile(magnitude, 90))] = 1

    return finalImage

getSobelDetectedImage = []
getSobelDetectedImages = []

for gray_img in gray_images:
    getSobelDetectedImage = sobel_edge(gray_img)
    getSobelDetectedImages.append(getSobelDetectedImage)


with open('sobel_Edge.txt', 'w') as f:
    for i, getSobelDetectedImage in enumerate(getSobelDetectedImages):
        f.write(f'sobel filter for image {i+1}: {getSobelDetectedImage}\n')
        print(f'sobel filter for image {i+1}: {getSobelDetectedImage}\n')


def dilation(img, kernel):
    pad = np.pad(img, (kernel.shape[0] // 2), mode='constant', constant_values=0)
    final_Image = np.zeros_like(img) # creating output image of the same size as the padded image

    for i in range(kernel.shape[0] // 2,(img.shape[0]+(kernel.shape[0] // 2)) ):
        for j in range(kernel.shape[0] // 2, (img.shape[0]+(kernel.shape[0] // 2))):
            kernel_centered = pad[i-(kernel.shape[0] // 2):i+(kernel.shape[0] // 2)+1, j-(kernel.shape[0] // 2):j+(kernel.shape[0] // 2)+1]
            final_Image[i-(kernel.shape[0] // 2), j-(kernel.shape[0] // 2)] = np.max(kernel_centered * kernel)
            print(i)
    return final_Image


def erosion(img, kernel):
    img_padded = np.pad(img, (kernel.shape[0] // 2), mode='constant', constant_values=1)
    final_Image = np.zeros_like(img)

    for i in range((kernel.shape[0] // 2), img.shape[0]+(kernel.shape[0] // 2)):
        for j in range((kernel.shape[0] // 2), img.shape[1]+(kernel.shape[0] // 2)):
            kernel_centered = img_padded[i-(kernel.shape[0] // 2):i+(kernel.shape[0] // 2)+1, j-(kernel.shape[0] // 2):j+(kernel.shape[0] // 2)+1]
            final_Image[i-(kernel.shape[0] // 2), j-(kernel.shape[0] // 2)] = np.min(kernel_centered * kernel)
            print(i)

    return final_Image


getDilutedImage = []
getDilutedImages = []
kernel = np.ones((5,5), np.uint8)
for gray_img in gray_images:
    getDilutedImage = dilation(gray_img,kernel=kernel)
    getDilutedImages.append(getDilutedImage)

with open('dilation.txt', 'w') as f:
    for i, getDilutedImage in enumerate(getDilutedImages):
        f.write(f'dilation filter for image {i+1}: {getDilutedImage}\n')
        print(f'dilation filter for image {i+1}: {getDilutedImage}\n')


geterosionImage = []
geterosionImages = []

for gray_img in gray_images:
    geterosionImage = erosion(gray_img,kernel=kernel)
    geterosionImages.append(geterosionImage)

with open('erosion.txt', 'w') as f:
    for i, geterosionImage in enumerate(geterosionImages):
        f.write(f'erosion filter for image {i+1}: {geterosionImage}\n')
        print(f'erosion filter for image {i+1}: {geterosionImage}\n')




# getKmeansImages = []
# for i,gray_img in enumerate(gray_images):
#     getKmeansImage = kmeans_segmentation(gray_img,2,i+1)
#     getKmeansImages.append(getKmeansImage)

# with open('kmeans.txt', 'w') as f:
#     for i, getKmeansImage in enumerate(getKmeansImages):
#         f.write(f'kmeans clustering filter for image {i+1}: {getKmeansImage}\n')
#         print(f'kmeans clustering for image {i+1}: {getKmeansImage}\n')

