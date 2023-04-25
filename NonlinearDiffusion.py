import cv2
import numpy as np
import os
import sys
from tkinter import filedialog
from tkinter import *
import imageio
from PIL import Image, ImageTk
import random

def nonlinear_diffusion_filter(img, filter_size, n_iter, time_step=0.01):
    img = img.astype(np.float32)
    sobel_x = get_sobel_x(filter_size)
    sobel_y = get_sobel_y(filter_size)
    for i in range(n_iter):
        # Compute gradients in x and y directions
        grad_x = cv2.filter2D(src=img, ddepth=-1, kernel=sobel_x)
        grad_y = cv2.filter2D(src=img, ddepth=-1, kernel=sobel_y)
        kappa = 50;
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        diffusivity = 1 / (1 + (grad_magnitude/kappa)**2)
        
        img = img + time_step * (cv2.filter2D(grad_x * diffusivity, ddepth=-1, kernel=sobel_x) + cv2.filter2D(grad_y * diffusivity, ddepth=-1, kernel=sobel_y))
        img = np.clip(img, 0, 255)
        
    return img.astype(np.uint8)

def add_gaussian_noise(img, mean, variance, probability):
    row, col = img.shape
    number_of_pixels = row * col
    
    noisy_image = img.copy().astype(np.float32)
    
    for i in range(int(number_of_pixels * probability)):
        y = random.randint(0, row - 1)
        x = random.randint(0, col - 1)
        noise = np.random.normal(mean, variance)
        noisy_image[y, x] = np.clip(noisy_image[y, x] + noise, 0, 255)
        
    return noisy_image.astype(np.uint8)

## Since OpenCV does not provide 15x15 Sobel Filter for gradient calculation of Nonlinear Diffusion,
## I've implemented my own.
## Source: 
# Expansion and Implementation of a 3x3 Sobel and Prewitt Edge
# Detection Filter to a 5x5 Dimension Filter
# M.Sc. Rana Abdul Rahman Lateef
# Baghdad College of Economic Sciences University
# URL: https://www.iasj.net/iasj/download/3e89a315575ed098

def get_sobel_x(size_n):
    #Will be used to calculate x gradient
    custom_sobel = np.ndarray((size_n,size_n))
    for i in range(size_n):
        for j in range(size_n):
            if j != size_n//2:
                custom_sobel[i,j] = (j-size_n//2)/((i-size_n//2)**2 + (j-size_n//2)**2)
            else:
                custom_sobel[i,j] = 0
    return custom_sobel

def get_sobel_y(size_n):
    #Will be used to calculate y gradient
    custom_sobel = np.ndarray((size_n,size_n))
    for i in range(size_n):
        for j in range(size_n):
            if i != size_n//2:
                custom_sobel[i,j] = (i-size_n//2)/((i-size_n//2)**2 + (j-size_n//2)**2)
            else:
                custom_sobel[i,j] = 0
    return -1*custom_sobel

### GUI PART ###

def load_image():
    global image, image_path
    file_path = filedialog.askopenfilename()
    image = imageio.imread(file_path)
    image = np.array(image)[:, :, ::-1]  # Convert from RGB to BGR for OpenCV compatibility
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_path.set(file_path)

    mean = float(gaussian_mean_entry.get())
    variance = float(gaussian_variance_entry.get())
    probability = float(gaussian_probability_entry.get())

    image = add_gaussian_noise(image, mean, variance, probability)
    display_image(image, f"Original Image (Gray Scaled)\nWith Gaussian Noise Added\nMean: {mean}, Var: {variance}, Prob: {probability}", canvas1)

    
def apply_filter():
    filter_size = int(diameter_entry.get())
    n_iter = int(iter_count_entry.get())
    time_step = float(time_step_entry.get())
    #filtered_image = cv2.bilateralFilter(image, filter_size, sigma_color, sigma_space)
    filtered_image = nonlinear_diffusion_filter(image, filter_size, n_iter , time_step)
    display_image(filtered_image, "Filtered Image (Non-linear Diffusion Filter)\nFilter Size:"+str(filter_size) +
                  "; Time Step:"+str(time_step) + "; Iteration:"+str(n_iter), canvas2)

def display_image(img, title, canvas):
    if len(img.shape) == 2:  # Check if the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB for display
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_pil.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk
    canvas.create_text(200, 450, text=title, font=("Helvetica", 16))
    


root = Tk()
root.title("Smoother GUI")

image_path = StringVar()

# Create UI elements
frame1 = Frame(root)
frame1.pack(side=TOP)

gaussian_frame = Frame(frame1, width=200, height=200)

load_button = Button(frame1, text="Load Image", command=load_image)
load_button.pack(side=LEFT, padx=0)

gaussian_label = Label(frame1, text="Gaussian Noise:")
gaussian_label.pack(side=LEFT, padx=5)

gaussian_mean_label = Label(frame1, text="Mean:")
gaussian_mean_label.pack(side=LEFT, padx=5)
gaussian_mean_entry = Entry(frame1, width=5)
gaussian_mean_entry.insert(0, "128")
gaussian_mean_entry.pack(side=LEFT, padx=5)

gaussian_variance_label = Label(frame1, text="Variance:")
gaussian_variance_label.pack(side=LEFT, padx=5)
gaussian_variance_entry = Entry(frame1, width=5)
gaussian_variance_entry.insert(0, "10")
gaussian_variance_entry.pack(side=LEFT, padx=5)

gaussian_probability_label = Label(frame1, text="Probability:")
gaussian_probability_label.pack(side=LEFT, padx=5)
gaussian_probability_entry = Entry(frame1, width=5)
gaussian_probability_entry.insert(0, "0.3")
gaussian_probability_entry.pack(side=LEFT, padx=5)

diameter_label = Label(frame1, text="Filter Size:")
diameter_label.pack(side=LEFT, padx=5)
diameter_entry = Entry(frame1, width=5)
diameter_entry.insert(0, "5")
diameter_entry.pack(side=LEFT, padx=5)

iter_count_label = Label(frame1, text="Iteration Count:")
iter_count_label.pack(side=LEFT, padx=5)
iter_count_entry = Entry(frame1, width=5)
iter_count_entry.insert(0, "30")
iter_count_entry.pack(side=LEFT, padx=5)

time_step_label = Label(frame1, text="Time Step:")
time_step_label.pack(side=LEFT, padx=5)
time_step_entry = Entry(frame1, width=5)
time_step_entry.insert(0, "0.01")
time_step_entry.pack(side=LEFT, padx=5)

apply_button = Button(frame1, text="Apply Nonlinear Diffusion Filter", command=apply_filter)
apply_button.pack(side=LEFT, padx=10)



frame2 = Frame(root)
frame2.pack(side=TOP, pady=10)

canvas1 = Canvas(frame2, width=500, height=500)
canvas1.pack(side=LEFT, padx=10)

canvas2 = Canvas(frame2, width=500, height=500)
canvas2.pack(side=LEFT, padx=10)

root.mainloop()
