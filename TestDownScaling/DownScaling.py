import cv2
import numpy as np
from PIL import Image
import random
from scipy.signal import convolve2d

def gaussian_kernel(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    gaussian = k @ k.T
    return gaussian

def apply_motion_blur(image, kernel):
    return convolve2d(image, kernel, mode='same', boundary='wrap')

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def jpeg_compression(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encoded_image, 1)
    return compressed_image

def downscale_image(image, scale_factor):
    height, width = image.shape[:2]
    new_dimensions = (int(width // scale_factor), int(height // scale_factor))
    downscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    return downscaled_image

def degradation_model(image_path, gaussian_blur_sigma=1.5, motion_blur_kernels=None, scale_factor=4, noise_sigma=10, jpeg_quality=30):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if random.random() > 0.5:
        kernel_size = int(2 * round(3 * gaussian_blur_sigma) + 1)
        kernel = gaussian_kernel(kernel_size, gaussian_blur_sigma)
        blurred_image = cv2.filter2D(image, -1, kernel)
    else:
        kernel = random.choice(motion_blur_kernels)
        blurred_image = apply_motion_blur(image, kernel)
    
    downscaled_image = downscale_image(blurred_image, scale_factor)
    
    noisy_image = add_gaussian_noise(downscaled_image, noise_sigma)
    
    degraded_image = jpeg_compression(noisy_image, jpeg_quality)
    
    return degraded_image

motion_blur_kernels = [np.eye(9), np.fliplr(np.eye(9))]

image_path = "9.png"
degraded_image = degradation_model(image_path, gaussian_blur_sigma=random.uniform(1, 3),
                                   motion_blur_kernels=motion_blur_kernels, scale_factor=random.uniform(1, 8),
                                   noise_sigma=random.uniform(0, 15), jpeg_quality=random.randint(10, 60))

cv2.imwrite("low_quality_9.png", cv2.cvtColor(degraded_image, cv2.COLOR_RGB2BGR))
