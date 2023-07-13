import numpy as np              # Will handle how data is structured
import matplotlib.pyplot as plt # Will be used for visual representation
from PIL import Image

# Part 1: Scale-Space Extrema Detection

###########################
# Guassian Blur
###########################

# Guassian Blur of image
def Guassian_Scale_Space(sigma: float | int, filter_shape: list | tuple | None): # G(x,y,sigma)

     m, n = filter_shape
     m_half = m // 2       # Take the floor of the result (rows)  
     n_half = n // 2       # Take the floor of the result (columns)

     guassian_filter = np.zeros((m,n), np.float32)  # Make a zero matrix based on the dimensions data from fitler shape

     # This is the Guassian Blur equation (Refer to page 2 of the SIFT Research pape)
     for y in range(-m_half, m_half):          # Will take care of the y values
        for x in range(-n_half, n_half):       # Will take care of the x value
            normal = 1 / (2.0 * np.pi * sigma**2.0)                     # 1 / 2 * pi * sigma^2 
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))  # -(x^2 + y^2) / 2(sigma)^2 | NOTE: np.exp() is set to base e in this case
            guassian_filter[y+m_half, x+n_half] = normal * exp_term     # (1 / 2 * pi * sigma^2)^(-(x^2 + y^2) / 2(sigma)^2)

     return guassian_filter # return the filter after caculation was made. 

# Convolution 
def convolution(image: np.ndarray, kernel: list | tuple) -> np.ndarray: # L(x,y,sigma) = G(x,y,sigma) * I(x,y) (convolution) 
    
     if len(image.shape) == 3:             
          m_i, n_i, c_i = image.shape
     elif len(image.shape) == 2:
          image = image[..., np.newaxis]
          m_i, n_i, c_i = image.shape
     else:
          raise Exception('Shape of image not supported')  # Usually for uneven images

     m_k, n_k = kernel.shape

     y_strides = m_i - m_k + 1  # possible number of strides in y direction
     x_strides = n_i - n_k + 1  # possible number of strides in x direction

     img = image.copy()
     output_shape = (m_i-m_k+1, n_i-n_k+1, c_i)
     output = np.zeros(output_shape, dtype=np.float32)

     count = 0  # taking count of the convolution operation happening

     output_tmp = output.reshape(
        (output_shape[0]*output_shape[1], output_shape[2])
     )

     for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[i:i+m_k, j:j+n_k, c]

                output_tmp[count, c] = np.sum(sub_matrix * kernel)

            count += 1

     output = output_tmp.reshape(output_shape)

     return output.astype(np.uint8)

###########################
# Difference of Guassian
###########################

def diffOfGuassian(lower_laplacian: np.ndarray, upper_laplacian: np.ndarray) -> np.ndarray: # D(x,y,sigma)
    output = upper_laplacian - lower_laplacian
    return output.astype(np.uint8)


# main function that can be used for testing
if __name__ == '__main__':
     image = np.array(Image.open('table.jpg')) # I(x,y)

     blurred_images = {5} # Set of Guassian scale spaces at const sigma value
     diff = {5} # set of Laplacian scale spaces at const sigma value
     sigma = 5

     # Guassian of difference for one octave and 5 scale spaces
     for i in range(5):
         blurred_images[sigma].append(Guassian_Scale_Space(sigma, (40, 40)))

     # Get the difference of Guassian between them
     for j in range(4):
         diff[sigma].append(diffOfGuassian(blurred_images[i, sigma], blurred_images[i+1, sigma]))
         

     # plotting Guassian Blur test with single image
     #plt.subplot(122)
     #plt.imshow(image.astype(np.uint8))
     #plt.subplot(122)
     #plt.imshow(blur_image)
     #plt.tight_layout()
     #plt.show()



     
     
     
     