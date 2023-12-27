import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    img_fft = np.fft.fft2(img)  # pretvaranje slike u frekventni domen
    img_fft = np.fft.fftshift(img_fft) # pomeranje u centar
    return img_fft

def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft)) #  iz frekventnog u prostorni domen
                                                 # nasa slika je moduo np.abs()

    return img_filtered


def low_pass_filter(img, center, radius):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > radius*radius: #sve izvan kruga nula
                img_fft_log[x,y] = 0

    #plt.imshow(img_fft_log)
    #plt.show()

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    return img_filtered

img = cv2.imread("slika_1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

center = (256, 256)
radius = 50


img_low_pass = low_pass_filter(img, center, radius)
plt.imshow(img_low_pass, cmap='gray')
plt.show()
