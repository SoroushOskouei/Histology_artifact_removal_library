import cv2
import numpy as np
from matplotlib import pyplot as plt

class ColorBackgroundRemover:
    def __init__(self, radius=50, threshold=0.1, amount=1):
        self.radius = radius
        self.threshold = threshold
        self.amount = amount

    def remove_background(self, img):
        # Compute the Fourier Transform of each channel of the image
        fshifts = []
        for i in range(3):
            f = np.fft.fft2(img[:,:,i])
            fshift = np.fft.fftshift(f)
            fshifts.append(fshift)

        # Finding the peak frequency in the Fourier Transform
        rows, cols = img.shape[:2]
        crow, ccol = rows//2, cols//2
        dft_magnitudes = [4 * np.log(np.abs(fshift)) for fshift in fshifts]
        peak_frequencies = [np.unravel_index(np.argmax(dft_magnitude), dft_magnitude.shape) for dft_magnitude in dft_magnitudes]
        cys, cxs = zip(*peak_frequencies)

        # Creating a mask that separates the background from the foreground
        mask = np.zeros((rows, cols), dtype=np.uint8)
        for cy, cx in zip(cys, cxs):
            cv2.circle(mask, (cx, cy), self.radius, 255, -1)
        mask = cv2.dilate(mask, None, iterations=5)

        # Applying the mask
        masked = np.zeros_like(img)
        for i in range(3):
            masked[:,:,i] = cv2.bitwise_and(img[:,:,i], img[:,:,i], mask=mask)



        # Adjusting the saturation
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.where(v <= 255*self.threshold, v*self.amount, v)
        hsv = cv2.merge((h, s, v))
        masked = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        masked[np.where((masked<[50,50,50]).all(axis=-1))] = [255,255,255]

        return masked



class ImageSharpener:
    def __init__(self, radius=3, threshold=10, amount=1):
        self.radius = radius
        self.threshold = threshold
        self.amount = amount



img0 = cv2.imread('/content/drive/MyDrive/FogRemoveDeconv/Foggy2.png')

img = img0[0:1200, 200:2300, :]



remover = ColorBackgroundRemover(radius=700, threshold=0.87, amount=5)
masked = remover.remove_background(img)


# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
axs[1].set_title('Processed Image')

plt.show()
