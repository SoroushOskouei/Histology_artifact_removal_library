# Reading the foggy image
img0 = cv2.imread('/Foggy.png')

img = img0[200:1200, 1300:2300, :]


class ImageSharpener:
    def __init__(self, radius=3, threshold=10, amount=1):
        self.radius = radius
        self.threshold = threshold
        self.amount = amount

    def sharpen_image(self, img):
        blurred = cv2.GaussianBlur(img, (self.radius, self.radius), 0)
        sharpened = np.float32(img) + self.amount * (np.float32(img) - np.float32(blurred))
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        mask = cv2.cvtColor(cv2.absdiff(img, sharpened), cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, self.threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.merge([mask, mask, mask])
        not_mask = cv2.bitwise_not(mask)
        white_mask = cv2.merge([not_mask[:,:,0], not_mask[:,:,0], not_mask[:,:,0]])
        sharpened = cv2.bitwise_and(sharpened, mask)
        sharpened = cv2.bitwise_or(sharpened, white_mask)
        return sharpened


# An instance of the ImageSharpener class
sharpener = ImageSharpener(radius=37, threshold=10, amount=30)

# Sharpening the image
sharpened = sharpener.sharpen_image(img)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
axs[1].set_title('Filtered Image')
plt.show()
