from mnist import read, show, normalize
import numpy as np
import matplotlib.pyplot as plt

train = list(read('train'))
data = train[0]

label = data[0]
pixels = data[1:]

pixels = np.array(pixels, dtype='uint8')
pixels = pixels.reshape((28, 28))


plt.title('Example of MNIST pattern')
plt.imshow(pixels, cmap='gray')
plt.show()