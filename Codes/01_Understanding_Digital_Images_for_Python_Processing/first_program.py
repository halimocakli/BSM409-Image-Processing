from skimage import io
from skimage import img_as_float
import numpy as np
from matplotlib import pyplot as plt

# Test imajımızı import edelim
test_image_uint8 = io.imread("..\Images\Lenna.png")

# Test imajını yazdıralım, bir numpy dizisi ile karşılaşacağız
print(test_image_uint8)

# Test imajını görüntüleyelim
plt.imshow(test_image_uint8)
plt.show()

# Rastgele imaj yaratalım
random_image = np.random.random([500, 500])

# Yarattığımız rastgele imajı görüntüleyelim
plt.imshow(random_image)
plt.show()

# random_image imajını yazdıralım
print(random_image)

# random_image minimum ve maksimum değerlerini görelim
print(f"Max Value of random image is {random_image.max()}\
      \nMin Value of random image is {random_image.min()}")
      
# Test imajının minimum ve maksimum değerlerini görelim
print(f"Max Value of test image is {test_image_uint8.max()}\
      \nMin Value of test image is {test_image_uint8.min()}")
      
# uint8 türünde olan test imajını float'a dönüştürelim
test_image_float = img_as_float(test_image_uint8)

# Float test imajını yazdıralım, bir numpy dizisi ile karşılaşacağız
print(test_image_float)

# Float test imajını görüntüleyelim
# uint8 türündeki test imajı ile görüntü bakımından hiçbir farkı yok
plt.imshow(test_image_float)
plt.show()

# Float test imajının minimum ve maksimum değerlerini görelim
print(f"Max Value of test image is {test_image_float.max()}\
      \nMin Value of test image is {test_image_float.min()}")
      
"""
BU NOKTAYA KADAR YAPTIĞIMIZ İŞLEMLER SONUCUNDA ANLIYORUZ Kİ İMAJLAR ASLINDA
BİRER NUMPY DİZİSİNDEN YA DA MATRİS'TEN BAŞKA BİR ŞEY DEĞİL. BUNA GÖRE, İMAJLARI
BİRBİRLERİ İLE ÇARPABİLİR, BİR İMAJI DİĞERİNE EKLEYEBİLİR YA DA DİĞERİDEN ÇIKARABİLİRİZ.
"""

# Float test imajını 0.5 ile çarparak herbir piksel değerini ikiye bölelim
dark_image = test_image_float * 0.5

# dark_image'ı yazdıralım
print(dark_image)

# dark_image'ı görüntüleyelim
# Orijinal imaja göre parlaklığı azaldı
plt.imshow(dark_image)
plt.show()

# dark_image imajının minimum ve maksimum değerlerini görelim
print(f"Max Value of test image is {dark_image.max()}\
      \nMin Value of test image is {dark_image.min()}")
      
      
# uint8 tipindeki test imajının belli bir kısmını kırmızı yapalım
test_image_uint8_red = test_image_uint8
test_image_uint8_red[10:200, 10:200, :] = [255, 0, 0]

# Ürettiğimiz imajı görüntüleyelim
plt.imshow(test_image_uint8_red)
plt.show()

# uint8 tipindeki test imajının belli bir kısmını yeşil yapalım
test_image_uint8_green = test_image_uint8
test_image_uint8_green[10:200, 10:200, :] = [0, 255, 0]

# Ürettiğimiz imajı görüntüleyelim
plt.imshow(test_image_uint8_green)
plt.show()

# uint8 tipindeki test imajının belli bir kısmını mavi yapalım
test_image_uint8_blue = test_image_uint8
test_image_uint8_blue[10:200, 10:200, :] = [0, 0, 255]

# Ürettiğimiz imajı görüntüleyelim
plt.imshow(test_image_uint8_blue)
plt.show()

# uint8 tipindeki test imajının belli bir kısmını beyaz yapalım
test_image_uint8_white = test_image_uint8
test_image_uint8_white[10:200, 10:200, :] = [255, 255, 255]

# Ürettiğimiz imajı görüntüleyelim
plt.imshow(test_image_uint8_white)
plt.show()