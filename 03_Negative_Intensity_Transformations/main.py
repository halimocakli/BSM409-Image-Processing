import cv2
import matplotlib.pyplot as plt
import numpy as np


# NOT: İmaj boyutu büyükse ve bu imajı görüntülemek istiyorsak, ekrandan taşmaması için Matplotlib'in imshow() metodunu kullanmalıyız.
# Bazı durumlarda Matplotlib'in imshow() metodu daha avantajlıdır. Mesela, plt.imshow() kullanınca açılan ekranda imajın boyutlarını görebiliriz.

def image_negative_builder(image):
    """
    image:param -> Matris formatındaki girdi görüntüsü.
    negative_image:return -> Negatif formattaki çıktı görüntüsü.

    Negatif Dönüşüm Fonlsiyonu : s = L - 1 - r
    L = 255 (8 Bitlik Görüntünün Maksimum Yoğunluk Değeri)
    L - 1 : 8 Bitlik görüntünün maksimum yoğunluk değeri 255 olması beklenir.
            Ancak 2^8 = 256 olduğu için 1 çıkartılır. Python'da buna ihtiyaç yoktur.
    r : Girdi görüntüsü içerisindeki her bir yoğunluk değerlerini tutan 2D Array.
    """
    L = np.max(image)
    negative_image = L - image
    return negative_image


if __name__ == "__main__":
    breast_XRAY = cv2.imread("./Images/breast_digital_Xray.tif")

    # Göğüs mamografisi aslında renkli bir imajdır. Boyutlarına bakarak bunu anlayabiliriz.
    print(f"Shape of Colorful Breast Image: {breast_XRAY.shape}")

    # Kolaylık olması bakımından, üzerinde çalışmadan önce imajı siyah-beyaz olacak şekilde import edelim.
    breast_XRAY = cv2.imread("./Images/breast_digital_Xray.tif", cv2.IMREAD_GRAYSCALE)

    # Yeni görüntümüzün boyutlarını inceleyelim.
    print(f"Shape of Grayscale Breast Image: {breast_XRAY.shape}")

    # Orijinal imajın negatifini oluşturalım.
    negative_breast_XRAY = image_negative_builder(breast_XRAY)

    # Negatif imajı kaydedelim
    plt.imsave("./Outputs/breast_digital_Xray_negative.png", negative_breast_XRAY, cmap="gray")

    breast_XRAY_original_negative_adjoined = np.hstack((breast_XRAY, negative_breast_XRAY))

    plt.imsave("./Outputs/breast_XRAY_original_negative_adjoined.png", breast_XRAY_original_negative_adjoined,
               cmap="gray")

    cv2.imshow("Breast XRAY", breast_XRAY)
    cv2.imshow("Negative Breast XRAY Image", negative_breast_XRAY)
    cv2.imshow("Breast XRAY - NEGATIVE BREAST XRAY", breast_XRAY_original_negative_adjoined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imshow(breast_XRAY, cmap="gray")
    plt.show()
