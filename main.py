import numpy as np
import cv2 as cv

img = cv.imread('D:\\Dados\\larissa\\imagens\\tyre_data\\IMG_1636.JPG', 0)
tmp = cv.imread('./images/unwrap-corte2.png', 0)
"""
preciso fazer a copia pois vamos desenhar retangulos
"""

width = 500
height = 300
dim = (width, height)

img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
H, W = img.shape
h, w = tmp.shape

methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]

for method in methods:
    im2 = img.copy()

    result = cv.matchTemplate(im2, tmp, method)
    (W - w + 1, H - h + 1)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    print(min_loc, max_loc)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    bottom_right = (location[0] + w, location[1] + h)
    cv.rectangle(im2, location, bottom_right, 255, 5)
    cv.imshow('Match', im2)
    cv.waitKey(0)
    cv.destroyAllWindows()    



