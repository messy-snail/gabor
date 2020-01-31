import numpy as np
import cv2

theta_arr = np.arange(0, np.pi, np.pi / 16)

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        # kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern = cv2.getGaborKernel((ksize, ksize), 2.0, theta, 20.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

#accum : mag
#aacum2 : ang
def process(img, filters):
    accum = np.zeros_like(img)
    accum2 = np.zeros_like(img)
    accum2 = np.float32(accum2)
    # for kern in filters:
    for idx, kern in enumerate(filters):
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        ck = np.greater(fimg, accum)
        np.maximum(accum, fimg, accum)
        angle=np.full_like(accum2, theta_arr[idx] * 180 / np.pi )
        angle = np.float32(angle)
        ck = np.float32(ck)
        angle = ck*angle
        np.maximum(accum2, angle, accum2)


    return accum, accum2


if __name__ == '__main__':
    import sys

    print(__doc__)
    try:
        img_fn = sys.argv[1]
    except:
        img_fn = 'hair2.png'
        # img_fn = 'Hair-PNG-Transparent-Picture-1.png'

    img_org = cv2.imread(img_fn)
    cv2.imshow('original', img_org)
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(img_org)
    hsv[..., 1] = 255
    if img is None:
        print(
        'Failed to load image file:', img_fn)
        sys.exit(1)

    filters = build_filters()

    res1, hue= process(img, filters)
    hsv[..., 0] = hue
    hsv[...,2] = cv2.normalize(res1,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    colmap = cv2.applyColorMap(res1, cv2.COLORMAP_HSV)
    colmap2 = cv2.applyColorMap(res1, cv2.COLORMAP_JET)
    colmap3 = cv2.applyColorMap(res1, cv2.COLORMAP_SPRING)
    cv2.imshow('result_hue', bgr)
    cv2.imshow('result_colormap', colmap)
    cv2.imshow('result_colormap2', colmap2)
    cv2.imshow('result_colormap3', colmap3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()