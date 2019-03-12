import numpy as np
import cv2
from matplotlib import pyplot as plt


#img = cv2.imread('dogs.jpg')
img = cv2.imread('dogs3.jpg')
gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

x,y,z = img.shape

def fftforimage(magnitude_spectrum):
    manipulated = magnitude_spectrum - np.min(magnitude_spectrum)
    manipulated = (magnitude_spectrum/np.max(magnitude_spectrum))*255
    manipulateduint = manipulated.astype('uint8')
    
    return manipulateduint

def fftcorrection(gs):
    dft = cv2.dft(np.float32(gs),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_end = dft_shift
    valuecut = np.max(dft_end)*(0.05/100)
    dft_end[np.logical_and(dft_end>-valuecut, dft_end<valuecut)] = 0
    rows, cols = gs.shape
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    img_back = fftforimage(img_back)
    return img_back

B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

reconstruct = np.zeros([x,y,z])
reconstruct[:,:,0] = fftcorrection(B)
reconstruct[:,:,1] = fftcorrection(G)
reconstruct[:,:,2] = fftcorrection(R)

reconstruct = reconstruct.astype('uint8')

trying = fftcorrection(G)

show_imagename = ['Original Image',
                  'GrayScale',
                  'Blue',
                  'Green',
                  'Red',
                  'trying',
                  'reconsruct'
                  ]
show_image = [img,
              gs,
              B,
              G,
              R,
              trying,
              reconstruct
              ]


#
#







#
#
#
#
#dft = cv2.dft(np.float32(gs),flags = cv2.DFT_COMPLEX_OUTPUT)
#dft_shift = np.fft.fftshift(dft)
#magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#
#dft_end = dft_shift
#valuecut = np.max(dft_end)*(0.05/100)
#dft_end[np.logical_and(dft_end>-valuecut, dft_end<valuecut)] = 0
#
#rows, cols = gs.shape
#crow,ccol = rows/2 , cols/2
#
#
#f_ishift = np.fft.ifftshift(dft_shift)
#img_back = cv2.idft(f_ishift)
#img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#
#magnitude_spectrum_end = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#
#initial_fft = fftforimage(magnitude_spectrum)
#end_fft = fftforimage(magnitude_spectrum_end)
#img_back = fftforimage(img_back)
#dftending = fftforimage(dft_end)
#
#
#
#
#
##Show Image List
#show_imagename = ['Original Image',
#                  'GrayScale',
#                  'initial_fft',
#                  'end_fft',
#                  'image back',
#                  'trying'
#                  ]
#show_image = [img,
#              gs,
#              initial_fft,
#              end_fft,
#              img_back,
#              trying
#              ]
##
#
#
#
#
#
#










n_showimg = len(show_image)



resize = False
rwidth = 700
rheight = 500
#Image Showing Sequencing
for k in range (0,n_showimg):
    if resize == True:
        cv2.namedWindow(show_imagename[k],cv2.WINDOW_NORMAL)    
        cv2.resizeWindow(show_imagename[k],rwidth,rheight)    
    cv2.imshow(show_imagename[k],show_image[k])

# SHOWING IMAGE ALGORITHMS ---------------------------------------------------------------------- END


cv2.waitKey(0)
cv2.destroyAllWindows()
