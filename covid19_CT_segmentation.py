import numpy as np
import cv2
from skimage import img_as_ubyte
import time

img = cv2.imread('archive/CT_Snapshot/130.png')

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (85,205,335,225)

start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(img, mask, rect, bgdModel,
	fgdModel,5, mode=cv2.GC_INIT_WITH_RECT)
end = time.time()

print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))
values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)
for (name, value) in values:
	# construct a mask that for the current value
	print("[INFO] showing mask for '{}'".format(name))
	valueMask = (mask == value).astype("uint8") * 255
	# display the mask so we can visualize it
	cv2.imshow(name, valueMask)
	cv2.waitKey(0)

#cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
res = img*mask2[:,:,np.newaxis]

gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
blurred  = cv2.GaussianBlur(gray, (11, 11), 0)
fi = blurred / 255.0
gamma = 0.7
out = np.power(fi, gamma)
img_8bit = img_as_ubyte(out)
print(img_8bit.dtype)
print(img_8bit.ndim) 
print(img_8bit.shape)

ret, th1 = cv2.threshold(img_8bit, 70, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
img2 = img.copy()
index = -1
thickness = 1
color = (0,30,255)
#cv2.drawContours(img2,contours,index,color,thickness)

print ("contours 數量：",len(contours))
list=[]
n = 0;
for idx,cnt in enumerate(contours):
    print("Contour No.%d"%(idx))
    #    print (cnt)
#    perimeter = cv2.arcLength(cnt, True) # 計算周長
#    print(f'輪廓周長為:{perimeter}')
    area = cv2.contourArea(cnt) # 計算面積
    print(f'輪廓面積為:{area}')
    print ("contours[n]點的個數：",len(contours[n]))
    if area >= 200 and area <= 5000:
        list.append(idx)
    print(list)
    
for i in list:
    cv2.drawContours(img2,contours,i,color,-1)

cv2.imshow('Contours',img2)
#cv2.imwrite('PPT_Image/image.jpg', img)
#cv2.imwrite('PPT_Image/gamma.jpg', img_8bit)
cv2.waitKey(0)