import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
imgs=[]
imgs_grey=[]
imgs_kp=[]
imgs_res=[]
np.random.seed(sum([ord(c) for c in 'kishandh']))
colors=[tuple(np.random.randint(0,255,3).tolist()) for i in range(10)]
def drawlines(img1,img2,lines,pts1,pts2):
    r,c,_ = img1.shape
    for i,(r,pt1,pt2) in enumerate(zip(lines,pts1,pts2)):
        color = colors[i]
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1.flatten()),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2.flatten()),5,color,-1)
    return img1,img2
filename=['tsucuba_left.png', 'tsucuba_right.png']
for i in range(1,3):
    img=cv2.imread(filename[i-1])
    imgs.append(img)
    img_grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs_grey.append(img_grey)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points, res = sift.detectAndCompute(img_grey,None)
    imgs_kp.append(key_points)
    imgs_res.append(res)
    final = cv2.drawKeypoints(img,key_points,None)
    cv2.imwrite('task2_sift'+str(i)+'.jpg',final)

matcher = cv2.BFMatcher()
count=0
best=[]
good=[]
matches=matcher.knnMatch(imgs_res[0], imgs_res[1], 2)
for m,n in matches:
    if m.distance < 0.75*n.distance:
        best.append([m])
        good.append(m)

res_msift=cv2.drawMatchesKnn(imgs[0],imgs_kp[0],imgs[1],imgs_kp[1], best, None,flags=2)
cv2.imwrite('task2_matches_knn.jpg', res_msift)
img_left = np.int32([ imgs_kp[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
img_right = np.int32([ imgs_kp[1][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
F, mask = cv2.findFundamentalMat(img_left, img_right, cv2.FM_RANSAC)
print(F)

rand_indx=[np.random.randint(0,len(mask)-1) for i in range(10)]
img_left = img_left[mask.ravel()==1]
img_left=np.array([img_left[i] for i in rand_indx])
img_right = img_right[mask.ravel()==1]
img_right=np.array([img_right[i] for i in rand_indx])
lines1 = cv2.computeCorrespondEpilines(img_right, 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(imgs[0],imgs[1],lines1,img_left,img_right)

lines2 = cv2.computeCorrespondEpilines(img_left, 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(imgs[1],imgs[0],lines2,img_right,img_left)

cv2.imwrite('task2_epi_left.jpg', img5)
cv2.imwrite('task2_epi_right.jpg', img3)

stereo = cv2.StereoBM_create(numDisparities=96, blockSize=31)
disparity = stereo.compute(imgs_grey[0], imgs_grey[1])
norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(norm_image,'gray')
plt.show()
