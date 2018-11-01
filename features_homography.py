import numpy as np
import cv2
import random
imgs=[]
imgs_grey=[]
imgs_kp=[]
imgs_res=[]

def warpTwoImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result


for i in range(1,3):
    img=cv2.imread('mountain'+str(i)+'.jpg')
    imgs.append(img)
    img_grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs_grey.append(img_grey)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points, res = sift.detectAndCompute(img_grey,None)
    imgs_kp.append(key_points)
    imgs_res.append(res)
    final = cv2.drawKeypoints(img,key_points,None)
    cv2.imwrite('task1_sift'+str(i)+'.jpg',final)

matcher = cv2.BFMatcher()
count=0
best=[]
good=[]
matches=matcher.knnMatch(imgs_res[0], imgs_res[1], 2)
for m,n in matches:
    if m.distance < 0.75*n.distance:
        best.append([m])
        good.append(m)
res_msift=cv2.drawMatchesKnn(imgs[0],imgs_kp[0],imgs[1],imgs_kp[1], random.sample(best, 10), None,flags=2)
cv2.imwrite('task1_matches_knn.jpg', res_msift)
src_pts = np.float32([ imgs_kp[1][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ imgs_kp[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
print('Homography matrix H', H)
matchesMask = mask.ravel().tolist()

draw_params = dict(matchColor = (0,0,255),
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(imgs[0],imgs_kp[0],imgs[1],imgs_kp[1],good,None,**draw_params)
cv2.imwrite('task1_matches.jpg',img3)

#wrap_img = cv2.warpPerspective(imgs[1], H, (imgs[0].shape[1]+imgs[1].shape[1],imgs[0].shape[0]))
cv2.imwrite('task1_pano.jpg', warpTwoImages(imgs[0], imgs[1], H))
