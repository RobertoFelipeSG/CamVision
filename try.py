import numpy as np
import cv2
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10

image1 = cv2.imread('1.jpg') # querdyImage
image2 = cv2.imread('2.jpg') # trainImage

img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

fig = plt.figure(1)
plt.subplot(121),plt.imshow(img1)
plt.subplot(122),plt.imshow(img2)
plt.show()

# DEFINE ROI
face_cascade = cv2.CascadeClassifier('/usr/local/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/opencv/data/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(img1, 1.3, 5)
for (x,y,w,h) in faces:
#    cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
    roi1 = img1[y-100:y+h, x-40:x+w]
    roi2 = img2[y-100:y+h, x-40:x+w]
    eyes1 = eye_cascade.detectMultiScale(roi1)
    eyes2 = eye_cascade.detectMultiScale(roi2)
    for (ex,ey,ew,eh) in eyes1:
#        cv2.rectangle(roi1,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        roi3 = roi1[0:ey, 0:x+w]
    for (ex,ey,ew,eh) in eyes2:
#   	cv2.rectangle(roi2,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	roi4 = roi2[0:ey, 0:x+w]
   
fig = plt.figure(2)
plt.subplot(121),plt.imshow(roi3)
plt.subplot(122),plt.imshow(roi4)
plt.show()
 


# FEATURES SEARCH 
# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create()
# Initiate SURF detector
surf = cv2.xfeatures2d.SURF_create()
 
# Find the keypoints and descriptors with SURF/SIFT
kp1, des1 = surf.detectAndCompute(roi3,None)
kp2, des2 = surf.detectAndCompute(roi4,None)

# FLANN parameters = Approxim. Nearest k=2 Neighbors 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)



# FEATURES MATCHING
# Need to draw only good matches, so create a mask
#matchesMask = [[0,0] for i in xrange(len(matches))]
 
# Ratio test as per Lowe's paper
#for i,(m,n) in enumerate(matches):
#    if m.distance < 0.8*n.distance:
#        matchesMask[i]=[1,0]
 
#draw_params = dict(matchColor = (0,255,0),
#                   singlePointColor = (255,0,0),
#                   matchesMask = matchesMask,
#                   flags = 0)
 
#img3 = cv2.drawMatchesKnn(roi1,kp1,roi2,kp2,matches,None,**draw_params)
 
#plt.imshow(img3,),plt.show()



# HOMOGRAPHY
# Refining matches by RANSAC (Filtering out geometrically incorrect matches)
# Store all the good matches as per Lowe's ratio test.
good = []
pts1 = []
pts2 = []
for m,n in matches:
   if m.distance < 0.8*n.distance:
       good.append(m)
       pts2.append(kp2[m.trainIdx].pt)
       pts1.append(kp1[m.queryIdx].pt)

# Find homography matrix and get inliers mask
#if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)  # Data Preparation
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # RANSAC: Random sample consensus
#     matchesMask = mask.ravel().tolist() # Here is the mask
 
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
 
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
 
#else:
#     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#     matchesMask = None

#draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
 
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#plt.imshow(img3, 'gray'),plt.show()



# HOMOGRAPHY REFINEMENT?



# FUNDAMENTAL MATRIX
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
  
print F

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1] 


def drawlines(roi3,roi4,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
	r,c = roi3.shape
	img1 = cv2.cvtColor(roi3,cv2.COLOR_GRAY2BGR)
    	img2 = cv2.cvtColor(roi4,cv2.COLOR_GRAY2BGR)
    	for r,pt1,pt2 in zip(lines,pts1,pts2):
        	color = tuple(np.random.randint(0,255,3).tolist())
        	x0,y0 = map(int, [0, -r[2]/r[1] ])
        	x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        	img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        	img1 = cv2.circle(roi3,tuple(pt1),5,color,-1)
        	img2 = cv2.circle(roi4,tuple(pt2),5,color,-1)
 	return img1,img2

# Find epilines from points in right image (2nd image) and drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(roi3,roi4,lines1,pts1,pts2)
 
# Find epilines from points in left image (1st image) and drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(roi4,roi3,lines2,pts2,pts1)

fig = plt.figure(3)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

# Essential Matrix --> Extract cameras 
#Essential = K.t() * F * K;

#a = cv2.triangulatePoints(F,F,pts1,pts2)

#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
#disparity = stereo.compute(lines1,lines2)
#plt.imshow(disparity,'gray')
#plt.show()
