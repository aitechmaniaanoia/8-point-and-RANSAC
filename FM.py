'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''
#import sys
#print(sys.path)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
#import random

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default = 1)
parser.add_argument("--image1", type=str,  default='data/myleft.jpg' )
parser.add_argument("--image2", type=str,  default='data/myright.jpg' )
args = parser.parse_args()

print(args)

## normalize points before calculate F
def normalize(points):
    mean_x = np.mean(points[:,0])
    mean_y = np.mean(points[:,1])
    
    std = np.mean(np.sqrt((points[:,0] - mean_x)**2 + (points[:,1] - mean_y)**2))
    
    scale = np.sqrt(2)/std
    
    translate_x = -scale*mean_x
    translate_y = -scale*mean_y
    
    T = [[scale,   0,     translate_x],
         [0,       scale, translate_y],
         [0,       0,     1]]
    return np.array(T)


def FM_by_normalized_8_point(pts1,  pts2):
    #F1, _ = cv2.findFundamentalMat(pts1, pts2,  cv2.FM_8POINT )
    #comment out the above line of code. 
	
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 

    # F:  fundmental matrix    
    n = len(pts1)
    ## normalize point coordinates
    T1 = normalize(pts1)
    T2 = normalize(pts2)
    
    ## add one column become [x y 1] and change the shape to 3*N 
    p1 = np.hstack((pts1, np.ones((n,1)))).T
    p2 = np.hstack((pts2, np.ones((n,1)))).T
    
    pt1 = np.dot(T1, p1)
    pt2 = np.dot(T2, p2)
    
    ##  build matrix for equations
    ##  each row in the A matrix below is constructed as 
    ##  [x*x', x*y', x, y*x', y*y', y, x', y', 1]
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [pt1[0,i]*pt2[0,i], pt1[0,i]*pt2[1,i], pt1[0,i], 
                pt1[1,i]*pt2[0,i], pt1[1,i]*pt2[1,i], pt1[1,i],
                pt2[0,i], pt2[1,i], 1]
    #print(A)
    ## SVD of A
    # A = U * sigma * V.T
    u, s, v = np.linalg.svd(A)
    
    ## Entries of F are the elements of column vector of V corresponding to the smallest singular value
    F = v.T[:,-1].reshape((3,3), order = 'F')
    #print(F)
    #print(v[-1])
    ## enforce rank 2 constraint on F
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    #print(S)
    F = np.dot(U, np.dot(np.diag(S), V))
    ## Un-normalize F
    F = np.dot(T2.T, F.dot(T1))
    F = F/F[2,2]
    
    return  F


def FM_by_RANSAC(pts1,  pts2):
    np.random.seed(10)
    #F1, mask1 = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )	
    #comment out the above line of code. 
	
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 
	
    # F:  fundmental matrix
    #n = len(pts1) * 0.76
    n = 0
    iters = 1000
    F = np.zeros((3,3))
    
    ## remove duplicate coordinates points
    pts = np.concatenate((pts1, pts2), axis = 1)
    sorted_idx = np.lexsort(pts.T)
    sorted_data =  pts[sorted_idx,:]
    
    ### Get unique row mask
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    
    ### Get unique rows
    pts = sorted_data[row_mask]
    pt_1 = pts[:,0:2]
    pt_2 = pts[:,2:]
    
    #print(pt_1)
    
    for i in range(iters):

        ## choose 8 pairs of matching points randomly
        idx = np.random.choice(np.arange(len(pt_1)), 8, replace = False)
        pt1 = pt_1[idx,:]
        pt2 = pt_2[idx,:]
        #print(pt1)
        ## fundamental matrix
        F_i = FM_by_normalized_8_point(pt1, pt2)
        #F_i, _ = cv2.findFundamentalMat(pt1, pt2,  cv2.FM_8POINT)
        
        ## compute the number of inliers n_i
        n_i = 0
        
        tolerance = 50
        
        mask = np.zeros((len(pts1),1))
        
        p1 = np.hstack((pts1, np.ones((len(pts1),1))))
        p2 = np.hstack((pts2, np.ones((len(pts2),1))))
        #print(p1)
        
        #t_p2 = np.dot(p1, F_i)
        #print(t_p2)
        
        #lines = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F_i)
        #lines = lines.reshape(-1,3)
        #print(lines)
        #lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F_i)
        #lines2 = lines2.reshape(-1,3)

        ### compute the Sampson Distance of each points
        for j in range(len(pts1)):
            
            ## calculate R
            R = p2[j].dot(np.dot(F_i, p1[j].T))
            #print(R)
            ## calculate Lines L = F*P
            lines1 = np.dot(F_i, p2[j,:].T)
            lines2 = p1[j,:].dot(F_i)
            #print(lines1)
            
            ## normalize point coordinate
            #lines1[0:2] = lines1[0:2] / (abs(lines1[0]) + abs(lines1[1]))
            #lines2[0:2] = lines2[0:2] / (abs(lines2[0]) + abs(lines2[1]))
            #print(lines1)
            
            SED_2 = R**2/(lines1[0]**2 + lines1[1]**2) + R**2/(lines2[0]**2 + lines2[1]**2)
            #SED = SED_2**0.5
            #print(SED_2)

            if SED_2 < tolerance:
                n_i = n_i + 1
            # mask:   whetheter the points are inliers
                mask[j] = 1
                
        ## refine the line    
        if n_i > n:
            n = n_i
            #F = F_i
            
            index = [k for k in range(len(mask)) if mask[k] == 1]

            # fit with all inliers
            F, _ = cv2.findFundamentalMat(pts1[index,:], pts2[index,:],  cv2.FM_8POINT )
            
        #break
    #print(i)
    
    return  F, mask

	
img1 = cv2.imread(args.image1,0) 
img2 = cv2.imread(args.image2,0)  

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    F1, mask1, F,  mask = FM_by_RANSAC(pts1, pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]	
else:
    F = FM_by_normalized_8_point(pts1, pts2)
	

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	
	
# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)
plt.show()

cv2.imwrite('output_1.jpg', img5)
#cv2.imwrite('8_points_2.jpg', img6)

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img4)
plt.subplot(122),plt.imshow(img3)
plt.show()

cv2.imwrite('output_2.jpg', img3)