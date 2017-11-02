import cv2, math, time
import sys
a = [-10.0, 9.9, -10.2]

direc = sys.argv[1]
print(direc)

#directory = 'calibrate'
#directory = '/home/robert/ProjectVenom/Modeling/PY-GOTURN/Datasets/path_prob_logs/gt/data6'
directory = direc
fileName = direc.split('/')[-1]

iterations = 100
log = open(directory+'/actions.txt', 'r')
last = None
centers = log.readlines()
gt = open(directory+'.ann', 'w')

for i in range(0,iterations):
	pt = centers[i][1:-2].split(',')
	pt[0] = int((float(pt[0]))*256)
	pt[1] = int((float(pt[1]))*144)
	print(pt)
	img = cv2.imread(directory+'/'+str(i)+'.png',cv2.IMREAD_COLOR)
# Draw Cross
	cv2.line(img, (pt[0]-5, pt[1]), (pt[0]+5, pt[1]), (0,0,255))
	cv2.line(img, (pt[0], pt[1]-5), (pt[0], pt[1]+5), (0,0,255))
# Draw Box

	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#	cv2.resizeWindow('image', 960,540)
	cv2.imshow('image',img)
	cv2.waitKey(300)
	cv2.destroyAllWindows()
#	cv2.imwrite('labeled/'+str(i)+'.jpg',img)
	
