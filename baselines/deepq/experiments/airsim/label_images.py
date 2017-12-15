import cv2, math, time
import sys
a = [-10.0, 9.9, -10.2]

direc = sys.argv[1]
print(direc)

#directory = 'calibrate'
#directory = '/home/robert/ProjectVenom/Modeling/PY-GOTURN/Datasets/path_prob_logs/gt/data6'
directory = direc
fileName = direc.split('/')[-1]

iterations = 200
log = open(directory+'/actions.txt', 'r')
last = None
centers = log.readlines()

for i in range(0,iterations):
	pt = int(centers[i][1:-2])
	print(pt)
        dx = int(256/3)
        dy = int(144/3)
        x = int(pt%3)*dx
        y = int(pt/3)*dy
        #print((x,y))
	img = cv2.imread(directory+'/'+str(i)+'.png',cv2.IMREAD_COLOR)
        height, width = img.shape[:2]
        print((width,height))
# Draw Cross
	cv2.line(img, (x, y), (x+dx, y), (0,0,255))
	cv2.line(img, (x+dx, y), (x+dx, y+dy), (0,0,255))
	cv2.line(img, (x+dx, y+dy), (x, y+dy), (0,0,255))
	cv2.line(img, (x, y+dy), (x, y), (0,0,255))
# Draw Box

#	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#	cv2.resizeWindow('image', 960,540)
#	cv2.imshow('image',img)
#	cv2.waitKey(250)
#	cv2.destroyAllWindows()
	cv2.imwrite('labeled/'+str(i)+'.jpg',img)
	
