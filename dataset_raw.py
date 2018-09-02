import numpy as np
import os
import cv2

def show_joints(img, predictions, box, wt=0, name='img'):
	# imghm = img.astype(np.float32)/255
	BLACK = (0, 0, 255)
	BLUE = (255, 0, 0)

	idx0 = np.array([[0,1],[0,3],[2,1],[2,3]])
	idx1 = np.array([[0,3],[2,3],[0,1],[2,1]])
	for n in range(len(idx0)):
		pt0 = (box[idx0[n][0]], box[idx0[n][1]])
		pt1 = (box[idx1[n][0]], box[idx1[n][1]])
		cv2.line(img, pt0, pt1, BLUE, 2)

	for coord in predictions:
		keypt = (int(coord[0]), int(coord[1]))
		if keypt == (-1, -1):
			continue
		cv2.circle(img, keypt, 3, BLACK, -1)

	cv2.imshow(name, img)
	cv2.waitKey(wt)
	return img


if __name__ == '__main__':
	ifile_name = 'infos/dlib_20160805_total.txt'
	ofile_name = 'infos/face_landmarks_dlib.txt'
	filedir = '../photos'
	input_file = open(ifile_name, 'r')
	output_file = open(ofile_name, 'w')
	index = 0
	elines = 76
	pr = 2
	ns = 0
	height = 0
	width = 0
	win = 0
	inside = True
	joints = np.zeros((74,2), dtype=np.int64)
	left = 100000
	right = -1
	top = 100000
	down = -1
	for line in input_file:
		line = line.strip()
		if index == 0:
			imgname = line
			img = cv2.imread(os.path.join(filedir, line))
			width = img.shape[1]
			height = img.shape[0]
			if img.shape[2] == 1:
				img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			assert img.shape[2] == 3
			# print('w: ',width, ' h: ', height)
			
		elif index == 1:
			line = line.split(' ')
			box = list(map(int, line))
			center_x = int((box[0]+box[2])/2)
			center_y = int((box[1]+box[3])/2)
			scale = pr*(box[2]-box[0])/200
			# output_file.write(str(center_x)+' ')
			# output_file.write(str(center_y)+' ')
			# output_file.write(str(round(scale, 6))+' ')
			win = pr*(box[2]-box[0])/2
		else:
			line = line.split(' ')
			c = int(line[0])
			r = int(line[1])
			# print('c: ',c, ' r: ',r)
			if c >= width or r >= height:
				c = -1
				r = -1
			if c != -1 and r != -1:
				if c < left:
					left = c
				if c > right:
					right = c
				if r < top:
					top = r
				if r > down:
					down = r
			joints[index-2] = [c, r]
			if inside and c != -1 and r != -1:
				dist = [c-center_x, r-center_y]
				if dist[0] > win or dist[1] > win:
					print('outside the box: {}'.format(imgname))
					inside = False
			# output_file.write(str(c)+' ')
			# output_file.write(str(r))
			# if index == elines-1:
			# 	output_file.write('\n')
			# else:
			# 	output_file.write(' ')
		index = index + 1
		if index == elines:
			index = 0

			# center_x = int((left+right)/2)
			# center_y = int((top+down)/2)
			# win = max([right-left, down-top])/2
			# scale = pr*win/100
			# box = np.array([int(center_x - pr*win), int(center_y - pr*win), int(center_x + pr*win), int(center_y + pr*win)])
			# box[box<0] = 0
			# if box[2] > width-1:
			# 	box[2] = width-1
			# if box[3] > height-1:
			# 	box[3] = height-1
			show_joints(img, joints, box)
			if not inside:
				# print('window: ',win)
				# print(box)
				# show_joints(img, joints, box)
				inside = True
			left = 100000
			right = -1
			top = 100000
			down = -1

			output_file.write(imgname+' ')
			output_file.write(str(center_x)+' ')
			output_file.write(str(center_y)+' ')
			output_file.write(str(round(scale, 6))+' ')
			for pt in joints:
				output_file.write(str(pt[0])+' ')
				output_file.write(str(pt[1])+' ')
			output_file.write('\n')


			# ns = ns + 1
			# if ns == 10:
			# 	break
		
	input_file.close()
	output_file.close()

