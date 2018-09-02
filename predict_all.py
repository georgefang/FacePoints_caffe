import sys
sys.path.append('./')

import time
import numpy as np
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import os
import glob
import h5py
import threading
from xml.dom import minidom
from datagen import DataGenerator
from utils import show_joints
import inference


class PredictAll():

	#-------------------------INITIALIZATION METHODS---------------------------
	def __init__(self, model, resize=False, hm=True):
		self.model = model
		self.resize = resize
		self.hm = hm
		self.originSize = False
		self.standard = False
		self.joint2dNum = 74
		self.box = None

	def set_originSize(self, orig):
		self.originSize = orig

	def set_standardSize(self, size, std=True):
		self.standard = std
		self.stdSize = size


	def predict_general( self, img, wt=0):
		imgOrig = np.copy(img)
		if self.resize:
			img = cv2.resize(img, (256,256))
		else:
			img, scale, _ = self.model.preProcessImage(img)
		test_img = np.copy(img)

		test_img = test_img.astype(np.float32) / 255
		startTime = time.time()
		joints, hms = self.model.predictJointsFromImageByMean(test_img)
		print('predict time is: ', 1000*(time.time()-startTime), 'ms')
		# flipRef = np.array([6,5,4,3,2,1,7,8,9,10,16,15,14,13,12,11])-1
		# joints = joints[flipRef, :]
		# hms = hms[:,:,flipRef]
		# img = cv2.flip(test_img, 1)
		test_img = test_img * 255
		test_img = test_img.astype(np.uint8)
		if self.hm:
			imgPred = show_joints(test_img, joints, hms, wt)
		elif not self.originSize:
			imgPred = show_joints(test_img, joints, wt=wt)

		if self.standard:
			joints = joints * self.stdSize / 256.0
			imgStd = cv2.resize(test_img, (self.stdSize, self.stdSize))
			imgPred = show_joints(imgStd, joints, wt=wt, name='std')

		if self.originSize:
			origShape = imgOrig.shape[0:2]
			msize = np.amax(origShape)
			if self.resize:
				joints[:,0] = joints[:,0] * origShape[1] / 256.0
				joints[:,1] = joints[:,1] * origShape[0] / 256.0
			else:
				scale = float(msize) / self.model.params['img_size']
				joints = (joints+0.5) * scale
				if origShape[0] < origShape[1]:
					joints[:,1] = joints[:,1] - (origShape[1]-origShape[0])/2
				else:
					joints[:,0] = joints[:,0] - (origShape[0]-origShape[1])/2

			imgPred = show_joints(imgOrig, joints, wt=wt, name='orgin')

		return joints, imgPred, scale

	def predict_track( self, img, wt=0):
		inp_size = 128
		if self.box is None:
			self.box = np.array([0,0,0,0], np.int)
			leng = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]
			self.box[0] = img.shape[1] - leng
			self.box[1] = img.shape[0] - leng
			self.box[2] = img.shape[1] + leng
			self.box[3] = img.shape[0] + leng
			self.box = self.box / 2
		
		img_crop = self.model.crop_image(img, self.box)
		scale = float(inp_size) / img_crop.shape[0]
		# print(img_crop.shape)
		img_crop = cv2.resize(img_crop, (inp_size,inp_size))

		test_img = np.copy(img_crop)
		test_img = test_img.astype(np.float32) / 255
		startTime = time.time()
		joints, hms = self.model.predictJointsFromImageByMean(test_img)
		print('predict time is: ', 1000*(time.time()-startTime), 'ms')
		joints = (joints + 0.5) / scale
		joints[:,0] = joints[:,0] + self.box[0]
		joints[:,1] = joints[:,1] + self.box[1]
		imgPred = show_joints(img, joints, wt=wt, name='track')
		return joints, imgPred

	def update_box( self, joints, shape):
		left  = np.amin(joints[:,0])
		right = np.amax(joints[:,0])
		up    = np.amin(joints[:,1])
		down  = np.amax(joints[:,1])
		centerx = int((left + right)/2)
		centery = int((up + down)/2)
		leng = right - left if (right-left) > (down-up) else down - up
		maxleng = shape[0] if shape[0] > shape[1] else shape[1]
		if leng > maxleng:
			leng = maxleng
		leng = int(leng)
		self.box = [centerx-leng, centery-leng, centerx+leng, centery+leng]

	def predict_image( self, imgname):
		img = cv2.imread(imgname)
		self.predict_general( img )


	def predict_camera( self, camidx):
		cam = cv2.VideoCapture(camidx)
		while True:
			# startTime = time.time()
			ret_val, img = cam.read()
		 	img = cv2.flip(img, 1)
		 	jts, _ = self.predict_track(img, wt=1)
		 	self.update_box(jts, img.shape)
		 	if cv2.waitKey(1) == 27:
		 		cv2.destroyAllWindows()
		 		break

	def predict_video( self, videoname, savename ):
		stop = False
		save = False
		while True:
			fourcc = cv2.VideoWriter_fourcc(*'MP4V')
			if savename is not None:
				save = True
				savefile = cv2.VideoWriter(savename, fourcc, 30.0, (256, 256))

			video = cv2.VideoCapture(videoname)
			if not video.isOpened():
				print("fail to read video file: {}".format(videoname))
				return
			joints2dQueue = []
			while True:
				# startTime = time.time()
				ret, frame = video.read()
				if ret == True:
					shape = frame.shape
					# frame = frame[int(0.2*shape[0]):int(0.8*shape[0])]
					joints, framePred, scale = self.predict_general(frame, wt=10)
					joints2dQueue.append(np.reshape(joints, [-1]))
					if save:
						# print(framePred.shape)
						savefile.write(framePred)
					if cv2.waitKey(1) == 27:
						stop = True
						break
				elif save:
					stop = True
					break

			if save or stop:
				video.release()
				if save:
					savefile.release()
				cv2.destroyAllWindows()
				return
