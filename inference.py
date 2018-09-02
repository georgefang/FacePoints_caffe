# -*- coding: utf-8 -*-
"""
Face Landmarks Estimation

Project by Xiao-Zhi Fang
AvatarWorks Lab
Huanshi Ltd.

@author: Xiao-Zhi Fang
@mail : george.fang@avatarworks.com

Abstract:
	This python code creates a Stacked Hourglass Model
	(Credits : A.Newell et al.)
	(Paper : https://arxiv.org/abs/1603.06937)
	
	Code translated from 'anewell' github
	Torch7(LUA) --> Caffe(PYTHON)
	(Code : https://github.com/anewell/pose-hg-train)

"""
import sys
sys.path.append('./')

from hourglass import HourglassModel
from time import time, clock
import numpy as np
sys.path.insert(0, '/home/george/caffe/python')
import caffe
import scipy.io
import cv2
from datagen import DataGenerator


class Inference():
	""" Inference Class
	Use this file to make your prediction
	Easy to Use
	Images used for inference should be RGB images (int values in [0,255])
	Methods:

	"""
	def __init__(self, params, model):
		""" Initilize the Predictor
		Args:
			config_file 	 	: *.cfg file with model's parameters
			model 	 	 	 	: *.index file's name. (weights to load) 
		"""
		t = time()
		self.params = params
		self.HG = HourglassModel(params=params, dataset=None, training=False)
		self.HG.generate_model_inference(model, gpu=False)
		print('Done: ', time() - t, ' sec.')

	
	#----------------------------PREDICTION METHODS---------------------------
	
	def pred(self, img):
		""" Given a 256 x 256 image, Returns prediction Tensor
		This prediction method returns values in [0,1]
		Use this method for inference
		Args:
			img		: Image -Shape (256 x256 x 3) -Type : float32
		Returns:
			out		: Array -Shape (64 x 64 x outputDim) -Type : float32
		"""
		if len(img.shape) == 3:
			img = np.expand_dims(img, axis=0)
		img = np.transpose(img, (0, 3, 1, 2))
		self.HG.Net.blobs['Data'].data[...] = img
		self.HG.Net.forward()
		out_valid = self.HG.Net.blobs['Deconvolution5'].data
		out_valid = np.exp(out_valid)
		return out_valid
	

	# -----------------------------Image Prediction----------------------------
	def predictJointsFromImage(self, img):
		image = np.copy(img)
		hms = self.pred(image)
		hm = hms[0]
		hmshape = hm.shape
		assert len(hmshape)==3
		joints = np.zeros((hmshape[-1], 2), dtype=np.int64)
		for i in range(0, hmshape[-1]):
			resh = np.reshape(hm[:,:,i], [-1])
			arg = np.argmax(resh)
			#print("hm: {}".format(outhm[:,:,i]))
			joints[i][0] = arg % hmshape[1]
			joints[i][1] = arg // hmshape[1]
			#print("joint {0}: ({1}, {2})".format(i, joints[i][0], joints[i][1]))
			
		joints = joints * image.shape[0] / hmshape[0]
		return joints, hm

	# -----------------------------Image Prediction By Mean Method----------------
	def predictJointsFromImageByMean(self, img):
		image = np.copy(img)
		hms = self.pred(image)
		hms = np.transpose(hms, (0,2,3,1))
		hm = hms[0]
		hmshape = hm.shape
		assert len(hmshape)==3
		joints = np.zeros((hmshape[-1], 2), dtype=np.float32)
		INDEX = np.arange(hmshape[0])
		sum_all = np.sum(hm, axis=(0,1))
		sum_row = np.sum(hm, axis=0)
		sum_col = np.sum(hm, axis=1)
		joints[:,0] = sum_row.T.dot(INDEX) / sum_all
		joints[:,1] = sum_col.T.dot(INDEX) / sum_all

		joints = joints * image.shape[0] / hmshape[0]
		return joints, hm	



	def preProcessImage(self, img):
		""" RETURN THE RESIZE IMAGE WHICH SIZE IS 256*256
		ARGS:
			img: input image
		"""
		shape = img.shape[0:2]
		assert len(shape) == 2
		sizeNor = self.params['img_size']
		msize = np.amax(shape)
		scale = float(sizeNor) / msize
		shape_new = np.array([int(shape[0]*scale), int(shape[1]*scale)])
		imgre = cv2.resize(img, (int(shape_new[1]), int(shape_new[0])))
		imgsq = np.zeros((sizeNor, sizeNor, 3), dtype=np.uint8)
		leftup = [0,0]
		leftup[0] = sizeNor/2 - shape_new[0]/2
		leftup[1] = sizeNor/2 - shape_new[1]/2
		imgsq[leftup[0]:leftup[0]+imgre.shape[0], leftup[1]:leftup[1]+imgre.shape[1], :] = np.copy(imgre) 
		return imgsq, scale, leftup

		
	def crop_image(self, img, box):
		part = np.copy(box)
		# print(img.shape)
		# print(box)
		img_crop = np.zeros((box[3]-box[1]+1, box[2]-box[0]+1,3), dtype=np.uint8)
		if part[0] < 0:
			part[0] = 0
		if part[1] < 0:
			part[1] = 0
		if part[2] > img.shape[1]-1:
			part[2] = img.shape[1]-1
		if part[3] > img.shape[0]-1:
			part[3] = img.shape[0]-1

		img_crop[part[1]-box[1]:part[3]-box[1]+1,part[0]-box[0]:part[2]-box[0]+1] = np.copy(img[part[1]:part[3]+1,part[0]:part[2]+1])
		return img_crop	
		
		