# -*- coding: utf-8 -*-
"""
Face Landmarks Estimation

Project by Xiao-Zhi Fang
Avatar Works
Created on May 15th 2018

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
from utils import show_joints
import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
import cv2
sys.path.insert(0, '/home/george/caffe/python')
import caffe
import google.protobuf as pb2

class HourglassModel():
	""" HourglassModel class: (to be renamed)
	Generate TensorFlow model to train and predict Face Landmarks from images, videos or webcams
	Please check README.txt for further information on model management.
	"""
	def __init__(self, params, dataset=None, training=True, w_summary=True):
		""" Initializer
		Args:
			params: config parameters
			dataset: mpii image dataset
			training: training or inferening
		"""
		self.outDim = params['num_joints']
		self.training = training
		self.dataset = dataset
		self.save_dir = params['saver_directory']
		self.accIdx = np.arange(self.outDim)
		self.solver_file = params['solver_file']
		self.deploy_file = params['deploy_file']
	# ACCESSOR
	
	
	def generate_model(self, load=None, gpu=True):
		caffe.set_device(0)
		if gpu:
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()
		self.solver = caffe.RMSPropSolver(self.solver_file)
		self.solver_param = caffe.proto.caffe_pb2.SolverParameter()
		if load is not None:
			self.solver.net.copy_from(load)
		with open(self.solver_file, 'rt') as fd:
			pb2.text_format.Merge(fd.read(), self.solver_param)

		self.batchSize_train = self.solver.net.blobs['Data'].data.shape[0]
		self.batchSize_test = self.solver.test_nets[0].blobs['Data'].data.shape[0]

		# print('input label shape: {}'.format(self.solver.net.blobs['label'].data.shape))

	def generate_model_inference(self, load=None, gpu=False):
		if gpu:
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()
		
		self.Net = caffe.Net(self.deploy_file, load, caffe.TEST)


	def visualize_valid(self, imgi, hm, set, name=None):
		img = np.copy(imgi)
		img = np.transpose(img, (1, 2, 0))
		hm = np.transpose(hm, (1, 2, 0))
		outhm = np.copy(hm)
		img = cv2.resize(img, (256,256))
		img = img * 255
		img = img.astype(np.uint8)
		outshape = outhm.shape
		assert len(outshape) == 3
		joints = np.zeros((outshape[-1], 2), dtype=np.float32)
		if set == 'train':
			for i in range(0, outshape[-1]):
				resh = np.reshape(outhm[:,:,i], [-1])
				arg = np.argmax(resh)
				#print("hm: {}".format(outhm[:,:,i]))
				joints[i][0] = arg % outshape[1]
				joints[i][1] = arg // outshape[1]
				#print("joint {0}: ({1}, {2})".format(i, joints[i][0], joints[i][1]))
		elif set == 'valid':
			outhm = np.exp(outhm)
			INDEX = np.arange(outshape[0])
			sum_all = np.sum(outhm, axis=(0,1))
			sum_row = np.sum(outhm, axis=0)
			sum_col = np.sum(outhm, axis=1)
			joints[:,0] = sum_row.T.dot(INDEX) / sum_all
			joints[:,1] = sum_col.T.dot(INDEX) / sum_all

		joints = joints * img.shape[0] / outhm.shape[0]
		show_joints(img, joints, name=name)

	def train_model(self, nEpochs = 10, epochSize = 1000, saveStep = 500, validIter = 10):
		"""
		"""
		startTime = time.time()
		self.resume = {}
		self.resume['accur'] = []
		self.resume['loss'] = []
		self.resume['err'] = []
		self.img_save_dir = os.path.join(self.save_dir, 'image')
		if not os.path.exists(self.img_save_dir):
			os.makedirs(self.img_save_dir)
		nEpochs = self.solver_param.max_iter / self.solver_param.stepsize
		epochSize = self.solver_param.stepsize
		# validIter = self.solver_param.test_interval
		validIter = 100
		for epoch in range(nEpochs):
			ebidx = 0
			randlist = self.dataset.epochsize_cat(epochSize, self.batchSize_train, sample = 'train')
			epochstartTime = time.time()
			avg_cost = 0.
			cost = 0.
			print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
			# Training Set
			for i in range(epochSize):
				# DISPLAY PROGRESS BAR
				# TODO : Customize Progress Bar
				percent = ((i+1.0)/epochSize) * 100
				num = np.int(20*percent/100)
				tToEpoch = int((time.time() - epochstartTime) * (100 - percent)/(percent))
				sys.stdout.write('\r Train: {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
				sys.stdout.flush()
				c = 0
				img_train, gt_train, weight_train = self.dataset._sam_generator(randlist[ebidx:ebidx+self.batchSize_train], self.batchSize_train, normalize = True, sample_set = 'train')
				# print(img_train.flags['C_CONTIGUOUS'])
				# print(gt_train.flags['C_CONTIGUOUS'])
				img_train = np.transpose(img_train, (0, 3, 1, 2))
				gt_train = np.transpose(gt_train, (0, 3, 1, 2))
				
				self.solver.net.blobs['Data'].data[...] = img_train
				self.solver.net.blobs['Label'].data[...] = gt_train
				# self.solver.net.set_input_arrays(img_train2, useless1)
				# self.solver.net.set_input_arrays(gt_train2, useless2)

				self.solver.step(self.solver_param.average_loss)
				c = self.solver.net.blobs['SigmoidCrossEntropyLoss1'].data
				c = c / (self.batchSize_train * self.outDim)
				cost += c
				avg_cost += c/epochSize
				ebidx = ebidx + self.batchSize_train
			epochfinishTime = time.time()
			#Save Weight (axis = epoch)
			
			print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime-epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime-epochstartTime)/epochSize))[:4] + ' sec.')
			self.resume['loss'].append(cost)
			# Validation Set
			ebidx = 0
			accuracy_array = np.array([0.0]*len(self.accIdx))
			randlist = self.dataset.epochsize_cat(validIter, self.batchSize_test, sample = 'valid')
			for i in range(validIter):
				img_valid, gt_valid, w_valid = self.dataset._sam_generator(randlist[ebidx:ebidx+self.batchSize_test], self.batchSize_test, normalize = True, sample_set = 'valid')
				# img_valid, gt_valid, w_valid = next(self.valid_gen)
				img_valid = np.transpose(img_valid, (0, 3, 1, 2))
				gt_valid = np.transpose(gt_valid, (0, 3, 1, 2))
				# self.solver.test_nets[0].set_input_array(img_valid, gt_valid)
				self.solver.test_nets[0].blobs['Data'].data[...] = img_valid
				self.solver.test_nets[0].blobs['Label'].data[...] = gt_valid
				self.solver.test_nets[0].forward()
				out_valid = self.solver.test_nets[0].blobs['Deconvolution5'].data
				# accuracy_pred = self.Session.run(self.joint_accur, feed_dict = {self.img : img_valid, self.gtMaps: gt_valid})
				# accuracy_array += np.array(accuracy_pred, dtype = np.float32) / validIter
				accuracy_pred = self.accuracy_compute(gt_valid, out_valid)
				accuracy_array += np.array(accuracy_pred, dtype=np.float32) / validIter
				ebidx = ebidx + self.batchSize_test
			# accuracy_array = accuracy_array
			print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%' )
			self.resume['accur'].append(accuracy_pred)
			self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
			num_show = 10 if self.batchSize_test > 10 else self.batchSize_test
			for i in range(num_show):
				# jts_t = self.visualize_valid(img_valid[i], gt_valid[i], 'train', os.path.join(self.img_save_dir, 'label_{}.jpg'.format(i)))
				jts_v = self.visualize_valid(img_valid[i], out_valid[i], 'valid', os.path.join(self.img_save_dir, 'infer_{}.jpg'.format(i)))
			# self.record_valid_diff(jts_v, jts_t)
			point_accur= {}
			for i in self.accIdx:
				point_accur[i] = round(100*accuracy_array[i],3)
			# print('pt accuracy: ',point_accur)
		print('Training Done')
		print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize_train) )
		print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
		print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
		print('  Training Time: ' + str( datetime.timedelta(seconds=time.time() - startTime)))
	
	def accuracy_compute(self, gt, out):
		assert( gt.shape == out.shape)
		shape = gt.shape
		joint_accur = []
		assert( shape[1] == len(self.accIdx) )
		for j in self.accIdx:
			err = 0.0
			for i in range(shape[0]):
				resh_out = np.reshape(out[i, j], [-1])
				resh_gt  = np.reshape(gt[i, j], [-1])
				arg_out = np.argmax(resh_out)
				arg_gt  = np.argmax(resh_gt)
				x_out, y_out = arg_out % shape[3], arg_out // shape[3]
				x_gt,  y_gt  = arg_gt % shape[3], arg_gt // shape[3]
				dis = np.sqrt( np.square(x_out.astype(np.float32)-x_gt.astype(np.float32)) + np.square(y_out.astype(np.float32)-y_gt.astype(np.float32)) )
				dis = dis / shape[3]
				err += dis
			joint_accur.append(1.0-err/shape[0])
		return joint_accur
