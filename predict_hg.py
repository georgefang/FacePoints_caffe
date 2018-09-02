import os
import time
from inference import Inference
from hourglass import HourglassModel
from datagen import DataGenerator
from utils import process_config
import numpy as np
import configparser
import argparse
import cv2
from predict_all import PredictAll

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', dest='model_dir', default='trained', type=str,
					 help='pose model directory')
parser.add_argument('--config_file', dest='config_file', default='config_dlib.cfg', type=str,
					 help='config file name')
parser.add_argument('--model_file', dest='model_file', default='solver_iter_100000.caffemodel', type=str,
					 help='pose model file name')
parser.add_argument('--resize', dest='resize', default=False, type=bool,
					 help='whether to resize the image to 256*256 directly')
parser.add_argument('--hm', dest='hm', default=False, type=bool,
					 help='whether to show the heat maps')
parser.add_argument('--image_file', dest='image_file', default=None, type=str,
					 help='image file name')
parser.add_argument('--camera', dest='camera', default=None, type=int,
					 help='whether to predict camera')
parser.add_argument('--video', dest='video', default=None, type=str,
					 help='whether to predict video')
parser.add_argument('--video_save', dest='video_save', default=None, type=str,
					 help='whether to save video predict result')

args = parser.parse_args()


if __name__ == '__main__':
	print('--Parsing Config File')

	modeldir = args.model_dir
	configfile = os.path.join(modeldir, args.config_file)
	modelfile = os.path.join(modeldir, args.model_file)
	print(modelfile)

	params = process_config(configfile)
	model = Inference(params=params, model=modelfile)
	predict = PredictAll(model=model, resize=args.resize, hm=args.hm)

	if args.image_file is not None:
		# single image prediction
		predict.predict_image(args.image_file)
	elif args.camera is not None:
		predict.predict_camera(args.camera)
	elif args.video is not None:
		predict.predict_video(args.video, args.video_save)
