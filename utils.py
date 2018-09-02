import configparser
import numpy as np
import cv2

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


def show_joints(img, predictions, hms=None, wt=10, name='img'):
	imghm = img.astype(np.float32)/255
	WHITE = (255, 255, 255)

	for coord in predictions:
		keypt = (int(coord[0]), int(coord[1]))
		cv2.circle(img, keypt, 3, WHITE, -1)

	cv2.imshow(name, img)

	if hms is not None:
		hms[hms<0] = 0
		hms = cv2.resize(hms, (imghm.shape[0], imghm.shape[1]))
		hmsimage = np.zeros((4*hms.shape[0], 4*hms.shape[1], 3), np.float32)
		hmc = np.zeros((hms.shape[0], hms.shape[1],3), np.float32)
		for n in np.arange( hms.shape[2] ):
			row = n // 4
			col = n % 4
			hm = hms[:, :, n]
			for j in np.arange(3):
				hmc[:,:,j] = hm
			up = row * hms.shape[0]
			down = (row+1) * hms.shape[0]
			left = col * hms.shape[1]
			right = (col+1) * hms.shape[1]
			hmsimage[up:down, left:right, ] = hmc * 0.7 + imghm * 0.3
		# hmsimage = hmsimage * 0.7 + img * 0.3
		cv2.imshow('heatmap', hmsimage)
	cv2.waitKey(wt)
	return img