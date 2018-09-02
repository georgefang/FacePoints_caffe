"""
Face Alignment
"""

from hourglass import HourglassModel
from datagen import DataGenerator
from utils import process_config
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--configfile', dest='configfile', default='config/config_dlib.cfg', type=str,
					 help='config file name')
parser.add_argument('--loadmodel', dest='loadmodel', default=None, type=str,
					 help='model name of continuing training')
args = parser.parse_args()
print(args.configfile)
print(args.loadmodel)

if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config( args.configfile )
	os.system('mkdir -p {}'.format(params['saver_directory']))
	os.system('cp {0} {1}'.format(args.configfile, params['saver_directory']))
	
	print('--Creating Dataset')
	dataset = DataGenerator(params['num_joints'], params['img_directory'], params['training_txt_file'], params['img_size'])
	dataset._create_train_table()
	dataset._randomize()
	dataset._create_sets()
	
	model = HourglassModel(params=params, dataset=dataset, training=True)
	model.generate_model(load=args.loadmodel)
	# model.restore('trained/tiny_200/hourglass_tiny_200_200')
	model.train_model()
	
