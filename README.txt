####################################################
Face Feature Points Detection (caffe version)

@author: Xiao-Zhi Fang
@mail: georgexzfang@163.com

Worked based on:
1. Stacked Hourglass Network for Human Pose Estimation
2. How far are we from solving the 2D & 3D Face Alignment problem?
####################################################

I. CONFIG FILE
	A 'config_dlib.cfg' is present in the directory.
	It contains all the variables needed to tweak the model.

	training_txt_file : Path to TEXT file containing information about images
	img_directory : Path to images dataset
	img_size : Size of input Image /!\ DO NOT CHANGE THIS PARAMETER (128 default value)
	hm_size : Size of output heatMap /!\ DO NOT CHANGE THIS PARAMETER (64 default value)
	num_joints : Number of joints considered
	solver_file: solver prototxt of caffe training
	deploy_file: network prototxt of caffe inference
	gpu: whether to use gpu to train model
	saver_directory: where to save trained model

II. DATASET
	To create a dataset you need to put every images of your set on the 'img_directory'.
	Add information about your images into the 'training_txt_file':

	EXAMPLE:
		100032540_1.jpg 835 807 4.11 566 637 561 715 566 813 568 891 580 963 612 1030 680 1060 760 1065 871 1072 961 1054 1026 1000 1070 917 1081 830 1095 745 1109 654 1028 580 975 556 908 556 864 578 910 578 975 578 608 570 647 543 711 547 744 574 711 570 650 563 635 639 691 613 739 654 677 654 977 660 926 619 871 658 926 667 767 647 746 708 718 763 730 791 778 793 836 793 855 763 834 715 834 652 746 761 813 761 681 891 718 852 751 843 781 845 804 843 852 867 889 900 846 920 810 927 767 929 726 923 696 910 732 882 776 891 813 887 813 880 776 878 735 874 776 884 778 726 658 617 649 654 709 654 718 630 899 637 901 665 954 669 961 626
		100040721_1.jpg 358 332 1.545 259 260 255 294 255 328 259 360 270 390 291 414 318 430 350 435 386 433 416 418 439 395 452 366 459 333 461 300 460 267 441 258 427 239 398 233 380 247 398 247 424 252 280 249 296 231 323 229 341 242 322 242 296 243 291 266 312 255 331 265 312 272 424 269 405 260 385 268 403 272 345 267 342 291 327 308 334 320 356 323 377 322 387 310 372 292 371 268 342 312 373 315 312 357 320 342 342 338 355 340 368 339 389 348 396 363 389 374 370 378 353 378 338 375 321 370 335 362 353 365 373 364 373 355 353 352 336 350 353 358 360 304 301 258 301 271 323 271 323 258 394 261 393 271 413 272 415 262

	In this example we consider 74 joints

	The text file is formalized as follow:
		image_name x_center y_center scale x1 y1 x2 y2 x3 y3 ...
		image_name is the file name
		(x_center y_center) Is the center of cropped image
		(scale) Is the size of cropped image related to 100
		(x1 y1 x2 y2 x3 y3 ...) is the list of coordinates of every joints
	/!\
	Missing part or values must be marked as -1

III. TRAINING
	To train a model, make sure to have a 'config_dlib.cfg' file in your main directory and a text file with regard to your dataset. Then run train_hg.py
	It will run the training.
	On a TITAN GTX for mini_batches of 16 images on 100 epochs of 1000 iterations: 2 days of training (1.6 million images)
	Training Parameters:
	'configfile': name of config file which stated in Part I
	'loadmodel': trained model if you want to continue training


IV. SAVING AND RESTORING MODELS
	Saving is automatically done when training.
	In the 'saver_directory' you will find several files:
	'name'_'epoch'.data-00000-of-00001
	'name'_'epoch'.index
	'name'_'epoch'.meta

	You can manually load the graph from *.meta file using TensorFlow methods. (http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)
	Or you can use the Restore Method in hourglass.py
	To do so, you first need to create the same graph as the saved one. To do so use the exact same 'config_dlib.cfg' that you used to train your model.
	Then use HourglassModel('config_dlib.cfg').restore(modelToLoad) to restore pretrained model.
	modelToLoad: 'saver_directory'/'name'_'epoch'
	/!\ BE SURE TO USE THE SAME CONFIG.CFG FILE OR THE METHOD WON'T BE ABLE TO ASSIGN THE RIGHT WEIGHTS

V. PREDICTION
	prediction is implemented in 'predict_hg.py', which contains some predict methods such as image, video, camera, and so on (only one method each run time). You can extend to other methods if you want.
	Prediction Parameters:
	'model_dir': directory of 2D trained model
	'config_file': name of 2D config file in 'model_dir'
	'model_file": file name of 2D trained model'
	'resize': whether to resize the input image to 256*256
	'hm': whether to show the predicted heat map
	'image_file': set image name to predict 2D pose from some image
	'camera': set web camera index to predict 2D pose from some camera
	'video': set video name to predict 2D pose from some video
	'video_save': set save name to store the result of video prediction

在跑范例之前，先执行DepthwiseConvolution文件夹下的README.md文件中How to build部分。
如果想要自己写网络prototxt文件（用到分通道卷积网络层的话），需要阅读README的其他部分。

预测运行范例：
python predict_hg.py --camera 0
或者
python predict_hg.py --image_file [your_image_file_name]
工程给出了默认的已训练好的模型，在trained/solver_iter_100000.caffemodel.
第一个例子运行的是摄像头实时跟踪和预测人脸特征点，没有使用人脸检测框，确保人脸在屏幕内，且一开始不能过小。
第二个例子运行的是预测单张图片的人脸特征点，同样没有使用人脸检测框，适用于自拍等人脸在照片中占比较大的图片。

训练运行范例：
python train_hg.py

训练时，需要在配置文件.cfg（默认是config/config_dlib.cfg）里修改图片数据集所在的文件夹路径：img_directory。192.168.11.200服务器上的图片数据集目录为：/home/research/disk1/george/FaceAlignment/photos

如果有新的训练数据，还需要更改标注信息文件名：training_txt_file
标注信息文件的生成在dataset_raw.py文件中，把ifile_name修改成原标注文件名，ofile_name修改成生成的标注文件名，filedir是图片数据集路径。直接运行该python文件，把最后生成的标注文件拷贝到想要保存的路径下。
