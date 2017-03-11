import os
import re
import sys
import random
from shutil import copyfile
from PIL import Image
import numpy as np
from subprocess import call

folder_list = [ ]  # put list of folders to process here...

def process_image_folder(folder):
	print folder
	dst_folder = folder+'_24x24'
	if not os.path.exists(dst_folder): os.makedirs(dst_folder)
	for subfolder in os.listdir(folder):
		subfolder_path = folder+'/'+subfolder
		if os.path.isdir(subfolder_path):
			print "  ",subfolder
			dst_subfolder_path = dst_folder+'/'+subfolder
			dst_subfolder_path_bin= dst_folder+'/'+subfolder+'_bin'
			if not os.path.exists(dst_subfolder_path): os.makedirs(dst_subfolder_path)
			if not os.path.exists(dst_subfolder_path_bin): os.makedirs(dst_subfolder_path_bin)
			for imgfile in os.listdir(subfolder_path):
				if imgfile[-4:]!='.jpg': continue
				imgfile_path = subfolder_path+'/'+imgfile
				dst_imgfile_path = dst_subfolder_path+'/'+imgfile
				call(["convert", imgfile_path, "-colorspace", "rgb", "-type", "TrueColor", "-scale", "24x24", dst_imgfile_path])
				label = [int(imgfile.split('_')[0])]
				im = np.array(Image.open(dst_imgfile_path))
				r = im[:,:,0].flatten()
				g = im[:,:,1].flatten()
				b = im[:,:,2].flatten()
				im_bin = np.array(list(label)+list(r)+list(g)+list(b), np.uint8)
				#print dst_subfolder_path_bin+'/'+imgfile[:-4]+'.bin'
				im_bin.tofile(dst_subfolder_path_bin+'/'+imgfile[:-4]+'.bin')
			#call(['/bin/cat',dst_subfolder_path_bin+'/*.bin','>',dst_subfolder_path+'.bin'],shell=True)

for f in folder_list:
	process_image_folder(f)
