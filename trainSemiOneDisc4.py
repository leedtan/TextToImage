import tensorflow as tf
import numpy as np
import modelSemiOneDisc

import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import time
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()

prince = True
#TO try: just reduced caption vector length, also remove gradient clipping. and increase learning rate *2.
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=10,
					   help='Noise dimension')

	parser.add_argument('--t_dim', type=int, default= 10,#1024#LEE,#default=256,
					   help='Text feature dimension')

	parser.add_argument('--batch_size', type=int, default=8,#LEECHANGE default=64,
					   help='Batch Size')

	parser.add_argument('--image_size', type=int, default=64,
					   help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=8,#64,
					   help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=8,#128,
					   help='Number of conv in the first layer discr.')

	parser.add_argument('--gfc_dim', type=int, default=1024,
					   help='Dimension of gen untis for for fully connected layer 1024')

	parser.add_argument('--caption_vector_length', type=int, default=100,#4096 - zdim (30)#2400Lee
					   help='Caption Vector Length')

	parser.add_argument('--data_dir', type=str, default="Data",
					   help='Data Directory')

	parser.add_argument('--learning_rate', type=float,default=1e-4,#1e-4 or 1e-5LEECHANGE default=0.0002,
					   help='Learning Rate')

	parser.add_argument('--beta1', type=float, default =.5,#LEECHANGE default=0.5,
					   help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=6000,
					   help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=2,
					   help='Save Model/Samples every x iterations over batches')

	parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--data_set', type=str, default="flowers",
                       help='Dat set: MS-COCO, flowers')
	
	parser.add_argument('--save_epoch', type=list, default=[10,20,60,100,300,600,1000,1500,2000,3000],
						help='Save model in specified epoch')

	args = parser.parse_args()
	model_options = {
		'z_dim' : args.z_dim,
		't_dim' : args.t_dim,
		'batch_size' : args.batch_size,
		'image_size' : args.image_size,
		'gf_dim' : args.gf_dim,
		'df_dim' : args.df_dim,
		'gfc_dim' : args.gfc_dim,
		'caption_vector_length' : args.caption_vector_length
	}
	beta2 = .9
	
	gan = modelSemiOneDisc.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model(args.beta1, beta2, args.learning_rate)
	
	g_optim = gan.g_optim
	d_optim = gan.d_optim
	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
	#sess = tf.InteractiveSession()
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	if prince:
		sess.run(tf.global_variables_initializer())
	else:
		tf.initialize_all_variables().run()
	
	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)
	
	loaded_data = load_training_data(args.data_dir, args.data_set)
	init = 0
	d_avg_full, d_avg_mid, d_avg_sml = 0,0,0
	lb = 0#-.3
	disc_break = -.3
	t1 = 0
	t2 = 0
	d_loss_noise = g_loss_noise = 0
	for i in range(args.epochs):
		noise_gen = 1/( 10 + np.power(i, 1/3))
		noise_disc = 2/( 10 + np.power(i, 1/3))
		t = time.time()
		img_idx = 0
		batch_no = 0
		caption_vectors = 0
		while batch_no*args.batch_size < 5:
			if init == 0:
				last_captions = np.random.rand(args.batch_size, args.caption_vector_length)/100
			else:
				last_captions = caption_vectors
			real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size, 
				args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)
			t1 += time.time()-t
			t = time.time()
			# DISCR UPDATE
			lambdaDAE = 1
			
			sess.run([g_optim],
				feed_dict = {
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
					input_tensors['l2reg']: 0,
					input_tensors['LambdaDAE']: lambdaDAE,
					input_tensors['noise_indicator'] : 0,
					input_tensors['noise_gen'] : noise_gen,
					input_tensors['noise_disc'] : noise_disc
				})
			sess.run([g_optim],
				feed_dict = {
					input_tensors['t_real_caption'] : np.random.rand(
						args.batch_size, args.caption_vector_length)*.02,
					input_tensors['t_z'] : z_noise,
					input_tensors['l2reg']: 0,
					input_tensors['LambdaDAE']: lambdaDAE,
					input_tensors['noise_indicator'] : 1,
					input_tensors['noise_gen'] : noise_gen,
					input_tensors['noise_disc'] : noise_disc
				})
			
			
			if np.random.rand() > 0.5 or init == 0:
				init = 1
				last_img = 'real'
				_, _, d_loss, g_loss, g1_loss, g2_loss, g3_loss, g4_loss, g5_loss, \
					d1_loss, d2_loss, d3_loss, d4_loss, d5_loss, img1, img2,img3,img4,img5   = sess.run(
					[g_optim, d_optim, loss['d_loss'], loss['g_loss'], 
					 loss['g1_loss'], loss['g2_loss'], loss['g3_loss'], loss['g4_loss'], loss['g5_loss'],
					 loss['d1_loss'], loss['d2_loss'], loss['d3_loss'], loss['d4_loss'], loss['d5_loss'],
					outputs['img1'],outputs['img2'], outputs['img3'],outputs['img4'], outputs['img5']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
						input_tensors['l2reg']: 0,
						input_tensors['LambdaDAE']: lambdaDAE,
						input_tensors['noise_indicator'] : 0,
						input_tensors['noise_gen'] : noise_gen,
						input_tensors['noise_disc'] : noise_disc
					})
				sess.run([g_optim],
					feed_dict = {
						input_tensors['t_real_caption'] : np.random.rand(
							args.batch_size, args.caption_vector_length)*.02,
						input_tensors['t_z'] : z_noise,
						input_tensors['l2reg']: 0,
						input_tensors['LambdaDAE']: lambdaDAE,
						input_tensors['noise_indicator'] : 1,
						input_tensors['noise_gen'] : noise_gen,
						input_tensors['noise_disc'] : noise_disc
					})
			elif np.random.rand() > 0:
				last_img = 'noise'
				_, _, d_loss_noise, g_loss_noise, img1, img2,img3,img4,img5   = sess.run(
					[g_optim, d_optim, loss['d_loss'], loss['g_loss'],
					outputs['img1'],outputs['img2'], outputs['img3'],outputs['img4'], outputs['img5']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : np.random.rand(
							args.batch_size, args.caption_vector_length)*.02,
						input_tensors['t_z'] : z_noise,
						input_tensors['l2reg']: 0,
						input_tensors['LambdaDAE']: lambdaDAE,
						input_tensors['noise_indicator'] : 1,
						input_tensors['noise_gen'] : noise_gen,
                        input_tensors['noise_disc'] : noise_disc
					})
				sess.run([g_optim],
					feed_dict = {
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
						input_tensors['l2reg']: 0,
						input_tensors['LambdaDAE']: lambdaDAE,
						input_tensors['noise_indicator'] : 0,
						input_tensors['noise_gen'] : noise_gen,
                        input_tensors['noise_disc'] : noise_disc
					})
			t2 += time.time()-t
			t = time.time()
			if batch_no % 10 == 0:
				img1, img2,img3,img4,img5   = sess.run(
                    [outputs['img1'],outputs['img2'], outputs['img3'],outputs['img4'], outputs['img5']],
                    feed_dict = {
                        input_tensors['t_real_caption'] : caption_vectors,
                        input_tensors['t_z'] : z_noise,
                        input_tensors['noise_indicator'] : 0,
                        input_tensors['noise_gen'] : 0
                    })	
				idx = np.random.randint(1,3)
				image1 = img1[idx,:,:,:]
				image2 = img2[idx,:,:,:]
				image3 = img3[idx,:,:,:]
				image4 = img4[idx,:,:,:]
				image5 = img5[idx,:,:,:]
				real_full = real_images[idx,:,:,:]
				folder = 'images_Semi4/'
				scipy.misc.imsave(folder + str(i) + '_img_idx:' + str(batch_no) + 'stage1' + last_img + '.jpg',image1)
				scipy.misc.imsave(folder + str(i) + '_img_idx:' + str(batch_no) + 'stage2' + last_img + '.jpg',image2)
				scipy.misc.imsave(folder + str(i) + '_img_idx:' + str(batch_no) + 'stage3' + last_img + '.jpg',image3)
				scipy.misc.imsave(folder + str(i) + '_img_idx:' + str(batch_no) + 'stage4' + last_img + '.jpg',image4)
				scipy.misc.imsave(folder + str(i) + '_img_idx:' + str(batch_no) + 'stage5' + last_img + '.jpg',image5)
				scipy.misc.imsave(folder + str(i) + '_img_idx:' + str(batch_no) + last_img + '.jpg',real_full)
				
				img_idx += 1

			print('epoch:', i, 'batch:', batch_no)
			print ('d_loss', d_loss, d1_loss, d2_loss, d3_loss, d4_loss, d5_loss)
			print ('g_loss', g_loss, g1_loss, g2_loss, g3_loss, g4_loss, g5_loss)
			
			print ('d_loss_noise', d_loss_noise)
			print ('g_loss_noise', g_loss_noise)
			if 0:
				print('t1', t1)
				print('t2', t2)
				
			batch_no += 1
			if (batch_no % args.save_every) == 0:
				#Lee commented the following line out because it crashed. No idea what it was trying to do.
				#save_for_vis(args.data_dir, real_images, gen, image_files)
				print ("Saving Images, Model")
				save_path = saver.save(sess, "Data/Models/latest_model_Semi4_{}_temp.ckpt".format(args.data_set))			
		if i in args.save_epoch:
			save_path = saver.save(sess, "Data/ModelEval/Semi4_after_{}_epoch_{}.ckpt".format(i, args.data_set))

def load_training_data(data_dir, data_set):
	if data_set == 'flowers':
		h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
		flower_captions = {}
		for ds in h.iteritems():
			flower_captions[ds[0]] = np.array(ds[1])
		image_list = [key for key in flower_captions]
		image_list.sort()

		img_75 = int(len(image_list)*0.75)
		training_image_list = image_list[0:img_75]
		random.shuffle(training_image_list)
		
		return {
			'image_list' : training_image_list,
			'captions' : flower_captions,
			'data_length' : len(training_image_list)
		}
	
	else:
		with open(join(data_dir, 'meta_train.pkl')) as f:
			meta_data = pickle.load(f)
		# No preloading for MS-COCO
		return meta_data

def save_for_vis(data_dir, real_images, generated_images, image_files):
	
	shutil.rmtree( join(data_dir, 'samples') )
	os.makedirs( join(data_dir, 'samples') )

	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = (real_images[i,:,:,:])
		scipy.misc.imsave( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = (generated_images[i,:,:,:])
		scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim, 
	caption_vector_length, split, data_dir, data_set, loaded_data = None):
	if data_set == 'mscoco':
		with h5py.File( join(data_dir, 'tvs/'+split + '_tvs_' + str(batch_no))) as hf:
			caption_vectors = np.array(hf.get('tv'))
			caption_vectors = caption_vectors[:,0:caption_vector_length]
		with h5py.File( join(data_dir, 'tvs/'+split + '_tv_image_id_' + str(batch_no))) as hf:
			image_ids = np.array(hf.get('tv'))

		real_images = np.zeros((batch_size, 64, 64, 3))
		wrong_images = np.zeros((batch_size, 64, 64, 3))
		
		image_files = []
		for idx, image_id in enumerate(image_ids):
			image_file = join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, image_id) )
			image_array = image_processing.load_image_array(image_file, image_size)
			real_images[idx,:,:,:] = image_array
			image_files.append(image_file)
		
		# TODO>> As of Now, wrong images are just shuffled real images.
		first_image = real_images[0,:,:,:]
		for i in range(0, batch_size):
			if i < batch_size - 1:
				wrong_images[i,:,:,:] = real_images[i+1,:,:,:]
			else:
				wrong_images[i,:,:,:] = first_image

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])


		return real_images, wrong_images, caption_vectors, z_noise, image_files

	if data_set == 'flowers':
		real_images = np.zeros((batch_size, 64, 64, 3))
		wrong_images = np.zeros((batch_size, 64, 64, 3))
		captions = np.zeros((batch_size, caption_vector_length))

		cnt = 0
		image_files = []
		#caption_text = [None]*batch_size
		for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
			idx = i % len(loaded_data['image_list'])
			image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][idx])
			image_array = image_processing.load_image_array(image_file, image_size)
			real_images[cnt,:,:,:] = image_array
			
			# Improve this selection of wrong image
			wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
			wrong_image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][wrong_image_id])
			wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
			wrong_images[cnt, :,:,:] = wrong_image_array

			random_caption = random.randint(0,4)
			#caption_text[i] = random_caption
			captions[cnt,:] = loaded_data['captions'][ loaded_data['image_list'][idx] ][ random_caption ][0:caption_vector_length]
			image_files.append( image_file )
			cnt += 1

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return real_images, wrong_images, captions, z_noise, image_files#, caption_text

if __name__ == '__main__':
	main()
