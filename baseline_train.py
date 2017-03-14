import tensorflow as tf
import numpy as np
import baseline_model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()

prince = True

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=10,
					   help='Noise dimension')

	parser.add_argument('--t_dim', type=int, default= 100,#default=256,
					   help='Text feature dimension')

	parser.add_argument('--batch_size', type=int, default=64,#LEECHANGE default=64,
					   help='Batch Size')

	parser.add_argument('--image_size', type=int, default=64,
					   help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=10,
					   help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=4,#64 for real model
					   help='Number of conv in the first layer discr.')

	parser.add_argument('--gfc_dim', type=int, default=1024,
					   help='Dimension of gen untis for for fully connected layer 1024')

	parser.add_argument('--caption_vector_length', type=int, default=100,
					   help='Caption Vector Length')

	parser.add_argument('--data_dir', type=str, default="Data",
					   help='Data Directory')

	parser.add_argument('--learning_rate', type=float,default=1e-4,#LEECHANGE default=0.0002,
					   help='Learning Rate')

	parser.add_argument('--beta1', type=float, default =.5,#LEECHANGE default=0.5,
					   help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=6000,
					   help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=30,
					   help='Save Model/Samples every x iterations over batches')

	parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--data_set', type=str, default="flowers",
                       help='Dat set: MS-COCO, flowers')

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
	
	
	gan = baseline_model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()
	
	#d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1, beta2 = .9).minimize(loss['d_loss'], var_list=variables['d_vars'])
	#g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1, beta2 = .9).minimize(loss['g_loss'], var_list=variables['g_vars'])
	
	optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
	gvs = optimizer.compute_gradients(loss['d_loss'], var_list=variables['d_vars'])
	capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
	d_optim = optimizer.apply_gradients(capped_gvs)
	
	optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1 = args.beta1, beta2 = .9)
	gvs = optimizer.compute_gradients(loss['g_loss'], var_list=variables['g_vars'])
	capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
	g_optim = optimizer.apply_gradients(capped_gvs)
	cntr = 0
	sess = tf.InteractiveSession()
	if prince:
		sess.run(tf.global_variables_initializer())
	else:
		tf.initialize_all_variables().run()
	
	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)
	
	loaded_data = load_training_data(args.data_dir, args.data_set)
	for i in range(args.epochs):
		img_idx = 0
		batch_no = 0
		while batch_no*args.batch_size < loaded_data['data_length']:
			real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size, 
				args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)
			sess.run(
				[g_optim],
				feed_dict = {
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})
			if cntr == 0:
				cntr = 2
				_, _, fake_img, d_loss, g_loss= sess.run(
					[g_optim, d_optim, outputs['generator'], loss['d_loss'], loss['g_loss']],
					feed_dict = {
						input_tensors['t_real_image'] : real_images,
						input_tensors['t_wrong_image'] : wrong_images,
						input_tensors['t_real_caption'] : caption_vectors,
						input_tensors['t_z'] : z_noise,
					})
			cntr -= 1
			sess.run(
				[g_optim],
				feed_dict = {
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : caption_vectors,
					input_tensors['t_z'] : z_noise,
				})
			if batch_no % 10 == 0:
				
				idx = np.random.randint(1,30)
				img_full = fake_img[idx,:,:,:]
				
				scipy.misc.imsave('baseline_images2/' + str(i) + '_img_idx:' + str(batch_no) + 'full.jpg',img_full)
				
				img_idx += 1
				
		
			print "D_loss", d_loss
			print "G_loss", g_loss
			
				
			print "LOSSES", d_loss, g_loss, batch_no, i, len(loaded_data['image_list'])/ args.batch_size
			batch_no += 1
			if (batch_no % args.save_every) == 0:
				print "Saving Images, Model"
				#save_for_vis(args.data_dir, real_images, fake_img, image_files)
				save_path = saver.save(sess, "Data/Models/latest_model_baseline2_{}_temp.ckpt".format(args.data_set))
		if i%5 == 0:
			save_path = saver.save(sess, "Data/Models/baseline_model_after_{}_epoch_{}.ckpt".format(args.data_set, i))

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
			captions[cnt,:] = loaded_data['captions'][ loaded_data['image_list'][idx] ][ random_caption ][0:caption_vector_length]
			image_files.append( image_file )
			cnt += 1

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return real_images, wrong_images, captions, z_noise, image_files

if __name__ == '__main__':
	main()