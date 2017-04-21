import tensorflow as tf
import numpy as np
import modelServer99 as model
import sys
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
from scipy import ndimage
plt.ioff()

def random_blur(imgs):
    sig = np.random.uniform(1.2,1.6)##.8
    noi = np.random.uniform(.08,.12)#.1
    ofs = np.random.uniform(-.1,.1,3)
    return np.clip(ndimage.gaussian_filter(imgs, sigma=[0,sig,sig,0]) + ofs + 
                   np.random.randn(imgs.shape[0],imgs.shape[1],imgs.shape[2],imgs.shape[3])*noi, 0, 1)

def super_blur(imgs):
    ofs_lim = 1.5
    half_imgs = int(imgs.shape[0]/2)
    imgs1 = (imgs[:half_imgs,:,:,:]-0.5) / 10
    imgs2 = (imgs[half_imgs:,:,:,:]-0.5) / 10
    sig = np.random.uniform(4,10)##.8
    if sig < 6:
        noi = np.random.uniform(.1,.3)
    else:
        noi = np.random.uniform(.3,1.)#.1
    ofs = np.random.uniform(-.2,1.2,3)
    imgs1 = np.clip(ndimage.gaussian_filter(imgs1, sigma=[sig/4,sig,sig,0]) + ofs + 
                   np.random.randn(imgs1.shape[0],imgs1.shape[1],imgs1.shape[2],imgs1.shape[3])*noi, 0, 1)
    sig = np.random.uniform(4,10)##.8
    if sig < 6:
        noi = np.random.uniform(.1,.3)
    else:
        noi = np.random.uniform(.3,1.)#.1
    ofs = np.random.uniform(-.2,1.2,3)
    imgs2 = np.clip(ndimage.gaussian_filter(imgs2, sigma=[sig/4,sig,sig,0]) + ofs + 
                   np.random.randn(imgs2.shape[0],imgs2.shape[1],imgs2.shape[2],imgs2.shape[3])*noi, 0, 1)
    return np.concatenate((imgs1,imgs2))

prince = True
#TO try: just reduced caption vector length, also remove gradient clipping. and increase learning rate *2.
def main():
    
    class Logger(object):
        def __init__(self, filename="last_run_output.txt"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()
            
        def flush(self):
            self.log.flush()
            
    sys.stdout = Logger("logs/" + str(os.path.basename(sys.argv[0])) +
                        str(time.time()) + ".txt")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=10,
                       help='Noise dimension')

    parser.add_argument('--t_dim', type=int, default= 64,#1024#LEE,#default=256,
                       help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=64,#LEECHANGE default=64,
                       help='Batch Size')

    parser.add_argument('--image_size', type=int, default=64,
                       help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=16,#30
                       help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=16,#128,#12
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

    parser.add_argument('--epochs', type=int, default=11,
                       help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=60,
                       help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--data_set', type=str, default="flowers",
                       help='Dat set: MS-COCO, flowers')

    parser.add_argument('--save_epoch', type=list, default=[2,5,7,10],
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
    
    gan = model.GAN(model_options)
    input_tensors, variables, loss, outputs = gan.build_model(args.beta1, beta2, args.learning_rate)
    
    g_optim = gan.g_optim
    d_optim = gan.d_optim
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.InteractiveSession()
    if prince:
        sess.run(tf.global_variables_initializer())
        '''
        sess.run(tf.global_variables_initializer(),feed_dict = {
            input_tensors['t_real_image'] : np.zeros((64,64,64,3)),
            input_tensors['t_real_caption'] : np.zeros((64, 100)),
            input_tensors['t_z'] : np.zeros((64,10)),
            input_tensors['noise_gen'] : 0,
            input_tensors['noise_disc'] : 0})
        '''
    else:
        tf.initialize_all_variables().run()
    
    saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)
    
    loaded_data = load_training_data(args.data_dir, args.data_set)
    init = -1
    skip_n = 0
    d_avg_full, d_avg_mid, d_avg_sml = 0,0,0
    lb = 0#-.3
    disc_break = -.3
    t1 = 0
    t2 = 0
    d_loss_noise = g_loss_noise = 0
    d1_loss =d2_loss= d3_loss=3
    num_disc_steps = 0
    time_size = 10368#df dim 30 : 19440
    real_time = np.zeros(time_size)
    fake_time = np.zeros(time_size)
    for i in range(args.epochs):
        noise_gen = .1/np.sqrt(i+1)
        noise_disc = .1/np.sqrt(i+1)
        t = time.time()
        img_idx = 0
        batch_no = 0
        caption_vectors = 0
        mul1=1
        mul2=1
        mul3=1
        while batch_no < 2:#batch_no*args.batch_size < loaded_data['data_length']:
            trans_mult = 5/np.sqrt(25+i)
            lr = 1e-4/np.sqrt(1+i)
            real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size, 
                args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)
            c_off1 = np.random.randint(2,10)
            c_off2 = np.random.randint(11,20)
            caption_wrong1 = np.concatenate((caption_vectors[c_off1:], caption_vectors[:c_off1]))
            caption_wrong2 = np.concatenate((caption_vectors[c_off2:], caption_vectors[:c_off2]))
            t1 += time.time()-t
            t = time.time()
            # DISCR UPDATE
            lambdaDAE = 10
            horrible_images = super_blur(real_images)
            rt, ft, _, g_loss, g1_loss, g2_loss, g3_loss, \
                img1, img2,img3   = sess.run(
                [gan.real_acts, gan.fake_acts, g_optim, loss['g_loss'],
                 loss['g1_loss'], loss['g2_loss'], loss['g3_loss'],
                outputs['img1'],outputs['img2'], outputs['img3']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_real_caption'] : caption_vectors,
                    input_tensors['cap_wrong1'] : caption_wrong1,
                    input_tensors['cap_wrong2'] : caption_wrong2,
                    input_tensors['t_z'] : z_noise,
                    input_tensors['l2reg']: 0,
                    input_tensors['LambdaDAE']: lambdaDAE,
                    input_tensors['noise_indicator'] : 0,
                    input_tensors['noise_gen'] : noise_gen,
                    input_tensors['noise_disc'] : noise_disc,
                    input_tensors['mul1'] : mul1,
                    input_tensors['mul2'] : mul2,
                    input_tensors['mul3'] : mul3,
                    gan.past_reals: real_time,
                    gan.past_fakes: fake_time,
                    gan.trans_mult : trans_mult,
                    input_tensors['lr'] : lr
                })
            
            real_time = real_time * .9 + rt * .1
            fake_time = fake_time * .9 + ft * .1

            d1_loss, d2_loss, d3_loss   = sess.run(
                [loss['d1_loss_gen'], loss['d2_loss_gen'], loss['d3_loss_gen']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_real_caption'] : caption_vectors,
                    input_tensors['cap_wrong1'] : caption_wrong1,
                    input_tensors['cap_wrong2'] : caption_wrong2,
                    input_tensors['LambdaDAE']: lambdaDAE,
                    input_tensors['t_z'] : z_noise,
                    input_tensors['noise_indicator'] : 0,
                    input_tensors['noise_gen'] : noise_gen,
                    input_tensors['mul1'] : mul1,
                    input_tensors['mul2'] : mul2,
                    input_tensors['mul3'] : mul3,
                    input_tensors['gen_image1'] : img3,
                    input_tensors['gen_image2'] : img2,
                    input_tensors['gen_image4'] : img1,
                    input_tensors['noise_disc'] : noise_disc
                })
            if np.min([d1_loss, d2_loss, d3_loss])>1.1 or init < 0:
                if init < 0:
                    init += 1
                num_disc_steps += 1
                print('running real disc')
                if init < 0:
                    init += 1
                sess.run(
                        [d_optim],
                        feed_dict = {
                            input_tensors['t_real_image'] : real_images,
                            input_tensors['t_wrong_image'] : wrong_images,
                            input_tensors['t_horrible_image'] : horrible_images,
                            input_tensors['t_real_caption'] : caption_vectors,
                            input_tensors['cap_wrong1'] : caption_wrong1,
                            input_tensors['cap_wrong2'] : caption_wrong2,
                            input_tensors['t_z'] : z_noise,
                            input_tensors['l2reg']: 0,
                            input_tensors['LambdaDAE']: lambdaDAE,
                            input_tensors['noise_indicator'] : 0,
                            input_tensors['noise_gen'] : noise_gen,
                            input_tensors['noise_disc'] : noise_disc,
                            input_tensors['mul1'] : mul1,
                            input_tensors['mul2'] : mul2,
                            input_tensors['mul3'] : mul3,
                            gan.trans_mult : trans_mult,
                            input_tensors['gen_image1'] : img3,
                            input_tensors['gen_image2'] : img2,
                            input_tensors['gen_image4'] : img1,
                            input_tensors['lr'] : lr
                        })
            else:
                sess.run(gan.l2_disc)
                rt, ft, _, g_loss, g1_loss, g2_loss, g3_loss, \
                    img1, img2,img3   = sess.run(
                    [gan.real_acts, gan.fake_acts, g_optim, loss['g_loss'], 
                     loss['g1_loss'], loss['g2_loss'], loss['g3_loss'],
                    outputs['img1'],outputs['img2'], outputs['img3']],
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_real_caption'] : caption_vectors,
                        input_tensors['cap_wrong1'] : caption_wrong1,
                        input_tensors['cap_wrong2'] : caption_wrong2,
                        input_tensors['t_z'] : z_noise,
                        input_tensors['l2reg']: 0,
                        input_tensors['LambdaDAE']: lambdaDAE,
                        input_tensors['noise_indicator'] : 0,
                        input_tensors['noise_gen'] : noise_gen,
                        input_tensors['noise_disc'] : noise_disc,
                        input_tensors['mul1'] : mul1,
                        input_tensors['mul2'] : mul2,
                        input_tensors['mul3'] : mul3,
                        gan.past_reals: real_time,
                        gan.past_fakes: fake_time,
                        gan.trans_mult : trans_mult,
                        input_tensors['lr'] : lr
                    })
                real_time = real_time * .9 + rt * .1
                fake_time = fake_time * .9 + ft * .1
            
            ############NOISE!
            rt, ft, _, g_loss_noise, g1_loss_noise, g2_loss_noise, g3_loss_noise, \
                img1_noise, img2_noise,img3_noise   = sess.run(
                [gan.real_acts, gan.fake_acts, g_optim, loss['g_loss'], 
                 loss['g1_loss_noise'], loss['g2_loss_noise'], loss['g3_loss_noise'],
                outputs['img1'],outputs['img2'], outputs['img3']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_real_caption'] : np.random.rand(
                                args.batch_size, args.caption_vector_length)*.02,
                    input_tensors['cap_wrong1'] : np.random.rand(
                                args.batch_size, args.caption_vector_length)*.02,
                    input_tensors['cap_wrong2'] : np.random.rand(
                                args.batch_size, args.caption_vector_length)*.02,
                    input_tensors['t_z'] : z_noise,
                    input_tensors['l2reg']: 0,
                    input_tensors['LambdaDAE']: lambdaDAE,
                    input_tensors['noise_indicator'] : 1,
                    input_tensors['noise_gen'] : noise_gen,
                    input_tensors['noise_disc'] : noise_disc,
                    input_tensors['mul1'] : mul1,
                    input_tensors['mul2'] : mul2,
                    input_tensors['mul3'] : mul3,
                    gan.past_reals: real_time,
                    gan.past_fakes: fake_time,
                    gan.trans_mult : trans_mult,
                    input_tensors['lr'] : lr
                })
            real_time = real_time * .9 + rt * .1
            fake_time = fake_time * .9 + ft * .1
                
            d1_loss_noise, d2_loss_noise, d3_loss_noise   = sess.run(
                [loss['d1_loss_gen_noise'], loss['d2_loss_gen_noise'], loss['d3_loss_gen_noise']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_real_caption'] : np.random.rand(
                                args.batch_size, args.caption_vector_length)*.02,
                    input_tensors['cap_wrong1'] : np.random.rand(
                                args.batch_size, args.caption_vector_length)*.02,
                    input_tensors['cap_wrong2'] : np.random.rand(
                                args.batch_size, args.caption_vector_length)*.02,
                    input_tensors['LambdaDAE']: lambdaDAE,
                    input_tensors['t_z'] : z_noise,
                    input_tensors['noise_indicator'] : 1,
                    input_tensors['noise_gen'] : noise_gen,
                    input_tensors['mul1'] : mul1,
                    input_tensors['mul2'] : mul2,
                    input_tensors['mul3'] : mul3,
                    input_tensors['gen_image1'] : img3_noise,
                    input_tensors['gen_image2'] : img2_noise,
                    input_tensors['gen_image4'] : img1_noise,
                    input_tensors['noise_gen'] : noise_gen,
                    input_tensors['noise_disc'] : noise_disc
                })
            if np.min([d1_loss_noise, d2_loss_noise, d3_loss_noise])>.8:
                num_disc_steps += 1
                if init < 0:
                    init += 1
                print('running noise disc')
                horrible_images = super_blur(real_images)
                sess.run(
                        [d_optim],
                        feed_dict = {
                            input_tensors['t_real_image'] : real_images,
                            input_tensors['t_wrong_image'] : wrong_images,
                            input_tensors['t_horrible_image'] : horrible_images,
                            input_tensors['t_real_caption'] : np.random.rand(
                                args.batch_size, args.caption_vector_length)*.02,
                            input_tensors['cap_wrong1'] : np.random.rand(
                                        args.batch_size, args.caption_vector_length)*.02,
                            input_tensors['cap_wrong2'] : np.random.rand(
                                        args.batch_size, args.caption_vector_length)*.02,
                            input_tensors['t_z'] : z_noise,
                            input_tensors['l2reg']: 0,
                            input_tensors['LambdaDAE']: lambdaDAE,
                            input_tensors['noise_indicator'] : 1,
                            input_tensors['noise_gen'] : noise_gen,
                            input_tensors['noise_disc'] : noise_disc,
                            input_tensors['mul1'] : mul1,
                            input_tensors['mul2'] : mul2,
                            input_tensors['mul3'] : mul3,
                            gan.past_reals: real_time,
                            gan.past_fakes: fake_time,
                            gan.trans_mult : trans_mult,
                            input_tensors['gen_image1'] : img3_noise,
                            input_tensors['gen_image2'] : img2_noise,
                            input_tensors['gen_image4'] : img1_noise,
                            input_tensors['lr'] : lr
                        })
            else:
                sess.run(gan.l2_disc)
                rt, ft, _, g_loss_noise, g1_loss_noise, g2_loss_noise, g3_loss_noise, \
                    img1_noise, img2_noise,img3_noise   = sess.run(
                    [gan.real_acts, gan.fake_acts, g_optim, loss['g_loss'], 
                     loss['g1_loss_noise'], loss['g2_loss_noise'], loss['g3_loss_noise'],
                    outputs['img1'],outputs['img2'], outputs['img3']],
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_real_caption'] : np.random.rand(
                                    args.batch_size, args.caption_vector_length)*.02,
                        input_tensors['cap_wrong1'] : np.random.rand(
                                    args.batch_size, args.caption_vector_length)*.02,
                        input_tensors['cap_wrong2'] : np.random.rand(
                                    args.batch_size, args.caption_vector_length)*.02,
                        input_tensors['t_z'] : z_noise,
                        input_tensors['l2reg']: 0,
                        input_tensors['LambdaDAE']: lambdaDAE,
                        input_tensors['noise_indicator'] : 1,
                        input_tensors['noise_gen'] : noise_gen,
                        input_tensors['noise_disc'] : noise_disc,
                        gan.past_reals: real_time,
                        gan.past_fakes: fake_time,
                        input_tensors['mul1'] : mul1,
                        input_tensors['mul2'] : mul2,
                        input_tensors['mul3'] : mul3,
                        gan.trans_mult : trans_mult,
                        input_tensors['lr'] : lr
                    })
                
                real_time = real_time * .9 + rt * .1
                fake_time = fake_time * .9 + ft * .1
            
            
            t2 += time.time()-t
            t = time.time()
            
            if batch_no % 5 == 0:
                img1, img2,img3, trans1, trans2, trans3, trans21, trans22, trans23   = sess.run(
                    [outputs['img1'],outputs['img2'], outputs['img3'],outputs['trans1'],outputs['trans2'], outputs['trans3'],
                     outputs['trans21'],outputs['trans22'], outputs['trans23']],
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_real_caption'] : caption_vectors,
                        input_tensors['t_z'] : z_noise,
                        input_tensors['noise_indicator'] : 0,
                        input_tensors['noise_gen'] : 0,
                        input_tensors['noise_disc'] : 0,
                        input_tensors['gen_image1'] : img3,
                        input_tensors['gen_image2'] : img2,
                        input_tensors['gen_image4'] : img1
                    })
                idx = np.random.randint(1,33)
                image1 = img1[idx,:,:,:]
                image2 = img2[idx,:,:,:]
                image3 = img3[idx,:,:,:]
                trs1 = trans1[idx,:,:,:]
                trs2 = trans2[idx,:,:,:]
                trs3 = trans3[idx,:,:,:]
                trs21 = trans21[idx,:,:,:]
                trs22 = trans22[idx,:,:,:]
                trs23 = trans23[idx,:,:,:]
                real_full = real_images[idx,:,:,:]
                horrible_full = horrible_images[idx,:,:,:]
                folder = 'images_Server99/'
                if 0:
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'stage1.jpg',image1)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'stage2.jpg',image2)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'stage3.jpg',image3)
                if 0:
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'trans1.jpg',trs1)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'trans2.jpg',trs2)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'trans3.jpg',trs3)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'trans21.jpg',trs21)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'trans22.jpg',trs22)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'trans23.jpg',trs23)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'horrible.jpg',horrible_full)
                    scipy.misc.imsave(folder + str(i) + '_img_idx' + str(batch_no) + 'real.jpg',real_full)
                img_idx += 1
                
            print(num_disc_steps, ' disc steps so far')
            print('epoch:', i, 'batch:', batch_no)
            print ('d_loss', d1_loss, d2_loss, d3_loss)
            print ('g_loss', g1_loss, g2_loss, g3_loss)
            print ('d_loss_noise', d1_loss_noise, d2_loss_noise, d3_loss_noise)
            print ('g_loss_noise', g1_loss_noise, g2_loss_noise, g3_loss_noise)
            
                
            batch_no += 1
            if 0:#(batch_no % args.save_every) == 0:
                #Lee commented the following line out because it crashed. No idea what it was trying to do.
                #save_for_vis(args.data_dir, real_images, gen, image_files)
                save_path = saver.save(sess, "Data/Models/weitaobattle1.ckpt")
        if 0:#i in args.save_epoch:
            save_path = saver.save(sess, "Data/ModelEval/Server100_after_{}_epoch_{}.ckpt".format(i, args.data_set))

def load_training_data(data_dir, data_set):
    if data_set == 'flowers':
        h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
        flower_captions = {}
        for ds in h.items():
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
    img_size = 64
    shutil.rmtree( join(data_dir, 'samples') )
    os.makedirs( join(data_dir, 'samples') )

    for i in range(0, real_images.shape[0]):
        real_image_255 = np.zeros( (img_size,img_size,3), dtype=np.uint8)
        real_images_255 = (real_images[i,:,:,:])
        scipy.misc.imsave( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

        fake_image_255 = np.zeros( (img_size,img_size,3), dtype=np.uint8)
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

        real_images = np.zeros((batch_size, image_size, image_size, 3))
        wrong_images = np.zeros((batch_size, image_size, image_size, 3))
        
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
        real_images = np.zeros((batch_size, image_size, image_size, 3))
        wrong_images = np.zeros((batch_size, image_size, image_size, 3))
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
