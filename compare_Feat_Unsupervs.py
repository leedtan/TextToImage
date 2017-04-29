import tensorflow as tf
import argparse
import model1Feat as model1
import model2Unsupervised as model2
import random
import h5py
from os.path import join
import numpy as np
from Utils import image_processing
import pandas as pd


def main():
    prince = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=10,
                       help='Noise dimension')
    
    parser.add_argument('--t_dim', type=int, default=64,
                       help='Text feature dimension')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
    
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image Size a, a x a')
    
    parser.add_argument('--gf_dim', type=int, default=16,
                       help='Number of conv in the first layer gen.')
    
    parser.add_argument('--df_dim', type=int, default=16,
                       help='Number of conv in the first layer discr.')
    
    parser.add_argument('--gfc_dim', type=int, default=1024,
                       help='Dimension of gen untis for for fully connected layer 1024')
    
    parser.add_argument('--caption_vector_length', type=int, default=556,
                       help='Caption Vector Length')
    
    parser.add_argument('--data_dir', type=str, default="Data",
                       help='Data Directory')
    
    parser.add_argument('--beta1', type=float, default =.5,
                       help='Momentum for Adam Update')

    parser.add_argument('--data_set', type=str, default="flowers",
                       help='Data set: MS-COCO, flowers')

    args = parser.parse_args()
    
    save_epoch = [5,10,20,50]
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.54)
    
    cols = ['epoch_' + str(i) for i in save_epoch]
    df_loss = pd.DataFrame(index = cols, columns = cols)
    
    for i in range(len(save_epoch)):
        for j in range((len(save_epoch))):
            modelFile1 = "./Data/ModelEval/Feat_checkpoint_epoch_%d.ckpt" % (save_epoch[i])
            modelFile2 = "./Data/ModelEval/Unsupervs_checkpoint_epoch_%d.ckpt" % (save_epoch[j])
            
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
    
            gan1 = model1.GAN(model_options)
            input_tensors1, _, _, outputs1 = gan1.build_model(args.beta1, .9, 1e-4)                        
            sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            
            if prince:
                sess1.run(tf.global_variables_initializer())
            else:
                tf.initialize_all_variables().run()            
            saver = tf.train.Saver()
            # Restore the first model:
            saver.restore(sess1, modelFile1)        
            loaded_data = load_training_data(args.data_dir, args.data_set)
            batch_no = 0
            _, _, caption_vectors, z_noise, _ = get_training_batch(batch_no, args.batch_size, 
                args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)        
            # Get output image from first model
            g1_img3 = sess1.run(outputs1['img3'],
                feed_dict = {
                    input_tensors1['t_real_caption'] : caption_vectors,
                    input_tensors1['t_z'] : z_noise,
                    input_tensors1['noise_indicator'] : 0,
                    input_tensors1['noise_gen'] : 0
                })
            g1_d1_p_3_gen_img_logit, g1_d1_p_3_gen_txt_logit = sess1.run(
                [outputs1['output_p_3_gen_img_logit'],outputs1['output_p_3_gen_txt_logit']],
                feed_dict= {
                    input_tensors1['t_real_caption'] : caption_vectors,
                            input_tensors1['t_z'] : z_noise,
                            input_tensors1['noise_indicator'] : 0,
                            input_tensors1['gen_image1'] : g1_img3,
                            input_tensors1['noise_disc'] : 0,
                            input_tensors1['noise_gen'] : 0
                    })
            #print('g1_d1_p_3_gen_img_logit:')
            #print(g1_d1_p_3_gen_img_logit)
            g1_d1_loss = cross_entropy(g1_d1_p_3_gen_img_logit, np.ones((args.batch_size, 1))) + cross_entropy(g1_d1_p_3_gen_txt_logit, np.ones((args.batch_size, 1)))            
            tf.reset_default_graph()
            sess1.close()            
            # Create second model
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
            gan2 = model2.GAN(model_options)
            input_tensors2, _, _, outputs2 = gan2.build_model(args.beta1, .9, 1e-4)        
            sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        
            if prince:
                sess2.run(tf.global_variables_initializer())
            else:
                tf.initialize_all_variables().run()            
            saver2 = tf.train.Saver()            
            saver2.restore(sess2, modelFile2)            
            # Get logits from the second model
            g1_d2_p_3_gen_img_logit, g1_d2_p_3_gen_txt_logit = sess2.run(
                [outputs2['output_p_3_gen_img_logit'],outputs2['output_p_3_gen_txt_logit']],
                feed_dict= {
                    input_tensors2['t_real_caption'] : caption_vectors,
                            input_tensors2['t_z'] : z_noise,
                            input_tensors2['noise_indicator'] : 0,
                            input_tensors2['gen_image1'] : g1_img3,
                            input_tensors2['noise_disc'] : 0,
                            input_tensors2['noise_gen'] : 0
                    })            
            g1_d2_loss = cross_entropy(g1_d2_p_3_gen_img_logit, np.ones((args.batch_size, 1))) + cross_entropy(g1_d2_p_3_gen_txt_logit, np.ones((args.batch_size, 1)))            
            # Get output image from second model
            g2_img3 = sess2.run(outputs2['img3'],
                feed_dict = {
                    input_tensors2['t_real_caption'] : caption_vectors,
                    input_tensors2['t_z'] : z_noise,
                    input_tensors2['noise_indicator'] : 0,
                    input_tensors2['noise_gen'] : 0
                })            
            # Get logits from the second model
            g2_d2_p_3_gen_img_logit, g2_d2_p_3_gen_txt_logit = sess2.run(
                [outputs2['output_p_3_gen_img_logit'],outputs2['output_p_3_gen_txt_logit']],
                feed_dict= {
                    input_tensors2['t_real_caption'] : caption_vectors,
                            input_tensors2['t_z'] : z_noise,
                            input_tensors2['noise_indicator'] : 0,
                            input_tensors2['gen_image1'] : g2_img3,
                            input_tensors2['noise_disc'] : 0,
                            input_tensors2['noise_gen'] : 0
                    })            
            g2_d2_loss = cross_entropy(g2_d2_p_3_gen_img_logit, np.ones((args.batch_size, 1))) + cross_entropy(g2_d2_p_3_gen_txt_logit, np.ones((args.batch_size, 1)))            
            tf.reset_default_graph()
            sess2.close()            
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
            gan1 = model1.GAN(model_options)
            input_tensors1, _, _, outputs1 = gan1.build_model(args.beta1, .9, 1e-4)            
            sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))            
            if prince:
                sess1.run(tf.global_variables_initializer())
            else:
                tf.initialize_all_variables().run()          
            saver = tf.train.Saver()          
            saver.restore(sess1, modelFile1)
            # Get logits from the first model
            g2_d1_p_3_gen_img_logit, g2_d1_p_3_gen_txt_logit = sess1.run(
                [outputs1['output_p_3_gen_img_logit'],outputs1['output_p_3_gen_txt_logit']],
                feed_dict= {
                    input_tensors1['t_real_caption'] : caption_vectors,
                            input_tensors1['t_z'] : z_noise,
                            input_tensors1['noise_indicator'] : 0,
                            input_tensors1['gen_image1'] : g2_img3,
                            input_tensors1['noise_disc'] : 0,
                            input_tensors1['noise_gen'] : 0
                    })
            #print('g1_d1_p_3_gen_img_logit:')
            #print(g1_d1_p_3_gen_img_logit)
            #print('g2_d1_p_3_gen_img_logit:')
            #print(g2_d1_p_3_gen_img_logit)
            tf.reset_default_graph()
            sess1.close()
            g2_d1_loss = cross_entropy(g2_d1_p_3_gen_img_logit, np.ones((args.batch_size, 1))) + cross_entropy(g2_d1_p_3_gen_txt_logit, np.ones((args.batch_size, 1)))
            g1_wins_on_d2 = 0
            g2_wins_on_d1 = 0
            for idx in range(g2_d1_loss.shape[0]):
                # Compare loss on disc 1
                if g1_d1_loss[idx][0] > g2_d1_loss[idx][0]:
                    g2_wins_on_d1 += 1
                    print(g2_d1_loss[idx][0],'<', g1_d1_loss[idx][0], 'g2 wins on d1')
                else:
                    print(g2_d1_loss[idx][0],'>', g1_d1_loss[idx][0], 'g1 wins on d1')                
                
                # Compare loss on disc 2
                if g1_d2_loss[idx][0] < g2_d2_loss[idx][0]:
                    g1_wins_on_d2 += 1
                    print(g1_d2_loss[idx][0],'<', g2_d2_loss[idx][0], 'g1 wins on d2')
                else:
                    print(g1_d2_loss[idx][0],'>', g2_d2_loss[idx][0], 'g2 wins on d2')
                
            df_loss.loc[cols[i],cols[j]] = str(g2_wins_on_d1)+'/'+str(g1_wins_on_d2)
    df_loss.to_csv('Feat_Uns.csv')
    
def cross_entropy(logits, labels):
    loss = np.maximum(logits, 0) - logits * labels + np.log(1 + np.exp(-np.abs(labels)))
    return loss
            
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

def get_training_batch(batch_no, batch_size, image_size, z_dim, 
    caption_vector_length, split, data_dir, data_set, loaded_data = None):
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
