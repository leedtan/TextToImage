from os.path import join
from os import listdir
from Utils import image_processing
import tensorflow as tf
import numpy as np

output_dir = "./images_Many1/"
num_stages = 5
image_size = 64

#def load_test_data(output_dir, image_size, num_stages):
    #image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][idx])
image_list =  [f for f in listdir(output_dir) if f[-3:]=='jpg']
image_list.sort(key=lambda x:int(x[:x.index("_")]))
num_group = len(image_list) / (num_stages + 1)
for i in range(num_group):
    real_image_name = image_list[i*(num_stages + 1)]
    fake_image_name = image_list[i*(num_stages + 1) + 5]
    real_image = image_processing.load_image_array(join(output_dir, real_image_name), image_size)
    fake_image = image_processing.load_image_array(join(output_dir, fake_image_name), image_size)
    real_image = np.einsum('ijk->kij', real_image)
    fake_image = np.einsum('ijk->kij', fake_image)
    t_real_image = tf.placeholder('float32', [3, image_size, image_size], name = 'real_image')
    t_fake_image = tf.placeholder('float32', [3, image_size, image_size], name = 'fake_image')
    s_r = tf.svd(t_real_image, compute_uv=False)
    s_f = tf.svd(t_fake_image, compute_uv=False)
    with tf.Session() as sess:
        sess.run([s_r], feed_dict={t_real_image:real_image})
        sess.run([s_f], feed_dict={t_fake_image:fake_image})
    print fake_image
    