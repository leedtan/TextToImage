import tensorflow as tf
from Utils import ops
from tensorflow.contrib.distributions import MultivariateNormalDiag as tf_norm

prince = True

def reshape(x, arr):
    return tf.reshape(x, [int(a) for a in arr])

def tf_resize(x, size):
    return tf.image.resize_images(x, (size, size))

def spread(x):
    s = x.get_shape()[1].value
    means = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(x,axis=[1,2]),1),2), [1, s, s, 1])
    diffs_fourthed = tf.pow(x - means, 4)
    return tf.pow( tf.reduce_mean(diffs_fourthed,axis=[1,2]),1/4)

def transform_images(transform, gamma = .5):
    s = transform.get_shape()[1].value
    transform = tf.map_fn(lambda img: tf.image.random_saturation(img, .7, 1.3), transform)
    transform = tf.map_fn(lambda img: tf.image.random_hue(img, .03), transform)
    transform = tf.map_fn(lambda img: tf.image.random_contrast(img, .8, 1.3), transform)
    transform = tf.map_fn(lambda img: tf.random_crop(img, [int(s*7/8),int(s*7/8),3]), transform)
    transform = tf_resize(transform, s)
    transform = tf.map_fn(lambda img: tf.image.random_brightness(img, .1), transform)
    transform = tf.map_fn(lambda img: tf.image.adjust_gamma(img, gamma, 1), transform)
    transform = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), transform)
    return transform

def transform_images_gen(transform, gamma = .5):
    s = transform.get_shape()[1].value
    transform = tf.map_fn(lambda img: tf.image.random_saturation(img, .8, 1.3), transform)
    transform = tf.map_fn(lambda img: tf.image.random_hue(img, .02), transform)
    transform = tf.map_fn(lambda img: tf.random_crop(img, [int(s*7/8),int(s*7/8),3]), transform)
    transform = tf_resize(transform, s)
    transform = tf.map_fn(lambda img: tf.image.random_brightness(img, .05), transform)
    transform = tf.map_fn(lambda img: tf.image.adjust_gamma(img, gamma, 1), transform)
    transform = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), transform)
    return transform

def generate_noise_images(transform):
    s = transform.get_shape()[1].value
    bs = 64
    base = tf.random_uniform([bs,1,1,3],0,1)
    base = tf.tile(base,[1,64,64,1])
    random_mult = tf.random_uniform([bs,1,1,1],0,1)
    random_mult = tf.tile(random_mult,[1,64,64,3])
    return tf.clip_by_value(base + tf.random_uniform(transform.get_shape(),-1, 1) * random_mult,0,1)

def generate_gray_noise_images(transform):
    s = transform.get_shape()[1].value
    bs = 64
    base = tf.random_uniform([bs,1,1,3],.4,.6)
    base = tf.tile(base,[1,64,64,1])
    random_mult = tf.random_uniform([bs,1,1,1],0,.3)
    random_mult = tf.tile(random_mult,[1,64,64,3])
    return tf.clip_by_value(base + tf.random_uniform(transform.get_shape(),-1, 1) * random_mult,0,1)

def generate_noise_blobs(transform):
    s = transform.get_shape()[1].value
    bs = 64
    base = tf.random_uniform([bs,1,1,3],0.1,0.9)
    base = tf.tile(base,[1,64,64,1])
    random_mult = tf.random_uniform([bs,1,1,1],0,1)
    random_mult = tf.tile(random_mult,[1,64,64,3])
    blob =  generate_basic_images(transform) - 0.5
    random_mult_blob = tf.random_uniform([bs,1,1,1],0,0.5)
    random_mult_blob = tf.tile(random_mult_blob,[1,64,64,3])
    blob = blob * random_mult_blob
    return tf.clip_by_value(base + tf.random_uniform(transform.get_shape(),-1, 1) * random_mult + blob,0,1)

def generate_gray_noise_blobs(transform):
    s = transform.get_shape()[1].value
    bs = 64
    base = tf.random_uniform([bs,1,1,3],0.4,0.6)
    base = tf.tile(base,[1,64,64,1])
    random_mult = tf.random_uniform([bs,1,1,1],0,.3)
    random_mult = tf.tile(random_mult,[1,64,64,3])
    blob =  generate_basic_images(transform) - 0.5
    random_mult_blob = tf.random_uniform([bs,1,1,1],0,0.2)
    random_mult_blob = tf.tile(random_mult_blob,[1,64,64,3])
    blob = blob * random_mult_blob
    return tf.clip_by_value(base + tf.random_uniform(transform.get_shape(),-1, 1) * random_mult + blob,0,1)

def generate_basic_images(transform, gamma = .5):
    s = transform.get_shape()[1].value
    bs = 64
    transform = tf.ones_like(transform,dtype = tf.float32) * 0.5 #dtype = float32?
    idxs = tf.constant(list(range(64)),dtype = tf.float32)
    idxs1 = tf.expand_dims(idxs, 0)
    idxs1 = tf.tile(idxs1, [64,1])
    idxs2 = tf.expand_dims(idxs, 1)
    idxs2 = tf.tile(idxs2, [1,64])
    idxs1 = tf.expand_dims(idxs1, 0)
    idxs1 = tf.expand_dims(idxs1, 3)
    idxs1 = tf.tile(idxs1, [bs,1,1,3])
    idxs2 = tf.expand_dims(idxs2, 0)
    idxs2 = tf.expand_dims(idxs2, 3)
    idxs2 = tf.tile(idxs2, [bs,1,1,3])
    for _ in range(10):
        #loc1 = tf.random_uniform([bs], 0, 64)random_normal
        loc1 = tf.random_normal([bs], 32, 16)
        std1 = tf.random_normal([bs], 32, 16)
        #std1 = tf.random_uniform([bs], 20, 60)
        #loc2= tf.random_uniform([bs], 0, 64)
        loc2 = tf.random_normal([bs], 32, 16)
        std2 = tf.random_normal([bs], 32, 16)
        #std2 = tf.random_uniform([bs], 20, 60)
        loc1 = tf.expand_dims(loc1, 1)
        loc1 = tf.expand_dims(loc1, 1)
        loc1 = tf.expand_dims(loc1, 1)
        loc1 = tf.tile(loc1,[1,64,64,3])
        loc2 = tf.expand_dims(loc2, 1)
        loc2 = tf.expand_dims(loc2, 1)
        loc2 = tf.expand_dims(loc2, 1)
        loc2 = tf.tile(loc2,[1,64,64,3])
        
        std1 = tf.expand_dims(std1, 1)
        std1 = tf.expand_dims(std1, 1)
        std1 = tf.expand_dims(std1, 1)
        std1 = tf.tile(std1,[1,64,64,3])
        
        std2 = tf.expand_dims(std2, 1)
        std2 = tf.expand_dims(std2, 1)
        std2 = tf.expand_dims(std2, 1)
        std2 = tf.tile(std2,[1,64,64,3])
        
        clr = tf.random_uniform([bs,3], -10000, 10000)
        clr = tf.expand_dims(clr,1)
        clr = tf.expand_dims(clr,1)
        clr = tf.tile(clr,[1,64,64,1])
        
        transform = transform + clr * tf.exp(-1*tf.square((idxs1 - loc1)/std1)/2-tf.square((idxs2 - loc2)/std2)/2 )/std1/std2/6.5
    
    transform = tf.map_fn(lambda img: tf.image.random_hue(img, .2), transform)
    transform = tf.clip_by_value(transform, 0, 1)
    transform = tf.map_fn(lambda img: tf.random_crop(img, [int(s*7/8),int(s*7/8),3]), transform)
    transform = tf_resize(transform, s)
    return transform


if prince:
    def concat(dim, objects, name=None):
        if name is None:
            return tf.concat(objects, dim)
        else:
            return tf.concat(objects, dim, name = None)
    def c_e(logits, labels):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
    def pack(x):
        return tf.stack(x)
else:
    def concat(dim, objects, name=None):
        if name is None:
            return tf.concat(dim, objects)
        else:
            return tf.concat(dim, objects, name = None)
    def c_e(logits, labels):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
    def pack(x):
        return tf.pack(x)


class GAN:
    '''
    OPTIONS
    z_dim : Noise dimension 100
    t_dim : Text feature dimension 256
    image_size : Image Dimension 64
    gf_dim : Number of conv in the first layer generator 64
    df_dim : Number of conv in the first layer discriminator 64
    gfc_dim : Dimension of gen untis for for fully connected layer 1024
    caption_vector_length : Caption Vector Length 2400
    batch_size : Batch Size 64
    '''
    def d_bn(self):
        d_bn = ops.batch_norm(name='d_bn_clean' + str(self.d_bn_idx))
        self.d_bn_idx += 1
        return d_bn
    def __init__(self, options):
        self.options = options
        self.d_bn_idx = 0
        self.g_text_bn = ops.batch_norm(name='g_text_bn')
        self.d_text_bn = ops.batch_norm(name='d_text_bn')
        #Hope to remove these eventually, some are needed for code I havent refactored
        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')
        self.g_bn4 = ops.batch_norm(name='g_bn4')
        
        self.g_idx = 0
        self.g2_idx = 0
        self.g3_idx = 0
        
        self.d_idx = 0
        self.g_bn_idx = 0

    def g_bn(self):
        g_bn = ops.batch_norm(name='g_bn_clean' + str(self.g_bn_idx))
        self.g_bn_idx += 1
        return g_bn
        
    def build_model(self, beta1, beta2, lr):
        img_size = self.options['image_size']
        
        l2reg = tf.placeholder('float32', shape=())
        
        LambdaDAE = tf.placeholder('float32', shape=())
        
        t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
        t_trans_image = transform_images(t_real_image, gamma = .7)
        t_trans2_image = transform_images(t_real_image, gamma = 1.2)

        t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
        t_horrible_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'horrible_image')
        
        gen_image1 = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'gen_image1')
        gen_image2 = tf.placeholder('float32', [self.options['batch_size'],img_size/2, img_size/2, 3 ], name = 'gen_image2')
        gen_image4 = tf.placeholder('float32', [self.options['batch_size'],img_size/4, img_size/4, 3 ], name = 'gen_image4')
        self.trans_mult = tf.placeholder('float32', shape=())#tf.placeholder_with_default([.5],1)
        trans1_gen_img1 = transform_images_gen(gen_image1, .8)
        trans2_gen_img1 = transform_images_gen(gen_image1, 1.3)
        trans1_gen_img2 = transform_images_gen(gen_image2, .8)
        trans2_gen_img2 = transform_images_gen(gen_image2, 1.3)
        trans1_gen_img4 = transform_images_gen(gen_image4, .8)
        trans2_gen_img4 = transform_images_gen(gen_image4, 1.3)
        
        real2 = tf_resize(t_real_image, 32)
        real4 = tf_resize(t_real_image, 16)
        trans2 = transform_images(real2, gamma = .7)
        trans4 = transform_images(real4, gamma = .7)
        trans22 = transform_images(real2, gamma = 1.2)
        trans24 = transform_images(real4, gamma = 1.2)
        wrong2 = tf_resize(t_wrong_image, 32)
        wrong4 = tf_resize(t_wrong_image, 16)
        horrible2 = tf_resize(t_horrible_image, 32)
        horrible4 = tf_resize(t_horrible_image, 16)
        
        noise_indicator = tf.placeholder('float32', shape=())
        self.noise_indicator = noise_indicator
        self.noise_gen = tf.placeholder('float32', shape=())
        self.noise_disc = tf.placeholder('float32', shape=())
        mul1 = tf.placeholder('float32', shape=())
        mul2 = tf.placeholder('float32', shape=())
        mul3 = tf.placeholder('float32', shape=())
        self.lr = tf.placeholder('float32', shape=())
        
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], 
                    self.options['caption_vector_length']], name = 'real_caption_input')
        caption_wrong1 = tf.placeholder('float32', [self.options['batch_size'], 
                    self.options['caption_vector_length']], name = 'wrong_caption_input1')
        caption_wrong2 = tf.placeholder('float32', [self.options['batch_size'], 
                    self.options['caption_vector_length']], name = 'wrong_caption_input2')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
        #t_normal_caption = tf.nn.l2_normalize(t_real_caption,1)
        g_normal_caption = self.g_text_bn(t_real_caption)
        g_raw_normal_caption = tf.nn.l2_normalize(t_real_caption, 0)
        with tf.variable_scope('d_bn_scope'):
            tf.get_variable_scope()._reuse = None
            d_normal_caption = self.d_text_bn(t_real_caption)
            d_caption_wrong1 = self.d_text_bn(caption_wrong1)
            d_caption_wrong2 = self.d_text_bn(caption_wrong2)
        #fake_image, fake_small_image, fake_mid_image = self.generator(t_z, t_real_caption)
        fake_image1, fake_image2, fake_image3,\
            gDAE1Loss, gDAE2Loss, gDAE3Loss = self.generator(t_z, g_normal_caption, g_raw_normal_caption)
            
        trans1_img4 = transform_images_gen(fake_image1, .8)
        trans2_img4 = transform_images_gen(fake_image1, 1.1)
        trans1_img2 = transform_images_gen(fake_image2, .8)
        trans2_img2 = transform_images_gen(fake_image2, 1.1)
        trans1_img1 = transform_images_gen(fake_image3, .8)
        trans2_img1 = transform_images_gen(fake_image3, 1.1)
        img_noise = self.noise_disc
        d_noisy_caption = d_normal_caption#ops.noised(t_real_caption,.0001)
        
        
        noisy_img = ops.noised(t_real_image,img_noise)
        noisy_trans_img = ops.noised(t_trans_image,img_noise)
        noisy_trans2_img = ops.noised(t_trans2_image,img_noise)
        noisy_wrong_img = ops.noised(t_wrong_image,img_noise)
        noisy_horrible_img = ops.noised(t_horrible_image,img_noise)
        
        noisy_img2 = ops.noised(real2,img_noise)
        noisy_trans_img2 = ops.noised(trans2,img_noise)
        noisy_trans2_img2 = ops.noised(trans22,img_noise)
        noisy_wrong_img2 = ops.noised(wrong2,img_noise)
        noisy_horrible_img2 = ops.noised(horrible2,img_noise)
        
        noisy_img4 = ops.noised(real4,img_noise)
        noisy_trans_img4 = ops.noised(trans4,img_noise)
        noisy_trans2_img4 = ops.noised(trans24,img_noise)
        noisy_wrong_img4 = ops.noised(wrong4,img_noise)
        noisy_horrible_img4 = ops.noised(horrible4,img_noise)
        
        noisy_fake_img4 = ops.noised(fake_image1,img_noise)
        noisy_fake_img2 = ops.noised(fake_image2,img_noise)
        noisy_fake_img1 = ops.noised(fake_image3,img_noise)
        
        noisy_gen_img4 = ops.noised(gen_image4,img_noise)
        noisy_gen_img2 = ops.noised(gen_image2,img_noise)
        noisy_gen_img1 = ops.noised(gen_image1,img_noise)
        '''
        regex to write the following acts
        p_(\d)_([^,]*?)_txt, ([^,]*?)_acts
        p_\1_\2_txt, \2\1_acts
        '''
        with tf.variable_scope('scope_1'):
            p_1_real_img_logit, p_1_real_txt_logit, p_1_real_img, p_1_real_txt, real1_acts = self.discriminator1(noisy_img4, d_noisy_caption)
            p_1_wrongcap1_img_logit, p_1_wrongcap1_txt_logit, p_1_wrongcap1_img, p_1_wrongcap1_txt, wrongcap11_acts = self.discriminator1(noisy_img4, d_caption_wrong1)
            p_1_wrongcap2_img_logit, p_1_wrongcap2_txt_logit, p_1_wrongcap2_img, p_1_wrongcap2_txt, wrongcap21_acts = self.discriminator1(noisy_img4, d_caption_wrong2)
            p_1_trans_img_logit, p_1_trans_txt_logit, p_1_trans_img, p_1_trans_txt, trans1_acts = self.discriminator1(noisy_trans_img4, d_noisy_caption)
            p_1_trans2_img_logit, p_1_trans2_txt_logit, p_1_trans2_img, p_1_trans2_txt, trans21_acts = self.discriminator1(noisy_trans2_img4, d_noisy_caption)
            p_1_wrong_img_logit, p_1_wrong_txt_logit, p_1_wrong_img, p_1_wrong_txt, wrong1_acts = self.discriminator1(noisy_wrong_img4, d_noisy_caption, reuse = True)
            p_1_horrible_img_logit, p_1_horrible_txt_logit, p_1_horrible_img, p_1_horrible_txt, horrible1_acts = self.discriminator1(noisy_horrible_img4, d_noisy_caption, reuse = True)
            p_1_gen_img_logit, p_1_gen_txt_logit, p_1_gen_img, p_1_fake_txt, fake1_acts = self.discriminator1(noisy_gen_img4, d_noisy_caption, reuse = True)
            p_1_fake_img_logit, p_1_fake_txt_logit, p_1_fake_img, p_1_fake_txt, fake1_acts = self.discriminator1(noisy_fake_img4, d_noisy_caption, reuse = True)
            p_1_t1_gen_img_logit, p_1_t1_gen_txt_logit, p_1_t1_gen_img, p_1_t1_gen_txt, t1_gen1_acts = self.discriminator1(trans1_gen_img4, d_noisy_caption, reuse = True)
            p_1_t1_fake_img_logit, p_1_t1_fake_txt_logit, p_1_t1_fake_img, p_1_t1_fake_txt, t1_fake1_acts = self.discriminator1(trans1_img4, d_noisy_caption, reuse = True)
            p_1_t2_gen_img_logit, p_1_t2_gen_txt_logit, p_1_t2_gen_img, p_1_gen_txt, gen1_acts = self.discriminator1(trans2_gen_img4, d_noisy_caption, reuse = True)
            p_1_t2_fake_img_logit, p_1_t2_fake_txt_logit, p_1_t2_fake_img, p_1_fake_txt, fake1_acts = self.discriminator1(trans2_img4, d_noisy_caption, reuse = True)

        with tf.variable_scope('scope_2'):
            p_2_real_img_logit, p_2_real_txt_logit, p_2_real_img, p_2_real_txt, real2_acts = self.discriminator2(noisy_img2, d_noisy_caption)
            p_2_wrongcap1_img_logit, p_2_wrongcap1_txt_logit, p_1_wrongcap1_img, p_2_wrongcap1_txt, wrongcap12_acts  = self.discriminator2(noisy_img2, d_caption_wrong1)
            p_2_wrongcap2_img_logit, p_2_wrongcap2_txt_logit, p_2_wrongcap2_img, p_2_wrongcap2_txt, wrongcap22_acts  = self.discriminator2(noisy_img2, d_caption_wrong2)
            p_2_trans_img_logit, p_2_trans_txt_logit, p_2_trans_img, p_2_trans_txt, trans2_acts = self.discriminator2(noisy_trans_img2, d_noisy_caption)
            p_2_trans2_img_logit, p_2_trans2_txt_logit, p_2_trans2_img, p_2_trans2_txt, trans22_acts = self.discriminator2(noisy_trans2_img2, d_noisy_caption)
            p_2_wrong_img_logit, p_2_wrong_txt_logit, p_2_wrong_img, p_2_wrong_txt, wrong2_acts = self.discriminator2(noisy_wrong_img2, d_noisy_caption, reuse = True)
            p_2_horrible_img_logit, p_2_horrible_txt_logit, p_2_horrible_img, p_2_horrible_txt, horrible2_acts = self.discriminator2(noisy_horrible_img2, d_noisy_caption, reuse = True)
            p_2_fake_img_logit, p_2_fake_txt_logit, p_2_fake_img, p_2_fake_txt, fake2_acts = self.discriminator2(noisy_fake_img2, d_noisy_caption, reuse = True)
            p_2_gen_img_logit, p_2_gen_txt_logit, p_2_gen_img, p_2_gen_txt, gen2_acts = self.discriminator2(noisy_gen_img2, d_noisy_caption, reuse = True)
            p_2_t1_gen_img_logit, p_2_t1_gen_txt_logit, p_2_t1_gen_img, p_2_t1_gen_txt, t1_gen2_acts = self.discriminator2(trans1_gen_img2, d_noisy_caption, reuse = True)
            p_2_t1_fake_img_logit, p_2_t1_fake_txt_logit, p_2_t1_fake_img, p_2_t1_fake_txt, t1_fake2_acts = self.discriminator2(trans1_img2, d_noisy_caption, reuse = True)
            p_2_t2_gen_img_logit, p_2_t2_gen_txt_logit, p_2_t2_gen_img, p_2_t2_gen_txt, t2_gen2_acts = self.discriminator2(trans2_gen_img2, d_noisy_caption, reuse = True)
            p_2_t2_fake_img_logit, p_2_t2_fake_txt_logit, p_2_t2_fake_img, p_2_t2_fake_txt, t2_fake2_acts = self.discriminator2(trans2_img2, d_noisy_caption, reuse = True)

        with tf.variable_scope('scope_3'):
            p_3_real_img_logit, p_3_real_txt_logit, p_3_real_img, p_3_real_txt, real3_acts = self.discriminator3(noisy_img, d_noisy_caption)
            p_3_wrongcap1_img_logit, p_3_wrongcap1_txt_logit, p_3_wrongcap1_img, p_3_wrongcap1_txt, wrongcap13_acts  = self.discriminator3(noisy_img, d_caption_wrong1)
            p_3_wrongcap2_img_logit, p_3_wrongcap2_txt_logit, p_3_wrongcap2_img, p_3_wrongcap2_txt, wrongcap23_acts  = self.discriminator3(noisy_img, d_caption_wrong2)
            p_3_trans_img_logit, p_3_trans_txt_logit, p_3_trans_img, p_3_trans_txt, trans3_acts = self.discriminator3(noisy_trans_img, d_noisy_caption)
            p_3_trans2_img_logit, p_3_trans2_txt_logit, p_3_trans2_img, p_3_trans2_txt, trans23_acts = self.discriminator3(noisy_trans2_img, d_noisy_caption)
            p_3_wrong_img_logit, p_3_wrong_txt_logit, p_3_wrong_img, p_3_wrong_txt, wrong3_acts = self.discriminator3(noisy_wrong_img, d_noisy_caption, reuse = True)
            p_3_horrible_img_logit, p_3_horrible_txt_logit, p_3_horrible_img, p_3_horrible_txt, horrible3_acts = self.discriminator3(noisy_horrible_img, d_noisy_caption, reuse = True)
            p_3_fake_img_logit, p_3_fake_txt_logit, p_3_fake_img, p_3_fake_txt, fake3_acts = self.discriminator3(noisy_fake_img1, d_noisy_caption, reuse = True)
            p_3_gen_img_logit, p_3_gen_txt_logit, p_3_gen_img, p_3_gen_txt, gen3_acts = self.discriminator3(noisy_gen_img1, d_noisy_caption, reuse = True)
            p_3_t1_gen_img_logit, p_3_t1_gen_txt_logit, p_3_t1_gen_img, p_3_t2_gen_txt, t2_gen3_acts = self.discriminator3(trans1_gen_img1, d_noisy_caption, reuse = True)
            p_3_t1_fake_img_logit, p_3_t1_fake_txt_logit, p_3_t1_fake_img, p_3_t1_fake_txt, t1_fake3_acts = self.discriminator3(trans1_img1, d_noisy_caption, reuse = True)
            p_3_t2_gen_img_logit, p_3_t2_gen_txt_logit, p_3_t2_gen_img, p_3_gen_txt, gen3_acts = self.discriminator3(trans2_gen_img1, d_noisy_caption, reuse = True)
            p_3_t2_fake_img_logit, p_3_t2_fake_txt_logit, p_3_t2_fake_img, p_3_fake_txt, fake3_acts = self.discriminator3(trans2_img1, d_noisy_caption, reuse = True)
        
        
        
        pos_ex = tf.ones_like(p_1_fake_img_logit)
        neg_ex = tf.zeros_like(p_1_fake_img_logit)
        d_loss_real = tf.reduce_mean(c_e(p_1_real_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_1_real_txt_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_2_real_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_2_real_txt_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_3_real_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_3_real_txt_logit, pos_ex))
            
        d_loss_trans = tf.reduce_mean(c_e(p_1_trans_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_1_trans_txt_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_2_trans_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_2_trans_txt_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_3_trans_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_3_trans_txt_logit, pos_ex))
            
        d_loss_trans2 = tf.reduce_mean(c_e(p_1_trans2_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_1_trans2_txt_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_2_trans2_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_2_trans2_txt_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_3_trans2_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_3_trans2_txt_logit, pos_ex))
            
        not_noise = 1-noise_indicator
        
        d_loss_wrong = tf.reduce_mean(c_e(p_1_wrong_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_1_wrong_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_2_wrong_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_2_wrong_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_3_wrong_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_3_wrong_txt_logit, neg_ex))*not_noise
            
        d_loss_wrongcap1 = tf.reduce_mean(c_e(p_1_wrongcap1_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_1_wrongcap1_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_2_wrongcap1_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_2_wrongcap1_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_3_wrongcap1_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_3_wrongcap1_txt_logit, neg_ex))*not_noise
            
        d_loss_wrongcap2 = tf.reduce_mean(c_e(p_1_wrongcap2_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_1_wrongcap2_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_2_wrongcap2_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_2_wrongcap2_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_3_wrongcap2_img_logit, pos_ex)) + tf.reduce_mean(c_e(p_3_wrongcap2_txt_logit, neg_ex))*not_noise
        
        d_loss_horrible = tf.reduce_mean(c_e(p_1_horrible_img_logit, neg_ex)) + tf.reduce_mean(c_e(p_1_horrible_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_2_horrible_img_logit, neg_ex)) + tf.reduce_mean(c_e(p_2_horrible_txt_logit, neg_ex))*not_noise + \
            tf.reduce_mean(c_e(p_3_horrible_img_logit, neg_ex)) + tf.reduce_mean(c_e(p_3_horrible_txt_logit, neg_ex))*not_noise
            
        #d_loss_noise = tf.reduce_mean(c_e(p_noise_img_logit, pos_ex))
        d_loss_real_img = tf.reduce_mean(c_e(p_1_real_img_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_2_real_img_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_3_real_img_logit, pos_ex))
            
        d_loss_trans_img = tf.reduce_mean(c_e(p_1_trans_img_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_2_trans_img_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_3_trans_img_logit, pos_ex))
        
        d_loss_trans2_img = tf.reduce_mean(c_e(p_1_trans2_img_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_2_trans2_img_logit, pos_ex)) + \
            tf.reduce_mean(c_e(p_3_trans2_img_logit, pos_ex))
            
        
            
        d1_loss_noise = tf.reduce_mean(c_e(p_1_fake_img_logit, neg_ex))
        d2_loss_noise = tf.reduce_mean(c_e(p_2_fake_img_logit, neg_ex))
        d3_loss_noise = tf.reduce_mean(c_e(p_3_fake_img_logit, neg_ex))
        
        d1_loss_gen_noise = tf.reduce_mean(c_e(p_1_gen_img_logit, neg_ex))
        d2_loss_gen_noise = tf.reduce_mean(c_e(p_2_gen_img_logit, neg_ex))
        d3_loss_gen_noise = tf.reduce_mean(c_e(p_3_gen_img_logit, neg_ex))
        
        g1_loss_noise = tf.reduce_mean(c_e(p_1_fake_img_logit, pos_ex))
        g2_loss_noise = tf.reduce_mean(c_e(p_2_fake_img_logit, pos_ex))
        g3_loss_noise = tf.reduce_mean(c_e(p_3_fake_img_logit, pos_ex))
        
        g1_loss_noise += gDAE1Loss*0
        g2_loss_noise += gDAE2Loss*0
        g3_loss_noise += gDAE3Loss*0
        
        
        
        d1_loss = d1_loss_noise + tf.reduce_mean(c_e(p_1_fake_txt_logit, neg_ex))
        d2_loss = d2_loss_noise + tf.reduce_mean(c_e(p_2_fake_txt_logit, neg_ex))
        d3_loss = d3_loss_noise + tf.reduce_mean(c_e(p_3_fake_txt_logit, neg_ex))
        
        d1_loss_gen = d1_loss_gen_noise + tf.reduce_mean(c_e(p_1_gen_txt_logit, neg_ex))
        d2_loss_gen = d2_loss_gen_noise + tf.reduce_mean(c_e(p_2_gen_txt_logit, neg_ex))
        d3_loss_gen = d3_loss_gen_noise + tf.reduce_mean(c_e(p_3_gen_txt_logit, neg_ex))
        
        g1_loss = g1_loss_noise + tf.reduce_mean(c_e(p_1_fake_txt_logit, pos_ex))
        g2_loss = g2_loss_noise + tf.reduce_mean(c_e(p_2_fake_txt_logit, pos_ex))
        g3_loss = g3_loss_noise + tf.reduce_mean(c_e(p_3_fake_txt_logit, pos_ex))
        
        
        
        
        ###TRANS GENERATED LOSSES
        d1_t1_loss_noise = tf.reduce_mean(c_e(p_1_t1_fake_img_logit, neg_ex))
        d2_t1_loss_noise = tf.reduce_mean(c_e(p_2_t1_fake_img_logit, neg_ex))
        d3_t1_loss_noise = tf.reduce_mean(c_e(p_3_t1_fake_img_logit, neg_ex))
        
        d1_t1_loss_gen_noise = tf.reduce_mean(c_e(p_1_t1_gen_img_logit, neg_ex))
        d2_t1_loss_gen_noise = tf.reduce_mean(c_e(p_2_t1_gen_img_logit, neg_ex))
        d3_t1_loss_gen_noise = tf.reduce_mean(c_e(p_3_t1_gen_img_logit, neg_ex))
        
        g1_t1_loss_noise = tf.reduce_mean(c_e(p_1_t1_fake_img_logit, pos_ex))
        g2_t1_loss_noise = tf.reduce_mean(c_e(p_2_t1_fake_img_logit, pos_ex))
        g3_t1_loss_noise = tf.reduce_mean(c_e(p_3_t1_fake_img_logit, pos_ex))
        
        d1_t1_loss = d1_t1_loss_noise + tf.reduce_mean(c_e(p_1_t1_fake_txt_logit, neg_ex))
        d2_t1_loss = d2_t1_loss_noise + tf.reduce_mean(c_e(p_2_t1_fake_txt_logit, neg_ex))
        d3_t1_loss = d3_t1_loss_noise + tf.reduce_mean(c_e(p_3_t1_fake_txt_logit, neg_ex))
        
        d1_t1_loss_gen = d1_t1_loss_gen_noise + tf.reduce_mean(c_e(p_1_t1_gen_txt_logit, neg_ex))
        d2_t1_loss_gen = d2_t1_loss_gen_noise + tf.reduce_mean(c_e(p_2_t1_gen_txt_logit, neg_ex))
        d3_t1_loss_gen = d3_t1_loss_gen_noise + tf.reduce_mean(c_e(p_3_t1_gen_txt_logit, neg_ex))
        
        g1_t1_loss = g1_t1_loss_noise + tf.reduce_mean(c_e(p_1_t1_fake_txt_logit, pos_ex))
        g2_t1_loss = g2_t1_loss_noise + tf.reduce_mean(c_e(p_2_t1_fake_txt_logit, pos_ex))
        g3_t1_loss = g3_t1_loss_noise + tf.reduce_mean(c_e(p_3_t1_fake_txt_logit, pos_ex))
        
        
        d1_t2_loss_noise = tf.reduce_mean(c_e(p_1_t2_fake_img_logit, neg_ex))
        d2_t2_loss_noise = tf.reduce_mean(c_e(p_2_t2_fake_img_logit, neg_ex))
        d3_t2_loss_noise = tf.reduce_mean(c_e(p_3_t2_fake_img_logit, neg_ex))
        
        d1_t2_loss_gen_noise = tf.reduce_mean(c_e(p_1_t2_gen_img_logit, neg_ex))
        d2_t2_loss_gen_noise = tf.reduce_mean(c_e(p_2_t2_gen_img_logit, neg_ex))
        d3_t2_loss_gen_noise = tf.reduce_mean(c_e(p_3_t2_gen_img_logit, neg_ex))
        
        g1_t2_loss_noise = tf.reduce_mean(c_e(p_1_t2_fake_img_logit, pos_ex))
        g2_t2_loss_noise = tf.reduce_mean(c_e(p_2_t2_fake_img_logit, pos_ex))
        g3_t2_loss_noise = tf.reduce_mean(c_e(p_3_t2_fake_img_logit, pos_ex))
        
        d1_t2_loss = d1_t2_loss_noise + tf.reduce_mean(c_e(p_1_t2_fake_txt_logit, neg_ex))
        d2_t2_loss = d2_t2_loss_noise + tf.reduce_mean(c_e(p_2_t2_fake_txt_logit, neg_ex))
        d3_t2_loss = d3_t2_loss_noise + tf.reduce_mean(c_e(p_3_t2_fake_txt_logit, neg_ex))
        
        d1_t2_loss_gen = d1_t2_loss_gen_noise + tf.reduce_mean(c_e(p_1_t2_gen_txt_logit, neg_ex))
        d2_t2_loss_gen = d2_t2_loss_gen_noise + tf.reduce_mean(c_e(p_2_t2_gen_txt_logit, neg_ex))
        d3_t2_loss_gen = d3_t2_loss_gen_noise + tf.reduce_mean(c_e(p_3_t2_gen_txt_logit, neg_ex))
        
        g1_t2_loss = g1_t2_loss_noise + tf.reduce_mean(c_e(p_1_t2_fake_txt_logit, pos_ex))
        g2_t2_loss = g2_t2_loss_noise + tf.reduce_mean(c_e(p_2_t2_fake_txt_logit, pos_ex))
        g3_t2_loss = g3_t2_loss_noise + tf.reduce_mean(c_e(p_3_t2_fake_txt_logit, pos_ex))
        
        
        
        
        
        
        
        disc_real_losses = [concat(1,[c_e(p_1_real_img_logit, pos_ex), c_e(p_1_real_txt_logit, pos_ex)]),
                            concat(1,[c_e(p_2_real_img_logit, pos_ex), c_e(p_2_real_txt_logit, pos_ex)]),
                            concat(1,[c_e(p_3_real_img_logit, pos_ex), c_e(p_3_real_txt_logit, pos_ex)])]
        disc_real_vals = [concat(1,[p_1_real_img_logit, p_1_real_txt_logit]),
                            concat(1,[p_2_real_img_logit, p_2_real_txt_logit]),
                            concat(1,[p_3_real_img_logit, p_3_real_txt_logit])]
        
        disc_wrong_losses = [concat(1,[c_e(p_1_wrong_img_logit, pos_ex), c_e(p_1_wrong_txt_logit, pos_ex)]),
                            concat(1,[c_e(p_2_wrong_img_logit, pos_ex), c_e(p_2_wrong_txt_logit, pos_ex)]),
                            concat(1,[c_e(p_3_wrong_img_logit, pos_ex), c_e(p_3_wrong_txt_logit, pos_ex)])]
        disc_wrong_vals = [concat(1,[p_1_wrong_img_logit, p_1_wrong_txt_logit]),
                            concat(1,[p_2_wrong_img_logit, p_2_wrong_txt_logit]),
                            concat(1,[p_3_wrong_img_logit, p_3_wrong_txt_logit])]
        
        disc_fake_losses = [concat(1,[c_e(p_1_fake_img_logit, pos_ex), c_e(p_1_fake_txt_logit, pos_ex)]),
                            concat(1,[c_e(p_2_fake_img_logit, pos_ex), c_e(p_2_fake_txt_logit, pos_ex)]),
                            concat(1,[c_e(p_3_fake_img_logit, pos_ex), c_e(p_3_fake_txt_logit, pos_ex)])]
        disc_fake_vals = [concat(1,[p_1_fake_img_logit, p_1_fake_txt_logit]),
                            concat(1,[p_2_fake_img_logit, p_2_fake_txt_logit]),
                            concat(1,[p_3_fake_img_logit, p_3_fake_txt_logit])]
        
        
        real1_acts = concat(0, real1_acts)
        real2_acts = concat(0, real2_acts)/2
        real3_acts = concat(0, real3_acts)/4
        fake1_acts = concat(0, fake1_acts)
        fake2_acts = concat(0, fake2_acts)/2
        fake3_acts = concat(0, fake3_acts)/4
        gen1_acts = concat(0, gen1_acts)
        gen2_acts = concat(0, gen2_acts)/2
        gen3_acts = concat(0, gen3_acts)/4

        self.real_acts = concat(0, [real1_acts, real2_acts, real3_acts])
        self.fake_acts = concat(0, [fake1_acts, fake2_acts, fake3_acts])
        self.gen_acts = concat(0, [gen1_acts, gen2_acts, gen3_acts])
        
        real_means, real_stds = tf.nn.moments(self.real_acts,axes=[0])
        fake_means, fake_stds = tf.nn.moments(self.fake_acts,axes=[0])
        gen_means, gen_stds = tf.nn.moments(self.gen_acts,axes=[0])
        
        real_acts_disc = (self.real_acts - real_means) / (real_stds + 1e-10)
        fake_acts_disc = (self.fake_acts - fake_means) / (fake_stds + 1e-10)
        gen_acts_disc = (self.gen_acts - gen_means) / (gen_stds + 1e-10)
        self.past_reals = tf.placeholder('float32', shape=[None])
        self.past_fakes = tf.placeholder('float32', shape=[None])
        real_time = self.real_acts * .1 + self.past_reals * .9
        fake_time = self.fake_acts * .1 + self.past_fakes * .9
        lambda_act = 0.1
        lambda_act_time = 10
        lambda_act_disc = .0001
        
        act_regs = tf.reduce_mean(tf.square ( self.real_acts - self.fake_acts)) * lambda_act
        act_regs_disc = tf.reduce_mean(tf.abs (real_acts_disc - gen_acts_disc)) * lambda_act_disc
        act_time_regs = tf.reduce_mean(tf.square(fake_time - real_time)) * lambda_act_time
        
        d_loss_noise = d_loss_real_img + \
                .2*(d_loss_wrong + d_loss_wrongcap1 + d_loss_wrongcap2) + \
                d1_loss_gen_noise + d2_loss_gen_noise + d3_loss_gen_noise + \
        d_loss = d_loss_real + \
                .2*(d_loss_wrong + d_loss_wrongcap1 + d_loss_wrongcap2)  + \
                d1_loss_gen + d2_loss_gen + d3_loss_gen
        
        g_loss_noise = 1*(g1_loss_noise*mul1 + g2_loss_noise/2*mul2 + g3_loss_noise/4*mul3)
        
        g_loss = 1*(g1_loss*mul1 + g2_loss/2*mul2 + g3_loss/4*mul3)
               

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g1_vars = [var for var in t_vars if ('g1_' in var.name or 'g_' in var.name)]
        g2_vars = [var for var in t_vars if 'g2_' in var.name]
        g3_vars = [var for var in t_vars if 'g3_' in var.name]
        
        d_loss_reg = tf.reduce_sum([tf.reduce_sum(tf.square(var)) * 1e-5 for var in d_vars])
        g_loss_reg = tf.reduce_sum([tf.reduce_sum(tf.square(var))*1e-5 for var in g1_vars + g2_vars + g3_vars])
        self.l2_disc = [v.assign(v*.99) for v in d_vars]
        #self.wgan_clip_1eneg3 = [v.assign(tf.clip_by_value(v, -0.001, 0.001)) for v in d_vars]
        #self.wgan_clip_1eneg2 = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
        #self.wgan_clip_1eneg1 = [v.assign(tf.clip_by_value(v, -0.1, 0.1)) for v in d_vars]
        #self.wgan_clip_1 = [v.assign(tf.clip_by_value(v, -1, 1)) for v in d_vars]

        input_tensors = {
            'cap_wrong1' : caption_wrong1,
            'cap_wrong2' : caption_wrong2,
            't_real_image' : t_real_image,
            't_wrong_image' : t_wrong_image,
            't_horrible_image' : t_horrible_image,
            't_real_caption' : t_real_caption,
            'gen_image1' : gen_image1,
            'gen_image2' : gen_image2,
            'gen_image4' : gen_image4,
            't_z' : t_z,
            'l2reg' : l2reg,
            'LambdaDAE' : LambdaDAE,
            'noise_indicator' : noise_indicator,
            'noise_gen' : self.noise_gen,
            'noise_disc' : self.noise_disc,
            'mul1': mul1,
            'mul2': mul2,
            'mul3': mul3,
            'lr' : self.lr
        }
        variables = {
            'd_vars' : d_vars,
            'g1_vars' : g1_vars,
            'g2_vars' : g2_vars,
            'g3_vars' : g3_vars
        }

        loss = {
            'g_loss': g_loss,
            'd_loss' : d_loss,
            
            'g1_loss' : g1_loss,
            'g2_loss' : g2_loss,
            'g3_loss' : g3_loss,
            
            'g1_loss_noise' : g1_loss_noise,
            'g2_loss_noise' : g2_loss_noise,
            'g3_loss_noise' : g3_loss_noise,
            
            'd1_loss' : d1_loss,
            'd2_loss' : d2_loss,
            'd3_loss' : d3_loss,
            
            'd1_loss_gen' : d1_loss_gen,
            'd2_loss_gen' : d2_loss_gen,
            'd3_loss_gen' : d3_loss_gen,
            
            'd1_loss_gen_noise' : d1_loss_gen_noise,
            'd2_loss_gen_noise' : d2_loss_gen_noise,
            'd3_loss_gen_noise' : d3_loss_gen_noise,
            
            'd1_loss_noise' : d1_loss_noise,
            'd2_loss_noise' : d2_loss_noise,
            'd3_loss_noise' : d3_loss_noise,
            
            'd_loss_real_img' : d_loss_real_img,
            'd_loss_real' : d_loss_real,
            'd_loss_wrong' : d_loss_wrong,
            
            
            'g_loss_noise' : g_loss_noise,
            'd_loss_noise' : d_loss_noise
        }
        
        
        
        outputs = {
            'img1' : fake_image1,
            'img2' : fake_image2,
            'img3' : fake_image3,
            'trans3' : t_trans_image,
            'trans2' : trans2,
            'trans1' : trans4,
            'trans23' : t_trans2_image,
            'trans22' : trans22,
            'trans21' : trans24,
            'disc_real_losses' : disc_real_losses,
            'disc_wrong_losses': disc_wrong_losses,
            'disc_fake_losses' : disc_fake_losses,
            'disc_real_vals' : disc_real_vals,
            'disc_wrong_vals': disc_wrong_vals,
            'disc_fake_vals' : disc_fake_vals,
            'output_p_3_gen_txt_logit' : p_3_gen_txt_logit,
            'output_p_3_gen_img_logit' : p_3_gen_img_logit
        }

        
    
        gloss3 = loss['g3_loss']# + .5 * gloss4
        gloss2 = loss['g2_loss']# + .5 * gloss3
        gloss1 = loss['g1_loss']# + .5 * gloss2
        '''
        self.g_optim = tf.train.AdamOptimizer(lr, beta1 = beta1,beta2 = beta2).minimize(
            g_loss_noise * noise_indicator + g_loss * (1-noise_indicator),
            var_list=variables['g1_vars'] + variables['g2_vars'] + 
            variables['g3_vars'] + variables['g4_vars'] + variables['g5_vars'])
        
        self.d_optim = tf.train.AdamOptimizer(lr, beta1 = beta1,beta2 = beta2).minimize(
            d_loss_noise * noise_indicator + d_loss * (1-noise_indicator), var_list=variables['d_vars'])
        '''
        optimizer = tf.train.AdamOptimizer(self.lr, beta1 = beta1,beta2 = beta2)
        gvs = optimizer.compute_gradients(
            g_loss_noise * noise_indicator + g_loss * (1-noise_indicator) + g_loss_reg,
            var_list=variables['g1_vars'] + variables['g2_vars'] + 
            variables['g3_vars'])
        clip_max = 1
        clip = .1
        capped_gvs = [(tf.clip_by_value(grad, -1*clip,clip), var) for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_max), var) for grad, var in capped_gvs if grad is not None]
        self.g_optim = optimizer.apply_gradients(capped_gvs)
        self.g_gvs = [grad for grad, var in gvs if grad is not None]
        
        optimizer = tf.train.AdamOptimizer(1e-4, beta1 = beta1,beta2 = beta2)
        gvs = optimizer.compute_gradients(
            d_loss_noise * noise_indicator + d_loss * (1-noise_indicator) + d_loss_reg,
            var_list=variables['d_vars'])
        clip_max = 10
        clip = 1
        capped_gvs = [(tf.clip_by_value(grad, -1*clip,clip), var) for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_max), var) for grad, var in capped_gvs if grad is not None]
        self.d_optim = optimizer.apply_gradients(capped_gvs)
        self.d_gvs = [grad for grad, var in gvs if grad is not None] 
        
        return input_tensors, variables, loss, outputs

    def build_generator(self):
        img_size = self.options['image_size']
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
        fake_image = self.sampler(t_z, t_real_caption)
        
        input_tensors = {
            't_real_caption' : t_real_caption,
            't_z' : t_z
        }
        
        outputs = {
            'generator' : fake_image
        }

        return input_tensors, outputs

    # Sample Images for a text embedding
    def sampler(self, t_z, t_text_embedding):
        tf.get_variable_scope().reuse_variables()
        
        s = self.options['image_size']
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        
        reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g1_embedding') )
        z_concat = concat(1, [t_z, reduced_text_embedding])
        z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g1_h0_lin')
        h0 = reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
        h0 = ops.lrelu(self.g_bn0(h0, train = False))
        
        h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g1_h1')
        h1 = ops.lrelu(self.g_bn1(h1, train = False))
        
        h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g1_h2')
        h2 = ops.lrelu(self.g_bn2(h2, train = False))
        
        h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g1_h3')
        h3 = ops.lrelu(self.g_bn3(h3, train = False))
        
        h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g1_h4')
        
        return (tf.tanh(h4)/2. + 0.5)

    def g_name(self):
        self.g_idx += 1
        return 'g1_' + str(self.g_idx)
    def g2_name(self):
        self.g2_idx += 1
        return 'g2_' + str(self.g2_idx)
    def g3_name(self):
        self.g3_idx += 1
        return 'g3_' + str(self.g3_idx)
    def d_name(self):
        self.d_idx += 1
        return 'd_generatedname_' + str(self.d_idx)
    

    def minibatch_discriminate(self, inpt, num_kernels=5, kernel_dim=3, reuse = None, lin_name = None):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        x = ops.linear(inpt, num_kernels * kernel_dim, lin_name)
        activation = reshape(x, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return minibatch_features
    
    def add_residual_pre(self, prev_layer, z_concat, text_filters = None, k_h = 5, k_w = 5, hidden_text_filters = None,
                     hidden_filters = None,name_func=None):
        
        filters = prev_layer.get_shape()[3].value
        if name_func is None:
            name_func = self.g_name
        if hidden_filters ==None:
            hidden_filters = filters * 4
        if text_filters == None:
            text_filters = int(filters/2)
        if hidden_text_filters == None:
            hidden_text_filters = int(filters/8)
        s = prev_layer.get_shape()[1].value
        
        bn0 = self.g_bn()
        bn1 = self.g_bn()
        
        low_dim = ops.conv2d(ops.lrelu(bn0(prev_layer)), hidden_filters, k_h=k_h, k_w=k_w, name = name_func())
        
        residual = ops.deconv2d(ops.lrelu(bn1(low_dim), name=name_func()), 
            [self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_w, name=name_func())
        
        next_layer = prev_layer + residual
        return next_layer
    
    #Residual Block
    def add_residual(self, prev_layer, z_concat, text_filters = None, k_h = 5, k_w = 5, hidden_text_filters = None,
                     hidden_filters = None,name_func=None):
        
        filters = prev_layer.get_shape()[3].value
        if name_func is None:
            name_func = self.g_name
        if hidden_filters ==None:
            hidden_filters = filters * 2
        if text_filters == None:
            text_filters = int(filters/2)
        if hidden_text_filters == None:
            hidden_text_filters = int(filters/4)
        s = prev_layer.get_shape()[1].value
        
        bn0 = self.g_bn()
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        bn3 = self.g_bn()
        
        low_dim = ops.conv2d(ops.lrelu(bn0(prev_layer)), hidden_filters, k_h=k_h, k_w=k_w, name = name_func())
        
        if z_concat is not None:
            text_augment = reshape(
                ops.linear(ops.lrelu(bn1(z_concat)),  s/4* s/4* text_filters,name_func()),[-1, s/4, s/4, text_filters])
            
            text_augment = ops.deconv2d(ops.lrelu(bn2(text_augment)),
                [self.options['batch_size'], s/2, s/2, hidden_text_filters], k_h=k_h, k_w=k_w, name=name_func())
        
            concatenated = concat(3, [text_augment, low_dim], name=name_func())
        else:
            concatenated = prev_layer
            
        res_hidden = bn3(ops.deconv2d(concatenated, 
            [self.options['batch_size'], s/2, s/2, hidden_filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func()))
        
        residual = ops.deconv2d(ops.lrelu(res_hidden, name = name_func()),
            [self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, name=name_func())
        
        next_layer = prev_layer + residual
        return ops.lrelu(next_layer, name=name_func())
    
    #Residual Block in lower dimension
    def add_ld_residual(self, prev_layer, z_concat, text_filters = 3, k_h = 5, k_w = 5, hidden_text_filters = 20,name_func=None):
        if name_func is None:
            name_func = self.g_name
        
        s = prev_layer.get_shape()[1].value
        filters = prev_layer.get_shape()[3].value

        bn0 = self.g_bn()
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        hidden_layer = ops.conv2d(prev_layer, filters, name=name_func())
        if z_concat is not None:
            text_augment = reshape(ops.lrelu(bn1(
                ops.linear(z_concat,  s/4* s/4* text_filters,name_func())), name = name_func()),[-1, s/4, s/4, text_filters])
            
            text_augment = ops.lrelu(bn0(ops.deconv2d(text_augment,
                [self.options['batch_size'], s/2, s/2, hidden_text_filters], name=name_func())))
            
            concatenated = concat(3, [text_augment, hidden_layer], name=name_func())
        else:
            concatenated = hidden_layer
            
        res_hidden = bn2(ops.deconv2d(
            concatenated, [self.options['batch_size'], s/2, s/2, filters], 
                k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func()))
        
        residual = ops.deconv2d(ops.lrelu(res_hidden, name = name_func()),
            [self.options['batch_size'], s, s, filters],k_h=k_h, k_w=k_w, name=name_func())
        next_layer = prev_layer + residual
        return ops.lrelu(next_layer, name=name_func())
    
    
    
    #Residual Block Standard Version. Needed for very small images
    def add_residual_standard(self, prev_layer, z_concat, text_filters = None, k_h = 5, k_w = 5,
                              hidden_filters = None,name_func=None):
        
        filters = prev_layer.get_shape()[3].value
        if name_func is None:
            name_func = self.g_name
        if hidden_filters ==None:
            hidden_filters = filters * 2
        if text_filters == None:
            text_filters = int(filters/2)
        s = prev_layer.get_shape()[1].value
        
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        if z_concat is not None:
            text_augment = reshape(ops.lrelu(bn1(
                ops.linear(z_concat,  s* s* text_filters,name_func())), name = name_func()),[-1, s, s, text_filters])
            concatenated = concat(3, [text_augment, prev_layer], name=name_func())
        else:
            concatenated = prev_layer
        res_hidden = bn2(ops.deconv2d(concatenated,
            [self.options['batch_size'], s, s, hidden_filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func()))
        
        residual = ops.deconv2d(ops.lrelu(res_hidden, name = name_func()),
            [self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func())
        next_layer = prev_layer + residual
        return ops.lrelu(next_layer, name=name_func())
    
    #Residual Block Standard version in a reduced dimensionality space. Not currently used/needed.
    def add_ld_residual_standard(self, prev_layer, z_concat, text_filters = 1, k_h = 5, k_w = 5,name_func=None):
        if name_func is None:
            name_func = self.g_name
        
        s = prev_layer.get_shape()[1].value
        filters = prev_layer.get_shape()[3].value
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        text_augment = reshape(ops.lrelu(bn1(
            ops.linear(z_concat,  s/2* s/2* text_filters,name_func())), name = name_func()),[-1, s/2, s/2, text_filters])
        
        hidden_layer = ops.conv2d(prev_layer, filters, name=name_func())
        
        res_hidden = bn2(ops.deconv2d(
            concat(3, [text_augment, hidden_layer], name=name_func()), 
            [self.options['batch_size'], s/2, s/2, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func()))
        
        residual = ops.deconv2d(ops.lrelu(res_hidden, name = name_func()),
            [self.options['batch_size'], s, s, filters],k_h=k_h, k_w=k_w, name=name_func())
        next_layer = prev_layer + residual
        return ops.lrelu(next_layer, name=name_func())
        #Update to maintain indexing
    
    def add_stackGan(self, prev_layer, s2, z_concat, k_h = 5, k_w = 5,name_func=None):
        return prev_layer
    
    def add_stackGanImg(self, prev_layer, s2, z_concat, name_func=None):
        return prev_layer
    
    
    # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def generator(self, t_z, t_text_embedding, raw_caption):
        
        s = self.options['image_size']
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        
        bn20 = self.g_bn()
        bn21 = self.g_bn()
        bn22 = self.g_bn()
        bn30 = self.g_bn()
        bn31 = self.g_bn()
        bn32 = self.g_bn()
        bno1 = self.g_bn()
        bno2 = self.g_bn()
        bno3 = self.g_bn()
        
        bn11 = self.g_bn()
        bn12 = self.g_bn()
        bn13 = self.g_bn()
        bn13sat = self.g_bn()
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        bn3 = self.g_bn()
        
        bnt1 = self.g_bn()
        bnt2 = self.g_bn()
        bnt3 = self.g_bn()
        raw_caption_to_rec = concat(1, [t_z, raw_caption])
        z_rec_target = concat(1, [t_z, t_text_embedding])
        z_concat = ops.lrelu( bnt1(ops.linear(z_rec_target, self.options['t_dim'], 'g1_embedding')))
        z_concat2 = ops.lrelu( bnt2(ops.linear(z_rec_target, self.options['t_dim'], 'g2_embedding')))
        z_concat3 = ops.lrelu( bnt3(ops.linear(z_rec_target, self.options['t_dim'], 'g3_embedding')))
        z_ = ops.linear(z_concat, self.options['gf_dim']*16*s16*s16, 'g1_h0_lin')
        h0 = reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 16])
        h0 = ops.lrelu(self.g_bn0(h0), name = 'g1_pre2')
        
        #h0 = self.add_residual(h0, z_concat, name_func=self.g_name)
        
        #h0 = self.add_residual_standard(h0, None, k_h = 3, k_w = 3, text_filters=4)
        #h1 = tf.image.resize_nearest_neighbor(h0, [s8, s8])
        h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*8],name='g1_h1')
        h1 = ops.lrelu(self.g_bn1(h1), name = 'g1_pre43234')
        
        stack1 = self.add_stackGan(h1, s8, z_concat, name_func=self.g_name)
        
        stack1out = ops.deconv2d(bno1(stack1), [self.options['batch_size'], s8, s8, self.options['gf_dim']*16],d_h=1, d_w=1,
                                 k_h = 3, k_w = 3,name=self.g_name())
        stack1out = ops.lrelu(bn11(stack1out))
        stack1out = ops.deconv2d(stack1out, [self.options['batch_size'], s4, s4, 3], k_h = 3, k_w = 3,name=self.g_name())
        
        stack2 = ops.deconv2d(stack1, 
                              [self.options['batch_size'], s4, s4, self.options['gf_dim']*4],name=self.g2_name())
        
        stack2 = concat(3, [stack2, ops.noised(stack1out, self.noise_gen)])
        
        stack2 = ops.lrelu(bn20(stack2))
        '''
        stack2 = ops.deconv2d(stack2, [self.options['batch_size'], s4, s4, self.options['gf_dim']*16],
                                 k_h = 5, k_w = 5,d_h=1, d_w=1,name=self.g2_name())
        
        stack2 = ops.lrelu(bn21(stack2))
        '''
        stack2 = ops.deconv2d(stack2, [self.options['batch_size'], s4, s4, self.options['gf_dim']*4],
                                 k_h = 5, k_w = 5,d_h=1, d_w=1,name=self.g2_name())
        
        stack2 = ops.lrelu(bn22(stack2))
        
        stack2 = self.add_stackGan(stack2, s4, z_concat2, k_h = 7, k_w = 7, name_func=self.g2_name)

        stack2out = ops.deconv2d(bno2(stack2), [self.options['batch_size'], s4, s4, self.options['gf_dim']*8],
                                 k_h = 5, k_w = 5,d_h=1, d_w=1,name=self.g2_name())
        stack2out = ops.lrelu(bn12(stack2out))
        stack2out = ops.deconv2d(stack2out, [self.options['batch_size'], s2, s2, 3], k_h = 3, k_w = 3,name=self.g2_name())
        
        stack3 = ops.deconv2d(stack2, 
                              [self.options['batch_size'], s2, s2, self.options['gf_dim']*2],name=self.g3_name())
        
        stack3 = concat(3, [stack3, ops.noised(stack2out, self.noise_gen)])
        
        stack3 = ops.lrelu(bn30(stack3))
        '''
        stack3 = ops.deconv2d(stack3, [self.options['batch_size'], s2, s2, self.options['gf_dim']*8],
                                 k_h = 5, k_w = 5,d_h=1, d_w=1,name=self.g3_name())
        
        stack3 = ops.lrelu(bn31(stack3))
        '''
        stack3 = ops.deconv2d(stack3, [self.options['batch_size'], s2, s2, self.options['gf_dim']*2],
                                 k_h = 7, k_w = 7,d_h=1, d_w=1,name=self.g3_name())
        
        stack3 = ops.lrelu(bn32(stack3))
        ex_f = self.options['gf_dim']*8
        stack3 = self.add_stackGan(stack3, s2, z_concat3, k_h = 9, k_w = 9, name_func=self.g3_name)
        stack3sat = tf.nn.sigmoid(bn13sat(ops.deconv2d(stack3, [self.options['batch_size'], s, s, 1], 
                                 k_h = 9, k_w = 9,name=self.g3_name()))) * 2 + 1
        stack3sat = tf.tile(stack3sat, [1,1,1,3])

        stack3out = ops.deconv2d(bno3(stack3), [self.options['batch_size'], s2, s2, self.options['gf_dim']*8],d_h=1, d_w=1,
                                 k_h = 5, k_w = 5,name=self.g2_name())
        stack3out = ops.lrelu(bn13(stack3out))
        
        
        stack3out = ops.deconv2d(stack3out, [self.options['batch_size'], s, s, 3], 
                                 k_h = 3, k_w = 3,name=self.g3_name())
        stack3out = stack3sat * stack3out
        '''
        add_residual(self, prev_layer, z_concat, text_filters = None, k_h = 5, k_w = 5, hidden_text_filters = None,
                     hidden_filters = None,name_func=None):
        
        if name_func is None:
            name_func = self.g_name
        if hidden_filters ==None:
            hidden_filters = filters
        if text_filters == None:
            text_filters = int(filters/2)
        if hidden_text_filters == None:
            hidden_text_filters = int(filters/8)
        '''
        stack3out = (tf.tanh(stack3out)/2. + 0.5)
        #stack3white = tf.tile(stack3white, [1,1,1,3])
        #stack3black = tf.tile(stack3black, [1,1,1,3])
        #stack3out = (stack3out * (1-stack3black) * (1-stack3white) + stack3white)
        
        z_target_dim = 556+10
        rec_stacks = 30
        z_rec1 = ops.linear(reshape(ops.lrelu(bn1(ops.conv2d(stack1, rec_stacks, 
                    name=self.g_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g_name())
        
        z_rec2 = ops.linear(reshape(ops.lrelu(bn2(ops.conv2d(stack2, rec_stacks, 
                    name=self.g2_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g2_name())
        
        z_rec3 = ops.linear(reshape(ops.lrelu(bn3(ops.conv2d(stack3, rec_stacks, 
                    name=self.g3_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g3_name())
        
        return (tf.tanh(stack1out)/2. + 0.5), (tf.tanh(stack2out)/2. + 0.5), stack3out, \
                tf.reduce_mean(tf.square(z_rec1 - raw_caption_to_rec)),tf.reduce_mean(tf.square(z_rec2 - raw_caption_to_rec)),\
                tf.reduce_mean(tf.square(z_rec3 - raw_caption_to_rec))

    def discriminator1(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        
        s = image.get_shape()[1].value
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=3, kernel_dim=3, reuse = reuse, lin_name = 'd_minilin1')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        ddim = self.options['df_dim']
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h0_conv'), name = 'd_pre1') #32
        h1 = ops.lrelu(ops.conv2d(h0, ddim*2, name = 'd_h1_conv'), name = 'd_pre2') #16
        h2 = ops.lrelu(ops.conv2d(h1, ddim*4, name = 'd_h2_conv'), name = 'd_pre3') #8
        h3 = ops.lrelu(ops.conv2d(h2, ddim*8, name = 'd_h3_conv'), name = 'd_pre4') #4
        
        h3_raw_lin = reshape(h3, [self.options['batch_size'], -1])
        h2_lin = reshape(h2, [self.options['batch_size'], -1])
        
        if s>=32:
            h3_lin_pre = ops.lrelu(ops.conv2d(h3, ddim*16, name = 'd_h4_conv'), name = 'd_pre5') #4
            
            h3_lin = reshape(h3_lin_pre, [self.options['batch_size'], -1])
            h_out_lin = concat(1, [h3_lin, h3_raw_lin, h2_lin, mini_information], name='hs_concat')
        else:
            h_out_lin = concat(1, [h3_raw_lin, h2_lin, mini_information], name='hs_concat')
            
        h3_lin_hidden = ops.lrelu(ops.linear(h_out_lin, ddim*8,'d_hidden'))
        h3_lin_hidden = concat(1, [h3_lin_hidden,h_out_lin], 'hs_smart')
        
        h4 = ops.linear(h3_lin_hidden, 1, 'd_h3_lin_2')
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        txt3 = ops.linear(t_text_embedding, self.options['t_dim']*2,'d_embedding')
        txt3 = tf.expand_dims(txt3,1)
        txt3 = tf.expand_dims(txt3,2)
        tiled3 = ops.lrelu(tf.tile(txt3, [1,int(s/16),int(s/16),1], name='tiled_embeddings'))
        
        h3_concat = concat( 3, [h3, tiled3], name='h3_concat')
        h3_new = ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new') #4
        h3_new_out = reshape(h3_new, [self.options['batch_size'], -1])
        
        txt2 = ops.linear(t_text_embedding, self.options['t_dim']/2,'d_embedding2')
        txt2 = tf.expand_dims(txt2,1)
        txt2 = tf.expand_dims(txt2,2)
        tiled2 = ops.lrelu(tf.tile(txt2, [1,int(s/8),int(s/8),1], name='tiled_embeddings2'))
        
        h2_concat = concat( 3, [h2, tiled2], name='h2_concat')
        h2_new = ops.conv2d(h2_concat, self.options['df_dim']*4, 1,1,1,1, name = 'd_h2_conv_new') #4
        h2_new_out = reshape(h2_new, [self.options['batch_size'], -1])
        
        h_all_text = concat(1, [h3_new_out, h2_new_out], 'h_all_txt')
        
        h5 = ops.linear(ops.lrelu(h_all_text), 1, 'd_h3_lin')
        
        h0mean, h0std = tf.nn.moments(h0, axes=[1,2])
        h0spread = spread(h0)
        h0max = tf.reduce_max(h0, [1, 2])
        h1mean, h1std = tf.nn.moments(h1, axes=[1,2])
        h1spread = spread(h1)
        h1max = tf.reduce_max(h1, [1, 2])
        h2mean, h2std = tf.nn.moments(h2, axes=[1,2])
        h2spread = spread(h2)
        h2max = tf.reduce_max(h2, [1, 2])
        h3mean, h3std = tf.nn.moments(h3, axes=[1,2])
        h3spread = spread(h3)
        h3max = tf.reduce_max(h3, [1, 2])
        
        h2_txt_mean, h2_txt_std = tf.nn.moments(h2_new, axes=[1,2])
        h2_txt_spread = spread(h2_new)
        h2_txt_max = tf.reduce_max(h2_new, [1, 2])
        
        h3_txt_mean, h3_txt_std = tf.nn.moments(h3_new, axes=[1,2])
        h3_txt_spread = spread(h3_new)
        h3_txt_max = tf.reduce_max(h3_new, [1, 2])

        h2_txt_mean = h2_txt_mean * (1-self.noise_indicator)
        h2_txt_std = h2_txt_std * (1-self.noise_indicator)
        h2_txt_spread = h2_txt_spread * (1-self.noise_indicator)
        h2_txt_max = h2_txt_max * (1-self.noise_indicator)
        
        h3_txt_mean = h3_txt_mean * (1-self.noise_indicator)
        h3_txt_std = h3_txt_std * (1-self.noise_indicator)
        h3_txt_spread = h3_txt_spread * (1-self.noise_indicator)
        h3_txt_max = h3_txt_max * (1-self.noise_indicator)
        
        distribution_parameters = [h0mean, h0std, h0spread,h0max, h1mean, h1std, h1spread,h1max, h2mean, h2std, h2spread,h2max,
                   h3mean, h3std, h3spread,h3max,h2_txt_mean, h2_txt_std, h2_txt_spread,h2_txt_max,
                   h3_txt_mean, h3_txt_std, h3_txt_spread,h3_txt_max]
        
        dist_len = len(distribution_parameters)
        stds = [None]*dist_len
        means = [None]*dist_len
        for idx, dp in enumerate(distribution_parameters):
            m, s = tf.nn.moments(dp, axes=[0])
            means[idx] = m
            stds[idx] = s
        
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5), stds+means
    
    
    def discriminator2(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        
        s = image.get_shape()[1].value
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=3, kernel_dim=3, reuse = reuse, lin_name = 'd_minilin1')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        ddim = self.options['df_dim']
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h0_conv'), name = 'd_pre1') #32
        h1 = ops.lrelu(ops.conv2d(h0, ddim*2, name = 'd_h1_conv'), name = 'd_pre2') #16
        h2 = ops.lrelu(ops.conv2d(h1, ddim*4, name = 'd_h2_conv'), name = 'd_pre3') #8
        h3 = ops.lrelu(ops.conv2d(h2, ddim*8, name = 'd_h3_conv'), name = 'd_pre4') #4
        
        h3_raw_lin = reshape(h3, [self.options['batch_size'], -1])
        h2_lin = reshape(h2, [self.options['batch_size'], -1])
        
        if s>=32:
            h3_lin_pre = ops.lrelu(ops.conv2d(h3, ddim*16, name = 'd_h4_conv'), name = 'd_pre5') #4
            
            h3_lin = reshape(h3_lin_pre, [self.options['batch_size'], -1])
            h_out_lin = concat(1, [h3_lin, h3_raw_lin, h2_lin, mini_information], name='hs_concat')
        else:
            h_out_lin = concat(1, [h3_raw_lin, h2_lin, mini_information], name='hs_concat')
            
        h3_lin_hidden = ops.lrelu(ops.linear(h_out_lin, ddim*8,'d_hidden'))
        h3_lin_hidden = concat(1, [h3_lin_hidden,h_out_lin], 'hs_smart')
        
        h4 = ops.linear(h3_lin_hidden, 1, 'd_h3_lin_2')
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        txt3 = ops.linear(t_text_embedding, self.options['t_dim'],'d_embedding')
        txt3 = tf.expand_dims(txt3,1)
        txt3 = tf.expand_dims(txt3,2)
        tiled3 = ops.lrelu(tf.tile(txt3, [1,int(s/16),int(s/16),1], name='tiled_embeddings'))
        
        h3_concat = concat( 3, [h3, tiled3], name='h3_concat')
        h3_new = ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new') #4
        h3_new_out = reshape(h3_new, [self.options['batch_size'], -1])
        
        txt2 = ops.linear(t_text_embedding, self.options['t_dim']/4,'d_embedding2')
        txt2 = tf.expand_dims(txt2,1)
        txt2 = tf.expand_dims(txt2,2)
        tiled2 = ops.lrelu(tf.tile(txt2, [1,int(s/8),int(s/8),1], name='tiled_embeddings2'))
        
        h2_concat = concat( 3, [h2, tiled2], name='h2_concat')
        h2_new = ops.conv2d(h2_concat, self.options['df_dim']*4, 1,1,1,1, name = 'd_h2_conv_new') #4
        h2_new_out = reshape(h2_new, [self.options['batch_size'], -1])
        
        h_all_text = concat(1, [h3_new_out, h2_new_out], 'h_all_txt')
        
        h5 = ops.linear(ops.lrelu(h_all_text), 1, 'd_h3_lin')
        
        h0mean, h0std = tf.nn.moments(h0, axes=[1,2])
        h0spread = spread(h0)
        h0max = tf.reduce_max(h0, [1, 2])
        h1mean, h1std = tf.nn.moments(h1, axes=[1,2])
        h1spread = spread(h1)
        h1max = tf.reduce_max(h1, [1, 2])
        h2mean, h2std = tf.nn.moments(h2, axes=[1,2])
        h2spread = spread(h2)
        h2max = tf.reduce_max(h2, [1, 2])
        h3mean, h3std = tf.nn.moments(h3, axes=[1,2])
        h3spread = spread(h3)
        h3max = tf.reduce_max(h3, [1, 2])
        
        h2_txt_mean, h2_txt_std = tf.nn.moments(h2_new, axes=[1,2])
        h2_txt_spread = spread(h2_new)
        h2_txt_max = tf.reduce_max(h2_new, [1, 2])
        
        h3_txt_mean, h3_txt_std = tf.nn.moments(h3_new, axes=[1,2])
        h3_txt_spread = spread(h3_new)
        h3_txt_max = tf.reduce_max(h3_new, [1, 2])

        h2_txt_mean = h2_txt_mean * (1-self.noise_indicator)
        h2_txt_std = h2_txt_std * (1-self.noise_indicator)
        h2_txt_spread = h2_txt_spread * (1-self.noise_indicator)
        h2_txt_max = h2_txt_max * (1-self.noise_indicator)
        
        h3_txt_mean = h3_txt_mean * (1-self.noise_indicator)
        h3_txt_std = h3_txt_std * (1-self.noise_indicator)
        h3_txt_spread = h3_txt_spread * (1-self.noise_indicator)
        h3_txt_max = h3_txt_max * (1-self.noise_indicator)
        
        distribution_parameters = [h0mean, h0std, h0spread,h0max, h1mean, h1std, h1spread,h1max, h2mean, h2std, h2spread,h2max,
                   h3mean, h3std, h3spread,h3max,h2_txt_mean, h2_txt_std, h2_txt_spread,h2_txt_max,
                   h3_txt_mean, h3_txt_std, h3_txt_spread,h3_txt_max]
        
        dist_len = len(distribution_parameters)
        stds = [None]*dist_len
        means = [None]*dist_len
        for idx, dp in enumerate(distribution_parameters):
            m, s = tf.nn.moments(dp, axes=[0])
            means[idx] = m
            stds[idx] = s
        
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5), stds+means
    
    
    def discriminator3(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        
        s = image.get_shape()[1].value
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=3, kernel_dim=3, reuse = reuse, lin_name = 'd_minilin1')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        ddim = self.options['df_dim']
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h0_conv'), name = 'd_pre1') #32
        h1 = ops.lrelu(ops.conv2d(h0, ddim*2, name = 'd_h1_conv'), name = 'd_pre2') #16
        h2 = ops.lrelu(ops.conv2d(h1, ddim*4, name = 'd_h2_conv'), name = 'd_pre3') #8
        h3 = ops.lrelu(ops.conv2d(h2, ddim*8, name = 'd_h3_conv'), name = 'd_pre4') #4
        
        h3_raw_lin = reshape(h3, [self.options['batch_size'], -1])
        h2_lin = reshape(h2, [self.options['batch_size'], -1])
        
        if s>=32:
            h3_lin_pre = ops.lrelu(ops.conv2d(h3, ddim*16, name = 'd_h4_conv'), name = 'd_pre5') #4
            
            h3_lin = reshape(h3_lin_pre, [self.options['batch_size'], -1])
            h_out_lin = concat(1, [h3_lin, h3_raw_lin, h2_lin, mini_information], name='hs_concat')
        else:
            h_out_lin = concat(1, [h3_raw_lin, h2_lin, mini_information], name='hs_concat')
            
        h3_lin_hidden = ops.lrelu(ops.linear(h_out_lin, ddim*8,'d_hidden'))
        h3_lin_hidden = concat(1, [h3_lin_hidden,h_out_lin], 'hs_smart')
        
        h4 = ops.linear(h3_lin_hidden, 1, 'd_h3_lin_2')
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        txt3 = ops.linear(t_text_embedding, self.options['t_dim']/2,'d_embedding')
        txt3 = tf.expand_dims(txt3,1)
        txt3 = tf.expand_dims(txt3,2)
        tiled3 = ops.lrelu(tf.tile(txt3, [1,int(s/16),int(s/16),1], name='tiled_embeddings'))
        
        h3_concat = concat( 3, [h3, tiled3], name='h3_concat')
        h3_new = ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new') #4
        h3_new_out = reshape(h3_new, [self.options['batch_size'], -1])
        
        txt2 = ops.linear(t_text_embedding, self.options['t_dim']/8,'d_embedding2')
        txt2 = tf.expand_dims(txt2,1)
        txt2 = tf.expand_dims(txt2,2)
        tiled2 = ops.lrelu(tf.tile(txt2, [1,int(s/8),int(s/8),1], name='tiled_embeddings2'))
        
        h2_concat = concat( 3, [h2, tiled2], name='h2_concat')
        h2_new = ops.conv2d(h2_concat, self.options['df_dim']*4, 1,1,1,1, name = 'd_h2_conv_new') #4
        h2_new_out = reshape(h2_new, [self.options['batch_size'], -1])
        
        h_all_text = concat(1, [h3_new_out, h2_new_out], 'h_all_txt')
        
        h5 = ops.linear(ops.lrelu(h_all_text), 1, 'd_h3_lin')
        
        h0mean, h0std = tf.nn.moments(h0, axes=[1,2])
        h0spread = spread(h0)
        h0max = tf.reduce_max(h0, [1, 2])
        h1mean, h1std = tf.nn.moments(h1, axes=[1,2])
        h1spread = spread(h1)
        h1max = tf.reduce_max(h1, [1, 2])
        h2mean, h2std = tf.nn.moments(h2, axes=[1,2])
        h2spread = spread(h2)
        h2max = tf.reduce_max(h2, [1, 2])
        h3mean, h3std = tf.nn.moments(h3, axes=[1,2])
        h3spread = spread(h3)
        h3max = tf.reduce_max(h3, [1, 2])
        
        h2_txt_mean, h2_txt_std = tf.nn.moments(h2_new, axes=[1,2])
        h2_txt_spread = spread(h2_new)
        h2_txt_max = tf.reduce_max(h2_new, [1, 2])
        
        h3_txt_mean, h3_txt_std = tf.nn.moments(h3_new, axes=[1,2])
        h3_txt_spread = spread(h3_new)
        h3_txt_max = tf.reduce_max(h3_new, [1, 2])

        h2_txt_mean = h2_txt_mean * (1-self.noise_indicator)
        h2_txt_std = h2_txt_std * (1-self.noise_indicator)
        h2_txt_spread = h2_txt_spread * (1-self.noise_indicator)
        h2_txt_max = h2_txt_max * (1-self.noise_indicator)
        
        h3_txt_mean = h3_txt_mean * (1-self.noise_indicator)
        h3_txt_std = h3_txt_std * (1-self.noise_indicator)
        h3_txt_spread = h3_txt_spread * (1-self.noise_indicator)
        h3_txt_max = h3_txt_max * (1-self.noise_indicator)
        
        distribution_parameters = [h0mean, h0std, h0spread,h0max, h1mean, h1std, h1spread,h1max, h2mean, h2std, h2spread,h2max,
                   h3mean, h3std, h3spread,h3max,h2_txt_mean, h2_txt_std, h2_txt_spread,h2_txt_max,
                   h3_txt_mean, h3_txt_std, h3_txt_spread,h3_txt_max]
        
        dist_len = len(distribution_parameters)
        stds = [None]*dist_len
        means = [None]*dist_len
        for idx, dp in enumerate(distribution_parameters):
            m, s = tf.nn.moments(dp, axes=[0])
            means[idx] = m
            stds[idx] = s
        
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5), stds+means
    
    