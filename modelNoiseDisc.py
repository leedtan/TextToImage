import tensorflow as tf
from Utils import ops


prince = True

def reshape(x, arr):
    return tf.reshape(x, [int(a) for a in arr])


if prince:
    def concat(dim, objects, name=None):
        if name is None:
            return tf.concat(objects, dim)
        else:
            return tf.concat(objects, dim, name = None)
    def cross_entropy(logits, labels):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
    def pack(x):
        return tf.stack(x)
else:
    def concat(dim, objects, name=None):
        if name is None:
            return tf.concat(dim, objects)
        else:
            return tf.concat(dim, objects, name = None)
    def cross_entropy(logits, labels):
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
    def __init__(self, options):
        self.options = options

        #Hope to remove these eventually, some are needed for code I havent refactored
        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')
        self.g_bn4 = ops.batch_norm(name='g_bn4')
        self.g_bn1h = ops.batch_norm(name='g_bn1h')
        self.g_bn2h = ops.batch_norm(name='g_bn2h')
        self.g_bnresblock1 = ops.batch_norm(name='g_bnresblock1')
        self.g_bnresblock2 = ops.batch_norm(name='g_bnresblock2')
        self.g_bn_small = ops.batch_norm(name='g_bn_small')
        self.g_bn_mid = ops.batch_norm(name='g_bn_mid')


        self.d_bn_idx = 0
        self.g_idx = 0
        self.g2_idx = 0
        self.g3_idx = 0
        self.g4_idx = 0
        self.g5_idx = 0
        self.d_idx = 0
        self.g_bn_idx = 0
        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')
        self.d_bn3 = ops.batch_norm(name='d_bn3')
        self.d_bn4 = ops.batch_norm(name='d_bn4')
        self.d_bn4_small = ops.batch_norm(name='d_bn4_small')
        self.d_bn4_mid = ops.batch_norm(name='d_bn4_mid')
        
        self.bn0m = ops.batch_norm(name='d_bn_clean_dbg1')
        self.bn1m = ops.batch_norm(name='d_bn_clean_dbg2')
        self.bn2m = ops.batch_norm(name='d_bn_clean_dbg3')
        self.bn3m = ops.batch_norm(name='d_bn_clean_dbg4')
        self.bn4m = ops.batch_norm(name='d_bn_clean_dbg5')
        self.bn0s = ops.batch_norm(name='d_bn_clean_dbg6')
        self.bn1s = ops.batch_norm(name='d_bn_clean_dbg7')
        self.bn2s = ops.batch_norm(name='d_bn_clean_dbg8')
        self.bn3s = ops.batch_norm(name='d_bn_clean_dbg18')
        self.bn4s = ops.batch_norm(name='d_bn_clean_dbg9')
        self.bn0f = ops.batch_norm(name='d_bn_clean_dbg10')
        self.bn1f = ops.batch_norm(name='d_bn_clean_dbg11')
        self.bn2f = ops.batch_norm(name='d_bn_clean_dbg12')
        self.bn3f = ops.batch_norm(name='d_bn_clean_dbg13')
        self.bn4f = ops.batch_norm(name='d_bn_clean_dbg14')

    def g_bn(self):
        g_bn = ops.batch_norm(name='g_bn_clean' + str(self.g_bn_idx))
        self.g_bn_idx += 1
        return g_bn
    def d_bn(self):
        d_bn = ops.batch_norm(name='d_bn_clean' + str(self.d_bn_idx))
        self.d_bn_idx += 1
        return d_bn
        
    def build_model(self, beta1, beta2, lr):
        img_size = self.options['image_size']
        
        l2reg = tf.placeholder('float32', shape=())
        
        t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
        LambdaDAE = tf.placeholder('float32', shape=())
        t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
        
        noise_indicator = tf.placeholder('float32', shape=())
        
        
        
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
        
        #fake_image, fake_small_image, fake_mid_image = self.generator(t_z, t_real_caption)
        fake_image1, fake_image2, fake_image3, fake_image4, fake_image5 ,\
            gDAE1Loss, gDAE2Loss, gDAE3Loss, gDAE4Loss, gDAE5Loss, = self.generator(t_z, t_real_caption)
        img_noise = .1
        noisy_caption = ops.noised(t_real_caption,.0001)
        noisy_img = ops.noised(t_real_image,img_noise)
        noisy_wrong_img = ops.noised(t_wrong_image,img_noise)
        
        noisy_fake_img1 = ops.noised(fake_image1,img_noise)
        noisy_fake_img2 = ops.noised(fake_image2,img_noise)
        noisy_fake_img3 = ops.noised(fake_image3,img_noise)
        noisy_fake_img4 = ops.noised(fake_image4,img_noise)
        noisy_fake_img5 = ops.noised(fake_image5,img_noise)
        
        with tf.variable_scope('scope_1'):
            p_1_real_img_logit, p_1_real_txt_logit, p_1_real_img, p_1_real_txt    = self.discriminator1(noisy_img, noisy_caption)
            p_1_wrong_img_logit, p_1_wrong_txt_logit, p_1_wrong_img, p_1_wrong_txt   = self.discriminator1(noisy_wrong_img, noisy_caption, reuse = True)
            p_1_fake_img_logit, p_1_fake_txt_logit, p_1_fake_img, p_1_fake_txt   = self.discriminator1(noisy_fake_img1, noisy_caption, reuse = True)
        
        with tf.variable_scope('scope_2'):
            p_2_real_img_logit, p_2_real_txt_logit, p_2_real_img, p_2_real_txt   = self.discriminator2(noisy_img, noisy_caption)
            p_2_wrong_img_logit, p_2_wrong_txt_logit, p_2_wrong_img, p_2_wrong_txt   = self.discriminator2(noisy_wrong_img, noisy_caption, reuse = True)
            p_2_fake_img_logit, p_2_fake_txt_logit, p_2_fake_img, p_2_fake_txt   = self.discriminator2(noisy_fake_img2, noisy_caption, reuse = True)
            
        with tf.variable_scope('scope_3'):
            p_3_real_img_logit, p_3_real_txt_logit, p_3_real_img, p_3_real_txt   = self.discriminator3(noisy_img, noisy_caption)
            p_3_wrong_img_logit, p_3_wrong_txt_logit, p_3_wrong_img, p_3_wrong_txt   = self.discriminator3(noisy_wrong_img, noisy_caption, reuse = True)
            p_3_fake_img_logit, p_3_fake_txt_logit, p_3_fake_img, p_3_fake_txt   = self.discriminator3(noisy_fake_img3, noisy_caption, reuse = True)
            
        with tf.variable_scope('scope_4'):
            p_4_real_img_logit, p_4_real_txt_logit, p_4_real_img, p_4_real_txt   = self.discriminator4(noisy_img, noisy_caption)
            p_4_wrong_img_logit, p_4_wrong_txt_logit, p_4_wrong_img, p_4_wrong_txt   = self.discriminator4(noisy_wrong_img, noisy_caption, reuse = True)
            p_4_fake_img_logit, p_4_fake_txt_logit, p_4_fake_img, p_4_fake_txt   = self.discriminator4(noisy_fake_img4, noisy_caption, reuse = True)
            
        with tf.variable_scope('scope_5'):
            p_5_real_img_logit, p_5_real_txt_logit, p_5_real_img, p_5_real_txt   = self.discriminator5(noisy_img, noisy_caption)
            p_5_wrong_img_logit, p_5_wrong_txt_logit, p_5_wrong_img, p_5_wrong_txt   = self.discriminator5(noisy_wrong_img, noisy_caption, reuse = True)
            p_5_fake_img_logit, p_5_fake_txt_logit, p_5_fake_img, p_5_fake_txt   = self.discriminator5(noisy_fake_img5, noisy_caption, reuse = True)


        '''
        with tf.variable_scope('scope_mid'):
            disc_mid_real_image, disc_mid_real_image_logit   = self.discriminator_mid(real_mid_image, t_real_caption)
            disc_mid_wrong_image, disc_mid_wrong_image_logit   = self.discriminator_mid(wrong_mid_image, t_real_caption, reuse = True)
            disc_mid_fake_image, disc_mid_fake_image_logit   = self.discriminator_mid(fake_mid_image, t_real_caption, reuse = True)
        '''
        
        pos_examples = tf.ones_like(p_1_fake_img_logit)
        neg_examples = tf.zeros_like(p_1_fake_img_logit)
        d_loss_real = tf.reduce_mean(cross_entropy(p_1_real_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_1_real_txt_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_2_real_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_2_real_txt_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_3_real_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_3_real_txt_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_4_real_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_4_real_txt_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_5_real_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_5_real_txt_logit, pos_examples))
            
        d_loss_wrong = tf.reduce_mean(cross_entropy(p_1_wrong_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_1_wrong_txt_logit, neg_examples)) + \
            tf.reduce_mean(cross_entropy(p_2_wrong_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_2_wrong_txt_logit, neg_examples)) + \
            tf.reduce_mean(cross_entropy(p_3_wrong_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_3_wrong_txt_logit, neg_examples)) + \
            tf.reduce_mean(cross_entropy(p_4_wrong_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_4_wrong_txt_logit, neg_examples)) + \
            tf.reduce_mean(cross_entropy(p_5_wrong_img_logit, pos_examples)) + tf.reduce_mean(cross_entropy(p_5_wrong_txt_logit, neg_examples))
        #d_loss_noise = tf.reduce_mean(cross_entropy(p_noise_img_logit, pos_examples))
        
        d_loss_real_img = tf.reduce_mean(cross_entropy(p_1_real_img_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_2_real_img_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_3_real_img_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_4_real_img_logit, pos_examples)) + \
            tf.reduce_mean(cross_entropy(p_5_real_img_logit, pos_examples))
            
        d1_loss_noise = tf.reduce_mean(cross_entropy(p_1_fake_img_logit, neg_examples))
        d2_loss_noise = tf.reduce_mean(cross_entropy(p_2_fake_img_logit, neg_examples))
        d3_loss_noise = tf.reduce_mean(cross_entropy(p_3_fake_img_logit, neg_examples))
        d4_loss_noise = tf.reduce_mean(cross_entropy(p_4_fake_img_logit, neg_examples))
        d5_loss_noise = tf.reduce_mean(cross_entropy(p_5_fake_img_logit, neg_examples))
        
        g1_loss_noise = tf.reduce_mean(cross_entropy(p_1_fake_img_logit, pos_examples)) + gDAE1Loss*LambdaDAE
        g2_loss_noise = tf.reduce_mean(cross_entropy(p_2_fake_img_logit, pos_examples)) + gDAE2Loss*LambdaDAE
        g3_loss_noise = tf.reduce_mean(cross_entropy(p_3_fake_img_logit, pos_examples)) + gDAE3Loss*LambdaDAE
        g4_loss_noise = tf.reduce_mean(cross_entropy(p_4_fake_img_logit, pos_examples)) + gDAE4Loss*LambdaDAE
        g5_loss_noise = tf.reduce_mean(cross_entropy(p_5_fake_img_logit, pos_examples)) + gDAE5Loss*LambdaDAE
        
        
        
        d1_loss = d1_loss_noise + tf.reduce_mean(cross_entropy(p_1_fake_txt_logit, neg_examples))
        d2_loss = d2_loss_noise + tf.reduce_mean(cross_entropy(p_2_fake_txt_logit, neg_examples))
        d3_loss = d3_loss_noise + tf.reduce_mean(cross_entropy(p_3_fake_txt_logit, neg_examples))
        d4_loss = d4_loss_noise + tf.reduce_mean(cross_entropy(p_4_fake_txt_logit, neg_examples))
        d5_loss = d5_loss_noise + tf.reduce_mean(cross_entropy(p_5_fake_txt_logit, neg_examples))
        
        g1_loss = g1_loss_noise + tf.reduce_mean(cross_entropy(p_1_fake_txt_logit, pos_examples))
        g2_loss = g2_loss_noise + tf.reduce_mean(cross_entropy(p_2_fake_txt_logit, pos_examples))
        g3_loss = g3_loss_noise + tf.reduce_mean(cross_entropy(p_3_fake_txt_logit, pos_examples))
        g4_loss = g4_loss_noise + tf.reduce_mean(cross_entropy(p_4_fake_txt_logit, pos_examples))
        g5_loss = g5_loss_noise + tf.reduce_mean(cross_entropy(p_5_fake_txt_logit, pos_examples))
        
        d_loss_noise = d_loss_real_img + d1_loss_noise + d2_loss_noise + d3_loss_noise + d4_loss_noise + d5_loss_noise
        g_loss_noise = g1_loss_noise + g2_loss_noise + g3_loss_noise + g4_loss_noise + g5_loss_noise
        
        d_loss = d_loss_real + d_loss_wrong + d1_loss + d2_loss + d3_loss + d4_loss + d5_loss
        g_loss = g1_loss + g2_loss + g3_loss + g4_loss + g5_loss
        
        
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g1_vars = [var for var in t_vars if 'g1_' in var.name]
        g2_vars = [var for var in t_vars if 'g2_' in var.name]
        g3_vars = [var for var in t_vars if 'g3_' in var.name]
        g4_vars = [var for var in t_vars if 'g4_' in var.name]
        g5_vars = [var for var in t_vars if 'g5_' in var.name]
        
        
        self.wgan_clip_1eneg3 = [v.assign(tf.clip_by_value(v, -0.001, 0.001)) for v in d_vars]
        self.wgan_clip_1eneg2 = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
        self.wgan_clip_1eneg1 = [v.assign(tf.clip_by_value(v, -0.1, 0.1)) for v in d_vars]
        self.wgan_clip_1 = [v.assign(tf.clip_by_value(v, -1, 1)) for v in d_vars]

        input_tensors = {
            't_real_image' : t_real_image,
            't_wrong_image' : t_wrong_image,
            't_real_caption' : t_real_caption,
            't_z' : t_z,
            'l2reg' : l2reg,
            'LambdaDAE' : LambdaDAE,
            'noise_indicator' : noise_indicator
        }

        variables = {
            'd_vars' : d_vars,
            'g1_vars' : g1_vars,
            'g2_vars' : g2_vars,
            'g3_vars' : g3_vars,
            'g4_vars' : g4_vars,
            'g5_vars' : g5_vars
        }

        loss = {
            'g_loss': g_loss,
            'd_loss' : d_loss,
            
            'g1_loss' : g1_loss,
            'g2_loss' : g2_loss,
            'g3_loss' : g3_loss,
            'g4_loss' : g4_loss,
            'g5_loss' : g5_loss,
            
            'g1_loss_noise' : g1_loss_noise,
            'g2_loss_noise' : g2_loss_noise,
            'g3_loss_noise' : g3_loss_noise,
            'g4_loss_noise' : g4_loss_noise,
            'g5_loss_noise' : g5_loss_noise,
            
            'd1_loss' : d1_loss,
            'd2_loss' : d2_loss,
            'd3_loss' : d3_loss,
            'd4_loss' : d4_loss,
            'd5_loss' : d5_loss,
            
            'd1_loss_noise' : d1_loss_noise,
            'd2_loss_noise' : d2_loss_noise,
            'd3_loss_noise' : d3_loss_noise,
            'd4_loss_noise' : d4_loss_noise,
            'd5_loss_noise' : d5_loss_noise,
            
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
            'img4' : fake_image4,
            'img5' : fake_image5
        }

        
        d_optim = tf.train.AdamOptimizer(lr, beta1 = beta1,beta2 = beta2).minimize(loss['d_loss'], var_list=variables['d_vars'])
    
        gloss5 = loss['g5_loss']
        gloss4 = loss['g4_loss']# + .5 * gloss5
        gloss3 = loss['g3_loss']# + .5 * gloss4
        gloss2 = loss['g2_loss']# + .5 * gloss3
        gloss1 = loss['g1_loss']# + .5 * gloss2
        self.g_optim = tf.train.AdamOptimizer(lr, beta1 = beta1,beta2 = beta2).minimize(
            g_loss_noise * noise_indicator + g_loss * (1-noise_indicator),
            var_list=variables['g1_vars'] + variables['g2_vars'] + 
            variables['g3_vars'] + variables['g4_vars'] + variables['g5_vars'])
        
        self.d_optim = tf.train.AdamOptimizer(lr, beta1 = beta1,beta2 = beta2).minimize(
            d_loss_noise * noise_indicator/5.0 + d_loss/5.0 * (1-noise_indicator), var_list=variables['d_vars'])
    
            
        
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
    def g4_name(self):
        self.g4_idx += 1
        return 'g4_' + str(self.g4_idx)
    def g5_name(self):
        self.g5_idx += 1
        return 'g5_' + str(self.g5_idx)
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
    
    
    #Residual Block
    def add_residual(self, prev_layer, z_concat, text_filters = 3, k_h = 5, k_w = 5, hidden_text_filters = 20,name_func=None):
        if name_func is None:
            name_func = self.g_name
        
        s = prev_layer.get_shape()[1].value
        filters = prev_layer.get_shape()[3].value
        
        bn0 = self.g_bn()
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        bn3 = self.g_bn()
        
        if z_concat is not None:
            text_augment = reshape(ops.lrelu(bn1(
                ops.linear(z_concat,  s/2* s/2* text_filters,name_func()))),[-1, s/2, s/2, text_filters])
            
            text_augment = ops.lrelu(bn0(ops.deconv2d(text_augment,
                [self.options['batch_size'], s, s, hidden_text_filters], name=name_func())))
        
            concatenated = concat(3, [text_augment, prev_layer], name=name_func())
        else:
            concatenated = prev_layer
            
        res_hidden = bn2(ops.deconv2d(concatenated, 
            [self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func()))
        
        residual = ops.deconv2d(ops.lrelu(res_hidden, name = name_func()),
            [self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func())
        
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
        bn3 = self.g_bn()
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
    def add_residual_standard(self, prev_layer, z_concat, text_filters = 1, k_h = 5, k_w = 5,name_func=None):
        if name_func is None:
            name_func = self.g_name
        
        s = prev_layer.get_shape()[1].value
        filters = prev_layer.get_shape()[3].value
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        bn3 = self.g_bn()
        if z_concat is not None:
            text_augment = reshape(ops.lrelu(bn1(
                ops.linear(z_concat,  s* s* text_filters,name_func())), name = name_func()),[-1, s, s, text_filters])
            concatenated = concat(3, [text_augment, prev_layer], name=name_func())
        else:
            concatenated = prev_layer
        res_hidden = bn2(ops.deconv2d(concatenated,
            [self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=name_func()))
        
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
        bn3 = self.g_bn()
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
    
    def add_stackGan(self, prev_layer, s2, z_concat, name_func=None):
        bn1 = self.g_bn()
        bn2 = self.g_bn()
        prev_layer = self.add_residual(prev_layer, z_concat,name_func=name_func)
        prev_layer = self.add_residual(prev_layer, z_concat,name_func=name_func)
        prev_layer = self.add_residual_standard(prev_layer, z_concat,name_func=name_func)
        prev_layer = self.add_residual_standard(prev_layer, z_concat,name_func=name_func)
        prev_layer = self.add_residual(prev_layer, z_concat=None,name_func=name_func)
        prev_layer = self.add_residual(prev_layer, z_concat=None,name_func=name_func)
        prev_layer = self.add_residual(prev_layer, z_concat=None,name_func=name_func,k_h = 3, k_w = 3)
        return prev_layer
    
    
    # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def generator(self, t_z, t_text_embedding):
        
        s = self.options['image_size']
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        
        bn1 = self.g_bn()
        obn2 = self.g_bn()
        bn2 = self.g_bn()
        obn3 = self.g_bn()
        bn3 = self.g_bn()
        bn4 = self.g_bn()
        bn5 = self.g_bn()
        bn6 = self.g_bn()
        bn7 = self.g_bn()
        bn8 = self.g_bn()
        bn9 = self.g_bn()
        bn10 = self.g_bn()
        bn11 = self.g_bn()
        bn12 = self.g_bn()
        bn13 = self.g_bn()
        z_rec_target = concat(1, [t_z, t_text_embedding])
        reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g1_embedding'))
        reduced_text_embedding2 = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g2_embedding'))
        reduced_text_embedding3 = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g3_embedding'))
        reduced_text_embedding4 = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g4_embedding'))
        reduced_text_embedding5 = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g5_embedding'))
        z_concat = concat(1, [t_z, reduced_text_embedding])
        z_concat2 = concat(1, [t_z, reduced_text_embedding2])
        z_concat3 = concat(1, [t_z, reduced_text_embedding3])
        z_concat4 = concat(1, [t_z, reduced_text_embedding4])
        z_concat5 = concat(1, [t_z, reduced_text_embedding5])
        z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g1_h0_lin')
        h0 = reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
        h0 = ops.lrelu(self.g_bn0(h0) , name = 'g1_pre2')
        
        h0 = self.add_residual_standard(h0, z_concat, k_h = 3, k_w = 3, text_filters=4)
        
        #h0 = self.add_residual_standard(h0, None, k_h = 3, k_w = 3, text_filters=4)
        #h1 = tf.image.resize_nearest_neighbor(h0, [s8, s8])
        h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4],name='g1_h1')
        h1 = ops.lrelu(self.g_bn1(h1), name = 'g1_pre43234')
        #h1 = ops.deconv2d(h1, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], d_h=1, d_w=1,name=self.g_name())
        #h1 = ops.lrelu(bn1(h1), name = self.g_name())
        #h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g1_h1')
        
        h1 = self.add_residual(h1, z_concat)
        #h1 = self.add_residual(h1, None)
        
        #h2 = tf.image.resize_nearest_neighbor(h1, [s4, s4])
        h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2],name='g1_h2')
        h2 = ops.lrelu(self.g_bn2(h2), name = 'g1_pre432')
        #h2 = ops.deconv2d(h2, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], d_h=1, d_w=1,name=self.g_name())
        #h2 = ops.lrelu(obn2(h2), name = self.g_name())
        
        h2 = self.add_residual(h2, z_concat)
        #h2 = self.add_residual(h2, None)

        #h3 = tf.image.resize_nearest_neighbor(h2, [s2, s2])
        h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1],name='g1_h3')
        h3 = ops.lrelu(self.g_bn3(h3), name = 'g1_pre4534')
        #h3 = ops.deconv2d(h3, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], d_h=1, d_w=1,name=self.g_name())
        #h3 = ops.lrelu(obn3(h3), name = self.g_name())
        #h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*2], name='g1_h3')
        
        h3 = self.add_residual(h3, z_concat)
        #h3 = self.add_residual(h3, z_concat)
        
        stack1 = ops.deconv2d(h3,[self.options['batch_size'], s, s, self.options['gf_dim']/4], name='g1_h6_out')
        stack1 = ops.lrelu(self.g_bn4(stack1))
        
        stack1out = ops.deconv2d(stack1, [self.options['batch_size'], s, s, 3], d_h=1, d_w=1,name=self.g_name())
        
        stack2 = self.add_stackGan(stack1, s, z_concat2, name_func=self.g2_name)
        #stack2out = ops.deconv2d(stack2, [self.options['batch_size'], s, s, 3],name=self.g2_name())
        #stack2out = ops.lrelu(bn2(stack2out))
        stack2out = ops.deconv2d(stack2, [self.options['batch_size'], s, s, 3], d_h=1, d_w=1,name=self.g2_name())
        
        stack3 = self.add_stackGan(stack2, s, z_concat3, name_func=self.g3_name)
        #stack3out = ops.deconv2d(stack3, [self.options['batch_size'], s, s, 3],name=self.g3_name())
        #stack3out = ops.lrelu(bn3(stack3out))
        stack3out = ops.deconv2d(stack3, [self.options['batch_size'], s, s, 3], d_h=1, d_w=1,name=self.g3_name())
        
        stack4 = self.add_stackGan(stack3, s, z_concat4, name_func=self.g4_name)
        #stack4out = ops.deconv2d(stack4, [self.options['batch_size'], s, s, 3],name=self.g4_name())
        #stack4out = ops.lrelu(bn4(stack4out))
        stack4out = ops.deconv2d(stack4, [self.options['batch_size'], s, s, 3], d_h=1, d_w=1,name=self.g4_name())
        
        stack5 = self.add_stackGan(stack4, s, z_concat5, name_func=self.g5_name)
        #stack5out = ops.deconv2d(stack5, [self.options['batch_size'], s, s, 3],name=self.g5_name())
        #stack5out = ops.lrelu(bn5(stack5out))
        stack5out = ops.deconv2d(stack5, [self.options['batch_size'], s, s, 3], d_h=1, d_w=1,name=self.g5_name())
        
        z_target_dim = 1000+10
        
        z_rec1 = ops.linear(reshape(ops.lrelu(bn1(ops.conv2d(stack1, 5, 
                    name=self.g_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g_name())
        
        z_rec2 = ops.linear(reshape(ops.lrelu(bn2(ops.conv2d(stack2, 5, 
                    name=self.g2_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g2_name())
        
        z_rec3 = ops.linear(reshape(ops.lrelu(bn3(ops.conv2d(stack3, 5, 
                    name=self.g3_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g3_name())
        
        z_rec4 = ops.linear(reshape(ops.lrelu(bn4(ops.conv2d(stack4, 5, 
                    name=self.g4_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g4_name())
        
        z_rec5 = ops.linear(reshape(ops.lrelu(bn5(ops.conv2d(stack5, 5, 
                    name=self.g5_name()))), [self.options['batch_size'], -1]),z_target_dim,self.g5_name())
        ''' 
        
        z_rec2 = reshape(ops.deconv2d(stack2, [self.options['batch_size'], s, s, 1], 
                    d_h=1, d_w=1,name=self.g2_name()), [self.options['batch_size'], -1])
        
        z_rec3 = reshape(ops.deconv2d(stack3, [self.options['batch_size'], s, s, 1], 
                    d_h=1, d_w=1,name=self.g3_name()), [self.options['batch_size'], -1])
        
        z_rec4 = reshape(ops.deconv2d(stack4, [self.options['batch_size'], s, s, 1], 
                    d_h=1, d_w=1,name=self.g4_name()), [self.options['batch_size'], -1])
        
        z_rec5 = reshape(ops.deconv2d(stack5, [self.options['batch_size'], s, s, 1], 
                    d_h=1, d_w=1,name=self.g5_name()), [self.options['batch_size'], -1])
        '''
        
        return (tf.tanh(stack1out)/2. + 0.5), (tf.tanh(stack2out)/2. + 0.5), (tf.tanh(stack3out)/2. + 0.5), (tf.tanh(stack4out)/2. + 0.5), (tf.tanh(stack5out)/2. + 0.5), \
                tf.reduce_mean(tf.square(z_rec1 - z_rec_target)),tf.reduce_mean(tf.square(z_rec2 - z_rec_target)),\
                tf.reduce_mean(tf.square(z_rec3 - z_rec_target)), \
                tf.reduce_mean(tf.square(z_rec4 - z_rec_target)),tf.reduce_mean(tf.square(z_rec5 - z_rec_target))

    def discriminator1(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        s = image.get_shape()[1].value
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=10, reuse = reuse, lin_name = 'd_minilin1')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        ddim = self.options['df_dim']
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h0_conv'), name = 'd_pre1') #32
        h1 = ops.lrelu(self.bn1f(ops.conv2d(h0, ddim*2, name = 'd_h1_conv')), name = 'd_pre2') #16
        h2 = ops.lrelu(self.bn2f(ops.conv2d(h1, ddim*4, name = 'd_h2_conv')), name = 'd_pre3') #8
        h3 = ops.lrelu(self.bn3f(ops.conv2d(h2, ddim*8, name = 'd_h3_conv')), name = 'd_pre4') #4
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'],#self.options['t_dim']/64
                             'd_embedding'), name = 'd_pre5')
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
        
        h3_concat = concat( 3, [h3, tiled_embeddings], name='h3_concat')
        h3_new = ops.lrelu(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'), name = 'd_pre6') #4
        
        h4 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
        
        h5 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin_2')
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5)
    
    def discriminator2(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        nm = '2'
        s = image.get_shape()[1].value
        ddim = self.options['df_dim']*2
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=10, reuse = reuse, lin_name = 'd_minilin2')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h02_conv' + nm)) #32
        h1 = ops.lrelu(self.bn1f(ops.conv2d(h0, ddim*2, name = 'd_h12_conv' + nm))) #16
        h2 = ops.lrelu(self.bn2f(ops.conv2d(h1, ddim*4, name = 'd_h22_conv' + nm))) #8
        h3 = ops.lrelu(self.bn3f(ops.conv2d(h2, ddim*8, name = 'd_h32_conv' + nm))) #4
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'],'d_embedding'))
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings' + nm)
        
        h3_concat = concat( 3, [h3, tiled_embeddings], name='h3_concat2' + nm)
        h3_new = ops.lrelu(ops.conv2d(h3_concat, ddim*8, 1,1,1,1, name = 'd_h3_conv_new' + nm)) #4
        
        h4 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin2' + nm)
        
        h5 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin_2' + nm)
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5)
    
    def discriminator3(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        nm = '3'
        s = image.get_shape()[1].value
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=10, reuse = reuse, lin_name = 'd_minilin3')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        ddim = self.options['df_dim']*3
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h02_conv' + nm)) #32
        h1 = ops.lrelu(self.bn1f(ops.conv2d(h0, ddim*2, name = 'd_h12_conv' + nm))) #16
        h2 = ops.lrelu(self.bn2f(ops.conv2d(h1, ddim*4, name = 'd_h22_conv' + nm))) #8
        h3 = ops.lrelu(self.bn3f(ops.conv2d(h2, ddim*8, name = 'd_h32_conv' + nm))) #4
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'],'d_embedding'))
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings' + nm)
        
        h3_concat = concat( 3, [h3, tiled_embeddings], name='h3_concat2' + nm)
        h3_new = ops.lrelu(ops.conv2d(h3_concat, ddim*8, 1,1,1,1, name = 'd_h3_conv_new' + nm)) #4
        
        h4 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin2' + nm)
        
        h5 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin_2' + nm)
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5)
    
    def discriminator4(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        nm = '4'
        s = image.get_shape()[1].value
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=10, reuse = reuse, lin_name = 'd_minilin4')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        ddim = self.options['df_dim']*4
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h02_conv' + nm)) #32
        h1 = ops.lrelu(self.bn1f(ops.conv2d(h0, ddim*2, name = 'd_h12_conv' + nm))) #16
        h2 = ops.lrelu(self.bn2f(ops.conv2d(h1, ddim*4, name = 'd_h22_conv' + nm))) #8
        h3 = ops.lrelu(self.bn3f(ops.conv2d(h2, ddim*8, name = 'd_h32_conv' + nm))) #4
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'],'d_embedding'))
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings' + nm)
        
        h3_concat = concat( 3, [h3, tiled_embeddings], name='h3_concat2' + nm)
        h3_new = ops.lrelu(ops.conv2d(h3_concat, ddim*8, 1,1,1,1, name = 'd_h3_conv_new' + nm)) #4
        
        h4 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin2' + nm)
        
        h5 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin_2' + nm)
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5)
    
    def discriminator5(self, image, t_text_embedding, reuse=False):
        if reuse:
            if not prince:
                tf.get_variable_scope().reuse_variables()
        elif prince:
            tf.get_variable_scope()._reuse = None
        nm = '5'
        s = image.get_shape()[1].value
        mini_information = self.minibatch_discriminate(reshape(
            image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=10, reuse = reuse, lin_name = 'd_minilin5')*10
        t_text_embedding = concat(1, [t_text_embedding, mini_information])
        ddim = self.options['df_dim']*5
        h0 = ops.lrelu(ops.conv2d(image, ddim, name = 'd_h02_conv' + nm)) #32
        h1 = ops.lrelu(self.bn1f(ops.conv2d(h0, ddim*2, name = 'd_h12_conv' + nm))) #16
        h2 = ops.lrelu(self.bn2f(ops.conv2d(h1, ddim*4, name = 'd_h22_conv' + nm))) #8
        h3 = ops.lrelu(self.bn3f(ops.conv2d(h2, ddim*8, name = 'd_h32_conv' + nm))) #4
        
        # ADD TEXT EMBEDDING TO THE NETWORK
        reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'],'d_embedding'))
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings' + nm)
        
        h3_concat = concat( 3, [h3, tiled_embeddings], name='h3_concat2' + nm)
        h3_new = ops.lrelu(ops.conv2d(h3_concat, ddim*8, 1,1,1,1, name = 'd_h3_conv_new' + nm)) #4
                
        h4 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin2' + nm)

        h5 = ops.linear(reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin_2' + nm)
        
        return h4, h5, tf.nn.sigmoid(h4), tf.nn.sigmoid(h5)
    
    