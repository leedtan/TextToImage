import tensorflow as tf
from Utils import ops


prince = True


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
		
	def build_model(self):
		img_size = self.options['image_size']
		
		l2reg = tf.placeholder('float32', shape=())
		
		t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
		
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
		if prince:
			real_small_image = tf.image.resize_images(t_real_image, (img_size/4,img_size/4))
			wrong_small_image = tf.image.resize_images(t_wrong_image, (img_size/4,img_size/4))
			real_mid_image = tf.image.resize_images(t_real_image, (img_size/2,img_size/2))
			wrong_mid_image = tf.image.resize_images(t_wrong_image, (img_size/2,img_size/2))
		else:	
			real_small_image = tf.image.resize_images(t_real_image, img_size/4,img_size/4)
			wrong_small_image = tf.image.resize_images(t_wrong_image, img_size/4,img_size/4)
			real_mid_image = tf.image.resize_images(t_real_image, img_size/2,img_size/2)
			wrong_mid_image = tf.image.resize_images(t_wrong_image, img_size/2,img_size/2)
		
		
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		
		fake_image, fake_small_image, fake_mid_image = self.generator(t_z, t_real_caption)
		with tf.variable_scope('scope_full'):
			disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_real_caption)
			disc_wrong_image, disc_wrong_image_logits   = self.discriminator(t_wrong_image, t_real_caption, reuse = True)
			disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_real_caption, reuse = True)
		with tf.variable_scope('scope_small'):
			disc_small_real_image, disc_small_real_image_logits   = self.discriminator_small(real_small_image, t_real_caption)
			disc_small_wrong_image, disc_small_wrong_image_logits   = self.discriminator_small(wrong_small_image, t_real_caption, reuse = True)
			disc_small_fake_image, disc_small_fake_image_logits   = self.discriminator_small(fake_small_image, t_real_caption, reuse = True)
		with tf.variable_scope('scope_mid'):
			disc_mid_real_image, disc_mid_real_image_logits   = self.discriminator_mid(real_mid_image, t_real_caption)
			disc_mid_wrong_image, disc_mid_wrong_image_logits   = self.discriminator_mid(wrong_mid_image, t_real_caption, reuse = True)
			disc_mid_fake_image, disc_mid_fake_image_logits   = self.discriminator_mid(fake_mid_image, t_real_caption, reuse = True)
		
		
		g_loss_full = tf.reduce_mean(cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image)))
		
		d_loss1 = tf.reduce_mean(cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image)))
		d_loss2 = tf.reduce_mean(cross_entropy(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image)))
		d_loss3 = tf.reduce_mean(cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image)))
		
		
		g_loss_small = tf.reduce_mean(cross_entropy(disc_small_fake_image_logits, tf.ones_like(disc_small_fake_image)))
		
		d_loss1_small = tf.reduce_mean(cross_entropy(disc_small_real_image_logits, tf.ones_like(disc_small_real_image)))
		d_loss2_small = tf.reduce_mean(cross_entropy(disc_small_wrong_image_logits, tf.zeros_like(disc_small_wrong_image)))
		d_loss3_small = tf.reduce_mean(cross_entropy(disc_small_fake_image_logits, tf.zeros_like(disc_small_fake_image)))
		
		
		g_loss_mid = tf.reduce_mean(cross_entropy(disc_mid_fake_image_logits, tf.ones_like(disc_mid_fake_image)))

		d_loss1_mid = tf.reduce_mean(cross_entropy(disc_mid_real_image_logits, tf.ones_like(disc_mid_real_image)))
		d_loss2_mid = tf.reduce_mean(cross_entropy(disc_mid_wrong_image_logits, tf.zeros_like(disc_mid_wrong_image)))
		d_loss3_mid = tf.reduce_mean(cross_entropy(disc_mid_fake_image_logits, tf.zeros_like(disc_mid_fake_image)))
		
		
		'''
		g_loss_full = 1-tf.reduce_mean(disc_fake_image)
		
		d_loss1 = 1-tf.reduce_mean(disc_real_image)
		d_loss2 = tf.reduce_mean(disc_wrong_image)
		d_loss3 = tf.reduce_mean(disc_fake_image)

		
		
		g_loss_small = 1-tf.reduce_mean(disc_small_fake_image)
		
		d_loss1_small = 1-tf.reduce_mean(disc_small_real_image)
		d_loss2_small = tf.reduce_mean(disc_small_wrong_image)
		d_loss3_small = tf.reduce_mean(disc_small_fake_image)
		
		
		g_loss_mid = 1-tf.reduce_mean(disc_mid_fake_image)

		d_loss1_mid = 1-tf.reduce_mean(disc_mid_real_image)
		d_loss2_mid = tf.reduce_mean(disc_mid_wrong_image)
		d_loss3_mid = tf.reduce_mean(disc_mid_fake_image)
		'''
		
		
		
		d_loss_full = d_loss3 + d_loss2 + d_loss1
		d_loss_mid = d_loss3_mid + d_loss2_mid + d_loss1_mid
		d_loss_small = d_loss3_small + d_loss2_small + d_loss1_small
		d_loss_small_full = d_loss_full + d_loss_small
		
		g_loss_small_mid = g_loss_small + g_loss_mid

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]
		
		d_l2reg = tf.reduce_mean(pack([tf.reduce_mean(tf.square(var) + tf.pow(var,4)/100.) for var in d_vars]))
		g_l2reg = tf.reduce_mean(pack([tf.reduce_mean(tf.square(var) + tf.pow(var,4)/100.) for var in g_vars]))
		
		d_loss = d_loss1 + d_loss2 + d_loss3 + d_loss1_small  + d_loss2_small + d_loss3_small + \
			d_loss1_mid  + d_loss2_mid + d_loss3_mid# + d_l2reg*l2reg
		'''
		d_loss_without_small = d_loss1 + d_loss2 + d_loss3 + \
			d_loss1_mid  + d_loss2_mid + d_loss3_mid
		d_loss_without_small_mid = d_loss1 + d_loss2 + d_loss3
		'''
		g_loss = g_loss_full + g_loss_small + g_loss_mid# + g_l2reg*l2reg
		
		self.wgan_clip_1eneg3 = [v.assign(tf.clip_by_value(v, -0.001, 0.001)) for v in d_vars]
		self.wgan_clip_1eneg2 = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
		self.wgan_clip_1eneg1 = [v.assign(tf.clip_by_value(v, -0.1, 0.1)) for v in d_vars]
		self.wgan_clip_1 = [v.assign(tf.clip_by_value(v, -1, 1)) for v in d_vars]

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z,
			'l2reg' : l2reg
		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss,
			'g_loss_full' : g_loss_full,
			'd_loss_full' : d_loss_full,
			'g_loss_mid' : g_loss_mid,
			'd_loss_mid' : d_loss_mid,
			'g_loss_small' : g_loss_small,
			'd_loss_small' : d_loss_small,
			'g_loss_small_mid' : g_loss_small_mid,
			'd_loss_small_full' : d_loss_small_full
		}

		outputs = {
			'generator' : fake_image,
			'generator_small_image' : fake_small_image,
			'real_small_image' : real_small_image,
			'generator_mid_image' : fake_mid_image,
			'real_mid_image' : real_mid_image
		}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'd_loss_full' : d_loss_full,
			'd_loss_mid' : d_loss_mid,
			'd_loss_small' : d_loss_small,
			'g_loss_full' : g_loss_full,
			'g_loss_mid' : g_loss_mid,
			'g_loss_small' : g_loss_small,
			'disc_real_image_logits' : disc_real_image_logits,
			'disc_wrong_image_logits' : disc_wrong_image,
			'disc_fake_image_logits' : disc_fake_image_logits
		}
		
		return input_tensors, variables, loss, outputs, checks

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
		
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = ops.lrelu(self.g_bn0(h0, train = False))
		
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = ops.lrelu(self.g_bn1(h1, train = False))
		
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = ops.lrelu(self.g_bn2(h2, train = False))
		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = ops.lrelu(self.g_bn3(h3, train = False))
		
		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		
		return (tf.tanh(h4)/2. + 0.5)

	def g_name(self):
		self.g_idx += 1
		return 'g_' + str(self.g_idx)
	def d_name(self):
		self.d_idx += 1
		return 'd_generatedname_' + str(self.d_idx)
	

	def minibatch_discriminate_full(self, inpt, num_kernels=5, kernel_dim=3, reuse = None):
		if reuse:
			if not prince:
				tf.get_variable_scope().reuse_variables()
		elif prince:
			tf.get_variable_scope()._reuse = None
		x = ops.linear(inpt, num_kernels * kernel_dim, 'd_minibatch_full')
		activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
		diffs = tf.expand_dims(activation, 3) - \
		    tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
		abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
		minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
		return minibatch_features
	def minibatch_discriminate_mid(self, inpt, num_kernels=5, kernel_dim=3, reuse = None):
		if reuse:
			if not prince:
				tf.get_variable_scope().reuse_variables()
		elif prince:
			tf.get_variable_scope()._reuse = None
		x = ops.linear(inpt, num_kernels * kernel_dim, 'd_minibatch_mid')
		activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
		diffs = tf.expand_dims(activation, 3) - \
		    tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
		abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
		minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
		return minibatch_features
	def minibatch_discriminate_small(self, inpt, num_kernels=5, kernel_dim=3, reuse = None):
		if reuse:
			if not prince:
				tf.get_variable_scope().reuse_variables()
		elif prince:
			tf.get_variable_scope()._reuse = None
		x = ops.linear(inpt, num_kernels * kernel_dim, 'd_minibatch_small')
		activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
		diffs = tf.expand_dims(activation, 3) - \
		    tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
		abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
		minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
		return minibatch_features
	#Residual Block
	def add_residual(self, prev_layer, z_concat, text_filters = 3, k_h = 5, k_w = 5, hidden_text_filters = 20):
		
		s = prev_layer.get_shape()[1].value
		filters = prev_layer.get_shape()[3].value
		
		bn0 = self.g_bn()
		bn1 = self.g_bn()
		bn2 = self.g_bn()
		bn3 = self.g_bn()
		
		if z_concat is not None:
			text_augment = tf.reshape(ops.lrelu(bn1(ops.noised(
				ops.linear(z_concat,  s/2* s/2* text_filters,self.g_name())))),[-1, s/2, s/2, text_filters])
			
			text_augment = ops.lrelu(bn0(ops.deconv2d(text_augment,
				[self.options['batch_size'], s, s, hidden_text_filters], name=self.g_name())))
		
			concatenated = concat(3, [text_augment, prev_layer], name=self.g_name())
		else:
			concatenated = prev_layer
			
		res_hidden = bn2(ops.noised(ops.deconv2d(concatenated, 
			[self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=self.g_name())))
		
		residual = ops.deconv2d(ops.lrelu(res_hidden, name = self.g_name()),
			[self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=self.g_name())
		
		next_layer = prev_layer + residual
		return ops.lrelu(next_layer, name=self.g_name())
	
	#Residual Block in lower dimension
	def add_ld_residual(self, prev_layer, z_concat, text_filters = 3, k_h = 5, k_w = 5, hidden_text_filters = 20):
		
		s = prev_layer.get_shape()[1].value
		filters = prev_layer.get_shape()[3].value

		bn0 = self.g_bn()
		bn1 = self.g_bn()
		bn2 = self.g_bn()
		bn3 = self.g_bn()
		hidden_layer = ops.conv2d(prev_layer, filters, name=self.g_name())
		if z_concat is not None:
			text_augment = tf.reshape(ops.lrelu(bn1(ops.noised(
				ops.linear(z_concat,  s/4* s/4* text_filters,self.g_name()))), name = self.g_name()),[-1, s/4, s/4, text_filters])
			
			text_augment = ops.lrelu(bn0(ops.deconv2d(text_augment,
				[self.options['batch_size'], s/2, s/2, hidden_text_filters], name=self.g_name())))
			
			concatenated = concat(3, [text_augment, hidden_layer], name=self.g_name())
		else:
			concatenated = hidden_layer
			
		res_hidden = bn2(ops.noised(ops.deconv2d(
			concatenated, [self.options['batch_size'], s/2, s/2, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=self.g_name())))
		
		residual = ops.deconv2d(ops.lrelu(res_hidden, name = self.g_name()),
			[self.options['batch_size'], s, s, filters],k_h=k_h, k_w=k_w, name=self.g_name())
		next_layer = prev_layer + residual
		return ops.lrelu(next_layer, name=self.g_name())
	
	
	
	#Residual Block Standard Version. Needed for very small images
	def add_residual_standard(self, prev_layer, z_concat, text_filters = 1, k_h = 5, k_w = 5):
		
		s = prev_layer.get_shape()[1].value
		filters = prev_layer.get_shape()[3].value
		bn1 = self.g_bn()
		bn2 = self.g_bn()
		bn3 = self.g_bn()
		if z_concat is not None:
			text_augment = tf.reshape(ops.lrelu(bn1(ops.noised(
				ops.linear(z_concat,  s* s* text_filters,self.g_name()))), name = self.g_name()),[-1, s, s, text_filters])
			concatenated = concat(3, [text_augment, prev_layer], name=self.g_name())
		else:
			concatenated = prev_layer
		res_hidden = bn2(ops.noised(ops.deconv2d(concatenated,
			[self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=self.g_name())))
		
		residual = ops.deconv2d(ops.lrelu(res_hidden, name = self.g_name()),
			[self.options['batch_size'], s, s, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=self.g_name())
		next_layer = prev_layer + residual
		return ops.lrelu(next_layer, name=self.g_name())
	
	#Residual Block Standard version in a reduced dimensionality space. Not currently used/needed.
	def add_ld_residual_standard(self, prev_layer, z_concat, text_filters = 1, k_h = 5, k_w = 5):
		
		s = prev_layer.get_shape()[1].value
		filters = prev_layer.get_shape()[3].value
		bn1 = self.g_bn()
		bn2 = self.g_bn()
		bn3 = self.g_bn()
		text_augment = tf.reshape(ops.lrelu(bn1(ops.noised(
			ops.linear(z_concat,  s/2* s/2* text_filters,self.g_name()))), name = self.g_name()),[-1, s/2, s/2, text_filters])
		
		hidden_layer = ops.conv2d(prev_layer, filters, name=self.g_name())
		
		res_hidden = bn2(ops.noised(ops.deconv2d(
			concat(3, [text_augment, hidden_layer], name=self.g_name()), 
			[self.options['batch_size'], s/2, s/2, filters], k_h=k_h, k_w=k_h, d_h=1, d_w=1, name=self.g_name())))
		
		residual = ops.deconv2d(ops.lrelu(res_hidden, name = self.g_name()),
			[self.options['batch_size'], s, s, filters],k_h=k_h, k_w=k_w, name=self.g_name())
		next_layer = prev_layer + residual
		return ops.lrelu(next_layer, name=self.g_name())
		#Update to maintain indexing
		
	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding):
		
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		bn1 = self.g_bn()
		bn2 = self.g_bn()
		bn3 = self.g_bn()
		bn4 = self.g_bn()
		bn5 = self.g_bn()
		bn6 = self.g_bn()
		bn7 = self.g_bn()
		bn8 = self.g_bn()
		bn9 = self.g_bn()
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') , name = 'g_pre1')
		z_concat = concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = ops.lrelu(self.g_bn0(h0) , name = 'g_pre2')
		
		h0 = self.add_residual_standard(h0, z_concat, k_h = 3, k_w = 3, text_filters=4)
		#h0 = self.add_residual_standard(h0, None, k_h = 3, k_w = 3, text_filters=4)
		#h1 = tf.image.resize_nearest_neighbor(h0, [s8, s8])
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4],name='g_h1')
		h1 = ops.lrelu(self.g_bn1(h1), name = 'g_pre43234')
		#h1 = ops.deconv2d(h1, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], d_h=1, d_w=1,name=self.g_name())
		#h1 = ops.lrelu(bn1(h1), name = self.g_name())
		#h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		
		h1 = self.add_residual(h1, z_concat)
		#h1 = self.add_residual(h1, None)
		
		#h2 = tf.image.resize_nearest_neighbor(h1, [s4, s4])
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2],name='g_h2')
		h2 = ops.lrelu(self.g_bn2(h2), name = 'g_pre432')
		#h2 = ops.deconv2d(h2, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], d_h=1, d_w=1,name=self.g_name())
		#h2 = ops.lrelu(bn2(h2), name = self.g_name())
		
		h2 = self.add_residual(h2, z_concat)
		#h2 = self.add_residual(h2, None)

		#h3 = tf.image.resize_nearest_neighbor(h2, [s2, s2])
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1],name='g_h3')
		h3 = ops.lrelu(self.g_bn3(h3), name = 'g_pre4534')
		#h3 = ops.deconv2d(h3, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], d_h=1, d_w=1,name=self.g_name())
		#h3 = ops.lrelu(bn3(h3), name = self.g_name())
		#h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*2], name='g_h3')
		
		h3 = self.add_residual(h3, z_concat)
		#h3 = self.add_residual(h3, None)

		#maybe project straight to 3d.
		#h4 = tf.image.resize_nearest_neighbor(h3, [s, s])
		#h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, self.options['gf_dim']*1], name='g_h4')
		#h4 = ops.lrelu(self.g_bn4(h4), name = 'g_pre10')
		#h4 = self.add_ld_residual(h4, z_concat)
		#h4 = self.add_ld_residual(h4, None)
		
		h4 = ops.deconv2d(h3,[self.options['batch_size'], s, s, 3], name='g_h6_out')
		#h4 = ops.lrelu(self.g_bn4(h4), name = self.g_name())
		
		#h4 = ops.deconv2d(h4, [self.options['batch_size'], s, s, 3], d_h=1, d_w=1,name=self.g_name())
		#h4 = ops.lrelu(bn4(h4), name = self.g_name())
		h4 = self.add_ld_residual(h4, z_concat, hidden_text_filters=2)
		#h4 = self.add_ld_residual(h4, None)
		#h4 = self.add_residual(h4, z_concat, hidden_text_filters=1)
		#h4 = self.add_residual(h4, None)
		
		small_gen_hidden = ops.lrelu(self.g_bn_small(ops.noised(ops.deconv2d(
			h1, [self.options['batch_size'], s4, s4, 100], name='g_small_gen_hidden'))), name = 'g_pre13')
		mid_gen_hidden = ops.lrelu(self.g_bn_mid(ops.noised(ops.deconv2d(
			h2, [self.options['batch_size'], s2, s2, 100], name='g_midgen_hidden'))), name = 'g_pre14')
		
		#small_gen_hidden = self.add_residual(small_gen_hidden, z_concat)
		#mid_gen_hidden = self.add_residual(mid_gen_hidden, z_concat)
		
		small_generated = ops.deconv2d(small_gen_hidden, [self.options['batch_size'], s4, s4, 3], d_h=1, d_w=1, name='g_small_generated')
		mid_generated = ops.deconv2d(mid_gen_hidden, [self.options['batch_size'], s2, s2, 3], d_h=1, d_w=1, name='g_midgenerated')
		
		#small_generated = self.add_residual(small_generated, None)
		#mid_generated = self.add_residual(mid_generated, None)
		
		return (tf.tanh(h4)/2. + 0.5), (tf.tanh(small_generated)/2. + 0.5), (tf.tanh(mid_generated)/2. + 0.5)

	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			if not prince:
				tf.get_variable_scope().reuse_variables()
		elif prince:
			tf.get_variable_scope()._reuse = None
		s = image.get_shape()[1].value
		mini_information = self.minibatch_discriminate_full(tf.reshape(
			image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=5, reuse = reuse)*10
		t_text_embedding = concat(1, [t_text_embedding, mini_information])
		h0 = ops.lrelu(ops.conv2d(ops.noised(image, .5), self.options['df_dim'], name = 'd_h0_conv'), name = 'd_pre1') #32
		h1 = ops.lrelu(self.bn1f(ops.noised(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv'))), name = 'd_pre2') #16
		h2 = ops.lrelu(self.bn2f(ops.noised(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))), name = 'd_pre3') #8
		h3 = ops.lrelu(self.bn3f(ops.noised(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv'))), name = 'd_pre4') #4
		
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'],#self.options['t_dim']/64
							 'd_embedding'), name = 'd_pre5')
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		
		h3_concat = concat( 3, [h3, tiled_embeddings], name='h3_concat')
		h3_new = ops.lrelu(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'), name = 'd_pre6') #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		
		return tf.nn.sigmoid(h4), h4
	
	
	
	
	def discriminator_small(self, image, t_text_embedding, reuse=False):
		if reuse:
			if not prince:
				tf.get_variable_scope().reuse_variables()
		elif prince:
			tf.get_variable_scope()._reuse = None
		s = image.get_shape()[1].value
		mini_information = self.minibatch_discriminate_full(tf.reshape(
			image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=5, reuse = reuse)*10
		t_text_embedding = concat(1, [t_text_embedding, mini_information])
		h0 = ops.lrelu(ops.conv2d(ops.noised(image, .5), self.options['df_dim'], name = 'd_h0_conv_s'), name = 'd_pre7') #32
		h1 = ops.lrelu(self.bn1s(ops.conv2d(h0, self.options['df_dim'], name = 'd_h1_conv_s')), name = 'd_pre8') #16
		#h2 = ops.lrelu(self.bn2s(ops.conv2d(h1, self.options['df_dim']*2, name = 'd_h2_conv_s')), name = 'd_pre9') #8
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim']/64, 'd_embeddingsmall'), name = 'd_pre10')
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='d_tiled_embeddings_s')
		
		h3_concat = concat( 3, [h1, tiled_embeddings], name='h3_concat')
		h3_new = ops.lrelu(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'), name = 'd_pre11') #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin_s')
		
		return tf.nn.sigmoid(h4), h4
	
	
	
	
	def discriminator_mid(self, image, t_text_embedding, reuse=False):
		if reuse:
			if not prince:
				tf.get_variable_scope().reuse_variables()
		elif prince:
			tf.get_variable_scope()._reuse = None
		s = image.get_shape()[1].value
		mini_information = self.minibatch_discriminate_full(tf.reshape(
			image, [self.options['batch_size'], -1]), num_kernels=10, kernel_dim=5, reuse = reuse)*10
		t_text_embedding = concat(1, [t_text_embedding, mini_information])
		h0 = ops.lrelu(ops.conv2d(ops.noised(image, .5), self.options['df_dim'], name = 'd_h0_conv_m'), name = 'd_pre12') #32
		h1 = ops.lrelu(self.bn1m(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv_m')), name = 'd_pre13') #16
		h2 = ops.lrelu(self.bn2m(ops.conv2d(h1, self.options['df_dim']*2, name = 'd_h2_conv_m')), name = 'd_pre14') #8
		#h3 = ops.lrelu( self.bn3m(ops.conv2d(h2, self.options['df_dim']*4, name = 'd_h3_conv')), name = 'd_pre15') #4
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim']/64, 'd_embeddingmid'), name = 'd_pre16')
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		
		h3_concat = concat(3, [h2, tiled_embeddings], name='h3_concat_m')
		h3_new = ops.lrelu(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new_m'), name = 'd_pre17') #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin_m')
		
		return tf.nn.sigmoid(h4), h4
	