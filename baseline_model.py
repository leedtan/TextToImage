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
		
		
		t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
		
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
		
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		
		fake_image = self.generator(t_z, t_real_caption)
		with tf.variable_scope('scope_full'):
			disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_real_caption)
			disc_wrong_image, disc_wrong_image_logits   = self.discriminator(t_wrong_image, t_real_caption, reuse = True)
			disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_real_caption, reuse = True)
		g_loss_full = tf.reduce_mean(cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image)))
		
		d_loss1 = tf.reduce_mean(cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image)))
		d_loss2 = tf.reduce_mean(cross_entropy(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image)))
		d_loss3 = tf.reduce_mean(cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image)))

		
		d_loss_full = d_loss3 + d_loss2 + d_loss1
		
		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]
		
		d_loss = d_loss1 + d_loss2 + d_loss3
		
		g_loss = g_loss_full
		

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z,
		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss,
		}

		outputs = {
			'generator' : fake_image,
		}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3' : d_loss3,
			'd_loss' : d_loss_full,
			'g_loss' : g_loss_full,
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
		
		reduced_text_embedding = ops.parametric_relu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = ops.parametric_relu(self.g_bn0(h0, train = False))
		
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = ops.parametric_relu(self.g_bn1(h1, train = False))
		
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = ops.parametric_relu(self.g_bn2(h2, train = False))
		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = ops.parametric_relu(self.g_bn3(h3, train = False))
		
		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		
		return (tf.tanh(h4)/2. + 0.5)

	def g_name(self):
		self.g_idx += 1
		return 'g_' + str(self.g_idx)
	
	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding):
		
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		reduced_text_embedding = ops.parametric_relu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') , name = 'g_pre1')
		z_concat = concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = ops.parametric_relu(self.g_bn0(h0) , name = 'g_pre2')
		
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = ops.parametric_relu(self.g_bn1(h1), name = 'g_pre3')

		
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = ops.parametric_relu(self.g_bn2(h2), name = 'g_pre4')

		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*2], name='g_h3')
		h3 = ops.parametric_relu(self.g_bn3(h3), name = 'g_pre5')

		#maybe project straight to 3d.
		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, self.options['gf_dim']*1], name='g_h4')
		h4 = ops.parametric_relu(self.g_bn4(h4), name = 'g_pre10')
		
		h4 = ops.deconv2d(h4,[self.options['batch_size'], s, s, 3], k_h=5, k_w=5, d_h=1, d_w=1, name='g_h6_out')
		

		return (tf.tanh(h4)/2. + 0.5)

	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			if not prince:
				tf.get_variable_scope().reuse_variables()
		elif prince:
			tf.get_variable_scope()._reuse = None
		h0 = ops.parametric_relu(ops.conv2d(ops.noised(image, .5), self.options['df_dim'], name = 'd_h0_conv'), name = 'd_pre1') #32
		'''
		h1 = ops.parametric_relu(self.bn1f(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv')), name = 'd_pre2') #16
		h2 = ops.parametric_relu(self.bn2f(ops.noised(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))), name = 'd_pre3') #8
		h3 = ops.parametric_relu( self.bn3f(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv')), name = 'd_pre4') #4
		'''
		image_resized = tf.reshape(h0, [self.options['batch_size'], -1])
		image_text =  concat(1, [image_resized, t_text_embedding], name = 'd_image_text_concat')
		
		d_hidden = ops.parametric_relu(self.bn4f(ops.linear(image_text, 10, 'd_hidden_1')), name = 'd_pre3134')
		d_hidden2 = ops.linear(d_hidden, 1, 'd_hidden_2')
		
		return tf.nn.sigmoid(d_hidden2), d_hidden2
		'''
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.parametric_relu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'), name = 'd_pre5')
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,64,64,1], name='tiled_embeddings')
		
		h3_concat = concat( 3, [h0, tiled_embeddings], name='h3_concat')
		h3_new = ops.parametric_relu( self.bn4f(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new')), name = 'd_pre6') #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		
		return tf.nn.sigmoid(h4), h4
		'''
	
	
	
	