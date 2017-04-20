import tensorflow as tf
from Utils import ops

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

		self.d_bn1 = ops.batch_norm(name='d_bn1')
		self.d_bn2 = ops.batch_norm(name='d_bn2')
		self.d_bn3 = ops.batch_norm(name='d_bn3')
		self.d_bn4 = ops.batch_norm(name='d_bn4')
		self.d_bn4_small = ops.batch_norm(name='d_bn4_small')
		self.d_bn4_mid = ops.batch_norm(name='d_bn4_mid')


	def build_model(self):
		img_size = self.options['image_size']
		
		l2reg = tf.placeholder('float32', shape=())
		
		t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
		
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
		
		real_small_image = tf.image.resize_images(t_real_image, img_size/4,img_size/4)
		wrong_small_image = tf.image.resize_images(t_wrong_image, img_size/4,img_size/4)
		
		real_mid_image = tf.image.resize_images(t_real_image, img_size/2,img_size/2)
		wrong_mid_image = tf.image.resize_images(t_wrong_image, img_size/2,img_size/2)
		
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		
		fake_image, fake_small_image, fake_mid_image = self.generator(t_z, t_real_caption)
		with tf.variable_scope('scope'):
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
		
		
		g_loss_full = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.ones_like(disc_fake_image)))
		
		d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real_image_logits, tf.ones_like(disc_real_image)))
		d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_wrong_image_logits, tf.zeros_like(disc_wrong_image)))
		d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_image_logits, tf.zeros_like(disc_fake_image)))

		
		
		g_loss_small = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_small_fake_image_logits, tf.ones_like(disc_small_fake_image)))
		
		d_loss1_small = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_small_real_image_logits, tf.ones_like(disc_small_real_image)))
		d_loss2_small = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_small_wrong_image_logits, tf.zeros_like(disc_small_wrong_image)))
		d_loss3_small = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_small_fake_image_logits, tf.zeros_like(disc_small_fake_image)))
		
		
		g_loss_mid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_mid_fake_image_logits, tf.ones_like(disc_mid_fake_image)))

		d_loss1_mid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_mid_real_image_logits, tf.ones_like(disc_mid_real_image)))
		d_loss2_mid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_mid_wrong_image_logits, tf.zeros_like(disc_mid_wrong_image)))
		d_loss3_mid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_mid_fake_image_logits, tf.zeros_like(disc_mid_fake_image)))

		d_loss_full = d_loss3 + d_loss2 + d_loss1
		d_loss_mid = d_loss3_mid + d_loss2_mid + d_loss1_mid
		d_loss_small = d_loss3_small + d_loss2_small + d_loss1_small
		d_loss_small_full = d_loss_full + d_loss_small
		
		g_loss_small_mid = g_loss_small + g_loss_mid

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]
		
		d_l2reg = tf.reduce_mean(tf.pack([tf.reduce_mean(tf.square(var) + tf.pow(var,4)/100.) for var in d_vars]))
		g_l2reg = tf.reduce_mean(tf.pack([tf.reduce_mean(tf.square(var) + tf.pow(var,4)/100.) for var in g_vars]))
		
		d_loss = d_loss1 + d_loss2 + d_loss3 + d_loss1_small  + d_loss2_small + d_loss3_small + \
			d_loss1_mid  + d_loss2_mid + d_loss3_mid# + d_l2reg*l2reg
		'''
		d_loss_without_small = d_loss1 + d_loss2 + d_loss3 + \
			d_loss1_mid  + d_loss2_mid + d_loss3_mid
		d_loss_without_small_mid = d_loss1 + d_loss2 + d_loss3
		'''
		g_loss = g_loss_full + g_loss_small + g_loss_mid# + g_l2reg*l2reg
		

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
		
		reduced_text_embedding = ops.parametric_relu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
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

	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding):
		
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		reduced_text_embedding = ops.parametric_relu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = ops.parametric_relu(self.g_bn0(h0))
		
		h1 = ops.noised(ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1'))
		h1 = ops.parametric_relu(self.g_bn1(h1))
		
		
		h1res = ops.deconv2d(h1, [self.options['batch_size'], s8, s8, 1],d_h=1, d_w=1, name='g_h1res')
		h1res = ops.parametric_relu(self.g_bn1h(h1res))
		h1resflat = tf.reshape(h1res, [-1, s8*s8])
		
		res1_in = tf.concat( 1, [h1resflat, z_concat], name='h1_res_block')
		res1_layers = 100
		res1block = ops.parametric_relu(ops.noised(ops.linear(res1_in,  s4/2* s4/2* res1_layers,'g_res1block')))
		res1block = tf.reshape(res1block, [-1, s4/2, s4/2, res1_layers])
		res1block_applied = ops.deconv2d(res1block, [self.options['batch_size'], s4, s4,self.options['gf_dim']*2], name='g_res1applied')
		
		
		h2 = ops.noised(ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2'))
		h2 = ops.parametric_relu(self.g_bn2(h2 + res1block_applied))
		
		
		h2res = ops.conv2d(h2, [self.options['batch_size'], s8, s8, 20],d_h=1, d_w=1, name='g_h2res')
		h2res = ops.parametric_relu(self.g_bn2h(h2res))
		h2resflat = tf.reshape(h2res, [-1, s8*s8*20])
		
		res2_in = tf.concat( 1, [h2resflat, z_concat], name='h2_res_block')
		res2_layers = 50
		res2block = ops.parametric_relu(ops.noised(ops.linear(res2_in,  s2/2* s2/2* res2_layers,'g_res2block')))
		res2block = tf.reshape(res2block, [-1, s2/2, s2/2, res2_layers])
		res2block_applied = ops.deconv2d(res2block, [self.options['batch_size'], s2, s2,self.options['gf_dim']*1], name='g_res2applied')
		
		#h3 = h2 + ops.parametric_relu(self.g_bnresblock2(res2block_applied))
		#h3 = ops.parametric_relu(self.g_bn3(ops.noised(ops.deconv2d(
		#	h3, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3'))))
		
		h3 = ops.noised(ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3'))
		h3 = ops.parametric_relu(self.g_bn3(h3 + res2block_applied))
		
		h4 = ops.noised(ops.deconv2d(h3, [self.options['batch_size'], s, s, self.options['gf_dim']*1], name='g_h4'))
		h4 = ops.parametric_relu(self.g_bn4(h4))
				
		h5 = ops.deconv2d(h4, [self.options['batch_size'], s, s, 3], k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5')

		small_gen_hidden = ops.parametric_relu(self.g_bn_small(ops.noised(ops.deconv2d(
			h1, [self.options['batch_size'], s4, s4, 100], name='g_small_gen_hidden'))))
		mid_gen_hidden = ops.parametric_relu(self.g_bn_mid(ops.noised(ops.deconv2d(
			h2, [self.options['batch_size'], s2, s2, 100], name='g_mid_gen_hidden'))))
		
		small_generated = ops.deconv2d(small_gen_hidden, [self.options['batch_size'], s4, s4, 3], d_h=1, d_w=1, name='g_small_generated')
		mid_generated = ops.deconv2d(mid_gen_hidden, [self.options['batch_size'], s2, s2, 3], d_h=1, d_w=1, name='g_mid_generated')
		
		return (tf.tanh(h5)/2. + 0.5), (tf.tanh(small_generated)/2. + 0.5), (tf.tanh(mid_generated)/2. + 0.5)

	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		h0 = ops.parametric_relu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32
		h1 = ops.parametric_relu( self.d_bn1(ops.noised(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv')))) #16
		h2 = ops.parametric_relu( self.d_bn2(ops.noised(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv')))) #8
		h3 = ops.parametric_relu( self.d_bn3(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv'))) #4
		
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.parametric_relu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		
		h3_concat = tf.concat( 3, [h3, tiled_embeddings], name='h3_concat')
		h3_new = ops.parametric_relu( self.d_bn4(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		
		return tf.nn.sigmoid(h4), h4
	
	
	
	
	def discriminator_small(self, image, t_text_embedding, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		h0 = ops.parametric_relu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32
		#h1 = ops.parametric_relu( self.d_bn1(ops.conv2d(h0, self.options['df_dim']*2,name = 'd_h1_conv'))) #16
		h1 = ops.parametric_relu( self.d_bn1(ops.conv2d(h0, self.options['df_dim'], d_h=1, d_w=1, name = 'd_h1_conv'))) #16
		h2 = ops.parametric_relu( self.d_bn2(ops.conv2d(h1, self.options['df_dim']*2, name = 'd_h2_conv'))) #8
		#h3 = ops.parametric_relu( self.d_bn3(ops.conv2d(h2, self.options['df_dim']*2, name = 'd_h3_conv'))) #4
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.parametric_relu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		
		h3_concat = tf.concat( 3, [h2, tiled_embeddings], name='h3_concat')
		h3_new = ops.parametric_relu( self.d_bn4_small(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		
		return tf.nn.sigmoid(h4), h4
	
	
	
	
	def discriminator_mid(self, image, t_text_embedding, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		h0 = ops.parametric_relu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32
		h1 = ops.parametric_relu( self.d_bn1(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv'))) #16
		#h2 = ops.parametric_relu( self.d_bn2(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))) #8
		h2 = ops.parametric_relu( self.d_bn2(ops.conv2d(h1, self.options['df_dim']*2, d_h=1, d_w=1, name = 'd_h2_conv'))) #8
		h3 = ops.parametric_relu( self.d_bn3(ops.conv2d(h2, self.options['df_dim']*4, name = 'd_h3_conv'))) #4
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.parametric_relu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		
		h3_concat = tf.concat( 3, [h3, tiled_embeddings], name='h3_concat')
		h3_new = ops.parametric_relu( self.d_bn4_mid(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
		
		h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		
		return tf.nn.sigmoid(h4), h4
	