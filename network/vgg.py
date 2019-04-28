from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VggNet(object):
	"""docstring for VggNet"""
	def __init__(self, vggname,is_training,keep_prob = 0.5,num_classes=10, group=1, scale=1.):
		super(VggNet, self).__init__()
		self.vggname = vggname 
		self.num_classes = num_classes

		self.regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)
		#self.initializer = tf.contrib.layers.xavier_initializer()
		self.initializer = tf.contrib.layers.variance_scaling_initializer()

		self.pool_num = 0
		self.conv_num = 0
		self.is_training = is_training

		self.keep_prob = keep_prob

                self.group = group
                self.scale = scale
		
	def forward(self,input):
                hidden = int(4096*self.scale)

		out = self.make_layer(input,cfg[self.vggname])
		out = tf.layers.flatten(out,name='flatten')
                
                out = tf.layers.dense(out,units=hidden,activation=tf.nn.relu,
                        kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='fc_1')
                out = tf.layers.dropout(out,rate=self.keep_prob,name='dropout1')

                out = tf.layers.dense(out,units=hidden,activation=tf.nn.relu,
                        kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='fc_2')
                out = tf.layers.dropout(out,rate=self.keep_prob,name='dropout2')

		predicts = tf.layers.dense(out,units=self.num_classes,
                        kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='fc_3')
		softmax_out = tf.nn.softmax(predicts,name='output')
		return predicts,softmax_out


	def conv2d(self,inputs,out_channel):
                conv_layers = []
                channels = inputs.get_shape()[3].value

                if channels > 3:
                    group = self.group
                else:
                    group = 1

                in_sz = int(channels / group)
                out_sz = int(out_channel * self.scale / group)

                for g in range(group):
                    output = tf.layers.conv2d(inputs[:,:,:,in_sz*g:in_sz*(g+1)],filters=out_sz,kernel_size=3,padding='same',
                                            kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='conv_'+str(self.conv_num)+'_g'+str(g))
                    #inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn_'+str(self.conv_num))
                    conv_layers.append(output)
                conv_concat = tf.concat(conv_layers, axis=-1)
		self.conv_num+=1
		return tf.nn.relu(conv_concat)

	def make_layer(self,inputs,netparam):
		for param in netparam:
			if param=='M':
				inputs = tf.layers.max_pooling2d(inputs,pool_size=2,strides=2,padding='same',name='pool_'+str(self.pool_num))
				self.pool_num+=1
			else:
				inputs = self.conv2d(inputs,param)
		inputs = tf.layers.average_pooling2d(inputs,pool_size=1,strides=1)
		return inputs

	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		losses+=tf.add_n(l2_reg)
		return losses


def vgg11(is_training=True,keep_prob=0.5):
	net = VggNet(vggname='VGG11',is_training=is_training,keep_prob=keep_prob)
	return net 


def vgg13(is_training=True,keep_prob=0.5):
	net = VggNet(vggname='VGG13',is_training=is_training,keep_prob=keep_prob)
	return net 


def vgg16(is_training=True,keep_prob=0.5,group=1,scale=1.):
	net = VggNet(vggname='VGG16',is_training=is_training,keep_prob=keep_prob,group=group,scale=scale)
	return net 


def vgg19(is_training=True,keep_prob=0.5):
	net = VggNet(vggname='VGG19',is_training=is_training,keep_prob=keep_prob)
	return net 


if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = vgg16()
		data = np.random.randn(64,32,32,3)
		inputs = tf.placeholder(tf.float32,[64,32,32,3])
		predicts,softmax_out = net.forward(inputs)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		init = tf.global_variables_initializer()
		sess = tf.Session(config=config)
		sess.run(init)
		output = sess.run(predicts,feed_dict={inputs:data})
		print(output.shape)
		sess.close()
