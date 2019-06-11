import tensorflow as tf
class resnet:
    def __init__(self,image,label,train=True):
        self.image,self.label,self.train=image/255,label,train
        self.init=tf.variance_scaling_initializer
        self.regular=tf.contrib.layers.l2_regularizer(1e-4)
        self.bn_momentum=0.9
        with tf.variable_scope('resnet',reuse=not train):
            
            out = self._conv(self.image,16,3,name='blocka1')

            out = self.basicblock(out,16,name='blockb1')
            out = self.basicblock(out,16,name='blockb2')
            out = self.basicblock(out,16,name='blockb3')
            out = self.basicblock(out,16,name='blockb4')
            out = self.basicblock(out,16,name='blockb5')
            out = self.basicblock(out,16,name='blockb6')
            out = self.basicblock(out,16,name='blockb7')
            out = self.basicblock(out,16,name='blockb8')
            out = self.basicblock(out,16,name='blockb9')

            out = self.basicblock(out,32,init_stride=2,name='blockc1')
            out = self.basicblock(out,32,name='blockc2')
            out = self.basicblock(out,32,name='blockc3')
            out = self.basicblock(out,32,name='blockc4')
            out = self.basicblock(out,32,name='blockc5')
            out = self.basicblock(out,32,name='blockc6')
            out = self.basicblock(out,32,name='blockc7')
            out = self.basicblock(out,32,name='blockc8')
            out = self.basicblock(out,32,name='blockc9')

            out = self.basicblock(out,64,init_stride=2,name='blockd1')
            out = self.basicblock(out,64,name='blockd2')
            out = self.basicblock(out,64,name='blockd3')
            out = self.basicblock(out,64,name='blockd4')
            out = self.basicblock(out,64,name='blockd5')
            out = self.basicblock(out,64,name='blockd6')
            out = self.basicblock(out,64,name='blockd7')
            out = self.basicblock(out,64,name='blockd8')
            out = self.basicblock(out,64,name='blockd9')

            out = self._bn_relu(out)
            out = self._pool(out,8,8,mode='avg')

            out = self._fc(tf.layers.flatten(out),10,name='fc')
            self.logits = out

    def get_loss(self):
        self.loss=tf.losses.sparse_softmax_cross_entropy(self.label,self.logits)+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return self.loss
    def get_accuracy(self):
        self.scaled_logits=tf.nn.softmax(self.logits)
        self.pred=tf.math.argmax(self.scaled_logits,axis=-1,output_type=tf.int64)
        self.accuracy=tf.reduce_sum(tf.cast(tf.equal(self.label,self.pred),dtype=tf.int32))
        return self.accuracy
    def basicblock(self,bottom,filters,init_stride=1,name='basic_block'):
        with tf.variable_scope(name):
            a1=self._bn_relu_conv(bottom,filters,3,init_stride,name='bn_relu_conv1')
            a2=self._bn_relu_conv(a1,filters,3,name='bn_relu_conv2')
            return self.shortcut(bottom,a2)
    def shortcut(self,bottom,residual,name='shortcut'):
        filters=residual.shape[-1]
        stride=bottom.shape[1]//residual.shape[1]
        with tf.variable_scope(name):
            if bottom.shape==residual.shape:
                shortcut=bottom
            else:
                shortcut=self._conv(bottom,filters,1,stride=stride.value)
            return shortcut+residual
    def _bn_relu_conv(self,bottom,filters,k,stride=1,name='bn_relu_conv'):
        with tf.variable_scope(name):
            bn_relu=self._bn_relu(bottom)
            conv=self._conv(bn_relu,filters,k,stride=stride)
            return conv    
    def _bn_relu(self,bottom,name='bn_relu'):
        with tf.variable_scope(name):
            norm=tf.layers.batch_normalization(bottom,training=self.train,momentum=self.bn_momentum,epsilon=1e-5)
            return tf.nn.relu(norm)

    
    def _conv(self,bottom,filters,k,stride=1,padding='same',activation=None,name=None):
        return tf.layers.conv2d(bottom,filters,k,stride,padding=padding,activation=activation,name=name,kernel_initializer=self.init,kernel_regularizer=self.regular)
    def _fc(self,bottom,filters,activation=None,name=None):
        return tf.layers.dense(bottom,filters,activation=activation,name=name,kernel_initializer=self.init,kernel_regularizer=self.regular)
    def _pool(self,bottom,k,stride,mode='max'):
        if mode=='max':
            return tf.nn.max_pool(bottom,[1,k,k,1],[1,stride,stride,1],padding='VALID')
        if mode=='avg':
            return tf.nn.avg_pool(bottom,[1,k,k,1],[1,stride,stride,1],padding='VALID')