import tensorflow as tf
from resnet import resnet
from cfg import cfg
from tqdm import tqdm
from input import get_data

model=resnet(*get_data())
loss=model.get_loss()

test_model=resnet(*get_data('test'),False)
accuracy=test_model.get_accuracy()

global_step=tf.Variable(0,trainable=False,name='global_step')
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss,global_step=global_step)
saver=tf.train.Saver(max_to_keep=1)
saver_max=tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    acc_max=0
    saver.restore(sess,'ckpt_best/0.9299')
    while 1:
        for i in tqdm(range(0,50000,cfg.batch_size),ncols=70):
            _,step=sess.run([train_step,global_step])
        saver.save(sess,cfg.dir_save+'model',global_step=step)
        acc=0
        for i in range(0,10000,100):
            acc+=accuracy.eval()
        acc=acc/10000 
        if acc>=acc_max:
            acc_max=acc
            saver_max.save(sess,cfg.dir_best+str(acc_max))
        print(step,'_准确率: ',acc ,  '最高准确率:',acc_max)
