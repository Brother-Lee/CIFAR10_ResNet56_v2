import tensorflow as tf
from cfg import cfg
def get_data(mode='train'):
    if mode=='train':
        classified_dataset=tf.data.TFRecordDataset(cfg.dir_train)
        dataset=classified_dataset.map(parser).map(preprocess).shuffle(1000).batch(cfg.batch_size,drop_remainder=True).repeat(-1)
    if mode=='test':
        test_dataset=tf.data.TFRecordDataset(cfg.dir_test)
        dataset=test_dataset.map(parser).batch(100,drop_remainder=True).repeat(-1)
    iterator=dataset.make_one_shot_iterator()
    image,label=iterator.get_next()
    return image,label

def parser(record):
    example=tf.parse_single_example(record,features={
        'image':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    })
    image=tf.reshape(tf.decode_raw(example['image'],tf.float32),[32,32,3])
    label=example['label']
    return image,label
def preprocess(image,label):
    image=tf.pad(image,[[4,4],[4,4],[0,0]])
    image=tf.image.random_crop(image,[32,32,3])
    image=tf.image.random_flip_left_right(image)
    return image,label

if __name__=='__main__':
    with tf.Session() as sess:
        import matplotlib.pyplot as plt
        image,label=get_data()
        while 1:
            a,b=sess.run([image,label])
            plt.imshow(a[0]/255)
            plt.title(b[0])
            plt.show()