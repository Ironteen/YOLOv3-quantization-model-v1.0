import sys
import os
import time
import tensorflow as tf
import numpy as np
import threading

from datetime import datetime
from core.config import cfg 
from core.darknet53 import darknet53_model
from dataset.ImageNet import load_ImageNet

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,7,8'

class pretrain_on_imagenet(object):
  def __init__(self):
    self.epochs                      = 100
    self.batch_size                  = 64
    self.learning_rate               = 0.005
    self.dropout                     = 0.5
    self.momentum                    = 0.9
    self.lmbda                       = cfg.WEIGHT_DECAY 
    self.reload                      = False
    self.imagenet_path               = cfg.IMAGENET_PATH 
    self.display_step                = 100
    self.test_step                   = 1000
    self.ckpt_path                   = './log/checkpoint_imagenet'
    self.imagenet                    = load_ImageNet()

    if not os.path.exists(self.ckpt_path):
      os.mkdir(self.ckpt_path)

  def train(self):
    train_img_path = os.path.join(self.imagenet_path, 'train')
    dataset_len = self.imagenet.imagenet_size(train_img_path)
    num_batches = int(float(dataset_len) / self.batch_size)

    train_labels, _ = self.imagenet.load_imagenet_meta(os.path.join(self.imagenet_path, 'data/meta.mat'))

    input_data = tf.placeholder(tf.float32, [None, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, 3])
    labels = tf.placeholder(tf.float32, [None, 1000])

    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # queue of examples being filled on the cpu
    with tf.device('/cpu:0'):
      queue = tf.FIFOQueue(self.batch_size * 3, [tf.float32, tf.float32], shapes=[[cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, 3], [1000]])
      enqueue_op = queue.enqueue_many([input_data, labels])
      input_batch, label_batch = queue.dequeue_many(self.batch_size)

    self.darknet53_model = darknet53_model(input_batch,trainable=True)
    result = self.darknet53_model.imagenet_recog(self.darknet53_model.darknet53_output)
    # result = self.darknet53_model.class_result
    # print('result:',result.get_shape().as_list())
    # print('label_batch:',label_batch.get_shape().as_list())

    # cross-entropy and weight decay
    with tf.name_scope('cross_entropy'):
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=label_batch, name='cross-entropy'))
    
    with tf.name_scope('l2_loss'):
      l2_loss = tf.reduce_sum(self.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
    
    with tf.name_scope('loss'):
      loss = tf.add(cross_entropy,l2_loss)

    # accuracy
    with tf.name_scope('accuracy'):
      correct = tf.equal(tf.argmax(result, 1), tf.argmax(label_batch, 1))
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    global_step = tf.Variable(0, trainable=False)
    epoch = tf.div(global_step, num_batches)
    
    # momentum optimizer
    with tf.name_scope('optimizer'):
      # print('loss:',loss)
      optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(loss, global_step=global_step)

    #save the parameters besides the last layer
    # checkpoint saver
    variables = tf.contrib.framework.get_variables_to_restore()
    # variables_to_resotre = [v for v in variables if v.name.split('/')[0]!='conv3']
    # saver_1 = tf.train.Saver(variables)
    saver = tf.train.Saver(variables)
    coord = tf.train.Coordinator()

    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto()) as sess:
      if self.reload:
        sess.run(init)
        ckpt_file = os.path.join(self.ckpt_path, 'model.ckpt')
        saver.restore(sess,ckpt_file)
        print('reload the %s successfully......'%ckpt_file)
      else:
        sess.run(init)

      # enqueuing batches procedure
      def enqueue_batches():
        while not coord.should_stop():
          image_data, labels_data = self.imagenet.read_batch(self.batch_size, train_img_path, train_labels)
          sess.run(enqueue_op, feed_dict={input_data: image_data,labels: labels_data})

      # creating and starting parallel threads to fill the queue
      num_threads = 3
      for i in range(num_threads):
        t = threading.Thread(target=enqueue_batches)
        t.setDaemon(True)
        t.start()

      start_time = time.time()
      for e in range(sess.run(epoch)+1, self.epochs):
        for i in range(1,num_batches+1):

          _, step = sess.run([optimizer, global_step], feed_dict={lr: self.learning_rate, keep_prob: self.dropout})
          #train_writer.add_summary(summary, step)

          # decaying learning rate
          if step == 80000 or step == 170000 or step == 350000:
            self.learning_rate /= 10

          # display current training informations
          if step % self.display_step == 0:
            c, a = sess.run([loss, accuracy], feed_dict={lr: self.learning_rate, keep_prob: 1.0})
            print (str(datetime.now())+' Epoch: {:03d} Step/Batch: {:05d}/{:05d} --- Loss: {:.7f} Training accuracy: {:.4f}'.format(e, step, num_batches, c, a))
              
          # make test and evaluate validation accuracy
          if step % self.test_step == 0:
            val_im, val_cls = self.imagenet.read_validation_batch(self.batch_size, os.path.join(self.imagenet_path, 'ILSVRC2012_img_val'), os.path.join(self.imagenet_path, 'data/ILSVRC2012_validation_ground_truth.txt'))
            v_a = sess.run(accuracy, feed_dict={input_batch: val_im, label_batch: val_cls, lr: self.learning_rate, keep_prob: 1.0})
            # intermediate time
            int_time = time.time()
            print ('Elapsed time: {}'.format(self.imagenet.format_time(int_time - start_time)))
            print ('Validation accuracy: {:.04f}'.format(v_a))
            # save weights to file
            save_path = saver.save(sess, os.path.join(self.ckpt_path, 'model.ckpt'))
            print('Variables saved in file: %s' % save_path)

      end_time = time.time()
      print ('Elapsed time: {}').format(self.imagenet.format_time(end_time - start_time))
      save_path = saver.save(sess, os.path.join(self.ckpt_path, 'model.ckpt'))
      print('Variables saved in file: %s' % save_path)

      coord.request_stop()


if __name__ == '__main__':
  #build dir needed
  imagenet_model = pretrain_on_imagenet()
  imagenet_model.train()