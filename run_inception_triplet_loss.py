# Neuon AI - PlantCLEF 2020

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import numpy as np
import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
from nets.inception_v4 import inception_v4
from nets import inception_utils
from database_module_triplet_loss import database_module

image_dir_parent_train = "PlantCLEF2020TrainingData"
image_dir_parent_test = "PlantCLEF2020TrainingData"

train_file1 = "list\clef2020_known_classes_herbarium_train_added.txt"
test_file1 = "list\clef2020_known_classes_herbarium_test.txt"

train_file2 = "list\clef2020_known_classes_field_train_added.txt"
test_file2 = "list\clef2020_known_classes_field_test.txt"

tensorboard_dir = "path_to_tensorboard_dir"

checkpoint_model1 = "plantclef2020\final.ckpt"
checkpoint_model2 = "plantclef2017\final.ckpt"
checkpoint_save_dir = "path_to_checkpoint_save_dir"

batch = 16
input_size = (299,299,3)
numclasses1 = 997
numclasses2 = 10000
learning_rate = 0.0001
iterbatch = 1
max_iter = 500000
val_freq = 60
val_iter = 20


class inception_module(object):
    def __init__(self,
                 batch,
                 iterbatch,
                 numclasses1,
                 numclasses2,
                 image_dir_parent_train,
                 image_dir_parent_test,
                 train_file1,
                 train_file2,
                 test_file1,
                 test_file2,
                 input_size,
                 checkpoint_model1,
                 checkpoint_model2,
                 learning_rate,
                 save_dir,
                 max_iter,
                 val_freq,
                 val_iter):
        
        self.batch = batch
        self.iterbatch = iterbatch
        self.image_dir_parent_train = image_dir_parent_train
        self.image_dir_parent_test = image_dir_parent_test
        self.train_file1 = train_file1
        self.train_file2 = train_file2
        self.test_file1 = test_file1
        self.test_file2 = test_file2
        self.input_size = input_size
        self.numclasses1 = numclasses1
        self.numclasses2 = numclasses2
        self.checkpoint_model1 = checkpoint_model1
        self.checkpoint_model2 = checkpoint_model2
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.max_iter = max_iter
        self.val_freq = val_freq
        self.val_iter = val_iter

        # ----- Database module ----- #
        self.train_database = database_module(
                image_source_dir = self.image_dir_parent_train,
                database_file1 = self.train_file1,
                database_file2 = self.train_file2,
                batch = self.batch,
                input_size = self.input_size,
                numclasses1 = self.numclasses1,
                numclasses2 = self.numclasses2,
                shuffle = True)

        self.test_database = database_module(
                image_source_dir = self.image_dir_parent_test,
                database_file1 = self.test_file1,
                database_file2 = self.test_file2,
                batch = self.batch,
                input_size = self.input_size,
                numclasses1 = self.numclasses1,
                numclasses2 = self.numclasses2,
                shuffle = True)
       
        # ----- Tensors ------ #
        print('Initiating tensors...')
        x1 = tf.placeholder(tf.float32,(None,) + self.input_size)
        x2 = tf.placeholder(tf.float32,(None,) + self.input_size)
        herbarium_embs = tf.placeholder(tf.float32)
        field_embs = tf.placeholder(tf.float32)
        feat_concat = tf.placeholder(tf.float32, shape=[None, 500])
        lbl_concat = tf.placeholder(tf.float32)
        y1 = tf.placeholder(tf.int32, (None,))
        y2 = tf.placeholder(tf.int32, (None,))
        self.is_training = tf.placeholder(tf.bool)
        is_train = tf.placeholder(tf.bool, name="is_training")
        
        # ----- Image pre-processing methods ----- #      
        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=True)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=False) 
        
        def data_in_train1():
            return tf.map_fn(fn = train_preproc,elems = x1,dtype=np.float32)
        
        def data_in_test1():
            return tf.map_fn(fn = test_preproc,elems = x1,dtype=np.float32)
        
        def data_in_train2():
            return tf.map_fn(fn = train_preproc,elems = x2,dtype=np.float32)
        
        def data_in_test2():
            return tf.map_fn(fn = test_preproc,elems = x2,dtype=np.float32)
        
        data_in1 = tf.cond(
                self.is_training,
                true_fn = data_in_train1,
                false_fn = data_in_test1
                )
        
        data_in2 = tf.cond(
                self.is_training,
                true_fn = data_in_train2,
                false_fn = data_in_test2
                )

        print('Constructing network...')        
        # ----- Network 1 construction ----- #
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_v4(data_in1,
                                            num_classes=self.numclasses1,
                                            is_training=self.is_training,
                                            scope='herbarium'
                                            )
            herbarium_embs = endpoints['PreLogitsFlatten']
            herbarium_bn = tf.layers.batch_normalization(herbarium_embs, training=is_train)
            herbarium_feat = tf.contrib.layers.fully_connected(
                            inputs=herbarium_bn,
                            num_outputs=500,
                            activation_fn=None,
                            normalizer_fn=None,
                            trainable=True,
                            scope='herbarium'                            
                    )
            herbarium_feat = tf.math.l2_normalize(
                                                herbarium_feat,
                                                axis=1      
                                            )
        
        # ----- Network 2 construction ----- #        
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits2,endpoints2 = inception_v4(data_in2,
                                            num_classes=self.numclasses2,
                                            is_training=self.is_training,
                                            scope='field'
                                            )
            field_embs = endpoints2['PreLogitsFlatten']
            field_bn = tf.layers.batch_normalization(field_embs, training=is_train)
            field_feat = tf.contrib.layers.fully_connected(
                            inputs=field_bn,
                            num_outputs=500,
                            activation_fn=None,
                            normalizer_fn=None,
                            trainable=True,
                            scope='field'                            
                    )            
            field_feat = tf.math.l2_normalize(
                                    field_feat,
                                    axis=1      
                                )
        
        feat_concat = tf.concat([herbarium_feat, field_feat], 0)
        lbl_concat = tf.concat([y1, y2], 0)

        # ----- Get all variables ----- #
        self.variables_to_restore = tf.trainable_variables()        
        self.variables_bn = [k for k in self.variables_to_restore if k.name.startswith('batch_normalization')]
        self.variables_herbarium = [k for k in self.variables_to_restore if k.name.startswith('herbarium')]
        self.variables_field = [k for k in self.variables_to_restore if k.name.startswith('field')]

        # ----- New variable list ----- #        
        self.var_list_front = self.variables_herbarium[0:-10] + self.variables_field[0:-10]        
        self.var_list_last = self.variables_herbarium[-10:] + self.variables_field[-10:] + self.variables_bn
        self.var_list_train = self.var_list_front + self.var_list_last
               
        # ----- Network losses ----- #
        with tf.name_scope("loss_calculation"): 
            with tf.name_scope("triplets_loss"):
                self.triplets_loss = tf.reduce_mean(
                        tf.contrib.losses.metric_learning.triplet_semihard_loss(
                                labels=lbl_concat, embeddings=feat_concat, margin=1.0))

            with tf.name_scope("L2_reg_loss"):
                self.regularization_loss = tf.add_n([ tf.nn.l2_loss(v) for v in self.var_list_train]) * 0.00004 
                
            with tf.name_scope("total_loss"):
                self.totalloss = self.triplets_loss + self.regularization_loss
                
        # ----- Create update operation ----- #
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)    
        self.vars_ckpt = slim.get_variables_to_restore()
        
        vars_ckpt_herbarium = [k for k in self.vars_ckpt if k.name.startswith('herbarium')]        
        vars_ckpt_field = [k for k in self.vars_ckpt if k.name.startswith('field')]

        # ----- Restore model 1 ----- #
        restore_fn1 = slim.assign_from_checkpoint_fn(
            self.checkpoint_model1, vars_ckpt_herbarium[:-2]) 
                
        # ----- Restore model 2 ----- #
        restore_fn2 = slim.assign_from_checkpoint_fn(
            self.checkpoint_model2, vars_ckpt_field[:-2]) 
       
        # ----- Training scope ----- #       
        with tf.name_scope("train"):
            loss_accumulator = tf.Variable(0.0, trainable=False)
            self.collect_loss = loss_accumulator.assign_add(self.totalloss)
            self.average_loss = tf.cond(self.is_training,
                                        lambda: loss_accumulator / self.iterbatch,
                                        lambda: loss_accumulator / self.val_iter)
            self.zero_op_loss = tf.assign(loss_accumulator,0.0)
            self.accum_train = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_train]                                               
            self.zero_ops_train = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train]
            
            # ----- Set up optimizer / Compute gradients ----- #
            with tf.control_dependencies(self.update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate * 0.1)                
                optimizer_lastlayers = tf.train.AdamOptimizer(self.learning_rate)                
                gradient1 = optimizer.compute_gradients(self.totalloss,self.var_list_front)
                gradient2 = optimizer_lastlayers.compute_gradients(self.totalloss,self.var_list_last)
                gradient = gradient1 + gradient2              
                gradient_only = [gc[0] for gc in gradient]
                gradient_only,_ = tf.clip_by_global_norm(gradient_only,1.25)
                
                self.accum_train_ops = [self.accum_train[i].assign_add(gc) for i,gc in enumerate(gradient_only)]

            # ----- Apply gradients ----- #
            self.train_step = optimizer.apply_gradients(
                    [(self.accum_train[i], gc[1]) for i, gc in enumerate(gradient)])
            
        # ----- Global variables ----- #
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]             
        var_list += bn_moving_vars
        
        # ----- Create saver ----- #
        saver = tf.train.Saver(var_list=var_list, max_to_keep=0)
        tf.summary.scalar('loss',self.average_loss) 
        self.merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
        
        # ----- Tensorboard writer--- #
        writer_train = tf.summary.FileWriter(tensorboard_dir+'/train')
        writer_test = tf.summary.FileWriter(tensorboard_dir+'/test')

        print('Commencing training...')        
        # ----- Create session ----- #
        with tf.Session() as sess:            
            sess.run(tf.global_variables_initializer())
            writer_train.add_graph(sess.graph)
            writer_test.add_graph(sess.graph)
            
            restore_fn1(sess)
            restore_fn2(sess)

            
            for i in range(self.max_iter+1):
                try:
                    sess.run(self.zero_ops_train)
                    sess.run([self.zero_op_loss])                    
                    
                    # ----- Validation ----- #
                    if i % self.val_freq == 0:                        
                        print('Start:%f'%sess.run(loss_accumulator))
                        for j in range(self.val_iter):
                            img1,img2,lbl1, lbl2 = self.test_database.read_batch()
                            sess.run(
                                        self.collect_loss,
                                        feed_dict = {x1 : img1,
                                                     x2 : img2,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     self.is_training : False,
                                                     is_train : False
                                        }                                  
                                    )
                            print('[%i]:%f'%(j,sess.run(loss_accumulator)))
                        print('End:%f'%sess.run(loss_accumulator))  
                        s,self.netLoss = sess.run(                        
                                [self.merged,self.average_loss],
                                    feed_dict = {
                                            self.is_training : False
                                    }                            
                                ) 
                        writer_test.add_summary(s, i)
                        print('[Valid] Epoch:%i Iter:%i Loss:%f'%(self.train_database.epoch,i,self.netLoss))

                        sess.run([self.zero_op_loss])
                        
                    # ----- Train ----- #
                    for j in range(self.iterbatch):
                        img1,img2,lbl1,lbl2 = self.train_database.read_batch()
    
                        sess.run(
                                    [self.collect_loss,self.accum_train_ops],
                                    feed_dict = {x1 : img1, 
                                                 x2 : img2,
                                                 y1 : lbl1,
                                                 y2 : lbl2,
                                                 self.is_training : True,
                                                 is_train : True
                                    }                                
                                )
                        
                    s,self.netLoss = sess.run(
                            [self.merged,self.average_loss],
                                feed_dict = {
                                        self.is_training : True
                                }                            
                            ) 
                    writer_train.add_summary(s, i)
                    
                    sess.run([self.train_step])
                        
                    print('[Train] Epoch:%i Iter:%i Loss:%f'%(self.train_database.epoch,i,self.netLoss))

                    if i % 5000 == 0:
                        saver.save(sess, os.path.join(self.save_dir,'%06i.ckpt'%i)) 
                    
                except KeyboardInterrupt:
                    print('Interrupt detected. Ending...')
                    break
                
            # ----- Save model --- #
            saver.save(sess, os.path.join(self.save_dir,'final.ckpt')) 
            print('Model saved')


# ----- Run Network ----- #
network = inception_module(
        batch = batch,
        iterbatch = iterbatch,
        numclasses1 = numclasses1,
        numclasses2 = numclasses2,
        input_size = input_size,
        image_dir_parent_train = image_dir_parent_train,
        image_dir_parent_test = image_dir_parent_test,
        train_file1 = train_file1,
        train_file2 = train_file2,
        test_file1 = test_file1,
        test_file2 = test_file2,        
        checkpoint_model1 = checkpoint_model1,
        checkpoint_model2 = checkpoint_model2,
        save_dir = checkpoint_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter
        )          
    
     

    
    
    
    
    
    