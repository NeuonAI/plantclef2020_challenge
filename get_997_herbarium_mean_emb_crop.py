# Neuon AI - PlantCLEF 2020

import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim
import numpy as np
import cv2
from nets.inception_v4 import inception_v4
from nets import inception_utils
from PIL import Image
from six.moves import cPickle
import os


# ============================================================= #
#                       Directories
# ============================================================= # 
image_dir_parent_train = "PlantCLEF2020TrainingData"
image_dir_parent_test = "PlantCLEF2020TrainingData"
train_herbarium_file = "list\clef2020_multilabel_train.txt"
herbarium_dir = "PlantCLEF2020TrainingData\herbarium"
checkpoint_model = "checkpoints\run16\040000.ckpt"
saved_pkl_file = "mean_emb_dict_997_herb_500_run16_40k_crops.pkl"


# ============================================================= #
#                         Parameters
# ============================================================= # 
batch = 5
numclasses1 = 997
numclasses2 = 10000
input_size = (299,299,3)

      
# ============================================================= #
#        Run network / Get herbarium mean embeddings
# ============================================================= #
# ----- Initiate tensors ----- #
x1 = tf.placeholder(tf.float32,(batch,) + input_size)
x2 = tf.placeholder(tf.float32,(batch,) + input_size)
y1 = tf.placeholder(tf.int32,(batch,))
y2 = tf.placeholder(tf.int32,(batch,))
is_training = tf.placeholder(tf.bool)
is_train = tf.placeholder(tf.bool, name="is_training")

# ----- Image preprocessing methods ----- #
train_preproc = lambda xi: inception_preprocessing.preprocess_image(
        xi,input_size[0],input_size[1],is_training=True)

test_preproc = lambda xi: inception_preprocessing.preprocess_image(
        xi,input_size[0],input_size[1],is_training=False)  

def data_in_train1():
    return tf.map_fn(fn = train_preproc,elems = x1,dtype=np.float32)      

def data_in_test1():
    return tf.map_fn(fn = test_preproc,elems = x1,dtype=np.float32)

def data_in_train2():
    return tf.map_fn(fn = train_preproc,elems = x2,dtype=np.float32)      

def data_in_test2():
    return tf.map_fn(fn = test_preproc,elems = x2,dtype=np.float32)

data_in1 = tf.cond(
        is_training,
        true_fn = data_in_train1,
        false_fn = data_in_test1
        )

data_in2 = tf.cond(
        is_training,
        true_fn = data_in_train2,
        false_fn = data_in_test2
        )

def get_herbarium_dict(train_herbarium_file):
    herbarium_dict = {}
    with open(train_herbarium_file,'r') as fid:
        h_lines = [x.strip() for x in fid.readlines()]
        
    herbarium_paths = [os.path.join(herbarium_dir,
                                    x.split(' ')[0]) for x in h_lines]
    herbarium_labels = [int(x.split(' ')[3]) for x in h_lines]
    
    for key, value in zip(herbarium_labels, herbarium_paths):
        if key not in herbarium_dict:
            herbarium_dict[key] = [] 
    
        herbarium_dict[key].append(value) 
        
    return herbarium_dict

def read_img(img_path):
    img = []
    try:
        current_img = img_path.replace("/", "\\")
        im = cv2.imread(current_img)
        
        if im is None:
           im = cv2.cvtColor(np.asarray(Image.open(current_img).convert('RGB')),cv2.COLOR_RGB2BGR)
        im = cv2.resize(im,(input_size[0:2]))
    
        if np.ndim(im) == 2:
            im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        else:
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)       

        # Center and Corner crops
        im1 = im[0:260,0:260,:]
        im2 = im[0:260,-260:,:]
        im3 = im[-260:,0:260,:]
        im4 = im[-260:,-260:,:]
        im5 = im[19:279,19:279,:]
        
        imtemp = [cv2.resize(ims,(input_size[0:2])) for ims in (im1,im2,im3,im4,im5)]
        [img.append(ims) for ims in imtemp]
        
    except:
        print('Exception found: Image not read...')
        pass 
    
    img = np.asarray(img,dtype=np.float32)/255.0
    return img

# ----- Construct network 1 ----- #
with slim.arg_scope(inception_utils.inception_arg_scope()):
    logits,endpoints = inception_v4(data_in1,
                                num_classes=numclasses1,
                                is_training=is_training,
                                scope='herbarium')
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
 
# ----- Construct network 2 ----- #   
with slim.arg_scope(inception_utils.inception_arg_scope()):
    logits2,endpoints2 = inception_v4(data_in2,
                                num_classes=numclasses2,
                                is_training=is_training,
                                scope='field')
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

variables_to_restore  = slim.get_variables_to_restore()    
restorer = tf.train.Saver(variables_to_restore)

# ----- Run session ----- #
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, checkpoint_model)
    herbarium_dict = get_herbarium_dict(train_herbarium_file)
    mean_emb_dict = {}
    counter = 0
    # ----- Iterate each class ----- #
    for key_class in herbarium_dict.keys():        
        selected_samples = []
        embedding_list = []
        embedding_np_list = []
        class_mean_embedding = []
        
        counter += 1
        print('\nCOUNTER:', counter, ' ----------- Current class:', key_class)
        
        current_class_files = herbarium_dict[key_class]
        current_class_files_len = len(herbarium_dict[key_class])       

        # Class selection percentage
        if current_class_files_len < 200:
            sample_number = 1
        elif current_class_files_len > 700:
            sample_number = 0.3
        else:
            sample_number = 0.5          

        # ----- Get class k samples ----- #
        selected_samples = np.random.choice(current_class_files, int(current_class_files_len * sample_number))
        print('Collected:', len(selected_samples), 'class samples')
        print('Getting class embeddings...')
        for fp in selected_samples:
            test_image = read_img(fp)
            sample_embedding = sess.run(
                        herbarium_feat,
                        feed_dict = {
                                        x1:test_image,
                                        is_training : False,
                                        is_train : False
                                }
                    )
            
            # Corner crops
            average_corner_crops = np.mean(sample_embedding, axis=0)
            reshaped_corner_crops = average_corner_crops.reshape(1,500)
            
            embedding_list.append(reshaped_corner_crops)

        # ----- Get class mean embs ----- #
        print('Getting class mean embeddings...')
        embedding_np_list = np.array(embedding_list)
        class_mean_embedding = embedding_np_list.mean(axis=0)
        
        # ----- Save mean embs into dict ----- #
        print('Saving class mean embeddings to dictionary...')
        if key_class not in mean_emb_dict:
            mean_emb_dict[key_class] = [] 
    
        mean_emb_dict[key_class].append(class_mean_embedding)  
 
    
# ----- Save mean embeddings ----- #
with open(saved_pkl_file,'wb') as fid:
    cPickle.dump(mean_emb_dict,fid,protocol=cPickle.HIGHEST_PROTOCOL)
    print('End. Mean emb pkl file created')
    





