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
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================= #
#                       Directories
# ============================================================= # 
image_dir_parent_train = "PlantCLEF2020TrainingData"
image_dir_parent_test = "PlantCLEF2020TrainingData"
checkpoint_model = "checkpoints\run16\040000.ckpt"
species_name_map_csv = "list\clef2020_herbarium_species.csv"
classmap_txt = "list\clef2020_herbarium_species_classid_map_to_index.txt"
herbarium_dictionary_file = "mean_emb_dict_997_herb_500_run16_40k_crops.pkl"
test_image = "PlantCLEF2020TrainingData\photo\373\5859.jpg"


# ============================================================= #
#                         Parameters
# ============================================================= # 
topN = 5 # Number of predictions to output
batch = 10 
# Assign batch = 10,
# 10 variations of flipped cropped imgs (center, top left, top right, bottom left, bottom right, 
# center flipped, top left flipped, top right flipped, bottom left flipped, bottom right flipped)
numclasses1 = 997 # Class number of Herbarium network
numclasses2 = 10000 # Class number of Field network
input_size = (299,299,3) # Image input size


# ============================================================= #
#                         Load data
# ============================================================= # 
# ----- Read herbarium dictionary pkl file ----- #
with open(herbarium_dictionary_file,'rb') as fid1:
	herbarium_dictionary = cPickle.load(fid1)

# ----- Map species index to folder ----- #
with open(classmap_txt,'r') as fid:
    classmap = [x.strip().split(' ')[0] for x in fid.readlines()]
    
# ----- Map species name to index ----- #    
species_name_map_df = pd.read_csv(species_name_map_csv, sep=',')
species_list = species_name_map_df['species'].to_list()


# ============================================================= #
#               Run network / validate image
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

def read_img(img_path):
    img = []
    try:
        current_img = img_path
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

        # Flip image
        flip_img = cv2.flip(im, 1)
        
        flip_im1 = flip_img[0:260,0:260,:]
        flip_im2 = flip_img[0:260,-260:,:]
        flip_im3 = flip_img[-260:,0:260,:]
        flip_im4 = flip_img[-260:,-260:,:]
        flip_im5 = flip_img[19:279,19:279,:]
        
        flip_imtemp = [cv2.resize(imf,(input_size[0:2])) for imf in (flip_im1,flip_im2,flip_im3,flip_im4,flip_im5)]
                
        [img.append(imf) for imf in flip_imtemp]        
        
    except:
        print("Exception found: Image not read...")
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
    
variables_to_restore = slim.get_variables_to_restore()
restorer = tf.train.Saver(variables_to_restore)

# ----- Run session ----- #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, checkpoint_model)
    
    test_image = read_img(test_image)
    
    sample_embedding = sess.run(
                field_feat,
                feed_dict = {
                                x2:test_image,
                                is_training : False,
                                is_train : False
                        }
            )   

    # Average center + corner crop embeddings
    averaged_flip = np.mean(sample_embedding, axis=0)
    reshaped_emb_sample = averaged_flip.reshape(1,500)       

    print('Getting herbarium dictionary...')    
    herbarium_emb_list = []
    for herbarium_class, herbarium_emb in herbarium_dictionary.items():
        herbarium_emb_list.append(np.squeeze(herbarium_emb))    
    herbarium_emb_list = np.array(herbarium_emb_list)
    
    print('Comparing sample embedding with herbarium distance...')
    similarity = cosine_similarity(reshaped_emb_sample, herbarium_emb_list)
        
    print('Getting probability distribution...') 
    similarity_distribution = []
    for sim in similarity:
        new_distribution = []
        for d in sim:
            new_similarity = 1 - d # 1 - cosine value (d)
            new_distribution.append(new_similarity)
        similarity_distribution.append(new_distribution)
    similarity_distribution = np.array(similarity_distribution)   
              
    # Apply inverse weighting with power of 5
    probabilty_list = []
    for d in similarity_distribution:
        inverse_weighting = (1/np.power(d,5))/np.sum(1/np.power(d,5))
        probabilty_list.append(inverse_weighting)    
    probabilty_list = np.array(probabilty_list)
     
    print('Getting topN predictions...')
    for prediction in probabilty_list:
        topN_class_list = prediction.argsort()[-topN:][::-1]
        topN_probability_list = np.sort(prediction)[-topN:][::-1]

    counter = 0 
    for cl, prob in zip(topN_class_list, topN_probability_list):
        counter += 1
        class_index = classmap[int(cl)]
        pred_name = species_list[int(cl)]
        print('\nPREDICTION:', counter)
        print('Species:', pred_name)
        print('Class index (folder):', class_index)
        print('Probability:', prob)
    
        
 
    