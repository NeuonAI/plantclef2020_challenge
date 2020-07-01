# Neuon AI - PlantCLEF 2020

import os
import numpy as np
from PIL import Image
import cv2
import random


class database_module(object):
    def __init__(
                self,
                image_source_dir,
                database_file1,
                database_file2,
                batch,
                input_size,
                numclasses1,
                numclasses2,
                shuffle = False
            ):
        
        
        print("Initialising database...")
        self.image_source_dir = image_source_dir
        self.database_file1 = database_file1
        self.database_file2 = database_file2
        self.batch = batch
        self.input_size = input_size
        self.numclasses1 = numclasses1
        self.numclasses2 = numclasses2
        self.shuffle = shuffle

        self.load_data_list()
        
        
    def load_data_list(self):        
        self.database1_dict = {}
        self.database2_dict = {}
        
        # ----- Dataset 1 ----- #
        with open(self.database_file1,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
        
        self.data_paths1 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels1 = [int(x.split(' ')[3]) for x in lines]
        
        for key, value in zip(self.data_labels1, self.data_paths1):
            if key not in self.database1_dict:
                self.database1_dict[key] = [] 
            self.database1_dict[key].append(value)
        
        # ----- Dataset 2 ----- #
        with open(self.database_file2,'r') as fid2:
            lines2 = [x.strip() for x in fid2.readlines()]          
        
        self.data_paths2 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines2]
        self.data_labels2 = [int(x.split(' ')[3]) for x in lines2]
        
        for key, value in zip(self.data_labels2, self.data_paths2):
            if key not in self.database2_dict:
                self.database2_dict[key] = [] 
            self.database2_dict[key].append(value)
        
        self.data_num1 = len(self.data_paths1)
        self.data_num2 = len(self.data_paths2)
        self.database1_dict_copy = self.database1_dict
        self.database2_dict_copy = self.database2_dict        

        self.unique_labels = list(set(self.data_labels1).intersection(self.data_labels2))
        self.epoch = 0
        self.reset_data_list()     
        
    
    def reset_data_list(self):
        
        self.database1_dict = {}
        self.database2_dict = {}
        
        # ----- Dataset 1 ----- #
        with open(self.database_file1,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
        
        self.data_paths1 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines]
        self.data_labels1 = [int(x.split(' ')[3]) for x in lines]
        
        for key, value in zip(self.data_labels1, self.data_paths1):
            if key not in self.database1_dict:
                self.database1_dict[key] = [] 
            self.database1_dict[key].append(value)
                
        # ----- Dataset 2 ----- #
        with open(self.database_file2,'r') as fid2:
            lines2 = [x.strip() for x in fid2.readlines()]
        
        self.data_paths2 = [os.path.join(self.image_source_dir,
                                        x.split(' ')[0]) for x in lines2]
        self.data_labels2 = [int(x.split(' ')[3]) for x in lines2]
        
        for key, value in zip(self.data_labels2, self.data_paths2):
            if key not in self.database2_dict:
                self.database2_dict[key] = [] 
            self.database2_dict[key].append(value)
  
        
        self.database1_dict_copy = self.database1_dict
        self.database2_dict_copy = self.database2_dict
        self.unique_labels = list(set(self.data_labels1).intersection(self.data_labels2))


    def read_img(self, fp):
        try:
            im = cv2.imread(fp)
            if im is None:
               im = cv2.cvtColor(np.asarray(Image.open(fp).convert('RGB')),cv2.COLOR_RGB2BGR)
            im = cv2.resize(im,(self.input_size[0:2]))
            if np.ndim(im) == 2:
                im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
            else:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        except:
            pass
        return im


    def get_random_class(self):
        random_class = random.choices(self.unique_labels, k=1)        
        return random_class    
    
    
    def get_paths(self, class_i):
        try:
            # ----- Anchor positive ----- #
            rand_anchor1 = random.choice(self.database1_dict_copy.get(class_i))
        except:
            rand_anchor1 = None
        try:
            # ----- Anchor negative ----- #
            rand_anchor2 = random.choice(self.database2_dict_copy.get(class_i))
        except:
            rand_anchor2 = None
        return rand_anchor1, rand_anchor2
         
    
    def get_random_anchors(self, class_i, rand_anchor1, rand_anchor2):        
        # ----- Random anchor positive ----- #
        im = self.read_img(rand_anchor1)
        if len(self.total_filepaths) < self.batch:
            self.img1.append(im)
            self.lbl1.append(class_i)
            self.total_filepaths.append(rand_anchor1)
            
            # Remove anchor from dictionary 1
            for key, value in self.database1_dict_copy.items():
                if key == class_i:
                    if rand_anchor1 in value:
                        value.remove(rand_anchor1)
                        
        # ----- Random anchor negative ----- #          
        im2 = self.read_img(rand_anchor2)
        if len(self.total_filepaths) < self.batch:
            self.img2.append(im2)
            self.lbl2.append(class_i)
            self.total_filepaths.append(rand_anchor2) 
            
            # Remove anchor from dictionary 2
            for key, value in self.database2_dict_copy.items():
                if key == class_i:
                    if rand_anchor2 in value:
                        value.remove(rand_anchor2)
          
        
    def read_batch(self):        
        self.total_filepaths = []
        self.img1 = []
        self.img2 = []
        self.lbl1 = [] 
        self.lbl2 = []

        current_class_labels = []

        while len(self.total_filepaths) < self.batch:
            
            try:
                # ----- Select random class ----- #
                class_i = self.get_random_class()[0]
            except:
                # ----- Insufficient data ----- #
                print("Resetting data list")
                self.reset_data_list()
                self.epoch += 1
                continue
                
            # ----- Current class ----- #
            if class_i not in current_class_labels:
                current_class_labels.append(class_i)
                
            # Check class has 4 samples
            class_i_count1 = self.lbl1.count(class_i)
            class_i_count2 = self.lbl2.count(class_i)
            class_i_count = class_i_count1 + class_i_count2
            if class_i_count >= 4:
                current_class_labels.remove(class_i)
                continue            

            # Iterate over current labels
            for class_i in current_class_labels:
                # Get anchor paths
                rand_anchor1, rand_anchor2 = self.get_paths(class_i)
                
                if rand_anchor1 is not None and rand_anchor2 is not None:
                    # Get anchor positive and negative
                    self.get_random_anchors(class_i, rand_anchor1, rand_anchor2)
                    
                if rand_anchor1 is None or rand_anchor2 is None:
                    # Remove class from list
                    self.unique_labels.remove(class_i)
                    current_class_labels.remove(class_i)
                    continue
                    
                # Check class has 4 samples
                class_i_count1 = self.lbl1.count(class_i)
                class_i_count2 = self.lbl2.count(class_i)
                class_i_count = class_i_count1 + class_i_count2
                if class_i_count >= 4:                    
                    current_class_labels.remove(class_i)                        
                        
            if len(self.unique_labels) < 4:
                # ----- Insufficient data ----- #
                print("Resetting data list")
                self.reset_data_list()
                self.epoch += 1
                continue

        self.img1 = np.asarray(self.img1,dtype=np.float32)/255.0
        self.img2 = np.asarray(self.img2,dtype=np.float32)/255.0            
        return (self.img1, self.img2, self.lbl1, self.lbl2)        
    
    
    
    
    