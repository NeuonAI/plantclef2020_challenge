# Neuon AI in PlantCLEF 2020
This repository contains the team's work in [PlantCLEF 2020](https://www.aicrowd.com/challenges/lifeclef-2020-plant).

We constructed a two-streamed Herbarium-Field Triplet Loss Network in tackling the challenge. The network is illustrated below:

![Figure 1](https://github.com/NeuonAI/plantclef2020_challenge/blob/master/neuon_triplet_network.png?raw=true "Herbarium-Field Triplet Loss Network")

## Repository Files
### Training scripts
- **run_inception_triplet_loss.py**
- **database_module_triplet_loss.py**

### Validation scripts
- **get_997_herbarium_mean_emb_crop.py**
  - gets the herbarium embedding representation for each class
- **validate_image.py**
  - validate a test image 
  
### Lists
- **clef2020_herbarium_species.csv**
  - list of species mapped to trained index
- **clef2020_herbarium_species_classid_map_to_index.txt**
  - list of class id (folder) mapped to trained index
- **clef2020_known_classes_herbarium_train.txt**
  - herbarium network training dataset
- **clef2020_known_classes_herbarium_train_added.txt**
  - herbarium network training dataset with added herbarium test dataset
- **clef2020_known_classes_herbarium_test.txt**
  - herbarium network test dataset
- **clef2020_known_classes_field_train.txt**
  - field network training dataset
- **clef2020_known_classes_field_train_added.txt**
  - field network training dataset with added field test dataset
- **clef2020_known_classes_field_test.txt**
  - field network test dataset
- **clef2020_multilabel_train.txt**
  - list of herbarium used in get_997_herbarium_mean_emb_crop.py
  
### Checkpoint
- **004000.ckpt.7z.001**
  - chekpoint of the trained network
