# Course Projects for NLP (COL 772)

# Info
* A1.1:
* A1.2:
* A2:

# Running instructions
First clone the repository:
```
https://github.com/MurtyShikhar/Natural-Language-Processing.git
```

Make sure you have scipy >= 0.17 and numpy >= 1.11.1 by running:
```
pip install -r requirements.txt
```
 

Warning: No Documentation. Code may be slightly unreadable
```
trainAtomicModels.py [-h] [-neg_samples NEG_SAMPLES]
                            [-vect_dim VECT_DIM] [-num_entities NUM_ENTITIES]
                            [-batch_size BATCH_SIZE]
                            [-num_relations NUM_RELATIONS] [-epochs NB_EPOCHS]
                            [-warm_start WARM_START] [-model MODEL]
                            [-dataset DATASET] [-optimizer OPTIMIZER]
                            [-rate LR] [-l2 L2]
```
where MODEL can be **distMult**, **E** , **EplusDistMult** or **deepDistMult**, DATASET is **WN18**, **FB15k**, or **FB15k-237** and OPTIMIZER is
either **Adagrad** or **RMSprop**

To make the train data again (in the form of matrices), run
```
dump_data.py [-h] [-neg_samples NEG_SAMPLES] [-dataset DATASET]
                    [-num_entities NUM_ENTITIES] [-path PATH]
```

where DATASET can be **FB15k**, **FB15k-237** or **WN18**, NEG_SAMPLES is the number of neg samples you want to create, and NUM_ENTITIES is the
number of entities in the dataset. 
