
#### Introduction ####

RNN-based methods tend to prioritize the modeling of predicate sequences while neglecting intermediate entities.Here we observe that these neglects can lead to bias phenomena.

![The disorder phenomena of RNN-based models](https://github.com/tinthin03/CbGAT/blob/master/bias.png "The ambiguity bias of RNN-based models")


Considering the limitations of RNNs in modeling mechanisms, this paper proposes a novel framework for logic rule learning based on Graph Neural Networks (GNNs), known as the $\underline{C}u\underline{b}ic \space \underline{G}raph \space \underline{At}tention \space Network$ (CbGAT).

CbGAT addresses these limitations by incorporating a new element called the $\textit{cubic relation}$, which exhibits duality with traditional predicate relations. While predicate relations describe the relationships between entities, cubic relations represent the relationship between pairs of predicates (i.e., the relationship of predicate relations) on specific entities.

![cubic_relations and entities](https://github.com/tinthin03/CbGAT/blob/master/cubic_relations.jpg "cubic_relations and entities")

#### Datasets ####

Open-world KGC tasks are commonly evaluated on Word-Net and Freebase subsets, such as WN18RR, and FB15k-237. To evaluate the reasoning ability of the rules mined by our method, we selected FB15K-237, WN18RR and UMLS as the experimental dataset. 

In order to test the inductive reasoning performance of our model, we introduce new datasets ILPC[1] and FB15k-237-Inductive.

[1] Mikhail Galkin, Max Berrendorf, and Charles Tapley Hoyt. An Open Challenge for Inductive Link Prediction on Knowledge.Mar 2022.

#### Usage ####

1. compile the codes by running the following command:
   > pip ./cppext/setup.py install 

2. Run rcb.sh for representation learning with GPU 0:
   > ./rcb.sh 0

3. run rem.sh for rule mining with GPU 0:
   > ./rem.sh 0


If you want to skip representation learning and directly conduct rule mining experiments, you can download a pre-trained model of the *ilpc-target* dataset (in the *checkpoints* folder) through the following link:
   > https://drive.google.com/drive/folders/1WXNAj1O4e5HdWgaYBfCPGUMbuvnOE6av?usp=sharing

The default dataset is *ilpc-target*, which can be selected by modifying the variable *exp* in *./main.py*.

The dataset needs to be downloaded in advance and stored in *./data* directory.
