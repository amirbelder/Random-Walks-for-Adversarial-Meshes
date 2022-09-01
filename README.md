# MeshWalker: Deep Mesh Understanding by Random Walks
Created by [Amir Belder](mailto:amirbelder5@gmail.com).

Based on: https://arxiv.org/abs/2202.07453

## Installation
In order to run the network you need to create and environment, we recommend a conda environment.
You have 2 choices, a 3.6 python env, and a 3.8 python env.
In both cases, go to the desired folder and:
For 3.6:
'pip install -r 3_6_requirements.txt'
For 3.8:
'pip install -r 3_8_requirements.txt'

#Source codes and Raw data
The source code of each attacked network, along with its raw data can be found at:
- [MeshCNN](https://github.com/ranahanocka/MeshCNN)
- [MeshWalker](https://github.com/alonlahav/meshWalker)
- [MeshNet](https://github.com/iMoonLab/MeshNet)
- [Pd-MeshNet](https://github.com/MIT-SPARK/PD-MeshNet)

## Data
We have 2 kinds of datasets, the raw of each network and thr adjusted walker datasets that were used to train the imitating networks.
The raw datsets can be found at the links above.
For each attacked network:
  - Our adjusted dataset is made out the raw data of each network.
  - On both the train set and test set, we took the vertices and faces of each network after its simplification.
  - We took the predicted labels of each of the train set meshes. 
  - We created a dataset in the MeshWalker format using these above collected data.  

These datasets can be found here:
Add link

The datasets could also be created by using 'datasets_prepare.py',
where you will find a function for each dataset.
You may need to adjust the raw datsets paths according to where you saved them on your computer.
Processing will rearrang dataset in `npz` files, labels included, vertex niebours added.

Some of our results can be found [here]( https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/MeshAdversarial/attacked_models_of_all_networks.zip).

You can also download it from our [raw_datasets]() folder.
Please run `bash ./scripts/download_raw_datasets.sh`.


### Processed
To prepare the data, run `python dataset_prepare.py <dataset>`

## Training Imitating Networks
```
python imitating_network_train.py 
```
In order to train the imitating networks you will have to change 3 values in the configuration YAML file:

- arch: set to one of: 'WALKER', 'MESHCNN', 'PDMESHNET', 'MESHNET'
- dataset: set to one of: 'SHREC11', 'MODELNET40'
- dataset_path: According to where you chose to put the data

Use tensorboard to show training results: `tensorboard <trained-model-folder>`
<img src='/doc/images/2nd_fig.png'>

## Attacking
To attack the data, run `python attack_mesh.py`
In order to attack the different mesh models, you again need to set 2 values in the configuration file:
- arch: set to one of: 'WALKER', 'MESHCNN', 'PDMESHNET', 'MESHNET'
- dataset: set to one of: 'SHREC11', 'MODELNET40'

Please notice that you have got to have the data you wish to attack and a trained imitating network.
You may need to change the directories of these two inside the attack_mesh file, according to where you saved them on your computer.

The attacked meshes can be found in the folder above the current working directory.
They will be saved according to the different networks, i.e.: '../attacks/imitating_network_name'

## Pretrained
All five pretrained imitating networks can be found here: [pretrained](https://technionmail-my.sharepoint.com/personal/alon_lahav_campus_technion_ac_il/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Falon%5Flahav%5Fcampus%5Ftechnion%5Fac%5Fil%2FDocuments%2Fmesh%5Fwalker%2Fpretrained)  models to run evaluation only. 

##Check Results
In order to check how the attacked affected the original SOTA system we advise to save the changed vertices and faces in an obj file.
Each of the SOTA systems uses these obj files while testing.
And so, by changing them, we assure that the networks perform all its needed precprocessing.
An example of such saving for SHREC11 can be found in the npz_to_obj.py file.
This is not necessary at the Walker attacked files, as they are already in format. 

## Troubleshooting
If rendering using `opengl` doesn't work, 
it might be because `LANG` environment parameter is not set to `en_US`.

To fix it just write: `LANG=en_US` in command line and then run the python script.

If you use PyCharm, go to: `Run -> Edit Configurations...` and add `
;LANG=en_US` to the `Environment variables:`
