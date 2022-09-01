import argparse
import utils

#get hyper params from yaml
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = utils.get_config(opts.config)

import numpy as np
import os
import re
import attack_single_mesh

#All data sets paths
meshCnn_and_Pd_meshNet_shrec_path ='datasets_processed/meshCNN_and_PD_meshNet_source_data/'
meshCnn_shrec_vertices_and_faces = 'datasets_processed/meshCNN_faces_vertices_labels/'
pd_MeshNet_shrec_vertices_and_faces = 'datasets_processed/Pd_meshNet_faces_vertices_labels/'
meshWalker_shrec_path = 'datasets_processed/walker_copycat_shrec11/'
meshWalker_model_net_path = 'datasets_processed/walker_copycat_modelnet40/'
mesh_net_path = 'datasets_processed/mesh_net_modelnet40/'

if config['gpu_to_use'] >= 0:
  utils.set_single_gpu(config['gpu_to_use'])

mesh_net_labels = ['night_stand', 'range_hood', 'plant', 'chair', 'tent',
    'curtain', 'piano', 'dresser', 'desk', 'bed',
    'sink',  'laptop', 'flower_pot', 'car', 'stool',
    'vase', 'monitor', 'airplane', 'stairs', 'glass_box',
    'bottle', 'guitar', 'cone',  'toilet', 'bathtub',
    'wardrobe', 'radio',  'person', 'xbox', 'bowl',
    'cup', 'door',  'tv_stand',  'mantel', 'sofa',
    'keyboard', 'bookshelf',  'bench', 'table', 'lamp']


walker_shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]

walker_model_net_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]

meshCNN_and_Pd_meshNet_shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]
meshCNN_and_Pd_meshNet_shrec11_labels.sort()


def get_dataset_path(config = None):
    if config is None:
        exit("Your configuration file is None... Exiting")
    if config['arch'] == 'WALKER' and config['dataset'] == 'MODELNET40':
      config['trained_model'] = 'trained_models/walker_modelnet_imitating_network'
      return meshWalker_model_net_path
    elif config['arch'] == 'MESHNET' and config['dataset'] == 'MODELNET40':
      config['trained_model'] = 'trained_models/mesh_net_imitating_network'
      return mesh_net_path
    if config['arch'] == 'WALKER' and config['dataset'] == 'SHREC11':
      config['trained_model'] = 'trained_models/walker_shrec11_imitating_network'
      return meshWalker_shrec_path
    elif config['arch'] == 'MESHCNN' and config['dataset'] == 'SHREC11':
      config['trained_model'] = 'trained_models/meshCNN_imitating_network'
      return meshCnn_shrec_vertices_and_faces
    elif config['arch'] == 'PDMESHNET' and config['dataset'] == 'SHREC11':
      config['trained_model'] = 'trained_models/pd_meshnet_imitating_network'
      return pd_MeshNet_shrec_vertices_and_faces
    else:
      exit("Please provide a valid dataset name in recon file.")


def attack_mesh_net_models(config=None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(2, 40):
        config['source_label'] = i
        name_of_class = mesh_net_labels[config['source_label']]
        model_net_files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in model_net_files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            num_of_models = [name for name in model_net_files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            name_parts = re.split(pattern='_', string=model_name)
            name_parts = [name for name in name_parts if name.isnumeric()]
            id = name_parts[0] #name_parts[1] + '_' + name_parts[2] + '_' + name_parts[-1][:-4]
            _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path+model_name, id=id, labels=mesh_net_labels)

    return


def attack_walker_model_net_models(config=None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(0, 40):
        config['source_label'] = i
        name_of_class = walker_model_net_labels[config['source_label']]
        model_net_files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in model_net_files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            num_of_models = [name for name in model_net_files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            name_parts = re.split(pattern='_', string=model_name)
            name_parts[-1] = name_parts[-1][:-4]
            id = ''
            for i in range(len(name_parts)):
                if name_parts[i].isdigit():
                    id = id +'_' + name_parts[i]

            _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path+model_name, id=id, labels=walker_model_net_labels)

    return


def attack_meshCNN_shrec11_models(config = None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(0, 30):
        config['source_label'] = i
        name_of_class = meshCNN_and_Pd_meshNet_shrec11_labels[config['source_label']]
        files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            num_of_models = [name for name in files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            name_parts = re.split(pattern='_', string=model_name)
            id = name_parts[2]
            _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path+model_name, id=id+'cnn', labels=meshCNN_and_Pd_meshNet_shrec11_labels)

    return


def attack_Pd_meshNet_shrec11_models(config = None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(0, 30):
        config['source_label'] = i
        name_of_class = meshCNN_and_Pd_meshNet_shrec11_labels[config['source_label']]
        files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            num_of_models = [name for name in files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            name_parts = re.split(pattern='_', string=model_name)
            id = name_parts[2]
            _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path+model_name, id=id+'_pd', labels=meshCNN_and_Pd_meshNet_shrec11_labels)

    return


def attack_walker_shrec11_models(config = None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(0, 30):
        config['source_label'] = i
        name_of_class = walker_shrec11_labels[config['source_label']]
        files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            num_of_models = [name for name in files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            name_parts = re.split(pattern='_', string=model_name)
            id = name_parts[2]
            _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path+model_name, id=id, labels=meshCNN_and_Pd_meshNet_shrec11_labels)

    return


def main():
  np.random.seed(0)
  utils.config_gpu(1, -1)
  #attack_meshCNN_shrec11_models(config=config)
  #attack_Pd_meshNet_shrec11_models(config=config)
  #attack_walker_shrec11_models(config=config)
  #attack_walker_model_net_models(config=config)
  attack_mesh_net_models(config=config)

  return 0


if __name__ == '__main__':
  main()
