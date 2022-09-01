import os, shutil, time, copy, glob

import yaml
from easydict import EasyDict
import json
import platform

import numpy as np
import tensorflow as tf
import trimesh, open3d
import pyvista as pv
import scipy
import pylab as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from tqdm import tqdm
import argparse

import rnn_model
import utils
import yaml
import dataset
import dataset_prepare

recon_training = True
timelog = {}
timelog['prep_model'] = []
timelog['fill_features'] = []

import yaml
def get_config(config):
  with open(config, 'r') as stream:
    return yaml.safe_load(stream)


"""
#get hyper params from yaml
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = get_config(opts.config)"""

def print_enters(to_print):
  print("\n\n\n\n")
  print(to_print)
  print("\n\n\n\n")

def get_model_names():
  part = 'test'
  model_fns = []
  for i, name in enumerate(dataset_prepare.model_net_labels):
    pathname_expansion = os.path.expanduser('~') + '/datasets_processed/ModelNet40/' + name + '/' + part + '/*.off'
    filenames = glob.glob(pathname_expansion)
    model_fns += filenames
  return model_fns

def show_walk(model, features, one_walk=False):
  for wi in range(features.shape[0]):
    walk = features[wi, :, -1].astype(np.int)
    jumps = features[wi, :, -2].astype(np.bool)
    utils.visualize_model_walk(model['vertices'], model['faces'], walk, jumps)
    if one_walk:
      break


def calc_accuracy_test(dataset_folder=False, logdir=None, labels=None, iter2use='last', classes_indices_to_use=None,
                       dnn_model=None, params=None, verbose_level=2, min_max_faces2use=[0, 5000], model_fn=None,
                       target_n_faces=['according_to_dataset'], n_walks_per_model=16, seq_len=None, data_augmentation={}):
  verbose_level = 2
  SHOW_WALK = 1
  WALK_LEN_PROP_TO_NUM_OF_TRIANLES = 0
  COMPONENT_ANALYSIS = False
  PRINT_CONFUSION_MATRIX = False
  np.random.seed(1)
  tf.random.set_seed(0)
  #classes2use = None #['desk', 'dresser', 'table', 'laptop', 'lamp', 'stool', 'wardrobe'] # or "None" for all
  #params.classes_indices_to_use = None #[15, 25]
  if params is None:
    classes2use = classes_indices_to_use
  else:
    classes2use = params.classes_indices_to_use


  print_details = verbose_level >= 2
  if params is None:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    if model_fn is not None:
      pass
    elif iter2use != 'last':
      model_fn = logdir + '/learned_model2keep--' + iter2use
      model_fn = model_fn.replace('//', '/')
    else:
      model_fn = tf.train.latest_checkpoint(logdir)
    if verbose_level and model_fn is not None:
      print(utils.color.BOLD + utils.color.BLUE + 'logdir : ', model_fn + utils.color.END)
  else:
    params = copy.deepcopy(params)
  params.batch_size = 1
  params.n_walks_per_model = n_walks_per_model

  if "modelnet" in params.logdir:
    model_name = "modelnet40"
  elif "mesh_net" in params.logdir:
    model_name = "mesh_net"
  elif "shrec" in params.logdir:
    model_name = "shrec11"
  else:
    print("Unknown model ! exiting...")
    exit(0)


  if 0:
    params.net_input.append('jump_indication')
  if 0:
    params.layer_sizes = None
    params.aditional_network_params = []

  if seq_len:
    params.seq_len = seq_len
  if verbose_level:
    print('params.seq_len:', params.seq_len, ' ; n_walks_per_model:', n_walks_per_model)

  #Amir - check 800 after training over 200
  if params is None:
    params.seq_len = 200

  if SHOW_WALK:
    params.net_input += ['vertex_indices']

  params.set_seq_len_by_n_faces = 1
  if dataset_folder:
    size_limit = np.inf # 200

    params.classes_indices_to_use = classes2use
    pathname_expansion = dataset_folder
    if 1:
      test_dataset, n_models_to_test = dataset.tf_mesh_dataset(params, pathname_expansion, mode=params.network_task,
                                                                shuffle_size=0, size_limit=size_limit, permute_file_names=True,
                                                                min_max_faces2use=min_max_faces2use, must_run_on_all=True,
                                                               data_augmentation=data_augmentation)
    else:
      test_dataset = dataset.mesh_dataset_iterator(params, pathname_expansion, mode=params.network_task,
                                                                         shuffle_size=0, size_limit=size_limit, permute_file_names=True,
                                                                         min_max_faces2use=min_max_faces2use)
      iter(test_dataset).__next__()
      n_models_to_test = 1000
  else:
    test_dataset = get_model_names()
    test_dataset = np.random.permutation(test_dataset)
    n_models_to_test = len(test_dataset)

  if dnn_model is None:
    if params.net == "RnnWalkNet":
      dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - SHOW_WALK, model_fn,
                                       model_must_be_load=True, dump_model_visualization=False)
    elif params.net == "Manifold_RnnWalkNet":
      dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, params.net_input_dim - SHOW_WALK, model_fn,
                                       model_must_be_load=True, dump_model_visualization=False)
    elif params.net == "Unsupervised_RnnWalkNet":
      dnn_model = rnn_model.Unsupervised_RnnWalkNet(params, params.n_classes, params.net_input_dim - SHOW_WALK, model_fn,
                                       model_must_be_load=True, dump_model_visualization=False)
    else:
      print("Net type is not familiar ! exiting..")
      exit(0)

  n_pos_all = 0
  n_classes = 40
  all_confusion = np.zeros((n_classes, n_classes), dtype=np.int)
  size_accuracy = []
  ii = 0
  tb_all = time.time()
  res_per_n_faces = {}
  pred_per_model_name = {}
  dnn_inference_time = [] # 150mSec for 64 walks of 200 steps
  bad_pred = EasyDict({'n_comp': [], 'biggest_comp_area_ratio': []})
  good_pred = EasyDict({'n_comp': [], 'biggest_comp_area_ratio': []})

  utils.print_labels_names_and_indices(model_name)
  for i, data in tqdm(enumerate(test_dataset), disable=print_details, total=n_models_to_test):
    name, ftrs, gt = data
    model_fn = name.numpy()[0].decode()
    model_name, n_faces = utils.get_model_name_from_npz_fn(model_fn)
    print(model_name)
    assert ftrs.shape[0] == 1, 'Must have one model per batch for test'
    if WALK_LEN_PROP_TO_NUM_OF_TRIANLES:
      n2keep = int(n_faces / 2.5)
      ftrs = ftrs[:, :, :n2keep, :]
    ftrs = tf.reshape(ftrs, ftrs.shape[1:])
    gt = gt.numpy()[0]
    predictions = None
    for i_f, this_target_n_faces in enumerate(target_n_faces):
      model = None
      if SHOW_WALK:
        if model is None:
          model = dataset.load_model_from_npz(model_fn)
        if model['vertices'].shape[0] < 1000:
          print(model_fn)
          print('nv: ', model['vertices'].shape[0])
          #Amir - removed for modelnet40
          #show_walk(model, ftrs.numpy(), one_walk=1)
        ftrs = ftrs[:, :, :-1]
      ftr2use = ftrs.numpy()
      for k in [1, -1][:1]: # test augmentation: flip X axis (did not help)
        ftr2use[:, :, 0] *= k
        tb = time.time()
        if 0:
          jumps = ftr2use[:,:,3]
          jumps = np.hstack((jumps, np.ones((jumps.shape[0], 1)))) # To promise that one jump is found
          ftr2use = ftr2use[:,:,:3]
          first_jumps = [np.where(j)[0][0]  for j in jumps]
          last_jumps  = [0,0,0] # [np.where(j)[0][-2] - np.where(j)[0][-3] for j in jumps]
          plt.hist(first_jumps)
          plt.hist(last_jumps)
          model = dataset.load_model_from_npz(model_fn)
          plt.title('#vertices / faces : ' + str(model['vertices'].shape[0]) + ' / ' + str(model['faces'].shape[0]))
          plt.show()
        predictions_ = dnn_model(ftr2use, classify=True, training=False).numpy()
        te = time.time() - tb
        dnn_inference_time.append(te / n_walks_per_model * 1000)
        if 0:#len(dnn_inference_time) == 10:
          print(dnn_inference_time)
          plt.hist(dnn_inference_time[1:])
          plt.xlabel('[mSec]')
          plt.show()
        if predictions is None:
          predictions = predictions_
        else:
          predictions = np.vstack((predictions, predictions_))

    mean_pred = np.mean(predictions, axis=0)
    max_hit = np.argmax(mean_pred)
    # Gals changes
    #model_name = labels[int(gt)]
    # End

    if model_name not in pred_per_model_name.keys():
      pred_per_model_name[model_name] = [gt, np.zeros_like(mean_pred)]
    pred_per_model_name[model_name][1] += mean_pred
    str2add = '; n.unique models: ' + str(len(pred_per_model_name.keys()))
    '''
    print("\n\n")
    print(pred_per_model_name)
    print("\n\n")
    '''

    if n_faces not in res_per_n_faces.keys():
      res_per_n_faces[n_faces] = [0, 0]
    res_per_n_faces[n_faces][0] += 1

    if COMPONENT_ANALYSIS:
      model = dataset.load_model_from_npz(model_fn)
      comp_summary = dataset_prepare.component_analysis(model['faces'], model['vertices'])
      comp_area = [a['area'] for a in comp_summary]
      n_components = len(comp_summary)
      biggest_comp_area_ratio = np.sort(comp_area)[-1] / np.sum(comp_area)

    if max_hit != gt:
      false_str = ' , predicted: ' + labels[int(max_hit)] + ' ; ' + model_fn
      if COMPONENT_ANALYSIS:
        bad_pred.n_comp.append(n_components)
        bad_pred.biggest_comp_area_ratio.append(biggest_comp_area_ratio)
    else:
      res_per_n_faces[n_faces][1] += 1
      false_str = ''
      if COMPONENT_ANALYSIS:
        good_pred.n_comp.append(n_components)
        good_pred.biggest_comp_area_ratio.append(biggest_comp_area_ratio)
    if print_details:
      print('  ', max_hit == gt, labels[int(gt)], false_str, 'n_vertices: ')#, model['vertices'].shape[0])
      if 0:#max_hit != gt:
        model = dataset.load_model_from_npz(model_fn)
        # Amir - only for model net
        #utils.visualize_model(model['vertices'], model['faces'], line_width=1, opacity=1)

    all_confusion[int(gt), max_hit] += 1
    n_pos_all += (max_hit == gt)
    ii += 1
    if print_details:
      print(i, '/', n_models_to_test, ')  Total accuracy: ', round(n_pos_all / ii * 100, 1), 'n_pos_all:', n_pos_all, str2add)

  if print_details:
    print(utils.color.BLUE + 'Total time, all:', time.time() - tb_all, utils.color.END)

  n_models = 0
  n_sucesses = 0
  all_confusion_all_faces = np.zeros((n_classes, n_classes), dtype=np.int)
  for k, v in pred_per_model_name.items():
    gt = v[0]
    pred = v[1]
    max_hit = np.argmax(pred)
    all_confusion_all_faces[gt, max_hit] += 1
    n_models += 1
    n_sucesses += max_hit == gt
  mean_accuracy_all_faces = n_sucesses / n_models
  if print_details:
    print('\n\n ---------------\nOn avarage, for all faces:')
    print('  Accuracy: ', np.round(mean_accuracy_all_faces * 100, 2), '% ; n models checkd: ', n_models)
    print('Results per number of faces:')
    print('  ', res_per_n_faces, '\n\n--------------\n\n')

  if 0:
    bins = [0, 700, 1500, 3000, 5000]
    accuracy_per_n_faces = []
    for b in range(len(bins) - 1):
      ks = [k for k in res_per_n_faces.keys() if k >= bins[b] and k < bins[b + 1]]
      attempts = 0
      successes = 0
      for k in ks:
        attempts_, successes_ = res_per_n_faces[k]
        attempts += attempts_
        successes += successes_
      if attempts:
        accuracy_per_n_faces.append(successes / attempts)
      else:
        accuracy_per_n_faces.append(np.nan)
    x = (np.array(bins[1:]) + np.array(bins[:-1])) / 2
    plt.figure()
    plt.plot(x, accuracy_per_n_faces)
    plt.xlabel('Number of faces')
    plt.ylabel('Accuracy')
    plt.show()

  if PRINT_CONFUSION_MATRIX:
    b = 0;
    e = 40
    utils.plot_confusion_matrix(all_confusion[b:e, b:e], labels[b:e], normalize=1, show_txt=0)

  # Print list of accuracy per model
  for confusion in [all_confusion, all_confusion_all_faces]:
    if print_details:
      print('------')
    acc_per_class = []
    for i, name in enumerate(labels):
      this_type = confusion[i]
      n_this_type = this_type.sum()
      accuracy_this_type = this_type[i] / n_this_type
      if n_this_type:
        acc_per_class.append(accuracy_this_type)
      this_type_ = this_type.copy()
      this_type_[i] = -1
      scnd_best = np.argmax(this_type_)
      scnd_best_name = labels[scnd_best]
      accuracy_2nd_best = this_type[scnd_best] / n_this_type
      if print_details:
        print(str(i).ljust(3), name.ljust(12), n_this_type, ',', str(round(accuracy_this_type * 100, 1)).ljust(5), ' ; 2nd best:', scnd_best_name.ljust(12), round(accuracy_2nd_best * 100, 1))
  mean_acc_per_class = np.mean(acc_per_class)

  if 0:
    print('Time Log:')
    for k, v in timelog.items():
      print('  ' , k, ':', np.mean(v))

  return [mean_accuracy_all_faces, mean_acc_per_class], dnn_model

def show_features_tsne(dataset_files_path, logdir, dnn_model=None, cls2show=None, n_iters=None, dataset_labels=None,
                       model_fn='', max_size_per_class=5):
  with open(logdir + '/params.txt') as fp:
    params = EasyDict(json.load(fp))
  params.network_task = 'classification'
  params.batch_size = 1
  params.one_label_per_model = True
  params.n_walks_per_model = 8
  params.logdir = logdir
  params.seq_len = 200
  params.new_run = 0
  if n_iters is None:
    n_iters = 1

  if "modelnet" in params.logdir:
    model_name = "modelnet40"
  elif "shrec" in params.logdir:
    model_name = "shrec11"
  else:
    print("Unknown model ! exiting...")
    exit(0)

  # choose between "last" and "features"
  layer_to_show = "last"
  layer_to_show = "features"

  utils.print_labels_names_and_indices(model_name)

  params.classes_indices_to_use = cls2show

  pathname_expansion = dataset_files_path

  test_dataset, n_items = dataset.tf_mesh_dataset(params, pathname_expansion, mode=params.network_task,
                                                            max_size_per_class=max_size_per_class)

  if params.net == 'RnnWalkNet':
    print('RnnWalkNet')
    # Gals changes
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)
    # BEFORE
    '''
    dnn_model = dnn_cad_seq.RnnWalkNet(params, params.n_classes, params.net_input_dim, model_fn,
                                        model_must_be_load=True, dump_model_visualization=False)
    '''
    # END
  elif params.net == 'Unsupervised_RnnWalkNet':
    dnn_model = rnn_model.Unsupervised_RnnWalkNet(params, params.n_classes, params.net_input_dim, model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)

  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  n_walks = 0
  model_fns = pred_all = lbl_all = None
  print('Calculating embeddings.')
  tb = time.time()
  for iter in range(n_iters):
    for name_, model_ftrs, labels in test_dataset:
      labels = labels.numpy()
      name = name_.numpy()[0].decode()
      print('  - Got data', name, labels)
      sp = model_ftrs.shape
      ftrs = tf.reshape(model_ftrs, (-1, sp[-2], sp[-1]))
      #print('  - Start Run Pred')
      if layer_to_show == "features":
        predictions_features = dnn_model(ftrs, training=False, classify=False).numpy()
        if 0:
          predictions_features = np.mean(predictions_features, axis=0)[None, :]
          name = [name]
        else:
          labels = np.repeat(labels, predictions_features.shape[0])
          name = [name] * predictions_features.shape[0]
        predictions_labels   = dnn_model(ftrs, training=False, classify=True).numpy()
      elif layer_to_show == "last":
        predictions_labels = dnn_model(ftrs, training=False, classify=True).numpy()
        predictions_features = predictions_labels
        if 0:
          predictions_features = np.mean(predictions_features, axis=0)[None, :]
          name = [name]
        else:
          labels = np.repeat(labels, predictions_features.shape[0])
          name = [name] * predictions_features.shape[0]
      else:
        print("layer_to_show is unvalid ! exiting..")
        exit(-1)
      #print('  - End Run Pred')
      pred_best = predictions_labels.argmax(axis=1)
      acc = np.mean(labels == pred_best)
      print('This batch accuracy:', round(100 * acc, 2))
      if pred_all is None:
        pred_all = predictions_features
        lbl_all = labels
        model_fns = name
      else:
        pred_all = np.vstack((pred_all, predictions_features))
        lbl_all = np.concatenate((lbl_all, labels))
        model_fns += name
      #if pred_all.shape[0] > 1200:
      #  break
      #break
  print('Feature calc time: ', round(time.time() - tb, 2))
  shape_fn_2_id = {}
  shape_fns = np.array(model_fns)
  for cls in np.unique(lbl_all):
    this_cls_idxs = np.where(lbl_all == cls)[0]
    shape_fn_this_cls = shape_fns[this_cls_idxs]
    shape_fn_2_id[cls] = {n: i for i, n in enumerate(list(set(shape_fn_this_cls)))}
  if 0:
    pred_all = pred_all[:1200, :20]
    lbl_all = lbl_all[:1200]
  print('Embedding shape:', pred_all.shape)
  print('t-SNE calculation')
  transformer = TSNE(n_components=2)

  ftrs_tsne = transformer.fit_transform(pred_all)
  print('  t-SNE calc finished')
  shps = '.1234+X|_'
  shps = '.<*>^vspPDd'
  colors = utils.colors_list
  plt.figure()
  i_cls = -1
  for cls, this_cls_shape_fns in shape_fn_2_id.items():
    i_cls += 1
    for i_shape, this_shape_fn in enumerate(this_cls_shape_fns):
      idxs = (shape_fns == this_shape_fn) * (lbl_all == cls)
      if idxs.size:
        clr = colors[i_cls % len(colors)]
        edgecolor = colors[(i_shape + 1) % len(colors)]
        mrkr = shps[i_shape % len(shps)]
        if i_shape == 0:
          label=dataset_labels[cls]
        else:
          label = None
        plt.scatter(ftrs_tsne[idxs, 0], ftrs_tsne[idxs, 1], color=clr, marker=mrkr, #edgecolor=edgecolor, linewidth=3,
                    s=100, label=label)
  plt.legend(fontsize=15)
  plt.axis('off')
  plt.show()

def check_rotation_weak_points():
  if 0:
    logdir = '/home/alonla/mesh_walker/runs_aug_45/0001-06.08.2020..17.40__shrec11_10-10_A/'
    logdir = '/home/alonla/mesh_walker/runs_aug_45/0002-06.08.2020..21.48__shrec11_10-10_B/'
  else:
    logdir = '/home/alonla/mesh_walker/runs_aug_360/0001-06.08.2020..17.40__shrec11_10-10_A/'
    logdir = '/home/alonla/mesh_walker/runs_aug_360_must/0001-23.08.2020..18.03__shrec11_10-10_A/'
    #logdir = '/home/alonla/mesh_walker/runs_aug_360/0002-06.08.2020..21.49__shrec11_10-10_B/'
  print(logdir)
  dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_A/test/*.*'
  #dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_B/test/*.*'

  # Gals changes
  #model_fn = logdir + 'learned_model2keep__00060003.keras'
  model_fn = glob.glob(logdir + "learned_model2keep__*.keras")[-1]
  # End

  dnn_model = Noe
  rot_angles = range(0, 360, 10)
  for axis in [0, 1, 2]:
    accs = []
    stds = []
    dataset.data_augmentation_rotation.test_rotation_axis = axis
    for rot in rot_angles:
      accs_this_rot = []
      for _ in range(5):
        acc, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder, n_walks_per_model=16,
                                            dnn_model=dnn_model, labels=dataset_prepare.shrec11_labels,
                                            model_fn=model_fn, verbose_level=0, data_augmentation={'rotation': rot})
        accs_this_rot.append(acc[0])
      accs.append(np.mean(accs_this_rot))
      stds.append(np.std(accs_this_rot))
      print(rot, accs, stds)
    plt.errorbar(rot_angles, accs, yerr=stds)
    plt.xlabel('Rotation [degrees]')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS rotation angles, axis = ' + str(axis))
  plt.legend(['axis=0', 'axis=1', 'axis=2'])
  plt.suptitle('/'.join(logdir.split('/')[-3:-1]))
  plt.show()

import sys

if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu(1)

  data_path = sys.argv[1]
  trained_model_path = sys.argv[2]
  dataset_name = sys.argv[3]
  print(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])
  accuracy_or_tsne = sys.argv[4]
  t_SNE = True if accuracy_or_tsne == "tsne" else False
  modelnet40 = True if dataset_name == "modelnet40" else False
  shrec11 = True if dataset_name == "shrec11" else False

  if len(sys.argv) == 6:
    config_path = sys.argv[6]
    config = get_config(config_path)
  else:
    config = None


  if trained_model_path == 'latest':
    trained_models_names = glob.glob("/home/galye/mesh_walker/runs_aug_360_must/*"+dataset_name+"*")
    trained_models_names.sort(key=os.path.getctime)
    trained_model_path = trained_models_names[-1]

  # for SHREC11
  if shrec11:
    #data_to_use = "10-10_A/test/*.npz"
    data_to_use = "16-04_a/test/*.npz"
    #data_to_use = "16-4_B/test/*.npz"
    #data_to_use = "16-4_C/test/*.npz"
    #data_to_use = "*test*.npz"

  # for modelnet40
  if modelnet40:
    data_to_use = "test*.npz"

  dataset_files_path = data_path + data_to_use
  logdir = trained_model_path
  print(logdir)
  print(glob.glob(logdir + "learned_model2keep__*.keras"))
  model_fn = glob.glob(logdir + "learned_model2keep__*.keras")[-1]

  if 0:
    check_rotation_weak_points()
    exit(0)

  #test_dataset()
  iter2use = 'last'
  classes_indices_to_use = None
  model_fn = None


  if t_SNE:   # t-SNE
    if shrec11:   # Use shrec model
      dataset_labels = dataset_prepare.shrec11_labels

      cls2show = range(30)[0:6]
      #cls2show = range(30)[-5:-1]
    elif modelnet40:   # Use ModelNet
      dataset_labels = dataset_prepare.model_net_labels

      cls2show = range(30)[-9:-1]
    else:
      print("No module specified !! exiting..")
      exit(0)

    show_features_tsne(dataset_files_path=dataset_files_path, logdir=logdir, n_iters=1, cls2show=cls2show, dataset_labels=dataset_labels,
                       model_fn=model_fn, max_size_per_class=3)
  elif modelnet40 : # ModelNet
    dataset_folder = dataset_files_path

    min_max_faces2use = [000, 4000]
    if 1:
      print(utils.color.BOLD + utils.color.BLUE + '6 classes used for fast run', utils.color.END)
      classes_indices_to_use = [39, 38, 37, 31, 28, 19]
    #classes_indices_to_use = range(30)
    accs, _ = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder,
                                 labels=dataset_prepare.model_net_labels, iter2use=iter2use,
                                 classes_indices_to_use=classes_indices_to_use,
                                 min_max_faces2use=min_max_faces2use, model_fn=model_fn, n_walks_per_model=16 * 4)
    print('Overall Accuracy / Mean Accuracy:', np.round(np.array(accs) * 100, 2))
  elif shrec11:  # SHREC11
    # Gals changes
    dataset_path_a = '16-04_a/test/*.npz'
    dataset_path_b = '16-4_B/test/*.npz'
    dataset_path_c = '16-4_C/test/*.npz'
    # END

    acc_all = []
    for curr_dataset_path in [dataset_path_a, dataset_path_b, dataset_path_c]:
      print("\n\nNew Iteration !")
      dataset_path = data_path + curr_dataset_path
      print("dataset_path=", dataset_path, "\n\n")

      if config is not None:
        if config['trained_only_2_classes'] == True:
          # params.classes_indices_to_use = (params.classes_indices_to_use)[0:2]
          first_label = min(config['source_label'], config['target_label'])
          sec_label = max(config['source_label'], config['target_label'])
          cls2show = [first_label, sec_label]
      else:
        cls2show = None
      cls2show = None #[15, 25]
      acc, _ = calc_accuracy_test(logdir=logdir,
                                  dataset_folder=dataset_path, classes_indices_to_use=cls2show, labels=dataset_prepare.shrec11_labels, iter2use=iter2use,
                                  model_fn=model_fn, n_walks_per_model=8)
      acc_all.append(acc)
      continue
    print(acc_all)
    print(np.mean(acc_all))
  elif 1: # Look for Rotation weekpoints
    if 0:
      logdir = '/home/alonla/mesh_walker/runs_aug_360/0004-07.08.2020..06.05__shrec11_16-04_A'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/16-04_a/test/*.*'
    else:
      if 0:
        logdir = '/home/alonla/mesh_walker/runs_aug_45/0001-06.08.2020..17.40__shrec11_10-10_A/'
      else:
        logdir = '/home/alonla/mesh_walker/runs_aug_360/0001-06.08.2020..17.40__shrec11_10-10_A/'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_A/test/*.*'
    model_fn = None # logdir + 'learned_model2keep__00200010.keras'
    tb = time.time()
    dnn_model = None
    accs = []
    rot_angles = range(0, 180, 10)
    for rot in rot_angles:
      acc, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder, n_walks_per_model=8,
                                          dnn_model=dnn_model, labels=dataset_prepare.shrec11_labels, iter2use=str(iter2use),
                                          model_fn=model_fn, verbose_level=0, data_augmentation={'rotation': rot})
      accs.append(acc[0])
      print(rot, accs)
    plt.plot(rot_angles, accs)
    plt.xlabel('Rotation [degrees]')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS rotation angles')
    plt.show()
  elif 1: # Check STD vs Number of walks
    if 1:
      logdir = '/home/alonla/mesh_walker/runs_aug_360/0004-07.08.2020..06.05__shrec11_16-04_A'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/16-04_a/test/*.*'
    else:
      logdir = '/home/alonla/mesh_walker/runs_aug_360/0001-06.08.2020..17.40__shrec11_10-10_A'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_A/test/*.*'
    model_fn = None # logdir + 'learned_model2keep__00200010.keras'
    tb = time.time()
    dnn_model = None
    for n_walks in [1, 2, 4, 8, 16, 32]:
      accs = []
      for _ in range(6):
        acc, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder, n_walks_per_model=n_walks,
                                            dnn_model=dnn_model, labels=dataset_prepare.shrec11_labels, iter2use=str(iter2use),
                                            model_fn=model_fn, verbose_level=0)
        accs.append(acc[0])
        #print('Run Time: ', time.time() - tb, ' ; Accuracy:', acc)
      print(n_walks, accs, 'STD:', np.std(accs))
  elif 1:
    r = 'cubes2keep/0016-03.04.2020..08.59__Cubes_NewPrms'
    logdir = os.path.expanduser('~') + '/mesh_walker/mesh_learning/' + r + '/'
    model_fn = logdir + 'learned_model2keep__00160080.keras'
    # Gals changes
    #model_fn = logdir + 'learned_model2keep__00160080.keras'
    model_fn = glob.glob(logdir + "learned_model2keep__*.keras")[-1]
    # End
    n_walks_to_check = [1, 2, 4, 8, 16, 32]
    acc_all = []
    for n_walks_per_model in n_walks_to_check:
      acc = calc_accuracy_test(logdir=logdir, model_fn=model_fn, target_n_faces=[1000],
                         from_cach_dataset='cubes/test*.npz', labels=dataset_prepare.cubes_labels, n_walks_per_model=n_walks_per_model)
      acc_all.append(acc[0][0])
    print('--------------------------------')
    print(acc_all)
    #[0.7708649468892261, 0.8482549317147192, 0.921092564491654, 0.952959028831563, 0.9742033383915023, 0.9787556904400607]
  #calc_accuracy_per_seq_len()
  #calc_accuracy_per_n_faces()
  #features_analysis()
