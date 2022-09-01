import argparse
import utils

#get hyper params from yaml
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = utils.get_config(opts.config)

if config['gpu_to_use'] >= 0:
  utils.set_single_gpu(config['gpu_to_use'])

import os, shutil, time
from easydict import EasyDict
import json
import cv2
import numpy as np
import tensorflow as tf
import pyvista as pv
import pylab as plt
import rnn_model
import utils
import dataset
import dataset_prepare


def dump_mesh(mesh_data, path, cpos, iter, x_server_exists):
  """
  Saves a picture of the mesh
  """
  if not os.path.isdir(path):
    os.makedirs(path)
  if x_server_exists:
    window_size = [512, 512]
    p = pv.Plotter(off_screen=1, window_size=(int(window_size[0]), int(window_size[1])))
    faces = np.hstack([[3] + f.tolist() for f in mesh_data['faces']])
    surf = pv.PolyData(mesh_data['vertices'], faces)
    p.add_mesh(surf, show_edges=False, color=None)
    p.camera_position = cpos
    p.set_background("#AAAAAA", top="White")
    rendered = p.screenshot()
    p.close()
    img = rendered.copy()
    my_text = str(iter)
    cv2.putText(img, my_text, (img.shape[1] - 100, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
               color=(0, 255, 255), thickness=2)
    cv2.imwrite(path + '/img_' + str(dump_mesh.i).zfill(5) + '.jpg', img)
  dump_mesh.i += 1
dump_mesh.i = 0


def deform_add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dump_model=True):
  """
  Saves the new attacked mesh model
  """
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels_fuzzy':
        m[field] = np.zeros((0,))
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query':
        dataset_prepare.prepare_edges_and_kdtree(m)

  if dump_model:
    np.savez(out_fn, **m)


def get_res_path(config, id = -1, labels = None):
  """
  Sets the result path in which the mesh and its pictures are saved
  """
  if labels is None:
    exit("Error, no shrec labels")

  net_name = config['trained_model'].split('/')[-1]
  if len(net_name) > 0:
    res_path = '../attacks/' + net_name +'/' + labels[config['source_label']]
  else:
    res_path = '../attacks/' + labels[config['source_label']]
  if id != -1:
    res_path+= '_'+str(id)
  return res_path, net_name


def plot_preditions(params, dnn_model, config, mesh_data, result_path, num_iter, x_axis, source_pred_list):
  """
  Saves as image a graph of the prediction of the network on the changed mesh
  """
  params.n_walks_per_model = 16
  features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
  ftrs = tf.cast(features[:, :, :3], tf.float32)
  eight_pred = dnn_model(ftrs, classify=True, training=False)
  sum_pred = tf.reduce_sum(eight_pred, 0)
  print("source_label number ", config['source_label'],  " over " + str(params.n_walks_per_model) + " runs is: ", (sum_pred.numpy())[config['source_label']] / params.n_walks_per_model)
  source_pred_list.append((sum_pred.numpy())[config['source_label']] / params.n_walks_per_model)
  params.n_walks_per_model = 8

  if not os.path.isdir(result_path + '/plots/'):
    os.makedirs(result_path + '/plots/')
  # plot the predictions
  x_axis.append(num_iter)

  plt.plot(x_axis, source_pred_list)
  plt.title(str(config['source_label']) + ": source pred")
  plt.savefig(result_path + '/plots/' + 'source_pred.png')
  plt.close()
  return


def define_network_and_its_params(config=None):
  """
  Defining the parameters of the network, called params, and loads the trained model, called dnn_model
  """
  with open(config['trained_model'] + '/params.txt') as fp:
    params = EasyDict(json.load(fp))
  model_fn = tf.train.latest_checkpoint(config['trained_model'])
  # Define network parameters
  params.batch_size = 1
  params.seq_len = config['walk_len']
  params.n_walks_per_model = 8
  params.set_seq_len_by_n_faces = False
  params.data_augmentaion_vertices_functions = []
  params.label_per_step = False
  params.n_target_vrt_to_norm_walk = 0
  params.net_input += ['vertex_indices']
  dataset.setup_features_params(params, params)
  dataset.mesh_data_to_walk_features.SET_SEED_WALK = False

  dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, 3, model_fn,
                                   model_must_be_load=True, dump_model_visualization=False)

  return params, dnn_model


def attack_single_mesh(config = None, source_mesh = None, id = -1, labels = None):
  if labels is None or config is None:
    exit(-1)

  # Defining network's parameters and model
  network_params, network_dnn_model = define_network_and_its_params(config=config)

  # Defining output path
  result_path, net_name = get_res_path(config=config, id=id, labels=labels)
  print("source label: ", config['source_label'], " output dir: ", result_path)
  if os.path.isdir(result_path) and config['use_last'] is False:
      shutil.rmtree(result_path)

  # Defining original mesh data - Either use the last saved in the folder or the original one
  orig_mesh_data_path = source_mesh
  if config['use_last'] is True:
    if os.path.exists(result_path+'/'+'last_model.npz'): # A previous model exists
      orig_mesh_data_path = result_path + '/last_model.npz'
    elif os.path.exists(source_mesh[0:-4] + '_attacked.npz'): # A previous attacked model exists
      orig_mesh_data_path = source_mesh[0:-4] + '_attacked.npz'

  orig_mesh_data = np.load(orig_mesh_data_path, encoding='latin1', allow_pickle=True)
  mesh_data = {k: v for k, v in orig_mesh_data.items()}

  # Defining parameters that keep track of the changes
  loss = []
  cpos = None
  last_dev_res = 0
  last_plt_res = 0
  fields_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'labels_fuzzy']
  source_pred_list = []
  x_axis = []
  vertices_counter = np.ones(mesh_data['vertices'].shape)
  vertices_gradient_change_sum = np.zeros(mesh_data['vertices'].shape)
  num_times_wrong_classification = 0

  # Defining the attack
  kl_divergence_loss = tf.keras.losses.KLDivergence()
  w = config['attacking_weight']
  if config['dataset'] == 'SHREC11':
    one_hot_original_label_vetor = tf.one_hot(config['source_label'], 30)
  elif config['dataset'] == 'MODELNET40':
    one_hot_original_label_vetor = tf.one_hot(config['source_label'], 40)
  else:
    one_hot_original_label_vetor = config['source_label']


  # Time measurment parameter
  start_time_100_iters = time.time()

  for num_iter in range(config['max_iter']):
    # Extract features and labels
    features, labels = dataset.mesh_data_to_walk_features(mesh_data, network_params)
    ftrs = tf.cast(features[:, :, :3], tf.float32) # The walks features
    v_indices = features[0, :, 3].astype(np.int) # the vertices indices of the walk

    with tf.GradientTape() as tape:
      tape.watch(ftrs)
      pred = network_dnn_model(ftrs, classify=True, training=False)

      # Produce the attack
      attack = -1 * w * kl_divergence_loss(one_hot_original_label_vetor, pred)

    # Check the prediction of the network
    pred = tf.reduce_sum(pred, 0)
    pred /= network_params.n_walks_per_model
    source_pred_brfore_attack = (pred.numpy())[config['source_label']]

    gradients = tape.gradient(attack, ftrs)
    ftrs_after_attack_update = ftrs + gradients

    new_pred = network_dnn_model(ftrs_after_attack_update, classify=True, training=False)
    new_pred = tf.reduce_sum(new_pred, 0)
    new_pred /= network_params.n_walks_per_model

    # Check to see that we didn't update too much
    # We don't want the change to be too big, as it may result in intersections.
    # And so, we check to see if the change caused us to get closer to the target by more than 0.01.
    # If so, we will divide the change so it won't change more than 0.01
    source_pred_after_attack = (new_pred.numpy())[config['source_label']]
    source_pred_abs_diff = abs(source_pred_brfore_attack - source_pred_after_attack)

    if source_pred_abs_diff > config['max_label_diff'] :
      # We update the gradients accordingly
        ratio = config['max_label_diff'] / source_pred_abs_diff
        gradients = gradients * ratio

    print("iter:", num_iter, " attack:", attack.numpy(), " w:", w, " source prec:", (pred.numpy())[config['source_label']],
          " max label:", np.argmax(pred))

    if np.argmax(pred) != config['source_label']:
      num_times_wrong_classification += 1
    else:
      num_times_wrong_classification = 0

    loss.append(attack.numpy())
    vertices_counter[v_indices] += 1
    vertices_gradient_change_sum[v_indices] += gradients[0].numpy()

    # Updating the mesh itself
    change = vertices_gradient_change_sum/vertices_counter
    mesh_data['vertices'] += change

    # If we got the wrong classification 10 times straight
    if num_times_wrong_classification > 10 * config['num_walks_per_iter']:
      if num_iter < 15:
        print("\n\nExiting.. Wrong model was loaded / Wrong labels were compared\n\n\n")
        return num_iter
      path = source_mesh if source_mesh is not None else None
      if result_path.__contains__('_meshCNN'):
        deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                       out_fn=path[0:-4] + 'meshCNN_attacked.npz')
      else:
        deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                       out_fn=path[0:-4] + '_attacked.npz')
      return num_iter

    # Saving pictures of the models
    if num_iter % 100 == 0:
      total_time_100_iters = time.time() - start_time_100_iters
      start_time_100_iters = time.time()

      preds_to_print_str = ''
      print('\n' + str(net_name) + '\n' + preds_to_print_str +'\n'
            + 'Time took for 100 iters: '+ str(total_time_100_iters) +'\n')

    curr_save_image_iter = num_iter - (num_iter % config['image_save_iter'])
    if curr_save_image_iter / config['image_save_iter'] >= last_dev_res + 1 or num_iter == 0:
      print(result_path)
      cpos = dump_mesh(mesh_data, result_path, cpos, num_iter, config['x_server_exists'])
      last_dev_res = num_iter / config['image_save_iter']
      deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed, out_fn=result_path + '/last_model.npz')#"+ str(num_iter))


    curr_plot_iter = num_iter - (num_iter % config['plot_iter'])
    if curr_plot_iter / config['plot_iter'] >= last_plt_res + 1 or num_iter == 0:
      plot_preditions(network_params, network_dnn_model, config, mesh_data, result_path, num_iter, x_axis, source_pred_list)
      last_plt_res = num_iter / config['plot_iter']


    if config['show_model_every'] > 0 and num_iter % config['show_model_every'] == 0 and num_iter > 0:
        plt.plot(loss)
        plt.show()
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])

  #res_path = config['result_path']
  cmd = f'ffmpeg -framerate 24 -i {result_path}img_%05d.jpg {result_path}mesh_reconstruction.mp4'
  os.system(cmd)
  return


def main():
  np.random.seed(0)
  utils.config_gpu(1, config['gpu_to_use'])
  attack_single_mesh(config=config, labels=dataset_prepare.model_net_labels)

  return 0

if __name__ == '__main__':
  main()
