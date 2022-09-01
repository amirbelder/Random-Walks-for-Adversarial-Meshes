#from utils import visualize_npz
import csv, glob, os, json
from easydict import EasyDict
from dataset import load_model_from_npz
import rnn_model, dataset
import tensorflow as tf
import numpy as np
import utils
from copy import deepcopy
import dataset_prepare


def show_walk(model, features, one_walk=False, weights=False, pred_cats=None, pred_val=None, labels=None, save_name=''):
  if weights is not False:
    walks = features[:,:,-1]
    for i, walk in enumerate(walks):
      name = '_rank_{}_weight_{:02d}%'.format(i+1, int(weights[i]*100))
      if labels:
        pred_label=labels[pred_cats[i]]
        pred_score=pred_val[i]
        title='{}: {:2.3f}\n weight: {:2.3f}'.format(pred_label, pred_score, weights[i])
      cur_color= 'cyan' if i < len(walks) //2 else 'magenta'  #'cadetblue'  #label2color[gt]
      rendered = utils.visualize_model(dataset.norm_model(model['vertices'], return_val=True),
                                       model['faces'], walk=[list(walk.astype(np.int32))],
                                       jump_indicator=features[i,:,-2],
                                       show_edges=True,
                                       opacity=0.5,
                                       all_colors=cur_color,
                                       edge_color_a='black',
                                       off_screen=True, save_fn=os.path.join(save_name, name), title=title)
    # TODO: save rendered to file
  else:
    for wi in range(features.shape[0]):
      walk = features[wi, :, -1].astype(np.int)
      jumps = features[wi, :, -2].astype(np.bool)
      utils.visualize_model_walk(model['vertices'], model['faces'], walk, jumps)
      if one_walk:
        break


def load_params(logdir):

  # ================ Loading parameters ============== #
  if not os.path.exists(logdir):
    raise(ValueError, '{} is not a folder'.format(logdir))
  try:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    params.net_input += ['vertex_indices']
    params.batch_size = 1
  except:
    raise(ValueError, 'Could not load params.txt from logdir')
  # ================================================== #
  return params

def load_model(params, model_fn=None):
  # ================ Loading architecture ============== #
  if not model_fn:
    model_fn = glob.glob(params.logdir + '/learned_model2keep__*.keras')
    model_fn.sort()
    model_fn = model_fn[-1]
  if params.net == 'HierTransformer':
    import attention_model
    dnn_model = attention_model.WalkHierTransformer(**params.net_params, params=params,
                                                    model_fn=model_fn, model_must_be_load=True)
  else:
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - 1,
                                     model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)
  return dnn_model


def predict_and_plot(models, logdir, logdir2=None):
  models.sort()
  params = load_params(logdir)
    # load all npzs in folder of filelist
  test_folder = os.path.dirname(params.datasets2use['test'][0])
  list_per_model = [glob.glob(test_folder + '/*{}*'.format(x)) for x in models]
  npzs = [item for sublist in list_per_model for item in sublist]
  test_dataset, n_models_to_test = dataset.tf_mesh_dataset(params, None, mode=params.network_task,
                                                           shuffle_size=0, permute_file_names=False, must_run_on_all=True,
                                                             filenames=npzs)
  dnn_model = load_model(params)
  if logdir2 is not None:
    params_2 = load_params(logdir2)
    dnn_model_2 = load_model(params_2)

  for i, data in enumerate(test_dataset):
    name, ftrs, gt = data
    ftrs = tf.reshape(ftrs, ftrs.shape[1:])
    ftr2use = ftrs[:, :, :-1].numpy()
    gt = gt.numpy()[0]
    model_fn = name.numpy()[0].decode()
    # forward pass through the model
    if params.cross_walk_attn:
      predictions_, weights, per_walk_predictions_ = [x.numpy() for x in dnn_model(ftr2use, classify='visualize', training=False)]
    else:
      predictions_ = dnn_model(ftr2use, classify=True, training=False).numpy()
    if logdir2 is not None:
      if params_2.cross_walk_attn:
        predictions_2, weights2, per_walk_predictions_2 = [x.numpy() for x in
                                                           dnn_model_2(ftr2use, classify='visualize', training=False)]
      else:
        predictions_2 = dnn_model_2(ftr2use, classify=True, training=False).numpy()
    if params.cross_walk_attn:
      # show only weights of walks where Alon's model failed
      # Showing walks with weighted attention - which walks recieved higher weights
      weights = weights.squeeze()
      if len(weights.shape) > 1:
        weights = np.sum(weights,axis=1)
        weights /= np.sum(weights)
      sorted_weights = np.argsort(weights)[::-1]
      sorted_features = ftrs.numpy()[sorted_weights]
      model = dataset.load_model_from_npz(model_fn)
      print(model_fn)
      print('nv: ', model['vertices'].shape[0])
      per_walk_pred = np.argmax(per_walk_predictions_[sorted_weights], axis=1)
      per_walk_scores = [per_walk_predictions_[i, j] for i,j in zip(sorted_weights, per_walk_pred)]
      # if 'modelnet40' in any(params.datasets2use.values()):
      labels = dataset_prepare.model_net_labels
      save_dir=os.path.join(params.logdir, 'plots', model_fn.split('/')[-1].split('.')[0])
      show_walk(model, sorted_features, weights=weights[sorted_weights],
                pred_cats=per_walk_pred, pred_val=per_walk_scores, labels=labels,
                save_name=save_dir)
      create_gif_from_preds(save_dir, title=model_fn.split('/')[-1].split('.')[0])
      with open(save_dir + '/pred.txt', 'w') as f:
        f.write('Predicted: {}'.format(labels[np.argmax(predictions_)]))
      # TODO: write prediction_2 scores to see the difference in prediction




def create_gif_from_preds(path, title=''):
  files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.png')]
  if not len(files):
    print('Did not find any .png images in {}'.format(path))
  sorted_indices = np.argsort([int(x.split('_')[-3]) for x in files])
  files = [files[i] for i in sorted_indices]
  from PIL import Image
  ims = [Image.open(x) for x in files]
  ims[0].save(os.path.join(path, '{}_animated.gif'.format(title)), save_all=True, append_images=ims[1:], duration=1000)


def compare_attention():
  attn_csv = '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/False_preds_9250.csv'
  orig_csv = '/home/ran/mesh_walker/runs_compare/0095-23.11.2020..15.31__modelnet/False_preds_9222.csv'
  attn_models = []
  orig_models = []
  with open(attn_csv, 'r') as f:
    for row in f:
      attn_models.append(row.split(',')[0])
  with open(orig_csv, 'r') as f:
    for row in f:
      orig_models.append(row.split(',')[0])

  fixed = [x for x in orig_models if x not in attn_models]
  ruined = [x for x in attn_models if x not in orig_models]


  first10_each_class_fp = [glob.glob('/home/ran/mesh_walker/datasets/modelnet40_walker/test_{}*'.format(c)) for c in dataset_prepare.model_net_labels]
  first10_each_class = ['_'.join(x.split('_')[3:5]) for y in first10_each_class_fp for x in y[:10] if len(y[0].split('_')) ==8]
  first10_each_class += ['_'.join(x.split('_')[3:6]) for y in first10_each_class_fp for x in y[:10] if len(y[0].split('_')) == 9]

  # predict_and_plot(fixed, '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/')
  # predict_and_plot(orig_models, '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/')
  predict_and_plot(first10_each_class, '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/')
  # predict_and_plot(fixed, '/home/ran/mesh_walker/runs_compare/0095-23.11.2020..15.31__modelnet')
  # attn_corrected = ['bed_0558', 'bookshelf_0633', 'bottle_0416', ]


if __name__ == '__main__':
  np.random.seed(4)
  compare_attention()
  # plot_attention()
  # npz_path = '/home/ran/mesh_walker/datasets/modelnet40_retrieval_split_0/train_desk_0018_000_simplified_to_4000.npz'
  # visualize_npz(npz_path)
