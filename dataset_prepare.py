import glob, os, shutil, sys, json
from pathlib import Path

import pylab as plt
import trimesh
import open3d
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import re

import utils


FIX_BAD_ANNOTATION_HUMAN_15 = 0

# Labels for all datasets_processed
# -----------------------
sigg17_part_labels = ['---', 'head', 'hand', 'lower-arm', 'upper-arm', 'body', 'upper-lag', 'lower-leg', 'foot']
sigg17_shape2label = {v: k for k, v in enumerate(sigg17_part_labels)}

model_net_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]
model_net_shape2label = {v: k for k, v in enumerate(model_net_labels)}

cubes_labels = [
  'apple',  'bat',      'bell',     'brick',      'camel',
  'car',    'carriage', 'chopper',  'elephant',   'fork',
  'guitar', 'hammer',   'heart',    'horseshoe',  'key',
  'lmfish', 'octopus',  'shoe',     'spoon',      'tree',
  'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}

coseg_labels = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}


def calc_mesh_area(mesh):
  t_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], process=False)
  mesh['area_faces'] = t_mesh.area_faces
  mesh['area_vertices'] = np.zeros((mesh['vertices'].shape[0]))
  for f_index, f in enumerate(mesh['faces']):
    for v in f:
      mesh['area_vertices'][v] += mesh['area_faces'][f_index] / f.size


def calc_vertex_labels_from_face_labels(mesh, face_labels):
  vertices = mesh['vertices']
  faces = mesh['faces']
  all_vetrex_labels = [[] for _ in range(vertices.shape[0])]
  vertex_labels = -np.ones((vertices.shape[0],), dtype=np.int)
  n_classes = int(np.max(face_labels))
  assert np.min(face_labels) == 1 # min label is 1, for compatibility to human_seg labels representation
  v_labels_fuzzy = -np.ones((vertices.shape[0], n_classes))
  for i in range(faces.shape[0]):
    label = face_labels[i]
    for f in faces[i]:
      all_vetrex_labels[f].append(label)
  for i in range(vertices.shape[0]):
    counts = np.bincount(all_vetrex_labels[i])
    vertex_labels[i] = np.argmax(counts)
    v_labels_fuzzy[i] = np.zeros((1, n_classes))
    for j in all_vetrex_labels[i]:
      v_labels_fuzzy[i, int(j) - 1] += 1 / len(all_vetrex_labels[i])
  return vertex_labels, v_labels_fuzzy


def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])]
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']])
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])


def add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' or field == 'edges':
        prepare_edges_and_kdtree(m)

  if dump_model:
    np.savez(out_fn, **m)

  return m


def get_sig17_seg_bm_labels(mesh, file, seg_path):
  # Finding the best match file name .. :
  in_to_check = file.replace('obj', 'txt')
  in_to_check = in_to_check.replace('off', 'txt')
  in_to_check = in_to_check.replace('_fix_orientation', '')
  if in_to_check.find('MIT_animation') != -1 and in_to_check.split('/')[-1].startswith('mesh_'):
    in_to_check = '/'.join(in_to_check.split('/')[:-2])
    in_to_check = in_to_check.replace('MIT_animation/meshes_', 'mit/mit_')
    in_to_check += '.txt'
  elif in_to_check.find('/scape/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/scape.txt'
  elif in_to_check.find('/faust/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/faust.txt'

  seg_full_fn = []
  for fn in Path(seg_path).rglob('*.txt'):
    tmp = str(fn)
    tmp = tmp.replace('/segs/', '/meshes/')
    tmp = tmp.replace('_full', '')
    tmp = tmp.replace('shrec_', '')
    tmp = tmp.replace('_corrected', '')
    if tmp == in_to_check:
      seg_full_fn.append(str(fn))
  if len(seg_full_fn) == 1:
    seg_full_fn = seg_full_fn[0]
  else:
    print('\nin_to_check', in_to_check)
    print('tmp', tmp)
    raise Exception('!!')
  face_labels = np.loadtxt(seg_full_fn)

  if FIX_BAD_ANNOTATION_HUMAN_15 and file.endswith('test/shrec/15.off'):
    face_center = []
    for f in mesh.faces:
      face_center.append(np.mean(mesh.vertices[f, :], axis=0))
    face_center = np.array(face_center)
    idxs = (face_labels == 6) * (face_center[:, 0] < 0) * (face_center[:, 1] < -0.4)
    face_labels[idxs] = 7
    np.savetxt(seg_full_fn + '.fixed.txt', face_labels.astype(np.int))

  return face_labels


def get_labels(dataset_name, mesh, file, fn2labels_map=None):
  v_labels_fuzzy = np.zeros((0,))
  if dataset_name == 'faust':
    face_labels = np.load('faust_labels/faust_part_segmentation.npy').astype(np.int)
    vertex_labels, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh, face_labels)
    model_label = np.zeros((0,))
    return model_label, vertex_labels, v_labels_fuzzy
  elif dataset_name.startswith('coseg') or dataset_name == 'human_seg_from_meshcnn':
    labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2] + '.eseg'
    e_labels = np.loadtxt(labels_fn)
    v_labels = [[] for _ in range(mesh['vertices'].shape[0])]
    faces = mesh['faces']

    fuzzy_labels_fn = '/'.join(file.split('/')[:-2]) + '/sseg/' + file.split('/')[-1].split('.')[-2] + '.seseg'
    seseg_labels = np.loadtxt(fuzzy_labels_fn)
    v_labels_fuzzy = np.zeros((mesh['vertices'].shape[0], seseg_labels.shape[1]))

    edge2key = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
      faces_edges = []
      for i in range(3):
        cur_edge = (face[i], face[(i + 1) % 3])
        faces_edges.append(cur_edge)
      for idx, edge in enumerate(faces_edges):
        edge = tuple(sorted(list(edge)))
        faces_edges[idx] = edge
        if edge not in edge2key:
          v_labels_fuzzy[edge[0]] += seseg_labels[edges_count]
          v_labels_fuzzy[edge[1]] += seseg_labels[edges_count]

          edge2key[edge] = edges_count
          edges.append(list(edge))
          v_labels[edge[0]].append(e_labels[edges_count])
          v_labels[edge[1]].append(e_labels[edges_count])
          edges_count += 1

    assert np.max(np.sum(v_labels_fuzzy != 0, axis=1)) <= 3, 'Number of non-zero labels must not acceeds 3!'

    vertex_labels = []
    for l in v_labels:
      l2add = np.argmax(np.bincount(l))
      vertex_labels.append(l2add)
    vertex_labels = np.array(vertex_labels)
    model_label = np.zeros((0,))

    return model_label, vertex_labels, v_labels_fuzzy
  else:
    tmp = file.split('/')[-1]
    model_name = '_'.join(tmp.split('_')[:-1])
    if dataset_name.lower().startswith('modelnet'):
      model_label = model_net_shape2label[model_name]
    elif dataset_name.lower().startswith('cubes'):
      model_label = cubes_shape2label[model_name]
    elif dataset_name.lower().startswith('shrec11'):
      model_name = file.split('/')[-3]
      if fn2labels_map is None:
        model_label = shrec11_shape2label[model_name]
      else:
        file_index = int(file.split('.')[-2].split('T')[-1])
        model_label = fn2labels_map[file_index]
    else:
      raise Exception('Cannot find labels for the dataset')
    vertex_labels = np.zeros((0,))
    return model_label, vertex_labels, v_labels_fuzzy

def fix_labels_by_dist(vertices, orig_vertices, labels_orig):
  labels = -np.ones((vertices.shape[0], ))

  for i, vertex in enumerate(vertices):
    d = np.linalg.norm(vertex - orig_vertices, axis=1)
    orig_idx = np.argmin(d)
    labels[i] = labels_orig[orig_idx]

  return labels

def get_faces_belong_to_vertices(vertices, faces):
  faces_belong = []
  for face in faces:
    used = np.any([v in vertices for v in face])
    if used:
      faces_belong.append(face)
  return np.array(faces_belong)


def remesh(mesh_orig, target_n_faces, add_labels=False, labels_orig=None):
  labels = labels_orig
  if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
    mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
    str_to_add = '_simplified_to_' + str(target_n_faces)
    mesh = mesh.remove_unreferenced_vertices()
    if add_labels and labels_orig.size:
      labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
  else:
    mesh = mesh_orig
    str_to_add = '_not_changed_' + str(np.asarray(mesh_orig.triangles).shape[0])

  return mesh, labels, str_to_add


def load_meshes(model_fns):
  f_names = glob.glob(model_fns)
  joint_mesh_vertices = []
  joint_mesh_faces = []
  for fn in f_names:
    mesh_ = trimesh.load_mesh(fn)
    vertex_offset = len(joint_mesh_vertices)
    joint_mesh_vertices += mesh_.vertices.tolist()
    faces = mesh_.faces + vertex_offset
    joint_mesh_faces += faces.tolist()

  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(joint_mesh_vertices)
  mesh.triangles = open3d.utility.Vector3iVector(joint_mesh_faces)

  return mesh


def load_mesh(model_fn, classification=True):
  if 1:  # To load and clean up mesh - "remove vertices that share position"
    if classification:
      mesh_ = trimesh.load_mesh(model_fn, process=True)
      mesh_.remove_duplicate_faces()
    else:
      mesh_ = trimesh.load_mesh(model_fn, process=False)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)
  else:
    mesh = open3d.io.read_triangle_mesh(model_fn)

  return mesh

def create_tmp_dataset(model_fn, p_out, n_target_faces):
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  mesh_orig = load_mesh(model_fn)
  mesh, labels, str_to_add = remesh(mesh_orig, n_target_faces)
  labels = np.zeros((np.asarray(mesh.vertices).shape[0],), dtype=np.int16)
  mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': 0, 'labels': labels})
  out_fn = p_out + '/tmp'
  add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, 'tmp')


#Gloabal variable that holds all the meshCNN and PD MESHNet vertices and faces npz's
#all_mesh_cnn_files = os.listdir(os.path.expanduser('~') + '/mesh_cnn_faces_and_vertices_npz')
def change_faces_and_vertices(mesh_data, file_name: str):
  name = (re.split(pattern=' |/', string=file_name))[-1]
  name = (re.split(pattern=' |\.', string=name))[0]
  path_to_meshCNN_file = [file for file in all_mesh_cnn_files if str(file).__contains__(name+'_')]
  mesh_cnn_raw_data = np.load(os.path.expanduser('~') + '/mesh_cnn_faces_and_vertices_npz/' + path_to_meshCNN_file[0])
  mesh_cnn_data = {k: v for k, v in mesh_cnn_raw_data.items()}
  mesh_data['vertices'] = mesh_cnn_data['vertices']
  mesh_data['faces'] = mesh_cnn_data['faces']
  mesh_data['label'] = mesh_cnn_data['label']
  return mesh_data


#Gloabal variable that holds all the copycat's npzs
#all_copycat_shrec11_files = os.listdir('datasets_processed/copycat_shrec11/')
def change_to_copycat_walker(mesh_data, file_name: str):
  name = (re.split(pattern=' |/', string=file_name))[-1]
  name = (re.split(pattern=' |\.', string=name))[0]
  path_to_meshCNN_file = [file for file in all_copycat_shrec11_files if str(file).__contains__(name+'_')]
  mesh_cnn_raw_data = np.load('datasets_processed/copycat_shrec11/' + path_to_meshCNN_file[0], encoding='latin1', allow_pickle=True)
  mesh_cnn_data = {k: v for k, v in mesh_cnn_raw_data.items()}
  mesh_data['label'] = mesh_cnn_data['label']
  return mesh_data


def prepare_directory_from_scratch(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                   size_limit=np.inf, fn_prefix='', verbose=True, classification=True, adversrial_data = None):
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  fileds_needed += ['labels_fuzzy']

  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None
      label, labels_orig, v_labels_fuzzy = get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      if adversrial_data == 'mesh_CNN' or adversrial_data == 'PD_MeshNet':
        change_faces_and_vertices(mesh_data, str(file))
      if adversrial_data == 'walker_copycat':
        change_to_copycat_walker(mesh_data, str(file))
      m = add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)

# ------------------------------------------------------- #

def prepare_modelnet40_walker():
  n_target_faces = [1000, 2000, 4000]
  labels2use = model_net_labels
  for i, name in tqdm(enumerate(labels2use)):
    for part in ['test', 'train']:
      pin = os.path.expanduser('~') + '/mesh_walker/datasets_raw/ModelNet40/' + name + '/' + part + '/'
      p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/modelnet40/'
      prepare_directory_from_scratch('modelnet40', pathname_expansion=pin + '*.off',
                                     p_out=p_out, add_labels='modelnet', n_target_faces=n_target_faces,
                                     fn_prefix=part + '_', verbose=False)


def prepare_cubes(labels2use=cubes_labels,
                  path_in=os.path.expanduser('~') + '/datasets_processed/cubes/',
                  p_out=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/cubes_tmp'):
  dataset_name = 'cubes'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    for part in ['test', 'train']:
      pin = path_in + name + '/' + part + '/'
      prepare_directory_from_scratch(dataset_name, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                                     classification=False)

def prepare_shrec11_from_raw():
  # Prepare labels per model name
  current_label = None
  model_number2label = [-1 for _ in range(600)]
  for line in open(os.path.expanduser('~') + '/Desktop/Shrec11/test.cla'):
    sp_line = line.split(' ')
    if len(sp_line) == 3:
      name = sp_line[0].replace('_test', '')
      if name in shrec11_labels:
        current_label = name
      else:
        raise Exception('?')
    if len(sp_line) == 1 and sp_line[0] != '\n':
      model_number2label[int(sp_line[0])] = shrec11_shape2label[current_label]


  # Prepare npz files
  p_in = os.path.expanduser('~') + '/Desktop/Shrec11/raw/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/shrec11_raw_500_meshCNN/'
  prepare_directory_from_scratch('shrec11', pathname_expansion=p_in + '*.off',
                                 p_out=p_out, add_labels=model_number2label, n_target_faces=[500])

def prepare_shrec11_meshCNN_from_raw():
  # Prepare labels per model name
  current_label = None
  model_number2label = [-1 for _ in range(600)]
  for line in open(os.path.expanduser('~') + '/Desktop/Shrec11/test.cla'):
    sp_line = line.split(' ')
    if len(sp_line) == 3:
      name = sp_line[0].replace('_test', '')
      if name in shrec11_labels:
        current_label = name
      else:
        raise Exception('?')
    if len(sp_line) == 1 and sp_line[0] != '\n':
      model_number2label[int(sp_line[0])] = shrec11_shape2label[current_label]


  # Prepare npz files
  p_in = os.path.expanduser('~') + '/Desktop/Shrec11/raw/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/shrec11_raw_500_meshCNN/'
  prepare_directory_from_scratch('shrec11', pathname_expansion=p_in + '*.off',
                                 p_out=p_out, add_labels=model_number2label, n_target_faces=[500], adversrial_data='mesh_CNN')

def prepare_shrec11_PD_MeshNet_from_raw():
  # Prepare labels per model name
  current_label = None
  model_number2label = [-1 for _ in range(600)]
  for line in open(os.path.expanduser('~') + '/Desktop/Shrec11/test.cla'):
    sp_line = line.split(' ')
    if len(sp_line) == 3:
      name = sp_line[0].replace('_test', '')
      if name in shrec11_labels:
        current_label = name
      else:
        raise Exception('?')
    if len(sp_line) == 1 and sp_line[0] != '\n':
      model_number2label[int(sp_line[0])] = shrec11_shape2label[current_label]


  # Prepare npz files
  p_in = os.path.expanduser('~') + '/Desktop/Shrec11/raw/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/shrec11_raw_500_meshCNN/'
  prepare_directory_from_scratch('shrec11', pathname_expansion=p_in + '*.off',
                                 p_out=p_out, add_labels=model_number2label, n_target_faces=[500], adversrial_data='PD_MeshNet')



def prepare_walker_shrec11_copycat_from_raw():
  # Prepare labels per model name
  current_label = None
  model_number2label = [-1 for _ in range(600)]
  for line in open(os.path.expanduser('~') + '/Desktop/Shrec11/test.cla'):
    sp_line = line.split(' ')
    if len(sp_line) == 3:
      name = sp_line[0].replace('_test', '')
      if name in shrec11_labels:
        current_label = name
      else:
        raise Exception('?')
    if len(sp_line) == 1 and sp_line[0] != '\n':
      model_number2label[int(sp_line[0])] = shrec11_shape2label[current_label]


  # Prepare npz files
  p_in = os.path.expanduser('~') + '/Desktop/Shrec11/raw/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/shrec11_raw_500_copycat/'
  prepare_directory_from_scratch('shrec11', pathname_expansion=p_in + '*.off',
                                 p_out=p_out, add_labels=model_number2label, n_target_faces=[500], meshCNN_data='walker_copycat')

  # Prepare split train / test
  change_train_test_split(p_out, 16, 4, '16-04_a')

def prepare_modelnet40_mesh_net():
  n_target_faces = [1024]
  labels2use = mesh_net_labels
  for i, name in tqdm(enumerate(labels2use)):
    for part in ['test', 'train']:
      p_in = os.path.expanduser('~') + '/meshNet_adverserial/ModelNet40/' + name + '/' + part + '/'
      p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/modelnet40_mesh_net/' #+ name + '/' + part + '/'
      mesh_net_vertices_and_faces_path = os.path.expanduser('~') + '/meshNet_adverserial/ModelNet40_MeshNet_raw/' + name + '/' + part + '/'
      mesh_net_labels_path = os.path.expanduser('~') + '/meshNet_adverserial/mesh_net_predicted_labels/' + name + '/' + part + '/'

      prepare_mesh_net_directory_from_scratch('mesh_net_modelnet40', pathname_expansion=p_in + '*.off',
                                     p_out=p_out, add_labels='modelnet', n_target_faces=n_target_faces,
                                     fn_prefix=part + '_', verbose=False, mesh_net_vertices_and_faces_path= mesh_net_vertices_and_faces_path,
                                             mesh_net_labels_path= mesh_net_labels_path)



def calc_face_labels_after_remesh(mesh_orig, mesh, face_labels):
  t_mesh = trimesh.Trimesh(vertices=np.array(mesh_orig.vertices), faces=np.array(mesh_orig.triangles), process=False)

  remeshed_face_labels = []
  for face in mesh.triangles:
    vertices = np.array(mesh.vertices)[face]
    center = np.mean(vertices, axis=0)
    p, d, closest_face = trimesh.proximity.closest_point(t_mesh, [center])
    remeshed_face_labels.append(face_labels[closest_face[0]])
  return remeshed_face_labels


def prepare_human_body_segmentation():
  dataset_name = 'sig17_seg_benchmark'
  labels_fuzzy = True
  human_seg_path = os.path.expanduser('~') + '/mesh_walker/datasets_raw/sig17_seg_benchmark/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/sig17_seg_benchmark-no_simplification/'

  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'face_labels']
  if labels_fuzzy:
    fileds_needed += ['labels_fuzzy']

  n_target_faces = [np.inf]
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  for part in ['test', 'train']:
    print('part: ', part)
    path_meshes = human_seg_path + '/meshes/' + part
    seg_path = human_seg_path + '/segs/' + part
    all_fns = []
    for fn in Path(path_meshes).rglob('*.*'):
      all_fns.append(fn)
    for fn in tqdm(all_fns):
      model_name = str(fn)
      if model_name.endswith('.obj') or model_name.endswith('.off') or model_name.endswith('.ply'):
        new_fn = model_name[model_name.find(part) + len(part) + 1:]
        new_fn = new_fn.replace('/', '_')
        new_fn = new_fn.split('.')[-2]
        out_fn = p_out + '/' + part + '__' + new_fn
        mesh = mesh_orig = load_mesh(model_name, classification=False)
        mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
        face_labels = get_sig17_seg_bm_labels(mesh_data, model_name, seg_path)
        labels_orig, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh_data, face_labels)
        if 0: # Show segment borders
          b_vertices = np.where(np.sum(v_labels_fuzzy != 0, axis=1) > 1)[0]
          vertex_colors = np.zeros((mesh_data['vertices'].shape[0],), dtype=np.int)
          vertex_colors[b_vertices] = 1
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=vertex_colors, point_size=2)
        if 0: # Show face labels
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], face_colors=face_labels, show_vertices=False, show_edges=False)
        if 0:
          print(model_name)
          print('min: ', np.min(mesh_data['vertices'], axis=0))
          print('max: ', np.max(mesh_data['vertices'], axis=0))
          cpos = [(-3.5, -0.12, 6.0), (0., 0., 0.1), (0., 1., 0.)]
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=labels_orig, cpos=cpos)
        add_labels = 1
        label = -1
        for this_target_n_faces in n_target_faces:
          mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
          if mesh == mesh_orig:
            remeshed_face_labels = face_labels
          else:
            remeshed_face_labels = calc_face_labels_after_remesh(mesh_orig, mesh, face_labels)
          mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices),
                                'faces': np.asarray(mesh.triangles),
                                'label': label, 'labels': labels,
                                'face_labels': remeshed_face_labels})
          if 1:
            v_labels, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh_data, remeshed_face_labels)
            mesh_data['labels'] = v_labels
            mesh_data['labels_fuzzy'] = v_labels_fuzzy
          if 0:  # Show segment borders
            b_vertices = np.where(np.sum(v_labels_fuzzy != 0, axis=1) > 1)[0]
            vertex_colors = np.zeros((mesh_data['vertices'].shape[0],), dtype=np.int)
            vertex_colors[b_vertices] = 1
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=vertex_colors, point_size=10)
          if 0:  # Show face labels
            utils.visualize_model(np.array(mesh.vertices), np.array(mesh.triangles), face_colors=remeshed_face_labels, show_vertices=False, show_edges=False)
          out_fc_full = out_fn + str_to_add
          if os.path.isfile(out_fc_full + '.npz'):
            continue
          add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
          if 0:
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                                  cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])


def prepare_seg_from_meshcnn(dataset, subfolder=None):
  if dataset == 'human_body':
    dataset_name = 'human_seg_from_meshcnn'
    p_in2add = 'human_seg'
    p_out_sub = p_in2add
    p_ext = ''
  elif dataset == 'coseg':
    p_out_sub = dataset_name = 'coseg'
    p_in2add = dataset_name + '/' + subfolder
    p_ext = subfolder

  path_in = os.path.expanduser('~') + '/mesh_walker/datasets_raw/from_meshcnn/' + p_in2add + '/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/' + p_out_sub + '_from_meshcnn/' + p_ext

  for part in ['test', 'train']:
    pin = path_in + '/' + part + '/'
    prepare_directory_from_scratch(dataset_name, pathname_expansion=pin + '*.obj',
                                   p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                                   classification=False)


def prepare_coseg(dataset_name='coseg',
                  path_in=os.path.expanduser('~') + '/datasets_processed/coseg/',
                  p_out_root=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/coseg_tmp2'):
  for sub_folder in os.listdir(path_in):
    p_out = p_out_root + '/' + sub_folder
    if not os.path.isdir(p_out):
      os.makedirs(p_out + '/' + sub_folder)

    for part in ['test', 'train']:
      pin = path_in + '/' + sub_folder + '/' + part + '/'
      prepare_directory_from_scratch(sub_folder, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf])

# ------------------------------------------------------- #

def map_fns_to_label(path=None, filenames=None):
  lmap = {}
  if path is not None:
    iterate = glob.glob(path + '/*.npz')
  elif filenames is not None:
    iterate = filenames

  for fn in iterate:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    label = int(mesh_data['label'])
    if label not in lmap.keys():
      lmap[label] = []
    if path is None:
      lmap[label].append(fn)
    else:
      lmap[label].append(fn.split('/')[-1])
  return lmap


def change_train_test_split(path, n_train_examples, n_test_examples, split_name):
  np.random.seed()
  fns_lbls_map = map_fns_to_label(path)
  for label, fns_ in fns_lbls_map.items():
    fns = np.random.permutation(fns_)
    assert len(fns) == n_train_examples + n_test_examples
    train_path = path + '/' + split_name + '/train'
    if not os.path.isdir(train_path):
      os.makedirs(train_path)
    test_path = path + '/' + split_name + '/test'
    if not os.path.isdir(test_path):
      os.makedirs(test_path)
    for i, fn in enumerate(fns):
      out_fn = fn.replace('train_', '').replace('test_', '')
      if i < n_train_examples:
        shutil.copy(path + '/' + fn, train_path + '/' + out_fn)
      else:
        shutil.copy(path + '/' + fn, test_path + '/' + out_fn)


# ------------------------------------------------------- #


def prepare_one_dataset(dataset_name, mode):
  dataset_name = dataset_name.lower()
  if dataset_name == 'modelnet40' or dataset_name == 'modelnet':
    prepare_modelnet40()

  if dataset_name == 'shrec11':
    pass

  if dataset_name == 'cubes':
    pass

  # Semantic Segmentations
  if dataset_name == 'human_seg':
    if mode == 'from_meshcnn':
      prepare_seg_from_meshcnn('human_body')
    else:
      prepare_human_body_segmentation()

  if dataset_name == 'coseg':
    prepare_seg_from_meshcnn('coseg', 'coseg_aliens')
    prepare_seg_from_meshcnn('coseg', 'coseg_chairs')
    prepare_seg_from_meshcnn('coseg', 'coseg_vases')


def vertex_pertubation(faces, vertices):
  n_vertices2change = int(vertices.shape[0] * 0.3)
  for _ in range(n_vertices2change):
    face = faces[np.random.randint(faces.shape[0])]
    vertices_mean = np.mean(vertices[face, :], axis=0)
    v = np.random.choice(face)
    vertices[v] = vertices_mean
  return vertices


def visualize_dataset(pathname_expansion):
  cpos = None
  filenames = glob.glob(pathname_expansion)
  while 1:
    fn = np.random.choice(filenames)
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    vertex_colors_idx = mesh_data['labels'].astype(np.int) if mesh_data['labels'].size else None
    vertices = mesh_data['vertices']
    #vertices = vertex_pertubation(mesh_data['faces'], vertices)
    utils.visualize_model(vertices, mesh_data['faces'], vertex_colors_idx=vertex_colors_idx, cpos=cpos, point_size=5)


if __name__ == '__main__':
  TEST_FAST = 0
  utils.config_gpu(False)
  np.random.seed(1)
  #prepare_shrec11_from_raw()
  #prepare_copycat_shrec11_from_raw()
  add_scale_to_dataset()

  #visualize_dataset('/home/alonlahav/mesh_walker/datasets_processed-tmp/sig17_seg_benchmark-no_simplification/*.npz')
  #visualize_dataset('/home/galye/mesh_walker/datasets_processed/shrec16/*.npz')
  '''
  dataset_name = 'human_seg'
  mode = 'from_raw'         # from_meshcnn / from_raw
  if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
  if len(sys.argv) > 2:
    mode = sys.argv[2]

  if dataset_name == 'all':
    for dataset_name_ in ['modelnet40', 'shrec11', 'cubes', 'human_seg', 'coseg']:
      prepare_one_dataset(dataset_name_)
  else:
    prepare_one_dataset(dataset_name, mode)

  if 0:
    prepare_shrec11_from_raw()
  elif 0:
    prepare_cubes()
  elif 0:
    prepare_cubes(dataset_name='shrec11', path_in=os.path.expanduser('~') + '/datasets_processed/shrec_16/',
                  p_out=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/shrec11_tmp',
                  labels2use=shrec11_labels)
  elif 0:
    prepare_coseg()
  elif 0:
    change_train_test_split(path=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/shrec11/',
                            n_train_examples=16, n_test_examples=4, split_name='16-04_C')
  elif 0:
    collect_n_models_per_class(in_path=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/coseg/coseg_vases/',
                               n_models4train=[1, 2, 4, 8, 16, 32])
  '''
