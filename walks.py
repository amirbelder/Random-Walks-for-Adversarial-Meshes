import os

from easydict import EasyDict
import numpy as np

import utils


def jump_to_closest_unviseted(model_kdtree_query, model_n_vertices, walk, enable_super_jump=True):
  for nbr in model_kdtree_query[walk[-1]]:
    if nbr not in walk:
      return nbr

  if not enable_super_jump:
    return None

  # If not fouind, jump to random node
  node = np.random.randint(model_n_vertices)

  return node


def get_seq_random_walk_no_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    if len(nodes_to_consider):
      to_add = np.random.choice(nodes_to_consider)
      jump = False
    else:
      if i > backward_steps:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = np.random.randint(n_vertices)
        jump = True
    seq[i] = to_add
    jumps[i] = jump
    visited[to_add] = 1

  return seq, jumps


def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
  MAX_BACKWARD_ALLOWED = np.inf # 25 * 2
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob) or (backward_steps > MAX_BACKWARD_ALLOWED)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps


def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  for i in range(1, seq_len + 1):
    b = min(0, i - 20)
    to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] = True

  return seq, jumps


def get_mesh():
  from dataset_prepare import prepare_edges_and_kdtree, load_mesh, remesh

  model_fn = os.path.expanduser('~') + '/datasets_processed/human_benchmark_sig_17/sig17_seg_benchmark/meshes/test/shrec/10.off'
  mesh = load_mesh(model_fn)
  mesh, _, _ = remesh(mesh, 4000)
  mesh = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'n_faces_orig': np.asarray(mesh.triangles).shape[0]})
  prepare_edges_and_kdtree(mesh)
  mesh['n_vertices'] = mesh['vertices'].shape[0]
  return mesh

def show_walk_on_mesh():
  walk, jumps = get_seq_random_walk_no_jumps(mesh, f0=0, seq_len=400)
  vertices = mesh['vertices']
  if 0:
    dxdydz = np.diff(vertices[walk], axis=0)
    for i, title in enumerate(['dx', 'dy', 'dz']):
      plt.subplot(3, 1, i + 1)
      plt.plot(dxdydz[:, i])
      plt.ylabel(title)
    plt.suptitle('Walk features on Human Body')
  utils.visualize_model(mesh['vertices'], mesh['faces'],
                               line_width=1, show_edges=1, edge_color_a='gray',
                               show_vertices=False, opacity=0.8,
                               point_size=4, all_colors='white',
                               walk=walk, edge_colors='red')


if __name__ == '__main__':
  utils.config_gpu(False)
  mesh = get_mesh()
  np.random.seed(1)
  show_walk_on_mesh()