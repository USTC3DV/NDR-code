import os
import argparse

# 0: just use depth map to initialize pose (only translation)
# 1: use Robust-ICP to registrate first frame to current (Recommended)
# 2: use Robust-ICP to registrate previous frame to current (Not Recommended)
rigid_registrate = 1 # hyper para

parser = argparse.ArgumentParser()
parser.add_argument('--depth_scale', type=float, default=1000.)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--registration_alg_path', type=str, default='./Fast-Robust-ICP/build/FRICP')
args = parser.parse_args()
depth_scale = args.depth_scale
data_path = args.data_path
registration_alg_path = args.registration_alg_path

os.system('python registrate.py --data_path {} --depth_scale {} --registration_alg_path {} --rigid_registrate {}'.format(data_path, depth_scale, registration_alg_path, rigid_registrate))
os.system('python visualization.py --data_path {} --depth_scale {}'.format(data_path, depth_scale))