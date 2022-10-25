import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--depth_scale', type=float, default=1000.)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--object_scale', type=float, default=1.05,
                        help='hyper-parameter for radius scaling.'
                        'Larger it is, smaller scaled object will be.')
args = parser.parse_args()
depth_scale = args.depth_scale
data_path = args.data_path
object_scale = args.object_scale

os.system('python create_camera.py --data_path {} --object_scale {}'.format(data_path, object_scale))
os.system('python visualization.py --data_path {} --depth_scale {}'.format(data_path, depth_scale))