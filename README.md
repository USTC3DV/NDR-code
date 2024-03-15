<p align="center">

  <h1 align="center">Neural Surface Reconstruction of Dynamic Scenes with Monocular RGB-D Camera</h1>
  <p align="center">
    <a href="https://rainbowrui.github.io/">Hongrui Cai</a>
    ·
    <a href="https://wanquanf.github.io/">Wanquan Feng</a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=5G-2EFcAAAAJ">Xuetao Feng</a>
    ·
    Yan Wang
    ·
    <a href="http://staff.ustc.edu.cn/~juyong/">Juyong Zhang</a>

  </p>
  <h2 align="center">NeurIPS 2022 (Spotlight)</h2>
  <h3 align="center"><a href="https://arxiv.org/pdf/2206.15258.pdf">Paper</a> | <a href="https://ustc3dv.github.io/ndr/">Project Page</a> | <a href="https://openreview.net/forum?id=8RKJj1YDBJT">OpenReview</a> | <a href="https://rainbowrui.github.io/data/NDR_poster.pdf">Poster</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./exhibition/teaser.gif" alt="Logo" width="90%">
  </a>
</p>

<p align="center">
  <a href="">
    <img src="./exhibition/teaser.png" alt="Logo" width="80%">
  </a>
</p>

<p align="center">
We propose Neural-DynamicReconstruction (NDR), a <b>template-free</b> method to recover high-fidelity geometry, motions and appearance of a <b>dynamic</b> scene from a <b>monocular</b> RGB-D camera.
</p>
<br>



## Usage

### Data Convention
The data is organized as [NeuS](https://github.com/Totoro97/NeuS#data-convention)

```
<case_name>
|-- cameras_sphere.npz    # camera parameters
|-- depth
    |-- # target depth for each view
    ...
|-- image
    |-- # target RGB each view
    ...
|-- mask
    |-- # target mask each view (For unmasked setting, set all pixels as 255)
    ...
```

Here `cameras_sphere.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world-to-image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Pre-processed Data
You can download a part of pre-processed [KillingFusion](https://campar.in.tum.de/personal/slavcheva/deformable-dataset/index.html) data [here](https://drive.google.com/file/d/1wQ4yB7r-a8sFkwB6bEJIDp14AhjG8g_B/view?usp=sharing) and unzip it into `./`.

<b><i>Important Tips:</i></b> If the pre-processed data is useful, please cite the related paper(s) and strictly abide by related open-source license(s).

### Setup
Clone this repository and create the environment (please notice CUDA version)
```shell
git clone https://github.com/USTC3DV/NDR-code.git
cd NDR-code
conda env create -f environment.yml
conda activate ndr
```

<details>
  <summary> Dependencies (click to expand) </summary>

  - torch==1.8.0
  - opencv_python==4.5.2.52
  - trimesh==3.9.8 
  - numpy==1.21.2
  - scipy==1.7.0
  - PyMCubes==0.1.2

</details>

### Running
- **Training**
```shell
python train_eval.py
```

- **Evaluating pre-trained model**
```shell
python pretrained_eval.py
```

### Data Pre-processing
To prepare your own data for experiments, please refer to [pose initialization](./pose_initialization/README.md).

### Geometric Projection
- **Compiling renderer**
```shell
cd renderer && bash build.sh && cd ..
```

- **Rendering meshes**

Input path of original data, path of results, and iteration number, e.g.
```shell
python geo_render.py ./datasets/kfusion_frog/ ./exp/kfusion_frog/result/ 120000
```
The rendered results will be put in dir ```[path_of_results]/validations_geo/```


## Todo List
- [x] Configure Files for [DeepDeform](https://github.com/AljazBozic/DeepDeform) Human Sequences and [KillingFusion](https://campar.in.tum.de/personal/slavcheva/deformable-dataset/index.html)
- [x] Code of Data Pre-processing
- [x] Code of Geometric Projection
- [x] Pre-trained Models and Evaluation Code
- [x] Training Code



## Acknowledgements
This project is built upon [NeuS](https://github.com/Totoro97/NeuS). Some code snippets are also borrowed from [IDR](https://github.com/lioryariv/idr) and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). The pre-processing code for camera pose initialization is borrowed from [Fast-Robust-ICP](https://github.com/yaoyx689/Fast-Robust-ICP). The evaluation code for geometry rendering is borrowed from [StereoPIFu_Code](https://github.com/CrisHY1995/StereoPIFu_Code). Thanks for these great projects. We thank all the authors for their great work and repos.


## Contact
If you have questions, please contact [Hongrui Cai](https://rainbowrui.github.io/).



## Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{Cai2022NDR,
  author    = {Hongrui Cai and Wanquan Feng and Xuetao Feng and Yan Wang and Juyong Zhang},
  title     = {Neural Surface Reconstruction of Dynamic Scenes with Monocular RGB-D Camera},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```

If you find our pre-processed data useful, please cite the related paper(s) and strictly abide by related open-source license(s).
