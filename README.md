# Gaussian Shadow Casting for Neural Characters
[Paper](https://arxiv.org/abs/2401.06116)

![Teaser](assets/relit_1.gif)

>**Gaussian Shadow Casting for Neural Characters**\
>[Luis Bolanos](https://github.com/LuisBolanos17/), [Shih-Yang Su](https://lemonatsu.github.io/), and [Helge Rhodin](http://helge.rhodin.de/)\
>CVPR 2024

## Setup

```
conda create -n gsc python=3.9
conda activate gsc

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Dataset

To showcase the learning of a neural avatar in intense outdoor illumination, while jointly learning the light source direction, we can train on the RANA dataset (subject 1). Config: `configs/danbo_gsc_lo_rana.yaml`

Download the RANA dataset from https://nvlabs.github.io/RANA/ and preprocess the data into our h5 format using the notebook `rana_to_h5.ipynb`. The notebook requires the SMPL_NEUTRAL.pkl file.

To showcase relighting on neural avatars learned in uniformly lit environments (ie. MonoPerfCap) please reach out to luisb[at]cs.ubc.ca or shihyang[at]cs.ubc.ca for the pretrained models as we are unable to share the pre-processed datasets due to licensing terms. Config for training while optimizing Gaussian lighting model: `configs/npc_aniso_gaussians.yaml`

## Training

RANA dataset:

`python train.py --config-name danbo_gsc_lo_rana expname=RANA_s1 dataset.subject=subject_01`

MonoPerfCap dataset:

`python train.py --config-name npc_aniso_gaussians expname=PerfCap_nadia dataset.subject=nadia`

## Relighting

RANA dataset:

`python run_render.py --config-name rana model_config=logs/RANA_s1/config.yaml +ckpt_path=[PATH TO .th CHECKPOINT] output_path=[PATH TO OUTPUT FOLDER]`

MonoPerfCap dataset:

`python run_render.py --config-name perfcap_relight model_config=logs/PerfCap_nadia/config.yaml +ckpt_path=[PATH TO .th CHECKPOINT] output_path=[PATH TO OUTPUT FOLDER]`

## Results

### Training Joint Optimization (Neural Field, Gaussians, Lighting)
![Training Progression GIF](assets/training.gif)

### Novel Pose Synthesis

Synthetic Sequence:
![Novel Pose Synthesis Syntetic Sequence GIF](assets/novelpose_synthetic.gif)

Real Sequence:
![Novel Pose Synthesis Real Sequence GIF](assets/novelpose_real.gif)

### HDRi Relighting

HDRis from https://polyhaven.com/hdris

autumn_field_4k:
![HDRi Relit Autumn Field 4K](assets/relit_1.gif)
veranda_4k:
![HDRi Relit Veranda 4K](assets/relit_2.gif)
kiara_8_sunset_4k:
![HDRi Relit Kiara 8 Sunset 4K](assets/relit_3.gif)
tricolor_points (custom):
![HDRi Relit Tricolor Points](assets/relit_4.gif)

## Citation

```
@inproceedings{bolanos2024gsc,
    title={Gaussian Shadow Casting for Neural Characters},
    author={Bolanos, Luis and Su, Shih-Yang and Rhodin, Helge},
    booktitle={The Conference on Computer Vision and Pattern Recognition},
    year={2024}
}

@inproceedings{su2023npc,
    title={NPC: Neural Point Characters from Video},
    author={Su, Shih-Yang and Bagautdinov, Timur and Rhodin, Helge},
    booktitle={International Conference on Computer Vision},
    year={2023}
}

@inproceedings{su2022danbo,
    title={DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks},
    author={Su, Shih-Yang and Bagautdinov, Timur and Rhodin, Helge},
    booktitle={European Conference on Computer Vision},
    year={2022}
}

```

## Acknowledgements

This work was supported in part by an NSERC Discovery Grant, an NSERC CGS-M Grant, and the computational resources and services provided by Advanced Research Computing at The University of British Columbia.
