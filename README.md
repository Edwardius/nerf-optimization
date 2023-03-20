# nerf-optimization
Containerized research environment for NeRF optimization.


## Setup
---
Run the following before using docker compose to fix ownership while inside a development container:

```bash
$ chmod +x initialize.sh
$ ./initialize.sh
```

This only needs to be done once. Confirm that you're project is initialized by checking the `.env` file. It should contain something like:

```
COMPOSE_PROJECT_NAME=nerf-optimization-e23zhou
FIXUID=1011
FIXGID=1014
GUI_TOOLS_VNC_PORT=20220
USERNAME=e23zhou
```

## Inside Instant NGP Container
---
```sh
cmake -DNGP_BUILD_WITH_GUI=off ./ -B ./build
cmake --build build --config RelWithDebInfo -j 16
```

to build and run the thing, do not use the `instant-ngp` executable, instead use

```sh
python3 scripts/run.py path/to/data_images
```

## Inside Kilonerf Container
---
**BEFORE YOU ENTER** make sure that you mount your downloaded nsvf dataset

```bash
wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip && unzip -n Synthetic_NSVF.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip && unzip -n Synthetic_NeRF.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip && unzip -n BlendedMVS.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip && unzip -n TanksAndTemple.zip
```
Inside compose, mount to `/home/docker/kilonerf/data/nsvf`

Inside the container, compile KiloNeRF's C++/CUDA code 
```
cd $KILONERF_HOME/cuda
python setup.py develop
```
To benchmark a trained model run:  
`bash benchmark.sh`

You can launch the **interactive viewer** by running:  
`bash render_to_screen.sh`

To train a model yourself run  
`bash train.sh`

The default dataset is `Synthetic_NeRF_Lego`, you can adjust the dataset by
setting the dataset variable in the respective script.

## Nerf PyTorch
---
You can follow the quickstart while inside the container

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```
After training for 100k iterations (~4 hours on a single 2080 Ti), you can find the following video at `logs/lego_test/lego_test_spiral_100000_rgb.mp4`.

## EfficientNeRF
```bash
 export DATA_DIR=/path/to/lego
 python3 train.py \
   --dataset_name blender \
   --root_dir $DATA_DIR \
   --N_samples 128 \
   --N_importance 5 --img_wh 800 800 \
   --num_epochs 16 --batch_size 4096 \
   --optimizer radam --lr 2e-3 \
   --lr_scheduler poly \
   --coord_scope 3.0 \
   --warmup_step 5000\
   --sigma_init 30.0 \
   --weight_threashold 1e-5 \
   --exp_name lego_coarse128_fine5_V384
```

## To copy things out
---
```
scp e23zhou@guacamole:/home/e23zhou/code/nerf-optimization/src/nerf-pytorch/logs/blender_paper_lego/blender_paper_lego_spiral_200000_rgb.mp4 /home/edwardius/code
```
I'm keeping this here for reference lmao. This is ran on your machine not in the one you are connected to.