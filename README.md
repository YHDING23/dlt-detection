## A Set of Object Detection Training Workloads for Alnair

This repo contains the supported code and configuration files to reproduce multiple CV training workload in object detection tasks. The repo heavily borrow the code from [MMdetection](https://github.com/open-mmlab/mmdetection), and you can find the original code from [here](https://github.com/open-mmlab/mmdetection/blob/31c84958f54287a8be2b99cbf87a6dcf12e57753/docs/en/1_exist_data_model.md). 

### Environment Setup with Docker

We provide a [Dockerfile](https://github.com/YHDING23/dlt-detection/blob/main/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03. The built image is pushed to `centaurusinfra/dlt-detection`. 

Run it with:

```shell
docker run --gpus all --shm-size=8g -it -v /nfs_1/data/coco:/data/coco centaurusinfra/dlt-detection
```

After you adjust your parameters, build your image with:

```shell
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
docker build -t your_image_for_detection docker/
```

Or you can also push it to our docker hub registry:

```shell
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
sudo docker image push centaurusinfra/your_image_for_detection:latest
```

### Train Predefined Models on COCO Datasets

This Repo provides out-of-the-box tools for training detection models. As you seen above, we here use the COCO dataset as an example, and it has been mounted as `/data/coco`.

#### Training on a single GPU

We provide `tools/train.py` to launch training jobs on a single GPU.
The basic usage is as follows.

```shell
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```

For example:
```angular2html
python tools/train.py configs/yolox/yolox_s_8x8_300e_coco.py --no-validate
```

Note a typical `CONFIG_FILE` always has a header pointing to the `configs/_base_` like:
```angular2html
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```
And they are the configuration files where you can change your backbone models, datasets (different tasks needs different annotation format), number of epochs (`schedules`) and micros. 

Plus, please Note the configuration of dataset in the same `CONFIG_FILE`, like:
```angular2html
# dataset settings
data_root = '/data/coco/'
dataset_type = 'CocoDataset'
```

During training, log files and checkpoints will be saved to the working directory, which is specified by `work_dir` in the config file or via CLI argument `--work-dir`. By default, the model is evaluated on the validation set every epoch. 

This tool accepts several optional arguments, including:

- `--no-validate` (**not suggested**): Disable evaluation during training.
- `--work-dir ${WORK_DIR}`: Override the working directory.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--options 'Key=value'`: Overrides other settings in the used config.

### Training on multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

For example:
```angular2html
bash ./tools/dist_train.sh configs/yolox/yolox_s_8x8_300e_coco.py 8 
```

Optional arguments remain the same as stated [above](#training-on-a-single-GPU).

### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```
