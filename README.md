# Multi-crop Fusion Strategy Based on Prototype Assignment for Remote Sensing Image Scene Classification
This code provides a PyTorch implementation and pretrained models for MFPC

Abstract-The gap between self-supervised visual representation learning and supervised learning is gradually closing. Self-supervised learning does not rely on a large amount of labeled data and reduces the loss of human labeled information. Compared with natural images, remote sensing images require rich samples and human annotation by experts. Moreover, many algorithms have poor interpretability and unconvincing results. Therefore, this paper proposes a self-supervised method based on prototype assignment by designing a pretext task so that the network maps features to prototypes in the process of learning, swaps the code corresponding to the obtained features, combines them with another data-enhancing feature, and then optimizes the network. The prototype is introduced to explain the clustering idea embodied in the whole process. Considering the existence of the scene information-rich characteristic of remote sensing images, we introduce multiple views with different resolutions to capture more detailed information on the images. Finally, if the data enhancement method is not powerful enough, the network can easily fall into an overfitting state, which prevents the network from learning subtle differences and detailed information. To address this shortcoming, we propose a fusion strategy to flatten the decision boundary of the framework so that the model can also learn the soft similarity between sample pairs. We name the whole framework MFPC. In extensive experiments conducted on three common remote sensing image datasets (i.e., UCMerced, AID, and NWPU45), MFPC achieves a maximum improvement of 4.3\% over some existing self-supervised algorithms, indicating that it can achieve good results.

# Running MFPC unsupervised training

## Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.4.0
- torchvision
- CUDA 10.1
- [Apex](https://github.com/NVIDIA/apex) with CUDA extension
- Other dependencies: scipy, pandas, numpy

## Singlenode training
MFPC is very simple to implement and experiment with.

For example, to train MFPC baseline on a single node with 4 gpus for 400 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=4 main_mfpc.py \
--data_path /path/to/your/own/dataset/train \
--dump_path /path/to/your/own/experiments/path \
--epochs 400 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 64 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 100 \
--dist_url tcp://localhost:10001 \
--world_size 1 \
--rank 0 \
--workers 10 \
--alpha 10.0 \
--beta 1.0 \
--do_unmixed True
```

# Evaluating models

## Evaluate models: Linear classification on your own dataset
To train a supervised linear classifier on frozen features/weights on a single node with 4 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py \
--data_path /path/to/your/own/dataset \
--dump_path /path/to/your/own/experiments/path \
--dist_url tcp://localhost:10002 \
--num_classes 45 \
--pretrained /path/to/checkpoints/your_own_model_name.pth.tar
```

# Common Issues

For help or issues using MFPC, please submit a GitHub issue.

## Citation
If you find this repository useful in your research, please cite:
```
Stay tuned!!!
```
