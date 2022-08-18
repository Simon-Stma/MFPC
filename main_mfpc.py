# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from src import utils
from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2, 6], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224, 96], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14, 0.05], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1., 0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")
parser.add_argument('--prob', default=0.5, type=float, help='prob for choosing region or global mixture')
parser.add_argument('--alpha', default=0.5, type=float, help='alpha for beta distribution')
parser.add_argument('--beta', default=0.5, type=float, help='beta for beta distribution')
parser.add_argument('--do_unmixed', default=False, type=bool, help='add unmixed loss')
parser.add_argument('--do_tsne', default=False, type=bool, help='visual clustering result')

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

# for tsne
parser.add_argument("--num_classes", type=int, default=45, help="cluster number")

def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue, embedding_tsne, labels_tsne = train(train_loader, model, optimizer, epoch, lr_schedule, queue)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch in [0, 99, 199, 299, 399] and args.do_tsne:
                tSNEVisualization(embedding_tsne, labels_tsne, args.num_classes, iter=epoch)

            if epoch in [0, 99, 199, 299, 399]:
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, f"checkpoint_{epoch + 1}.pth.tar"),
                )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)


def train(train_loader, model, optimizer, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ori = AverageMeter()
    losses_mix = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, [inputs, labels] in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ concat embedding for TSNE ... ============
        if it == 0:
            embedding_tsne = embedding[:bs].cuda()
            labels_tsne = labels.cuda()
        else:
            embedding_tsne = torch.cat((embedding_tsne, embedding[:bs].cuda()), dim=0).cuda()
            labels_tsne = torch.cat((labels_tsne, labels.cuda()), dim=0).cuda()

        # ============ swav loss ... ============
        loss_ori = 0
        loss_mix = 0

        # calculate lmd distribution
        r = np.random.rand(1)
        lam = np.random.beta(args.alpha, args.beta)

        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean( torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss_ori += subloss / (np.sum(args.nmb_crops) - 1)

            # calculate mixed loss
            if queue is not None and use_the_queue and args.do_unmixed:
                # mix loss
                im_1 = inputs[:len(args.crops_for_assign)][-(1 + crop_id)]
                # r = np.random.rand(1)
                # # args.beta = 1.0
                # lam = np.random.beta(args.alpha, args.beta)
                images_reverse = torch.flip(im_1, (0,))
                if r < args.prob:
                    mixed_images = lam * im_1 + (1 - lam) * images_reverse
                    mixed_images_flip = torch.flip(mixed_images, (0,))
                else:
                    mixed_images = im_1.clone()
                    bbx1, bby1, bbx2, bby2 = utils.rand_bbox(im_1.size(), lam)
                    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images_reverse[:, :, bbx1:bbx2, bby1:bby2]
                    mixed_images_flip = torch.flip(mixed_images, (0,))
                    # # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (im_1.size()[-1] * im_1.size()[-2]))
                loss_m11, loss_m12 = unMixedLoss(model, mixed_images, q, args)
                loss_mix += lam * loss_m11 + (1 - lam) * loss_m12
            else:
                loss_mix = 0.0

        loss_ori /= len(args.crops_for_assign)
        loss_mix /= len(args.crops_for_assign)
        loss = loss_ori + loss_mix
        if queue is not None and use_the_queue and args.do_unmixed:
            losses_ori.update(loss_ori.item(), inputs[0].size(0))
            losses_mix.update(loss_mix.item(), inputs[0].size(0))
        else:
            losses_ori.update(loss_ori.item(), inputs[0].size(0))
            losses_mix.update(0.0, inputs[0].size(0))


        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Loss_ori {loss_ori.val:.4f} ({loss_ori.avg:.4f})\t"
                "Loss_mix {loss_mix.val:.4f} ({loss_mix.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_ori=losses_ori,
                    loss_mix=losses_mix,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )

    embedding_tsne_gather = concat_all_gather(embedding_tsne)
    labels_tsne_gather = concat_all_gather(labels_tsne)

    return (epoch, losses.avg), queue, embedding_tsne_gather, labels_tsne_gather


def unMixedLoss(model, mixed_images, q, args):
    # compute mixed features
    _, z_mixed = model(mixed_images)
    z_mixed /= args.temperature
    # alternative implementation: q_mixed_flip = self.encoder_q(im1_mixed_re)
    z_mixed_flip = torch.flip(z_mixed, (0,))
    z_mixed = nn.functional.normalize(z_mixed, dim=1)
    z_mixed_flip = nn.functional.normalize(z_mixed_flip, dim=1)
    loss_m1 = -torch.mean(torch.sum(q * F.log_softmax(z_mixed, dim=1), dim=1))
    loss_m2 = -torch.mean(torch.sum(q * F.log_softmax(z_mixed_flip, dim=1), dim=1))

    return loss_m1, loss_m2


def tSNEVisualization(features, labels, num_cluster, iter):
    from sklearn.manifold import TSNE
    # import pandas as pd
    # import seaborn as sns
    import matplotlib.pyplot as plt

    labels = labels.cpu().numpy().tolist()

    # # We want to get TSNE embedding with 2 dimensions
    # n_components = 2
    # tsne = TSNE(n_components)
    # features = features.cpu().numpy()
    # tsne_result = tsne.fit_transform(features)
    # # tsne_result.shape
    # # (1000, 2)
    # # Two dimensions for each of our images
    #
    # # Plot the result of our TSNE with the label color coded
    # # A lot of the stuff here is about making the plot look pretty and not TSNE
    # tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': labels})
    # fig, ax = plt.subplots(1)
    # sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)
    # lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    # ax.set_xlim(lim)
    # ax.set_ylim(lim)
    # ax.set_aspect('equal')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # plt.show()

    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    features = features.cpu().numpy()
    X_tsne = tsne.fit_transform(features)  # dataset [N, dim]
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_cluster))  # label是标签，我这里每个标签的元素是一个元组
    #####----------------------#####
    for i in range(len(X_norm)):
        if labels[i] == -1:
            color = 'grey'
        else:
            color = colors[labels[i]]
        plt.text(X_norm[i, 0], X_norm[i, 1], 'o', color=color, fontdict={'weight': 'light', 'size': 5})
    #####  统计每个簇的中心坐标 ####
    group = [[] for _ in range(num_cluster)]
    for i in range(len(X_norm)):
        if labels[i] == -1:
            continue
        group[labels[i]].append(X_norm[i])
    id_posi = []
    for i in range(num_cluster):
        id_posi.append(np.mean(np.array(group[i]), 0))

    # id2name 是一个字典，根据id找到name
    for i in range(num_cluster):
        plt.text(id_posi[i][0], id_posi[i][1], str(i), color='black',
                 fontdict={'style': 'italic', 'weight': 'light', 'size': 6})

    plt.xticks([])
    plt.yticks([])
    plt.title(f'Iteration {iter}')
    # plt.savefig('cluster.jpg')
    plt.show()


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

if __name__ == "__main__":
    main()
