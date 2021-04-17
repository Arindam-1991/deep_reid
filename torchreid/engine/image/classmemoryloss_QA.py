from __future__ import division, print_function, absolute_import

import torch
from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss
from torchreid.losses import ClassMemoryLoss

from ..engine import Engine


class ImageQAConvEngine(Engine):
    r"""Triplet-loss engine for image-reid.
    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        matcher,
        margin = 0.3,
        weight_t=1,
        weight_clsm=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        mem_batch_size = 16,
    ):
        super(ImageQAConvEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_clsm >= 0
        assert weight_t + weight_clsm > 0
        self.weight_t = weight_t
        self.weight_clsm = weight_clsm

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_clsmloss = ClassMemoryLoss(matcher, datamanager.num_train_pids, mem_batch_size = mem_batch_size)
        if self.use_gpu:
            self.criterion_clsmloss = self.criterion_clsmloss.cuda()

#         print('... In QAConv Engine : {}, batch_size : {}'.format(self.datamanager.train_loader.sampler.num_instances, self.datamanager.train_loader.batch_size))

    def optim_param(self):
        lr = list(map(lambda group: group['lr'], self.optimizer.param_groups))
        print('... Current lr : {} \n', lr)


    def forward_backward(self, data):
        imgs, pids, camids, dsetids = self.parse_data_for_train_DG(data)
        # print('... camids ... {} '.format(camids))

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        features = self.model(imgs)
        # print('... Shape of features : {}'.format(features.size()))
        # t = torch.reshape(features, (int(32 / 4), 4, 2048))
        # print('... Shape of transformed features : {}'.format(t.size()))

        loss = 0
        loss_summary = {}
        # print("Algorithm is at epoch : {}".format(self.epoch))

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_clsm > 0:
            loss_clsm, acc = self.compute_loss(self.criterion_clsmloss, features, pids)
            loss += self.weight_clsm * loss_clsm
            loss_summary['loss_clsm'] = loss_clsm.item()
            loss_summary['acc'] = acc.item() #metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.optim_param()

        return loss_summary
