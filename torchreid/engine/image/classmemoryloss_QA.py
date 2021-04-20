from __future__ import division, print_function, absolute_import

import numpy as np
import torch #, sys
import os.path as osp
from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss
from torchreid.losses import ClassMemoryLoss

from ..engine import Engine
from ..pretrainer import PreTrainer

# Required for new engine run defination
import time, datetime
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)
from torch.utils.tensorboard import SummaryWriter
from torchreid.utils.serialization import load_checkpoint


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
        self.datamanager = datamanager
        self.model = model
        self.matcher = matcher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_clsm >= 0
        assert weight_t + weight_clsm > 0
        self.weight_t = weight_t
        self.weight_clsm = weight_clsm

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_clsmloss = ClassMemoryLoss(self.matcher, datamanager.num_train_pids, mem_batch_size = mem_batch_size)
        if self.use_gpu:
            self.criterion_clsmloss = self.criterion_clsmloss.cuda()

        # print('... In QAConv Engine : {}, batch_size : {}'.format(self.datamanager.train_loader.sampler.num_instances, self.datamanager.train_loader.batch_size))


    def save_model(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint(
            {
            'model': self.model.module.state_dict(),
            'criterion': self.criterion_clsmloss.module.state_dict(),
            'optim': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1
            }, 
            fpath = osp.join(output_dir, 'checkpoint.pth.tar'),
            is_best = is_best)


# --- Modifying run and train definations of engine (in below code)
    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10, # If print_freq is invalid if print_epoch is set true
        print_epoch=False, 
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        train_resume = False,
        pre_epochs = 1,
        pmax_steps = 2000,
        pnum_trials = 10,
        acc_thr = 0.6,
        enhance_data_aug = False,
        method_name = 'QAConv',
        sub_method_name = 'res50_layer3',
        qbatch_sz = None,
        gbatch_sz = None,
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """
        self.resume = train_resume
        self.pre_epochs = pre_epochs
        self.pmax_steps = pmax_steps
        self.pnum_trials = pnum_trials
        self.acc_thr = acc_thr
        self.enhance_data_aug = enhance_data_aug
        self.method_name = method_name
        self.sub_method_name = sub_method_name
        self.qbatch_sz = qbatch_sz
        self.gbatch_sz = gbatch_sz

        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True'
            )

        print(".... Running from new engine run defination ... !")
        # print(self.datamanager.transform_tr)
        # self.datamanager.QAConv_train_loader()
        # print(self.datamanager.transform_tr)

        # Pre-training the network for warm-start
        self.pretrain(test_only, save_dir) # test_only automatically loads model from checkpoint

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return


        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print('=> Start training')

        for self.epoch in range(self.start_epoch, self.max_epoch):
            info_dict = self.train(
                print_freq=print_freq,
                print_epoch = print_epoch,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )

            train_time = time.time() - time_start
            lr = info_dict['lr']
            if print_epoch:
                if self.weight_t > 0 and self.weight_clsm > 0:
                    print(
                        '* Finished epoch %d at lr=[%g, %g, %g]. Loss_t: %.3f. Loss_clsm: %.3f. Acc: %.2f%%. Training time: %.0f seconds.                  \n'
                        % (self.epoch + 1, lr[0], lr[1], lr[2], 
                            info_dict['loss_t_avg'], info_dict['loss_clsm_avg'], info_dict['prec_avg'] * 100, train_time))
                elif self.weight_t > 0:
                    print(
                        '* Finished epoch %d at lr=[%g, %g, %g]. Loss_t: %.3f. Training time: %.0f seconds.                  \n'
                        % (self.epoch + 1, lr[0], lr[1], lr[2], info_dict['loss_t_avg'], train_time))
                elif self.weight_clsm > 0:
                    print(
                        '* Finished epoch %d at lr=[%g, %g, %g]. Loss_clsm: %.3f. Acc: %.2f%%. Training time: %.0f seconds.                  \n'
                        % (self.epoch + 1, lr[0], lr[1], lr[2], info_dict['loss_clsm_avg'], info_dict['prec_avg'] * 100, train_time))


            if (self.epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (self.epoch+1) % eval_freq == 0 \
               and (self.epoch + 1) != self.max_epoch:
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                self.save_model(self.epoch, rank1, save_dir)

            # Modify transforms and re-initilize train dataloader
            if not self.enhance_data_aug and self.epoch < self.max_epoch - 1:
                if 'prec_avg' not in info_dict.keys():
                    self.enhance_data_aug = True
                    print('Start to Flip and Block only for triplet loss')
                    self.datamanager.QAConv_train_loader()

                elif info_dict['prec_avg'] > self.acc_thr:
                    self.enhance_data_aug = True
                    print('\nAcc = %.2f%% > %.2f%%. Start to Flip and Block.\n' % (info_dict['prec_avg']* 100, self.acc_thr *100))
                    self.datamanager.QAConv_train_loader()

                
        if self.max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            self.save_model(self.epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()


    def pretrain(self, test_only, output_dir):
        """
        This function either loads an already trained model or pre-trains a model before actual 
        training for a better starting point.
        """  
        if self.resume or test_only:
            print('Loading checkpoint...')
            if self.resume and (self.resume != 'ori'):
                checkpoint = load_checkpoint(self.resume)
            else:
                checkpoint = load_checkpoint(osp.join(output_dir, 'checkpoint.pth.tar'))
            self.model.load_state_dict(checkpoint['model'])
            self.criterion_clsmloss.load_state_dict(checkpoint['criterion'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            start_epoch = checkpoint['epoch']
            print("=> Start epoch {} ".format(start_epoch))
        elif self.pre_epochs > 0:
            pre_tr = PreTrainer(
                self.model, 
                self.criterion_clsmloss, 
                self.optimizer, 
                self.datamanager, 
                self.pre_epochs, 
                self.pmax_steps, 
                self.pnum_trials)
            result_file = osp.join(output_dir, self.method_name, 'pretrain_metric.txt')
            self.model, self.criterion_clsmloss, self.optimizer = pre_tr.train(result_file, self.method_name, self.sub_method_name)


    def train(self, print_freq=10, print_epoch=False, fixbase_epoch=0, open_layers=None):
        print(".... Calling train defination from new engine run ... !")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        info_dict = {} # Dictonary containing all the information (loss, accuracy, lr etc)
        if self.weight_t > 0:
            losses_t = AverageMeter()
            # info_dict['loss_t_avg'] = None

        if self.weight_clsm > 0:
            losses_clsm = AverageMeter()
            precisions = AverageMeter()
            # info_dict['loss_clsm_avg'] = None
            # info_dict['loss_acc_avg'] = None

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if self.weight_t > 0:
                losses_t.update(loss_summary['loss_t'], self.targets_sz)
          
            if self.weight_clsm > 0:
                losses_clsm.update(loss_summary['loss_clsm'], self.targets_sz)
                precisions.update(loss_summary['acc'], self.targets_sz)


            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    ), 
                    end='\r'
                )

            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar(
                    'Train/lr', self.get_current_lr(), n_iter
                )

            end = time.time()

        info_dict['lr'] = list(map(lambda group: group['lr'], self.optimizer.param_groups))
        self.update_lr()

        # Returing the relevant info in dictionary
        if self.weight_t > 0:
            info_dict['loss_t_avg'] = losses_t.avg
        if self.weight_clsm > 0:
            info_dict['loss_clsm_avg'] = losses_clsm.avg
            info_dict['prec_avg'] = precisions.avg
        return info_dict


    # Defining evaluation mechanism
    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
    ):
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.extract_features(imgs)
                batch_time.update(time.time() - end)
                features = features.data.cpu()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix is with metric={} ...'.format('QAConv_kernel')
        )
        distmat = metrics.pairwise_distance_using_QAmatcher(
            self.matcher, qf, gf, 
            prob_batch_size = self.qbatch_sz, 
            gal_batch_size = self.gbatch_sz)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.pairwise_distance_using_QAmatcher(
                self.matcher, qf, qf, 
                prob_batch_size = self.qbatch_sz, 
                gal_batch_size = self.qbatch_sz)
            distmat_gg = metrics.pairwise_distance_using_QAmatcher(
                self.matcher, gf, gf, 
                prob_batch_size = self.gbatch_sz, 
                gal_batch_size = self.gbatch_sz)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0], mAP

# --- Modifying run and train definations of engine (in upper code)


    def forward_backward(self, data):
        imgs, pids, camids, dsetids = self.parse_data_for_train_DG(data)
        # print('... camids ... {} '.format(camids))

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        self.targets_sz = pids.size(0)

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

        return loss_summary
