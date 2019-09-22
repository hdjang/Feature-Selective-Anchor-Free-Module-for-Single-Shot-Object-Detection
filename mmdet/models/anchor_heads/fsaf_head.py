import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses import sigmoid_focal_loss, FocalLoss, iou_loss
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
import pdb


@HEADS.register_module
class FSAFHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 feat_strides=[8, 16, 32, 64, 128],
                 eps_e=0.2,
                 eps_i=0.5,
                 FL_alpha=0.25,
                 FL_gamma=2.0,
                 bbox_offset_norm=4,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        
        self.feat_strides = feat_strides
        self.eps_e = eps_e
        self.eps_i = eps_i
        self.FL_alpha = FL_alpha
        self.FL_gamma = FL_gamma
        self.bbox_offset_norm = bbox_offset_norm
        
        super(FSAFHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.fsaf_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fsaf_reg = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fsaf_cls, std=0.01, bias=bias_cls)
        normal_init(self.fsaf_reg, std=0.1)
        
    
    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        cls_score = self.fsaf_cls(cls_feat)
        bbox_pred = self.relu(self.fsaf_reg(reg_feat))
        
        return cls_score, bbox_pred
    
    def loss_single(self,
             cls_scores,
             bbox_preds,
             cls_targets_list,
             reg_targets_list,
             level,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        device = cls_scores[0].device
        num_imgs = cls_targets_list.shape[0]
        
        # loss-cls
        scores = cls_scores.permute(0,2,3,1).reshape(-1,1)
        labels = cls_targets_list.permute(0,2,3,1).reshape(-1)
        valid_cls_idx = labels != -1
        loss_cls = sigmoid_focal_loss(scores[valid_cls_idx], labels[valid_cls_idx],
                           gamma=self.FL_gamma, alpha=self.FL_alpha, reduction='sum')
        norm_cls = (labels == 1).long().sum()
        #loss_cls /= norm_cls
        
        # loss-reg
        offsets = bbox_preds.permute(0,2,3,1).reshape(-1,4)
        gtboxes = reg_targets_list.permute(0,2,3,1).reshape(-1,4)
        valid_reg_idx = (gtboxes[:,0] != -1)
        if valid_reg_idx.long().sum() != 0:
            offsets = offsets[valid_reg_idx]
            gtboxes = gtboxes[valid_reg_idx]
            
            H,W = bbox_preds.shape[2:]
            y,x = torch.meshgrid([torch.arange(0,H), torch.arange(0,W)])
            y = (y.float() + 0.5) * self.feat_strides[level]
            x = (x.float() + 0.5) * self.feat_strides[level]
            xy = torch.cat([x.unsqueeze(2),y.unsqueeze(2)], dim=2).float().to(device)
            xy = xy.permute(2,0,1).unsqueeze(0).repeat(num_imgs,1,1,1)
            xy = xy.permute(0,2,3,1).reshape(-1,2)
            xy = xy[valid_reg_idx]
            w = (offsets[:,1] + offsets[:,3]).unsqueeze(1) * self.bbox_offset_norm * self.feat_strides[level]
            h = (offsets[:,0] + offsets[:,2]).unsqueeze(1) * self.bbox_offset_norm * self.feat_strides[level]
            bbox_xywh = torch.cat([xy,w,h], dim=1)  # (N,4)
            bbox_xyxy = self.xywh2xyxy(bbox_xywh)

            loss_reg = iou_loss(bbox_xyxy, gtboxes)
            norm_reg = valid_reg_idx.long().sum()
            #loss_reg /= norm_reg
        else:
            loss_reg = torch.tensor(0).float().to(device)
            norm_reg = torch.tensor(0).float().to(device)
        
        return loss_cls, loss_reg, norm_cls, norm_reg

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        
        cls_reg_targets = self.fsaf_target(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore)
        
        (cls_targets_list, reg_targets_list) = cls_reg_targets
        
        level_list = [i for i in range(len(self.feat_strides))]
        loss_cls, loss_reg, norm_cls, norm_reg = multi_apply(
             self.loss_single,
             cls_scores,
             bbox_preds,
             cls_targets_list,
             reg_targets_list,
             level_list,
             img_metas=img_metas,
             cfg=cfg,
             gt_bboxes_ignore=None)
        
        loss_cls = sum(loss_cls)/sum(norm_cls)
        loss_reg = sum(loss_reg)/sum(norm_reg)
        
        return dict(loss_cls=loss_cls, loss_bbox=loss_reg)
    
    
    '''
    #----------------------
    # loss w/o multi_apply()
    #----------------------
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        
        cls_reg_targets = self.fsaf_target(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore)
        
        (cls_targets_list, reg_targets_list) = cls_reg_targets
        
        device = cls_scores[0].device
        num_imgs = len(gt_bboxes)
        
        loss_cls = 0
        loss_reg = 0
        norm_cls = 0
        norm_reg = 0
        
        num_levels = len(cls_targets_list)
        for level in range(num_levels):
            # loss-cls
            scores = cls_scores[level].permute(0,2,3,1).reshape(-1,1)
            labels = cls_targets_list[level].permute(0,2,3,1).reshape(-1)
            valid_cls_idx = labels != -1
            loss_cls += sigmoid_focal_loss(scores[valid_cls_idx], labels[valid_cls_idx],
                               gamma=self.FL_gamma, alpha=self.FL_alpha, reduction='sum')
            norm_cls += (labels == 1).long().sum()
            
            # loss-reg
            offsets = bbox_preds[level].permute(0,2,3,1).reshape(-1,4)
            gtboxes = reg_targets_list[level].permute(0,2,3,1).reshape(-1,4)
            valid_reg_idx = (gtboxes[:,0] != -1)
            if valid_reg_idx.long().sum() != 0:
                offsets = offsets[valid_reg_idx]
                gtboxes = gtboxes[valid_reg_idx]

                H,W = bbox_preds[level].shape[2:]
                y,x = torch.meshgrid([torch.arange(0,H), torch.arange(0,W)])
                xy = torch.cat([x.unsqueeze(2),y.unsqueeze(2)], dim=2).float().to(device)
                xy = xy.permute(2,0,1).unsqueeze(0).repeat(num_imgs,1,1,1)
                xy = xy.permute(0,2,3,1).reshape(-1,2)
                xy = xy[valid_reg_idx]
                w = (offsets[:,1] + offsets[:,3]).unsqueeze(1) * self.bbox_offset_norm
                h = (offsets[:,0] + offsets[:,2]).unsqueeze(1) * self.bbox_offset_norm
                bbox_xywh = torch.cat([xy,w,h], dim=1)  # (N,4)
                bbox_xyxy = self.xywh2xyxy(bbox_xywh)

                loss_reg += iou_loss(bbox_xyxy, gtboxes)
                norm_reg += valid_reg_idx.long().sum()
            #print(loss_reg)
        loss_cls /= norm_cls
        loss_reg /= norm_reg
        #pdb.set_trace()
        return dict(loss_cls=loss_cls, loss_bbox=loss_reg)
    '''
    
    def fsaf_target(self,
                     cls_scores,
                     bbox_preds,
                     gt_bboxes,
                     gt_labels,
                     img_metas,
                     cfg,
                     gt_bboxes_ignore_list=None):
        
        device = cls_scores[0].device
        
        # target placeholder
        num_levels = len(cls_scores)
        cls_targets_list = []
        reg_targets_list = []
        for level in range(num_levels):
            cls_targets_list.append(torch.zeros_like(cls_scores[level]).long()) #  0 init
            reg_targets_list.append(torch.ones_like(bbox_preds[level]) * -1)    # -1 init
        
        # detached network prediction for online GT generation
        num_imgs = len(gt_bboxes)
        cls_scores_list = []
        bbox_preds_list = []
        for img in range(num_imgs):
            # detached prediction for online pyramid level selection
            cls_scores_list.append([lvl[img].detach() for lvl in cls_scores])
            bbox_preds_list.append([lvl[img].detach() for lvl in bbox_preds])
            
        # generate online GT
        num_imgs = len(gt_bboxes)
        for img in range(num_imgs):
            # sort objects according to its size
            gt_bboxes_img_xyxy = gt_bboxes[img]
            gt_bboxes_img_xywh = self.xyxy2xywh(gt_bboxes_img_xyxy)
            gt_bboxes_img_size = gt_bboxes_img_xywh[:,-2] * gt_bboxes_img_xywh[:,-1]
            _, gt_bboxes_img_idx = gt_bboxes_img_size.sort(descending=True)
            
            for obj_idx in gt_bboxes_img_idx:
                label = gt_labels[img][obj_idx]-1
                gt_bbox_obj_xyxy = gt_bboxes_img_xyxy[obj_idx]
                # get optimal online pyramid level for each object
                opt_level = self.get_online_pyramid_level(
                    cls_scores_list[img], bbox_preds_list[img], gt_bbox_obj_xyxy, label)
                
                # get the effective/ignore area
                H,W = cls_scores[opt_level].shape[2:]
                b_p_xyxy = gt_bbox_obj_xyxy / self.feat_strides[opt_level]
                e_spatial_idx, i_spatial_idx = self.get_spatial_idx(b_p_xyxy,W,H,device)
                
                # cls-GT
                # fill prob= 1 for the effective area
                cls_targets_list[opt_level][img, label, e_spatial_idx] = 1
                
                # fill prob=-1 for the ignoring area
                _i_spatial_idx = cls_targets_list[opt_level][img, label] * i_spatial_idx.long()
                i_spatial_idx = i_spatial_idx - (_i_spatial_idx == 1)
                cls_targets_list[opt_level][img, label, i_spatial_idx] = -1                
                
                # fill prob=-1 for the adjacent ignoring area; lower
                if opt_level != 0:
                    H_l,W_l = cls_scores[opt_level-1].shape[2:]
                    b_p_xyxy_l = gt_bbox_obj_xyxy / self.feat_strides[opt_level-1]
                    _, i_spatial_idx_l = self.get_spatial_idx(b_p_xyxy_l,W_l,H_l,device)
                    # preserve cls-gt that is already filled as effective area
                    _i_spatial_idx_l = cls_targets_list[opt_level-1][img, label] * i_spatial_idx_l.long()
                    i_spatial_idx_l = i_spatial_idx_l - (_i_spatial_idx_l == 1)
                    cls_targets_list[opt_level-1][img, label][i_spatial_idx_l] = -1
                    
                # fill prob=-1 for the adjacent ignoring area; upper
                if opt_level != num_levels-1:
                    H_u,W_u = cls_scores[opt_level+1].shape[2:]
                    b_p_xyxy_u = gt_bbox_obj_xyxy / self.feat_strides[opt_level+1]
                    _, i_spatial_idx_u = self.get_spatial_idx(b_p_xyxy_u,W_u,H_u,device)
                    # preserve cls-gt that is already filled as effective area
                    _i_spatial_idx_u = cls_targets_list[opt_level+1][img, label] * i_spatial_idx_u.long()
                    i_spatial_idx_u = i_spatial_idx_u - (_i_spatial_idx_u == 1)
                    cls_targets_list[opt_level+1][img, label][i_spatial_idx_u] = -1
                
                # reg-GT
                reg_targets_list[opt_level][img, :, e_spatial_idx] = gt_bbox_obj_xyxy.unsqueeze(1)
                
        return cls_targets_list, reg_targets_list
        
    def get_spatial_idx(self, b_p_xyxy, W, H, device):
        # zero-tensor w/ (H,W)
        e_spatial_idx = torch.zeros((H,W)).byte()
        i_spatial_idx = torch.zeros((H,W)).byte()
        
        # effective idx
        b_e_xyxy = self.get_prop_xyxy(b_p_xyxy, self.eps_e, W, H)
        e_xmin = b_e_xyxy[0]
        e_xmax = b_e_xyxy[2]+1
        e_ymin = b_e_xyxy[1]
        e_ymax = b_e_xyxy[3]+1
        e_spatial_idx[e_ymin:e_ymax, e_xmin:e_xmax] = 1
        
        # ignore idx
        b_i_xyxy = self.get_prop_xyxy(b_p_xyxy, self.eps_i, W, H)
        i_xmin = b_i_xyxy[0]
        i_xmax = b_i_xyxy[2]+1
        i_ymin = b_i_xyxy[1]
        i_ymax = b_i_xyxy[3]+1
        i_spatial_idx[i_ymin:i_ymax, i_xmin:i_xmax] = 1
        i_spatial_idx[e_ymin:e_ymax, e_xmin:e_xmax] = 0
        
        return e_spatial_idx.to(device), i_spatial_idx.to(device)
        
    def get_online_pyramid_level(self, cls_scores_img, bbox_preds_img, gt_bbox_obj_xyxy, gt_label_obj):
        device = cls_scores_img[0].device
        num_levels = len(cls_scores_img)
        level_losses = torch.zeros(num_levels)
        for level in range(num_levels):
            H,W = cls_scores_img[level].shape[1:]
            b_p_xyxy = gt_bbox_obj_xyxy / self.feat_strides[level]
            b_e_xyxy = self.get_prop_xyxy(b_p_xyxy, self.eps_e, W, H)
            
            # Eqn-(1)
            N = (b_e_xyxy[3]-b_e_xyxy[1]+1) * (b_e_xyxy[2]-b_e_xyxy[0]+1)
            
            # cls loss; FL
            score = cls_scores_img[level][gt_label_obj,b_e_xyxy[1]:b_e_xyxy[3]+1,b_e_xyxy[0]:b_e_xyxy[2]+1]
            score = score.contiguous().view(-1).unsqueeze(1)
            label = torch.ones_like(score).long()
            label = label.contiguous().view(-1)
            
            loss_cls = sigmoid_focal_loss(score, label, gamma=self.FL_gamma, alpha=self.FL_alpha, reduction='sum')
            loss_cls /= N
            
            # reg loss; IoU
            offsets = bbox_preds_img[level][:,b_e_xyxy[1]:b_e_xyxy[3]+1,b_e_xyxy[0]:b_e_xyxy[2]+1]
            offsets = offsets.contiguous().permute(1,2,0)  # (b_e_H,b_e_W,4)
            
            # predicted bbox
            y,x = torch.meshgrid([torch.arange(b_e_xyxy[1],b_e_xyxy[3]+1), torch.arange(b_e_xyxy[0],b_e_xyxy[2]+1)])
            y = (y.float() + 0.5) * self.feat_strides[level]
            x = (x.float() + 0.5) * self.feat_strides[level]
            xy = torch.cat([x.unsqueeze(2),y.unsqueeze(2)], dim=2).float().to(device)
            w = (offsets[:,:,1] + offsets[:,:,3]).unsqueeze(2) * self.bbox_offset_norm * self.feat_strides[level]
            h = (offsets[:,:,0] + offsets[:,:,2]).unsqueeze(2) * self.bbox_offset_norm * self.feat_strides[level]
            bbox_xywh = torch.cat([xy,w,h], dim=2)  # (b_e_H,b_e_W,4)
            bbox_xywh = bbox_xywh.view(-1,4)  # (-1,4)
            bbox_xyxy = self.xywh2xyxy(bbox_xywh)
            
            loss_reg = iou_loss(bbox_xyxy, gt_bbox_obj_xyxy.unsqueeze(0).repeat(N,1))
            loss_reg /= N
            
            loss = loss_cls + loss_reg
            
            level_losses[level] = loss
        min_level_idx = torch.argmin(level_losses)
        #print(level_losses, min_level_idx)
        return min_level_idx
            
    def get_prop_xyxy(self, xyxy, scale, w, h):
        # scale bbox
        xywh = self.xyxy2xywh(xyxy)
        xywh[2:] *= scale
        xyxy = self.xywh2xyxy(xywh)
        # clamp bbox
        xyxy[0] = xyxy[0].floor().clamp(0, w-2).int() # x1
        xyxy[1] = xyxy[1].floor().clamp(0, h-2).int() # y1
        xyxy[2] = xyxy[2].ceil().clamp(1, w-1).int()  # x2
        xyxy[3] = xyxy[3].ceil().clamp(1, h-1).int()  # y2
        return xyxy.int()
    
    def xyxy2xywh(self, xyxy):
        if xyxy.dim() == 1:
            return torch.cat((0.5 * (xyxy[0:2] + xyxy[2:4]), xyxy[2:4] - xyxy[0:2]), dim=0)
        else:
            return torch.cat((0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]), dim=1)
    
    def xywh2xyxy(self, xywh):
        if xywh.dim() == 1:
            return torch.cat((xywh[0:2] - 0.5 * xywh[2:4], xywh[0:2] + 0.5 * xywh[2:4]), dim=0)
        else:
            return torch.cat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4], xywh[:, 0:2] + 0.5 * xywh[:, 2:4]), dim=1)
        
        
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        # Only single-img evaluation is available now
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds) == num_levels
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype
        scale_factor = img_metas[0]['scale_factor']
        
        # generate center-points
        xy_list = []
        for level in range(num_levels):
            H,W = bbox_preds[level].shape[2:]
            y,x = torch.meshgrid([torch.arange(0,H), torch.arange(0,W)])
            y = (y.float() + 0.5) * self.feat_strides[level]
            x = (x.float() + 0.5) * self.feat_strides[level]
            xy = torch.cat([x.unsqueeze(2),y.unsqueeze(2)], dim=2).float().to(device)
            xy = xy.permute(2,0,1).unsqueeze(0)
            xy_list.append(xy)
            
        mlvl_bboxes = []
        mlvl_scores = []
        for level, (cls_score, bbox_pred, xy) in enumerate(zip(cls_scores, bbox_preds, xy_list)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score[0].permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)
            xy = xy[0].permute(1,2,0).reshape(-1,2)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                xy = xy[topk_inds, :]
            
            # decode predicted offsets to get final bbox
            w = (bbox_pred[:,1] + bbox_pred[:,3]).unsqueeze(1) * self.bbox_offset_norm * self.feat_strides[level]
            h = (bbox_pred[:,0] + bbox_pred[:,2]).unsqueeze(1) * self.bbox_offset_norm * self.feat_strides[level]
            bbox_xywh = torch.cat([xy,w,h], dim=1) # (N,4)
            bbox_xyxy = self.xywh2xyxy(bbox_xywh)
            
            mlvl_bboxes.append(bbox_xyxy)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return [(det_bboxes, det_labels)]
    
    
        
    
    
    
    
    
