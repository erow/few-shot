from typing import *
from torch.utils import *
from torch import nn
import torch,math,os
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import gin.torch

from DPGN.backbone import ResNet12, ConvNet
from DPGN.dpgn import DPGN
from DPGN.utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, preprocessing, \
    initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode

import logging
import argparse
import imp

class ModelModule(pl.LightningModule):
    def __init__(self, cifar_flag, config, train_opt,eval_opt, **kwargs: Any) -> None:
        """
        The Trainer of DPGN model
        :param enc_module: backbone network (Conv4, ResNet12, ResNet18, WRN)
        :param gnn_module: DPGN model
        :param data_loader: data loader
        :param log: logger
        :param arg: command line arguments
        :param config: model configurations
        :param best_step: starting step (step at best eval acc or 0 if starts from scratch)
        """
        super().__init__()
        self.train_opt = train_opt
        self.eval_opt = eval_opt
        self.save_hyperparameters()
        
        if config['backbone'] == 'resnet12':
            enc_module = ResNet12(emb_size=config['emb_size'], cifar_flag=cifar_flag)
            print('Backbone: ResNet12')
        elif config['backbone'] == 'convnet':
            enc_module = ConvNet(emb_size=config['emb_size'], cifar_flag=cifar_flag)
            print('Backbone: ConvNet')
        else:
            raise NotImplementedError('Invalid backbone: {}, please specify a backbone model from '
                        'convnet or resnet12.'.format(config['backbone']))

        gnn_module = DPGN(config['num_generation'],
                      train_opt['dropout'],
                      train_opt['num_ways'] * train_opt['num_shots'],
                      train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'] * train_opt['num_queries'],
                      train_opt['loss_indicator'],
                      config['point_distance_metric'],
                      config['distribution_distance_metric'])

        self.config = config
        

        
        # initialize variables
        self.tensors = allocate_tensors()
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.to(self.device)

        # set backbone and DPGN
        self.enc_module = enc_module
        self.gnn_module = gnn_module

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--cifar_flag", type=bool, default=True)
        return parent_parser

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters())]

    def on_train_start(self) -> None:
        self.num_supports, self.num_samples, self.query_edge_mask, self.evaluation_mask = \
            preprocessing(self.train_opt['num_ways'],
                          self.train_opt['num_shots'],
                          self.train_opt['num_queries'],
                          self.train_opt['batch_size'],
                          self.device)
        return super().on_train_start()

    def training_step(self, batch,batch_id):
        """
        train function
        :return: None
        """
        log = {}
        
        # initialize nodes and edges for dual graph model
        support_data, support_label, query_data, query_label, all_data, all_label_in_edge, node_feature_gd, \
        edge_feature_gp, edge_feature_gd = initialize_nodes_edges(batch,
                                                                    self.num_supports,
                                                                    self.tensors,
                                                                    self.train_opt['batch_size'],
                                                                    self.train_opt['num_queries'],
                                                                    self.train_opt['num_ways'],
                                                                    self.device)
        
        # use backbone encode image
        last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

        # run the DPGN model
        point_similarity, node_similarity_l2, distribution_similarities = self.gnn_module(second_last_layer_data,
                                                                                                last_layer_data,
                                                                                                node_feature_gd,
                                                                                                edge_feature_gd,
                                                                                                edge_feature_gp)

        # compute loss
        total_loss, query_node_cls_acc_generations, query_edge_loss_generations = \
            self.compute_train_loss_pred(all_label_in_edge,
                                            point_similarity,
                                            node_similarity_l2,
                                            self.query_edge_mask,
                                            self.evaluation_mask,
                                            self.num_supports,
                                            support_label,
                                            query_label,
                                            distribution_similarities)

        # log training info
        log['loss'] = total_loss
        log['loss/train_edge']=query_edge_loss_generations[-1]
        log['acc/node'] = query_node_cls_acc_generations[-1]

        self.log_dict(log)
        return total_loss



    def on_test_start(self) -> None:
        self.num_supports, self.num_samples, self.query_edge_mask, self.evaluation_mask = \
            preprocessing(self.train_opt['num_ways'],
                          self.train_opt['num_shots'],
                          self.train_opt['num_queries'],
                          self.train_opt['batch_size'],
                          self.device)

        return super().on_test_start()
    def test_step(self, batch,batch_id):
        """
        evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        :return: None
        """
        query_edge_loss_generations = []
        query_node_cls_acc_generations = []

        # initialize nodes and edges for dual graph model
        support_data, support_label, query_data, query_label, all_data, all_label_in_edge, node_feature_gd, \
        edge_feature_gp, edge_feature_gd = initialize_nodes_edges(batch,
                                                                    self.num_supports,
                                                                    self.tensors,
                                                                    self.eval_opt['batch_size'],
                                                                    self.eval_opt['num_queries'],
                                                                    self.eval_opt['num_ways'],
                                                                    self.device)


        last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

        # run the DPGN model
        point_similarity, _, _ = self.gnn_module(second_last_layer_data,
                                                    last_layer_data,
                                                    node_feature_gd,
                                                    edge_feature_gd,
                                                    edge_feature_gp)

        query_node_cls_acc_generations, query_edge_loss_generations = \
            self.compute_eval_loss_pred(query_edge_loss_generations,
                                        query_node_cls_acc_generations,
                                        all_label_in_edge,
                                        point_similarity,
                                        self.query_edge_mask,
                                        self.evaluation_mask,
                                        self.num_supports,
                                        support_label,
                                        query_label)

        log = {
            'loss/edge_test':query_edge_loss_generations,
            'acc/node_test':query_node_cls_acc_generations

        }
        self.log_dict(log)


        
    def compute_train_loss_pred(self,
                            all_label_in_edge,
                            point_similarities,
                            node_similarities_l2,
                            query_edge_mask,
                            evaluation_mask,
                            num_supports,
                            support_label,
                            query_label,
                            distribution_similarities):
        """
        compute the total loss, query classification loss and query classification accuracy
        :param all_label_in_edge: ground truth label in edge form of point graph
        :param point_similarities: prediction edges of point graph
        :param node_similarities_l2: l2 norm of node similarities
        :param query_edge_mask: mask for queries
        :param evaluation_mask: mask for evaluation (for unsupervised setting)
        :param num_supports: number of samples in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :param distribution_similarities: distribution-level similarities
        :return: total loss
                    query classification accuracy
                    query classification loss
        """

        # Point Loss
        total_edge_loss_generations_instance = [
            self.edge_loss((1 - point_similarity), (1 - all_label_in_edge))
            for point_similarity
            in point_similarities]

        # Distribution Loss
        total_edge_loss_generations_distribution = [
            self.edge_loss((1 - distribution_similarity), (1 - all_label_in_edge))
            for distribution_similarity
            in distribution_similarities]

        # combine Point Loss and Distribution Loss
        distribution_loss_coeff = 0.1
        total_edge_loss_generations = [
            total_edge_loss_instance + distribution_loss_coeff * total_edge_loss_distribution
            for (total_edge_loss_instance, total_edge_loss_distribution)
            in zip(total_edge_loss_generations_instance, total_edge_loss_generations_distribution)]

        pos_query_edge_loss_generations = [
            torch.sum(total_edge_loss_generation * query_edge_mask * all_label_in_edge * evaluation_mask)
            / torch.sum(query_edge_mask * all_label_in_edge * evaluation_mask)
            for total_edge_loss_generation
            in total_edge_loss_generations]

        neg_query_edge_loss_generations = [
            torch.sum(total_edge_loss_generation * query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)
            / torch.sum(query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)
            for total_edge_loss_generation
            in total_edge_loss_generations]

        # weighted edge loss for balancing pos/neg
        query_edge_loss_generations = [
            pos_query_edge_loss_generation + neg_query_edge_loss_generation
            for (pos_query_edge_loss_generation, neg_query_edge_loss_generation)
            in zip(pos_query_edge_loss_generations, neg_query_edge_loss_generations)]

        # (normalized) l2 loss
        query_node_pred_generations_ = [
            torch.bmm(node_similarity_l2[:, num_supports:, :num_supports],
                        one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.device))
            for node_similarity_l2
            in node_similarities_l2]

        # prediction
        query_node_pred_generations = [
            torch.bmm(point_similarity[:, num_supports:, :num_supports],
                        one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.device))
            for point_similarity
            in point_similarities]

        query_node_pred_loss = [
            self.pred_loss(query_node_pred_generation, query_label.long()).mean()
            for query_node_pred_generation
            in query_node_pred_generations_]

        # train accuracy
        query_node_acc_generations = [
            torch.eq(torch.max(query_node_pred_generation, -1)[1], query_label.long()).float().mean()
            for query_node_pred_generation
            in query_node_pred_generations]

        # total loss
        total_loss_generations = [
            query_edge_loss_generation + 0.1 * query_node_pred_loss_
            for (query_edge_loss_generation, query_node_pred_loss_)
            in zip(query_edge_loss_generations, query_node_pred_loss)]

        # compute total loss
        total_loss = []
        num_loss = self.config['num_loss_generation']
        for l in range(num_loss - 1):
            total_loss += [total_loss_generations[l].view(-1) * self.config['generation_weight']]
        total_loss += [total_loss_generations[-1].view(-1) * 1.0]
        total_loss = torch.mean(torch.cat(total_loss, 0))
        return total_loss, query_node_acc_generations, query_edge_loss_generations

    def compute_eval_loss_pred(self,
                               query_edge_losses,
                               query_node_accs,
                               all_label_in_edge,
                               point_similarities,
                               query_edge_mask,
                               evaluation_mask,
                               num_supports,
                               support_label,
                               query_label):
        """
        compute the query classification loss and query classification accuracy
        :param query_edge_losses: container for losses of queries' edges
        :param query_node_accs: container for classification accuracy of queries
        :param all_label_in_edge: ground truth label in edge form of point graph
        :param point_similarities: prediction edges of point graph
        :param query_edge_mask: mask for queries
        :param evaluation_mask: mask for evaluation (for unsupervised setting)
        :param num_supports: number of samples in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :return: query classification loss
                 query classification accuracy
        """

        point_similarity = point_similarities[-1]
        full_edge_loss = self.edge_loss(1 - point_similarity, 1 - all_label_in_edge)

        pos_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * all_label_in_edge * evaluation_mask) / torch.sum(
            query_edge_mask * all_label_in_edge * evaluation_mask)
        neg_query_edge_loss = torch.sum(
            full_edge_loss * query_edge_mask * (1 - all_label_in_edge) * evaluation_mask) / torch.sum(
            query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)

        # weighted loss for balancing pos/neg
        query_edge_loss = pos_query_edge_loss + neg_query_edge_loss

        # prediction
        query_node_pred = torch.bmm(
            point_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.eval_opt['num_ways'], support_label.long(), self.device))

        # test accuracy
        query_node_acc = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()

        query_edge_losses += [query_edge_loss.item()]
        query_node_accs += [query_node_acc.item()]

        return query_node_accs, query_edge_losses


