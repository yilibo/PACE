#!/usr/bin/env python3
"""Few-shot episodic sampler."""

import numpy as np
import torch


class FewShotEpisodeSampler(torch.utils.data.Sampler):
    """Sample N-way K-shot episodes from a classification dataset."""

    def __init__(self, dataset, cfg, mode, less_iters=False):
        self.dataset = dataset
        self.cfg = cfg
        self.mode = mode
        self.less_iters = less_iters

        self.num_way = int(cfg.FEW_SHOT.N_WAY)
        self.num_support = int(cfg.FEW_SHOT.K_SHOT)
        self.num_queries = int(
            cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS if mode == "train" else cfg.FEW_SHOT.TEST_QUERY_PER_CLASS
        )
        self.samples_per_class = self.num_support + self.num_queries

        self.labels = np.array(dataset._labels)
        self.class_ids = sorted(np.unique(self.labels).tolist())
        self.class_to_indices = {
            c: np.where(self.labels == c)[0].tolist() for c in self.class_ids
        }

    def __len__(self):
        div_factor = self.cfg.NUM_GPUS if not self.cfg.FEW_SHOT.TRAIN_OG_EPISODES else 1
        if self.mode == "train":
            return max(1, int(self.cfg.FEW_SHOT.TRAIN_EPISODES) // max(1, div_factor))
        test_eps = int(self.cfg.FEW_SHOT.TEST_EPISODES)
        if self.less_iters:
            return max(1, test_eps // max(1, self.cfg.NUM_GPUS) // 5)
        return max(1, test_eps // max(1, self.cfg.NUM_GPUS))

    def __iter__(self):
        for _ in range(len(self)):
            chosen_classes = np.random.choice(self.class_ids, self.num_way, replace=False)
            batch_indices = []
            sample_types = []
            batch_label = []

            for epi_cls, cls_id in enumerate(chosen_classes):
                pool = self.class_to_indices[cls_id]
                replace = len(pool) < self.samples_per_class
                selected = np.random.choice(pool, self.samples_per_class, replace=replace).tolist()
                st = (["support"] * self.num_support) + (["query"] * self.num_queries)
                batch_indices.extend(selected)
                sample_types.extend(st)
                batch_label.extend([epi_cls] * self.samples_per_class)

            order = np.random.permutation(len(batch_indices)).tolist()
            index_and_sample_info = [
                (batch_indices[i], batch_label[i], sample_types[i]) for i in order
            ]
            yield index_and_sample_info
