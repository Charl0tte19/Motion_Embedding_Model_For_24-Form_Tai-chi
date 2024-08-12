import torch
from torch import nn
import random


class Triplet_Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.triplet_margin = cfg.triplet_margin
        self.triplet_method = cfg.triplet_method
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.reduction = getattr(cfg, 'reduction', 'mean')

    def cosine_distance(self, p, sp):
        return (1.0 - self.cosine_similarity(p, sp)) / 2.0

    def rbf_distance(self, p, sp):
        gamma = 1.0
        # last dimension is channels
        return (1.0 - torch.exp(-gamma * torch.sum(torch.square(p - sp), dim=-1)))

    def calculate_triplet(self, anchors, positives, negatives, margin, dist_method, anchor_form, negative_form):
        ap_distance = dist_method(anchors, positives)
        an_distance = dist_method(anchors, negatives)
        hinge = torch.clamp(ap_distance - an_distance + margin, min=0.0)
        return hinge, 1.0-ap_distance, 1.0-an_distance

    def forward(self, output_embs, an_similarity, anchor_form, negative_form):
    
        anchors_embs, positives_embs, negatives_embs = output_embs[:,0], output_embs[:,1], output_embs[:,2]
        
        if self.triplet_method == "cosine":
            method = self.cosine_distance
        elif self.triplet_method == "rbf":
            method = self.rbf_distance

        if self.triplet_margin is None:
            loss, embs_ap_similarity, embs_an_similarity = self.calculate_triplet(anchors_embs, positives_embs, negatives_embs, 1.0-an_similarity, method, anchor_form, negative_form)
        else:
            loss, embs_ap_similarity, embs_an_similarity = self.calculate_triplet(anchors_embs, positives_embs, negatives_embs, self.triplet_margin, method, anchor_form, negative_form)
        
        if self.reduction=="mean":
            loss = torch.mean(loss)
        elif self.reduction=="sum":
            loss = torch.sum(loss)


        return loss, embs_ap_similarity, embs_an_similarity
