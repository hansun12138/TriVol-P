import torch
import torch.nn as nn
import torch.nn.functional as F


class TriVolumeLoss(nn.Module):
    """
    Minimal tri-volume contrastive loss for:
        (POI, PROTAC, E3)

    anchor_mode:
        - "protac": only use PROTAC as anchor
        - "poi": only use POI as anchor
        - "e3": only use E3 as anchor
        - "all": average over all three anchors
    """
    def __init__(self, temp=0.07, anchor_mode="protac", label_smoothing=0.0, eps=1e-12):
        super().__init__()
        assert anchor_mode in {"protac", "poi", "e3", "all"}
        self.temp = temp
        self.anchor_mode = anchor_mode
        self.label_smoothing = label_smoothing
        self.eps = eps

    def compute_3modal_volume_batch(self, anchor_feats, feat1_all, feat2_all):
        """
        Returns a [B, B] matrix where:
            volumes[i, j] = V(anchor_i, feat1_j, feat2_j)
        """
        batch_size1 = anchor_feats.shape[0]
        batch_size2 = feat1_all.shape[0]

        aa = torch.einsum('bi,bi->b', anchor_feats, anchor_feats).unsqueeze(1).expand(-1, batch_size2)
        af1 = anchor_feats @ feat1_all.T
        af2 = anchor_feats @ feat2_all.T

        f1f1 = torch.einsum('bi,bi->b', feat1_all, feat1_all).unsqueeze(0).expand(batch_size1, -1)
        f1f2 = torch.einsum('bi,bi->b', feat1_all, feat2_all).unsqueeze(0).expand(batch_size1, -1)

        f2f2 = torch.einsum('bi,bi->b', feat2_all, feat2_all).unsqueeze(0).expand(batch_size1, -1)

        G = torch.stack([
            torch.stack([aa,  af1,  af2],  dim=-1),
            torch.stack([af1, f1f1, f1f2], dim=-1),
            torch.stack([af2, f1f2, f2f2], dim=-1)
        ], dim=-2)  # [B, B, 3, 3]

        G = G.float()
        a = G[..., 0, 0]
        b = G[..., 0, 1]
        c = G[..., 0, 2]
        d = G[..., 1, 0]
        e = G[..., 1, 1]
        f = G[..., 1, 2]
        g = G[..., 2, 0]
        h = G[..., 2, 1]
        i = G[..., 2, 2]

        gram_det = (
            a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g)
        )
        volumes = torch.sqrt(torch.clamp(torch.abs(gram_det), min=self.eps))
        return volumes

    def single_anchor_loss(self, anchor_feats, feat1_feats, feat2_feats):
        """
        Symmetric batch contrastive loss for one anchor choice.
        """
        anchor_feats = F.normalize(anchor_feats, dim=-1)
        feat1_feats = F.normalize(feat1_feats, dim=-1)
        feat2_feats = F.normalize(feat2_feats, dim=-1)

        volumes = self.compute_3modal_volume_batch(anchor_feats, feat1_feats, feat2_feats)

        # smaller volume => higher similarity
        sim_forward = -volumes / self.temp
        sim_reverse = -volumes.T / self.temp

        targets = torch.arange(anchor_feats.size(0), device=anchor_feats.device)

        loss_forward = F.cross_entropy(sim_forward, targets, label_smoothing=self.label_smoothing)
        loss_reverse = F.cross_entropy(sim_reverse, targets, label_smoothing=self.label_smoothing)

        return 0.5 * (loss_forward + loss_reverse)

    def forward(self, z_poi, z_protac, z_e3):
        losses = {
            "poi": self.single_anchor_loss(z_poi, z_protac, z_e3),
            "protac": self.single_anchor_loss(z_protac, z_poi, z_e3),
            "e3": self.single_anchor_loss(z_e3, z_poi, z_protac),
        }

        if self.anchor_mode == "all":
            tri_loss = (losses["poi"] + losses["protac"] + losses["e3"]) / 3.0
        else:
            tri_loss = losses[self.anchor_mode]

        stats = {k: float(v.detach().item()) for k, v in losses.items()}
        return tri_loss, stats

class PairwiseContrastiveLoss(nn.Module):
    """
    Bidirectional pairwise contrastive loss for:
        - PROTAC <-> POI
        - PROTAC <-> E3

    pair_mode:
        - "poi": only PROTAC-POI
        - "e3": only PROTAC-E3
        - "both": average both pairs
    """
    def __init__(self, temp=0.07, pair_mode="both", label_smoothing=0.0):
        super().__init__()
        assert pair_mode in {"poi", "e3", "both"}
        self.temp = temp
        self.pair_mode = pair_mode
        self.label_smoothing = label_smoothing

    def bidirectional_pair_loss(self, feat_a, feat_b):
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)

        logits_ab = feat_a @ feat_b.T / self.temp
        logits_ba = feat_b @ feat_a.T / self.temp

        targets = torch.arange(feat_a.size(0), device=feat_a.device)

        loss_ab = F.cross_entropy(
            logits_ab, targets, label_smoothing=self.label_smoothing
        )
        loss_ba = F.cross_entropy(
            logits_ba, targets, label_smoothing=self.label_smoothing
        )

        return 0.5 * (loss_ab + loss_ba)

    def forward(self, h_poi, h_protac, h_e3):
        loss_protac_poi = self.bidirectional_pair_loss(h_protac, h_poi)
        loss_protac_e3 = self.bidirectional_pair_loss(h_protac, h_e3)

        if self.pair_mode == "poi":
            pair_loss = loss_protac_poi
        elif self.pair_mode == "e3":
            pair_loss = loss_protac_e3
        else:
            pair_loss = 0.5 * (loss_protac_poi + loss_protac_e3)

        stats = {
            "protac_poi": float(loss_protac_poi.detach().item()),
            "protac_e3": float(loss_protac_e3.detach().item()),
        }
        return pair_loss, stats
