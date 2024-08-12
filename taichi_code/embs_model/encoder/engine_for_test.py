import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import data.utils as kpts_utils


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def render_anchors_positives_negatives(anchor, positive, negative, similarity, similarity_negative, embs_similarity, embs_similarity_negative, kpts_type="3d"):
    if kpts_type == "3d":
        return render_keypoints_3d_for_tensorboard(anchor[..., [0,1,2,-1]].detach().cpu().numpy(), positive[..., [0,1,2,-1]].detach().cpu().numpy(), negative[..., [0,1,2,-1]].detach().cpu().numpy(), similarity, similarity_negative, embs_similarity, embs_similarity_negative)
    elif kpts_type == "2d":
        return render_keypoints_3d_for_tensorboard(anchor[..., [0,1,-1]].detach().cpu().numpy(), positive[..., [0,1,-1]].detach().cpu().numpy(), negative[..., [0,1,-1]].detach().cpu().numpy(), similarity, similarity_negative, embs_similarity, embs_similarity_negative, kpts_type="2d")
            

def test_step(model, dataloader, loss_fn, device, test_writer, epoch, cfg):
    model.eval()
    loss_fn.eval()

    losses = {}
    losses['triplet_loss'] = AverageMeter()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):

            apn_pairs, ap_similarity, an_similarity, clip_lens = batch_data
            apn_pairs = apn_pairs.to(device)
            an_similarity = an_similarity.to(device)

            # ==== if using stgcn ==== #
            if "joint" not in cfg.data_type_views:
                output_embs = model(apn_pairs[..., 3:])
            else:
                output_embs = model(apn_pairs)
            output_embs = torch.squeeze(output_embs, dim=(-2,-1))
            # ==== stgcn ==== #

            # ==== if using TemporalSimpleModel ==== #
            # if "joint" not in cfg.data_type_views:
            #     anchor_embs = model(apn_pairs[:,0,...,3:])
            #     positive_embs = model(apn_pairs[:,1,...,3:])
            #     negative_embs = model(apn_pairs[:,2,...,3:])
            # else:
            #     anchor_embs = model(apn_pairs[:,0])
            #     positive_embs = model(apn_pairs[:,1])
            #     negative_embs = model(apn_pairs[:,2])
            # output_embs = torch.stack((anchor_embs, positive_embs, negative_embs), dim=1)
            # ==== TemporalSimpleModel ==== #

            triplet_loss, embs_ap_similarity, embs_an_similarity = loss_fn(output_embs, an_similarity, None, None)

            timeline = epoch*len(dataloader)+(batch_idx+1)

            cnt = 0
            for clip in range(len(clip_lens)):
                for f in range(clip_lens[clip]):
                    test_writer.add_video(f"Test: clip_{batch_idx*len(clip_lens)+clip:03} anchor-positive-negative", render_anchors_positives_negatives(apn_pairs[cnt,0], apn_pairs[cnt,1], apn_pairs[cnt,2], ap_similarity[cnt].item(), an_similarity[cnt].item(), embs_ap_similarity[cnt].item(), embs_an_similarity[cnt].item(), kpts_type="3d"), f, fps=cfg.fps)
                    cnt += 1
            
            test_writer.add_scalar('Test: 01 Anchor-Positive Similarity', torch.mean(embs_ap_similarity), timeline)
            test_writer.add_scalar('Test: 01 Anchor-Negative Similarity', torch.mean(embs_an_similarity), timeline)
            test_writer.add_scalar('Test: Origin 01 Anchor-Positive Similarity', torch.mean(ap_similarity), timeline)
            test_writer.add_scalar('Test: Origin 01 Anchor-Negative Similarity', torch.mean(an_similarity), timeline)
            losses['triplet_loss'].update(triplet_loss.item())  
    
            test_writer.add_scalar('Iter Test: Triplet Loss', losses['triplet_loss'].val, timeline)

    return losses['triplet_loss'].avg



def test(model, test_dataloader, loss_fn, cfg, device):

    if cfg.kpts_type_for_train != "mediapipe":
        from data.rendering import render_keypoints_2d_for_tensorboard, render_keypoints_3d_for_tensorboard
        globals()['render_keypoints_2d_for_tensorboard'] = render_keypoints_2d_for_tensorboard
        globals()['render_keypoints_3d_for_tensorboard'] = render_keypoints_3d_for_tensorboard
    else:
        from data.rendering_mediapipe import render_keypoints_3d_for_tensorboard
        globals()['render_keypoints_3d_for_tensorboard'] = render_keypoints_3d_for_tensorboard

    writer = SummaryWriter(f'../logs_for_test')
    
    test_losses = {}
    test_losses['triplet_loss'] = AverageMeter()

    min_test_loss = torch.inf

    for epoch in tqdm(range(1)):
        triplet_loss_avg = test_step(model, test_dataloader, loss_fn, device, writer, epoch, cfg)
        test_losses['triplet_loss'].update(triplet_loss_avg)

        writer.add_scalar('Epoch: Test Triplet Loss', test_losses['triplet_loss'].val, epoch + 1)
    
    writer.add_scalar('Epoch: Avg Test Triplet Loss', test_losses['triplet_loss'].avg, 1)