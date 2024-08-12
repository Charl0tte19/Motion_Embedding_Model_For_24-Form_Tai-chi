import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import data.utils as kpts_utils
import random


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
            


def save_checkpoint(chk_path, epoch, lr, optimizer, model, loss):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'loss' : loss
    }, chk_path)


def train_step(model, dataloader, optimizer, loss_fn, device, train_writer, epoch, cfg):
    model.train()
    loss_fn.train()

    losses = {}
    losses['triplet_loss'] = AverageMeter()

    for batch_idx, batch_data  in enumerate(dataloader):
        
        apn_pairs, ap_similarity, an_similarity, anchor_form, negative_form = batch_data
        apn_pairs = apn_pairs.to(device)
        an_similarity = an_similarity.to(device)
        anchor_form = anchor_form.to(device)
        negative_form = negative_form.to(device)

        # exit_on = False 
        # if torch.isnan(batch_data).any():
        #     print(f"NaN detected in batch_data")
        #     exit_on = True
        
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
        
        triplet_loss, embs_ap_similarity, embs_an_similarity = loss_fn(output_embs, an_similarity, anchor_form, negative_form)

        # if exit_on:
        #     print("all_losses",all_losses)
        #     for name, param in model.named_parameters():
        #         if torch.isnan(param).any():
        #             print(f"NaN detected in parameter: {name}")
        #         if param.grad is not None and torch.isnan(param.grad).any():
        #             print(f"NaN detected in gradient of parameter: {name}")
        #     exit()

        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        
        timeline = epoch*len(dataloader)+(batch_idx+1)
        
        if (batch_idx+1) == len(dataloader):
            train_writer.add_video("Train: anchor-positive-negative", render_anchors_positives_negatives(apn_pairs[0,0], apn_pairs[0,1], apn_pairs[0,2], ap_similarity[0].item(), an_similarity[0].item(), embs_ap_similarity[0].item(), embs_an_similarity[0].item(), kpts_type="3d"), timeline, fps=cfg.fps)
        
        train_writer.add_scalar('Train: 01 Anchor-Positive Similarity', torch.mean(embs_ap_similarity), timeline)
        train_writer.add_scalar('Train: 01 Anchor-Negative Similarity', torch.mean(embs_an_similarity), timeline)
        train_writer.add_scalar('Train: Origin 01 Anchor-Positive Similarity', torch.mean(ap_similarity), timeline)
        train_writer.add_scalar('Train: Origin 01 Anchor-Negative Similarity', torch.mean(an_similarity), timeline)
        losses['triplet_loss'].update(triplet_loss.item())  
  
        train_writer.add_scalar('Iter Train: Triplet Loss', losses['triplet_loss'].val, timeline)

    return losses['triplet_loss'].avg


def test_step(model, dataloader, loss_fn, device, test_writer, epoch, cfg):
    model.eval()
    loss_fn.eval()

    losses = {}
    losses['triplet_loss'] = AverageMeter()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):

            apn_pairs, ap_similarity, an_similarity, anchor_form, negative_form = batch_data
            apn_pairs = apn_pairs.to(device)
            an_similarity = an_similarity.to(device)
            anchor_form = anchor_form.to(device)
            negative_form = negative_form.to(device)

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

            triplet_loss, embs_ap_similarity, embs_an_similarity = loss_fn(output_embs, an_similarity, anchor_form, negative_form)

            timeline = epoch*len(dataloader)+(batch_idx+1)
            if (batch_idx+1) == len(dataloader):
                idx = random.choice(range(len(apn_pairs)))
                test_writer.add_video("Test: anchor-positive-negative", render_anchors_positives_negatives(apn_pairs[idx,0], apn_pairs[idx,1], apn_pairs[idx,2], ap_similarity[idx].item(), an_similarity[idx].item(), embs_ap_similarity[idx].item(), embs_an_similarity[idx].item(), kpts_type="3d"), timeline, fps=cfg.fps)

            test_writer.add_scalar('Test: 01 Anchor-Positive Similarity', torch.mean(embs_ap_similarity), timeline)
            test_writer.add_scalar('Test: 01 Anchor-Negative Similarity', torch.mean(embs_an_similarity), timeline)
            test_writer.add_scalar('Test: Origin 01 Anchor-Positive Similarity', torch.mean(ap_similarity), timeline)
            test_writer.add_scalar('Test: Origin 01 Anchor-Negative Similarity', torch.mean(an_similarity), timeline)
            losses['triplet_loss'].update(triplet_loss.item())  
    

            test_writer.add_scalar('Iter Test: Triplet Loss', losses['triplet_loss'].val, timeline)

    return losses['triplet_loss'].avg


def train(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, cfg, device):

    if cfg.kpts_type_for_train != "mediapipe":
        from data.rendering import render_keypoints_2d_for_tensorboard, render_keypoints_3d_for_tensorboard
        globals()['render_keypoints_2d_for_tensorboard'] = render_keypoints_2d_for_tensorboard
        globals()['render_keypoints_3d_for_tensorboard'] = render_keypoints_3d_for_tensorboard
    else:
        from data.rendering_mediapipe import render_keypoints_3d_for_tensorboard
        globals()['render_keypoints_3d_for_tensorboard'] = render_keypoints_3d_for_tensorboard

    writer = SummaryWriter(f'../logs')
    
    train_losses = {}
    train_losses['triplet_loss'] = AverageMeter()

    test_losses = {}
    test_losses['triplet_loss'] = AverageMeter()

    min_test_loss = torch.inf

    for epoch in tqdm(range(cfg.num_epochs)):
        triplet_loss_avg = train_step(model, train_dataloader, optimizer, loss_fn, device, writer, epoch, cfg)
        train_losses['triplet_loss'].update(triplet_loss_avg)

        triplet_loss_avg = test_step(model, test_dataloader, loss_fn, device, writer, epoch, cfg)
        test_losses['triplet_loss'].update(triplet_loss_avg)

        writer.add_scalar('Epoch: Train Triplet Loss', train_losses['triplet_loss'].val, epoch + 1)

        writer.add_scalar('Epoch: Test Triplet Loss', test_losses['triplet_loss'].val, epoch + 1)

        if cfg.use_scheduler:
            scheduler.step()

        if (epoch+1)%50 == 0:
            save_checkpoint(f"../results/epoch_{epoch+1}.pth", epoch, optimizer.param_groups[0]["lr"], optimizer, model, test_losses['triplet_loss'].val)

        if min_test_loss > test_losses['triplet_loss'].val:
            min_test_loss = test_losses['triplet_loss'].val
            save_checkpoint(f"../results/best_epoch.pth", epoch, optimizer.param_groups[0]["lr"], optimizer, model, test_losses['triplet_loss'].val)
        