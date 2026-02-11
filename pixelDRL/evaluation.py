import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt

from model import PixelDRL  
from dataset_2d import BraTS2023_Slice_Dataset 

SPLIT = 'val_ratio'
NOISE = 0.0
save_path = f"/vol/biomedic2/bglocker_studproj/trm25/checkpoints_14/vis"

# Show over-segmentation

def compute_dice_score(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = BraTS2023_Slice_Dataset(root_dir=args.data_dir, split=SPLIT, noise=NOISE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Test Set: {len(test_dataset)} images.")

    model = PixelDRL(in_channels=2, n_actions=1).to(device) 
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval() 
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return


    dice_scores = []
    with torch.no_grad():
        for i, (image, ground_truth, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image = image.to(device)
            ground_truth = ground_truth.to(device)
            
            # Initialise Mask as ALL ONES (Pruning Strategy)
            # We start with everything selected and the agent removes background.
            current_mask = torch.ones_like(image).to(device)
            
            masks = []

            for t in range(args.t_max):
                state = torch.cat([image, current_mask], dim=1)
                
                policy_logits, value = model(state)
                
                probs = torch.sigmoid(policy_logits)
                
                action = (probs > 0.5).float()
                
                current_mask = current_mask * action 

                if args.save_viz:
                    save_visualization(image, ground_truth, current_mask, i, name_file=f"step_{t}", diff=True)
                masks.append(current_mask.cpu().squeeze().numpy())

            dice = compute_dice_score(current_mask, ground_truth)
            dice_scores.append(dice)
            
            # save_step_visualization(image, ground_truth, masks, i, name_file=f"progress")

    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print("-" * 30)
    print(f"RESULTS FOR {args.checkpoint}")
    if args.save_viz:
        print(f" Visualizations saved to {save_path}")
    print("-" * 30)
    print(f"Images Evaluated: {len(dice_scores)}")
    print(f"Mean Dice Score:  {mean_dice:.4f}")
    print(f"Std Dev:          {std_dice:.4f}")
    print("-" * 30)

# If diff=True, then mark in the visualization where the prediction differs from the ground truth (e.g., false positives in red, false negatives in blue).
def save_visualization(img, gt, pred, idx, name_file="result", diff=False):
    
    img = img.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    pred = pred.cpu().squeeze().numpy()
    
    plt.figure(figsize=(10, 3))
    
    plt.subplot(1, 3, 1)
    plt.title("MRI Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)

    if not diff:
        plt.title("PixelDRL-MG Prediction")
        plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    else:
        rgb_viz = np.zeros((gt.shape[0], gt.shape[1], 3))

        tp = (pred == 1) & (gt == 1)
        rgb_viz[tp] = [1, 1, 1] 

        fp = (pred == 1) & (gt == 0)
        rgb_viz[fp] = [1, 0, 0]

        fn = (pred == 0) & (gt == 1)
        rgb_viz[fn] = [0, 0, 1] 

        # (White=TP, Red=FP, Blue=FN)
        plt.title("PixelDRL-MG Prediction")
        plt.imshow(rgb_viz)
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{idx}_{name_file}.png")
    plt.close()

def save_step_visualization(img, gt, pred_steps, idx, name_file="step_evolution"):
    def to_numpy(x):
        return np.squeeze(x.cpu().numpy())

    img = to_numpy(img)
    gt = to_numpy(gt)

    preds_np = [to_numpy(p) for p in pred_steps]
    
    num_steps = len(preds_np)
    ncols = 2 + num_steps
    ncols = 6

    plt.figure(figsize=(3 * 6, 3 * 2))
    
    plt.subplot(2, ncols, 1)
    plt.title("MRI Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, ncols, 2)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    for i, curr_pred in enumerate(preds_np):
        ax_idx = 3 + i
        plt.subplot(2, ncols, ax_idx)
        
        curr_pred = (curr_pred > 0.5).astype(int)
        
        if i == 0:
            plt.title(f"Step {i}")
            plt.imshow(curr_pred, cmap='gray', vmin=0, vmax=1)
        else:
            prev_pred = (preds_np[i-1] > 0.5).astype(int)
            
            h, w = curr_pred.shape
            rgb_viz = np.zeros((h, w, 3))
            
            stable = (curr_pred == 1) & (prev_pred == 1)
            change = (curr_pred != prev_pred)
            
            rgb_viz[stable] = [1, 1, 1]  # White
            rgb_viz[change] = [1, 0, 0]  # Red
            
            plt.title(f"Step {i}")
            plt.imshow(rgb_viz)
        
        plt.axis('off')

    # 3. Save
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"{idx}_{name_file}.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to BraTS folder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--t_max", type=int, default=10, help="Number of steps for agent")
    parser.add_argument("--save_viz", action="store_true", help="Save PNGs of results")
    
    args = parser.parse_args()
    
    evaluate(args)