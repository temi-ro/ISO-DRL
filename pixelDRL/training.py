import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os  
from model import PixelDRL 
from dataset_2d import BraTS2023_Slice_Dataset
from torch.utils.data import DataLoader
import datetime

# --- CONFIGURATION ---
TUMOR_WEIGHT = 0.0 # Weight 30.0 balances ~3% tumor coverage vs 97% background
LR_STEP_SIZE = 20 # Learning rate scheduler step size
LR_INIT = 1e-4 # Initial Learning Rate
ENTROPY_WEIGHT = 0.03
BATCH_SIZE = 8 #
ALPHA_SUPERVISED_ANCHOR = 5.0
GPU = "A100"
ATTEMPT = "33"
TMAX = 10
SPLIT = 'train_ratio'

print(f"Configuration: TUMOR_WEIGHT={TUMOR_WEIGHT}, \nLR_INIT={LR_INIT}, \nENTROPY_WEIGHT={ENTROPY_WEIGHT}, \nBATCH_SIZE={BATCH_SIZE}, ALPHA_SUPERVISED_ANCHOR={ALPHA_SUPERVISED_ANCHOR}, \nGPU={GPU}, \nATTEMPT={ATTEMPT}, \nTMAX={TMAX}")

# --- HELPER: Dice Score Calculation ---
def calculate_dice(pred, target):
    """
    Computes the Dice Coefficient for a single batch.
    """
    if pred.dim() == 3: pred = pred.unsqueeze(1)
    if target.dim() == 3: target = target.unsqueeze(1)

    smooth = 1e-5
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def train_one_episode(model, optimizer, image, ground_truth, t_max=10, gamma=0.95):
    """
    Reinforcement Learning Phase: Iterative Pruning
    """
    model.train()
    batch_size = image.size(0)
    device = image.device
    
    optimizer.zero_grad()

    current_mask = torch.ones_like(image).to(device)
    
    log_probs = []
    values = []
    rewards = []
    entropies = []
    
    pixel_map = ground_truth * (TUMOR_WEIGHT - 1) + 1.0
    scale = 10.0

    supervised_loss_acum = 0.0
    bce_criterion = nn.BCEWithLogitsLoss()

    # --- EPISODE LOOP ---
    for t in range(t_max):
        state = torch.cat([image, current_mask], dim=1)
        
        policy_logits, value_map = model(state)
        
        probs = torch.sigmoid(policy_logits)
        dist = torch.distributions.Bernoulli(probs=probs)
        action = dist.sample() # [B, 1, H, W]
        
        new_mask = current_mask * action
        
        prev_diff = torch.square(current_mask - ground_truth)
        curr_diff = torch.square(new_mask - ground_truth)

        reward = (prev_diff - curr_diff) * pixel_map * scale 
        
        log_probs.append(dist.log_prob(action))
        values.append(value_map.squeeze(1)) 
        entropies.append(dist.entropy())
        rewards.append(reward.squeeze(1)) 

        current_mask = new_mask.detach()

        supervised_loss_acum += bce_criterion(policy_logits, ground_truth.unsqueeze(1))

    
    # --- LOSS CALCULATION ---
    R = torch.zeros_like(values[0])
    policy_loss = 0
    value_loss = 0
    
    for i in reversed(range(t_max)):
        R = rewards[i] + gamma * R
        advantage = R - values[i]
        
        policy_loss -= (log_probs[i] * advantage.detach()).mean()
        policy_loss -= ENTROPY_WEIGHT * entropies[i].mean()

        value_loss += torch.square(advantage).mean()

    supervised_loss = supervised_loss_acum / t_max
    
    total_loss = policy_loss + 0.5 * value_loss + ALPHA_SUPERVISED_ANCHOR * supervised_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return total_loss.item(), torch.stack(rewards).mean().item()

def debug_dice_check(model, dataloader, device, t_max=10):
    model.eval()
    try:
        images, ground_truths, _ = next(iter(dataloader))
        images = images.to(device)
        ground_truths = ground_truths.to(device)
        
        current_mask = torch.ones_like(images).to(device)
        
        with torch.no_grad():
            for t in range(t_max):
                state = torch.cat([images, current_mask], dim=1)
                policy_logits, _ = model(state)
                probs = torch.sigmoid(policy_logits)
                action = (probs > 0.5).float()
                current_mask = current_mask * action
        
        dice_score = calculate_dice(current_mask, ground_truths)
        print(f"[DEBUG] Val Dice: {dice_score:.4f} | GT Pixels: {ground_truths.sum().item():.0f} | Pred Pixels: {current_mask.sum().item():.0f}")
        
    except Exception as e:
        print(f"[DEBUG] Check failed: {e}")
    model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PixelDRL")
    parser.add_argument("--attempt", type=str, default="run1", help="Attempt name")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--resume", type=str, default=None, help="Resume path")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model: 2 Input Channels (Img+Mask), 1 Action Channel (Binary)
    model = PixelDRL(in_channels=2, n_actions=1).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    start_epoch = 0

    # Resume Logic
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            try:
                fname = os.path.basename(args.resume)
                if "epoch" in fname:
                    start_epoch = int(fname.split("epoch")[1].split(".pth")[0]) + 1
            except:
                pass
        else:
            print(f"Error: Checkpoint file '{args.resume}' not found!")
            exit()

    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.9)

    train_dataset = BraTS2023_Slice_Dataset(args.dataset, split=SPLIT)
    val_dataset = BraTS2023_Slice_Dataset(args.dataset, split='val_ratio')
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    # Batch size logic
    per_gpu_batch_size = BATCH_SIZE
    total_batch_size = per_gpu_batch_size * max(1, torch.cuda.device_count())
    
    train_dataloader = DataLoader(train_dataset, batch_size=total_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Starting Training (Attempt: {ATTEMPT})...")
    num_epochs = 200
    
    # To modify:
    save_dir = f"checkpoints_{ATTEMPT}"

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_reward = 0
        count = 0
        
        for batch_idx, (images, rl_targets, _) in enumerate(train_dataloader):
            images = images.to(device)
            rl_targets = rl_targets.to(device)
            
            loss, reward = train_one_episode(model, optimizer, images, rl_targets, t_max=TMAX, gamma=0.95)
            
            epoch_loss += loss
            epoch_reward += reward
            count += 1

        avg_loss = epoch_loss / count
        avg_rw = epoch_reward / count

        print(f"[{datetime.datetime.now()}][*]: Epoch [{epoch}/{num_epochs}] | Loss: {avg_loss:.4f} | Reward: {avg_rw:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        debug_dice_check(model, val_dataloader, device)

        scheduler.step()
        
        if epoch % 1 == 0:
            save_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(save_state, f"{save_dir}/pixelDRL_epoch{epoch}.pth")

    final_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(final_state, f"{save_dir}/pixelDRL_final.pth")


    