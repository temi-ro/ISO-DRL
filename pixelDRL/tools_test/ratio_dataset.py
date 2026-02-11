

# from tqdm import tqdm
# import torch
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

# from model import PixelDRL
# from dataset_2d import BraTS2023_Slice_Dataset

if __name__ == "__main__":
    # dataset_path = "/home/temi/ICL/ISO/brats2023/training_raw/Dataset001_BraTS2023/"
    # dataset = BraTS2023_Slice_Dataset(dataset_path, split='val')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ratios = []
    # for i, (images, rl_targets, binary_masks) in enumerate(tqdm(dataloader)):
    #     # Create a histogram of ratio of tumor pixels to total pixels in the ground truth masks
    #     ground_truth = binary_masks.squeeze(0)  # Remove batch dim
    #     total_pixels = ground_truth.numel()
    #     tumor_pixels = torch.sum(ground_truth).item()
    #     ratio = tumor_pixels / total_pixels
    #     ratios.append(ratio)

    # # Plot histogram
    # plt.figure(figsize=(8,6))
    # plt.hist(ratios, bins=20, color='blue', alpha=0.7)
    # plt.title("Histogram of Tumor Pixel Ratios in Ground Truth Masks")
    # plt.xlabel("Ratio of Tumor Pixels to Total Pixels")
    # plt.ylabel("Number of Images")
    # plt.grid(True)
    # # plt.savefig("./pixelDRL/ratio_data/val_tumor_pixel_ratio_histogram.png")
    # plt.close()

    # # Save ratios to a text file
    # with open("./pixelDRL/ratio_data/val_tumor_pixel_ratios.txt", "w") as f:
    #     for ratio in ratios:
    #         f.write(f"{ratio}\n")

    with open("./pixelDRL/ratio_data/val_tumor_pixel_ratios.txt", "r") as f:
        ratios = [float(line.strip()) for line in f.readlines()]
        print(f"Total samples: {len(ratios)}")
        print(f"Min ratio: {min(ratios):.6f}")
        print(f"Max ratio: {max(ratios):.6f}")
        print(f"Mean ratio: {sum(ratios)/len(ratios):.6f}")

        plt.figure(figsize=(8,6))
        plt.hist(ratios, bins=20, color='blue', alpha=0.7)
        plt.title("Histogram of Tumor Pixel Ratios in Ground Truth Masks")
        plt.xlabel("Ratio of Tumor Pixels to Total Pixels")
        plt.ylabel("Number of Images")
        plt.grid(True)
        # plt.savefig("./pixelDRL/ratio_data/val_tumor_pixel_ratio_histogram.png")

    print("\n---\n")
    print("Now checking train ratios...")

    with open("./pixelDRL/ratio_data/train_tumor_pixel_ratios.txt", "r") as f:
        ratios2 = [float(line.strip()) for line in f.readlines()]
        print(f"Total samples: {len(ratios2)}")
        print(f"Min ratio: {min(ratios2):.6f}")
        print(f"Max ratio: {max(ratios2):.6f}")
        print(f"Mean ratio: {sum(ratios2)/len(ratios2):.6f}")
        plt.figure(figsize=(8,6))
        plt.hist(ratios2, bins=20, color='green', alpha=0.7)
        plt.title("Histogram of Tumor Pixel Ratios in Ground Truth Masks (Train)")
        plt.xlabel("Ratio of Tumor Pixels to Total Pixels")
        plt.ylabel("Number of Images")
        plt.grid(True)

    # plt.show()

    # Total mean
    total_ratios = ratios + ratios2
    print(f"\nOverall Mean Ratio: {sum(total_ratios)/len(total_ratios):.6f}")
