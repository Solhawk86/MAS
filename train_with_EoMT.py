import lightly_train

checkpoint_path = "/mnt/mydata/DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
url = f"file://{checkpoint_path}"

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="/home/yangyan/hong/MAS/data/results/weights/2025.10.14",
        model="dinov3/vits16-eomt",
        model_args={
            # Replace with your own url
            "backbone_url": url,
        },
        data={
            "train": {
                "images": "/home/yangyan/hong/MAS/data/dataset/MAS3K/train/images",   # Path to training images
                "masks": "/home/yangyan/hong/MAS/data/dataset/MAS3K/train/masks",     # Path to training masks
            },
            "val": {
                "images": "/home/yangyan/hong/MAS/data/dataset/MAS3K/val/images",     # Path to validation images
                "masks": "/home/yangyan/hong/MAS/data/dataset/MAS3K/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset                    
                0: "background",
                255: "foreground",
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            # "ignore_classes": [0], 
        },
        transform_args={
            "image_size": (518, 518), # (height, width)
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    )
    lightly_train.train()