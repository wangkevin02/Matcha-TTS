# train.py
import os
import argparse
import json
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from matcha.models.components.my_flow_matching import CFM_Vec
# Import your modified model classes from the file where you saved them
# from my_model_file import CFM_Vec 
# For this example, I'll paste the classes here directly.
# In a real project, you would import them.
# <PASTE THE CFM_Vec, BASECFM_Vec, Decoder_Vec, etc. classes here>

def setup_distributed(rank, world_size):
    """Initializes the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed environment."""
    destroy_process_group()

class EmbeddingDataset(Dataset):
    def __init__(self, jsonl_path):
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        condition_path = item['pt_save_path']
        label_path = item['label']
        if isinstance(item['pt_save_path'], str):
            condition_emb = torch.load(condition_path, map_location='cpu')
        else: # list of float
            condition_emb = torch.tensor(item['pt_save_path'], dtype=torch.float32)
        if isinstance(item['label'], str):
            label_emb = torch.load(label_path, map_location='cpu')
        else: # list of float
            label_emb = torch.tensor(item['label'], dtype=torch.float32)

        # Create a mask for the condition (assuming 0 padding is not used, so mask is all ones)
        # If you have padding, you need to generate a proper mask here.
        mask = torch.ones(1, condition_emb.shape[1]).float()
        
        return condition_emb, mask, label_emb

def train(rank, world_size, args):
    print(f"Starting training on rank {rank}.")
    setup_distributed(rank, world_size)
    device = rank

    # --- 1. Create Dataset and DataLoader ---
    dataset = EmbeddingDataset(args.data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # --- 2. Define Model and Optimizer ---
    # Example parameters, you should load these from a config file (e.g., Hydra, YAML)
    cfm_params = argparse.Namespace(solver='euler', sigma_min=1e-4)
    decoder_params = {
        "channels": (256, 512, 1024),
        "n_blocks": 2,
        "num_mid_blocks": 2,
        "attention_head_dim": 64,
        "num_heads": 8,
    }

    model = CFM_Vec(
        text_emb_dim=args.text_emb_dim,
        output_dim=args.output_dim,
        cfm_params=cfm_params,
        decoder_params=decoder_params,
        n_spks=1 # Assuming no speaker embeddings for now
    ).to(device)

    model = DDP(model, device_ids=[device])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # --- 3. Training Loop ---
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        for i, (condition, mask, label) in enumerate(dataloader):
            condition = condition.to(device)
            mask = mask.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            
            # Use model.module to access methods of the original model
            loss, _ = model.module.compute_loss(x1=label, mask=mask, mu=condition)
            
            loss.backward()
            optimizer.step()

            if rank == 0 and i % args.log_interval == 0:
                print(f"Epoch: {epoch+1}/{args.epochs} | Batch: {i}/{len(dataloader)} | Loss: {loss.item():.6f}")

        # --- 4. Save Checkpoint ---
        if rank == 0:
            print(f"Saving checkpoint at the end of epoch {epoch+1}")
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            
            # Save the original model's state dict, not the DDP wrapper
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)

    cleanup()
    if rank == 0:
        print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Training for Sequence-to-Vector CFM")
    parser.add_argument('--data_path', type=str, default="./sample_data.jsonl", help='Path to the .jsonl data file.')
    parser.add_argument('--output_dir', type=str, default="./output", help='Directory to save checkpoints.')
    parser.add_argument('--text_emb_dim', type=int, default=1024, help='Dimension of text embeddings.')
    parser.add_argument('--output_dim', type=int, default=256, help='Dimension of the target label embedding.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--log_interval', type=int, default=50, help='Interval for logging training status.')
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")
    
    mp.spawn(train, args=(world_size, args), nprocs=world_size)