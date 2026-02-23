#!/usr/bin/env python3
import sys
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

def convert_lokr_to_lora(input_path, output_path, target_rank=None):
    print(f"Loading {input_path}...")
    state_dict = load_file(input_path)
    new_state_dict = {}
    
    # Gruplama için geçici sözlük
    groups = {}
    
    for key, value in state_dict.items():
        # LoKr anahtarlarını bul: lokr_w1, lokr_w2
        if "lokr_" in key:
            base_key = key.replace(".lokr_w1", "").replace(".lokr_w2", "").replace(".alpha", "")
            if base_key not in groups:
                groups[base_key] = {}
            
            if "lokr_w1" in key:
                groups[base_key]["w1"] = value
            elif "lokr_w2" in key:
                groups[base_key]["w2"] = value
            elif "alpha" in key:
                groups[base_key]["alpha"] = value
        else:
            # Normal anahtarları olduğu gibi aktar
            new_state_dict[key] = value

    print(f"Found {len(groups)} LoKr blocks. Converting to LoRA...")

    for base_key, parts in groups.items():
        if "w1" in parts and "w2" in parts:
            w1 = parts["w1"].float()
            w2 = parts["w2"].float()
            
            # Kronecker Product işlemi (LoKr'ın olayı budur)
            # W = w1 (kron) w2
            weight = torch.kron(w1, w2)
            
            # Şimdi bu devasa matrisi standart LoRA'ya (Low Rank) çevireceğiz
            # SVD (Singular Value Decomposition) kullanarak
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            
            # Rank belirle (Eğer belirtilmediyse w1'in boyutunu baz al)
            rank = target_rank if target_rank else min(w1.shape[0], w2.shape[1])
            
            # Matrisi kes (Truncate)
            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]
            
            # LoRA Down ve Up matrislerini oluştur
            # Down = U * sqrt(S)
            # Up = sqrt(S) * Vh
            sqrt_s = torch.diag(torch.sqrt(s))
            lora_down = torch.matmul(u, sqrt_s)
            lora_up = torch.matmul(sqrt_s, vh)
            
            # Anahtarları yeniden adlandır (Draw Things formatı)
            # lokr_w1/w2 yerine lora_down/lora_up
            
            # Genellikle Draw Things 'lora_unet_' veya 'lora_te_' bekler
            # Input key formatına göre ayarlama:
            new_key_base = base_key.replace("lokr_", "lora_")
            
            new_state_dict[f"{new_key_base}.lora_down.weight"] = lora_down.to(dtype=torch.float16)
            new_state_dict[f"{new_key_base}.lora_up.weight"] = lora_up.to(dtype=torch.float16)
            
            # Alpha varsa ekle
            if "alpha" in parts:
                new_state_dict[f"{new_key_base}.alpha"] = parts["alpha"].to(dtype=torch.float16)
            else:
                # Alpha yoksa rank'i alpha olarak ata (standart pratik)
                new_state_dict[f"{new_key_base}.alpha"] = torch.tensor(rank, dtype=torch.float16)
                
            print(f"Converted: {base_key} -> Rank {rank}")

    print(f"Saving to {output_path}...")
    save_file(new_state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lokr_convert.py <input.safetensors> [rank]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = Path(input_file).stem + "_fixed_lora.safetensors"
    
    target_rank = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    convert_lokr_to_lora(input_file, output_file, target_rank)