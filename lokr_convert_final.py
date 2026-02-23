#!/usr/bin/env python3
import sys
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import time

def convert_lokr_to_lora(input_path, output_path, target_rank=None):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Mac GPU (Metal) Hızlandırması AKTİF!")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU bulunamadı, CPU kullanılıyor.")

    print(f"Loading {input_path}...")
    state_dict = load_file(input_path)
    new_state_dict = {}
    groups = {}
    
    for key, value in state_dict.items():
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
            new_state_dict[key] = value

    total_blocks = len(groups)
    print(f"Found {total_blocks} LoKr blocks. Starting GPU conversion...")
    start_time = time.time()
    processed = 0

    for base_key, parts in groups.items():
        processed += 1
        if "w1" in parts and "w2" in parts:
            try:
                w1 = parts["w1"].to(device).float()
                w2 = parts["w2"].to(device).float()
                
                weight = torch.kron(w1, w2)
                
                try:
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                except Exception as e:
                    weight = weight.cpu()
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                    u, s, vh = u.to(device), s.to(device), vh.to(device)

                rank = target_rank if target_rank else min(w1.shape[0], w2.shape[1])
                
                u = u[:, :rank]
                s = s[:rank]
                vh = vh[:rank, :]
                
                sqrt_s = torch.diag(torch.sqrt(s))
                
                # DÜZELTME 1: YÖNLER DOĞRU BAĞLANDI
                # U * sqrt(S) = Çıkış Matrisi (lora_up) -> [out_dim, rank]
                # sqrt(S) * Vh = Giriş Matrisi (lora_down) -> [rank, in_dim]
                lora_up = torch.matmul(u, sqrt_s)     
                lora_down = torch.matmul(sqrt_s, vh)   
                
                new_key_base = base_key.replace("lokr_", "lora_")
                
                new_state_dict[f"{new_key_base}.lora_down.weight"] = lora_down.cpu().to(dtype=torch.float16)
                new_state_dict[f"{new_key_base}.lora_up.weight"] = lora_up.cpu().to(dtype=torch.float16)
                
                # DÜZELTME 2: 0 BOYUTLU (EMPTY) TENSOR HATASI GİDERİLDİ
                if "alpha" in parts:
                    alpha_t = parts["alpha"].cpu()
                    if alpha_t.dim() == 0:
                        alpha_t = alpha_t.unsqueeze(0) # Boş boyutu 1D array yapar (Draw Things çökmesini engeller)
                    new_state_dict[f"{new_key_base}.alpha"] = alpha_t.to(dtype=torch.float16)
                else:
                    new_state_dict[f"{new_key_base}.alpha"] = torch.tensor([rank], dtype=torch.float16)
                
                sys.stdout.write(f"\rConverting: {processed}/{total_blocks} blocks... ({(processed/total_blocks)*100:.1f}%)")
                sys.stdout.flush()
                
                del w1, w2, weight, u, s, vh, lora_down, lora_up
                if device.type == "mps":
                    torch.mps.empty_cache()

            except Exception as e:
                print(f"\nError converting block {base_key}: {e}")
                continue

    save_file(new_state_dict, output_path)
    print(f"\nDone! Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = Path(input_file).stem + "_fixed_lora.safetensors"
    target_rank = int(sys.argv[2]) if len(sys.argv) > 2 else None
    convert_lokr_to_lora(input_file, output_file, target_rank)