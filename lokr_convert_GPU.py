#!/usr/bin/env python3
import sys
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import time

def convert_lokr_to_lora(input_path, output_path, target_rank=None):
    # Mac GPU (MPS) hızlandırmasını kontrol et
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Mac GPU (Metal Performance Shaders) Hızlandırması AKTİF!")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU bulunamadı, CPU kullanılıyor (Yavaş olabilir).")

    print(f"Loading {input_path}...")
    state_dict = load_file(input_path)
    new_state_dict = {}
    
    groups = {}
    
    # Gruplama
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
                # Verileri GPU'ya taşı
                w1 = parts["w1"].to(device).float()
                w2 = parts["w2"].to(device).float()
                
                # Kronecker çarpımı (GPU üzerinde çok daha hızlıdır)
                weight = torch.kron(w1, w2)
                
                # SVD Ayrıştırması
                # Not: Çok büyük matrislerde MPS bazen hata verebilir, 
                # bu durumda otomatik CPU'ya düşecek bir koruma ekliyoruz.
                try:
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                except Exception as e:
                    print(f"\n⚠️ Block {processed}: GPU SVD failed, falling back to CPU for this block...")
                    weight = weight.cpu()
                    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                    u = u.to(device)
                    s = s.to(device)
                    vh = vh.to(device)

                # Rank belirle
                rank = target_rank if target_rank else min(w1.shape[0], w2.shape[1])
                
                # Kesme işlemi (Truncate)
                u = u[:, :rank]
                s = s[:rank]
                vh = vh[:rank, :]
                
                # Matris çarpımı (GPU'da şimşek hızındadır)
                sqrt_s = torch.diag(torch.sqrt(s))
                lora_down = torch.matmul(u, sqrt_s)
                lora_up = torch.matmul(sqrt_s, vh)
                
                # Sonucu CPU'ya geri al ve FP16'ya çevir (Hafıza tasarrufu)
                new_key_base = base_key.replace("lokr_", "lora_")
                
                new_state_dict[f"{new_key_base}.lora_down.weight"] = lora_down.cpu().to(dtype=torch.float16)
                new_state_dict[f"{new_key_base}.lora_up.weight"] = lora_up.cpu().to(dtype=torch.float16)
                
                if "alpha" in parts:
                    new_state_dict[f"{new_key_base}.alpha"] = parts["alpha"].to(dtype=torch.float16)
                else:
                    new_state_dict[f"{new_key_base}.alpha"] = torch.tensor(rank, dtype=torch.float16)
                
                # İlerleme çubuğu benzeri bilgi
                sys.stdout.write(f"\rConverting: {processed}/{total_blocks} blocks... ({(processed/total_blocks)*100:.1f}%)")
                sys.stdout.flush()
                
                # VRAM temizliği (Mac'in şişmemesi için)
                del w1, w2, weight, u, s, vh, lora_down, lora_up
                if device.type == "mps":
                    torch.mps.empty_cache()

            except Exception as e:
                print(f"\nError converting block {base_key}: {e}")
                continue

    print(f"\n\nSaving to {output_path}...")
    save_file(new_state_dict, output_path)
    elapsed = time.time() - start_time
    print(f"Done! Total time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lokr_convert.py <input.safetensors> [rank]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = Path(input_file).stem + "_fixed_lora.safetensors"
    
    target_rank = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    convert_lokr_to_lora(input_file, output_file, target_rank)