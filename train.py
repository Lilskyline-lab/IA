#!/usr/bin/env python3
"""
Script d'entraînement HessGpt ULTRA-OPTIMISÉ
Configuration: ~5M paramètres
Dataset: OASST2 (1000 messages)
Durée estimée: 5-15 minutes sur GPU T4
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
import time
from datasets import load_dataset
from tqdm import tqdm

sys.path.append('./Core/Model')
sys.path.append('./Core/Training')

from HessGpt import HessGPT
from transformers import AutoTokenizer

print("="*60)
print("🚀 ENTRAÎNEMENT HessGpt ULTRA-OPTIMISÉ")
print("="*60)

# ============================================
# DÉTECTION GPU + OPTIMISATIONS
# ============================================
if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # OPTIMISATIONS CRITIQUES POUR GPU
    torch.backends.cudnn.benchmark = True  # ← Accélération 10-20%
    torch.backends.cuda.matmul.allow_tf32 = True  # ← Accélération 20-30%
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"\n✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print("✅ Optimisations GPU activées (TF32, cuDNN benchmark)")
else:
    device = 'cpu'
    print("\n⚠️  CPU détecté - Utiliser GPU pour 50x plus rapide!")
    sys.exit(1)  # Forcer GPU car trop lent sur CPU

# ============================================
# CONFIGURATION OPTIMISÉE (5M paramètres)
# ============================================
CONFIG = {
    'vocab_size': 151665,
    'embed_dim': 512,      # ← Augmenté de 32 à 256 (meilleur ratio perf/vitesse)
    'num_heads': 8,        # ← Augmenté de 4 à 8
    'num_layers': 4,       # ← Garde 4 (bon compromis)
    'max_seq_len': 256,    # ← RÉDUIT de 1024 à 256 (2-3x plus rapide!)
    'batch_size': 4,      # ← AUGMENTÉ (très important pour GPU!)
    'num_epochs': 1,
    'learning_rate': 5e-4, # ← Augmenté (convergence plus rapide)
    'dropout': 0.1,
    'gradient_accumulation': 1,  # Pas besoin avec batch_size=32
    'max_texts': 1000,     # Limiter dataset (1000 messages suffisent)
}

# Auto-ajustement batch_size selon RAM GPU
if gpu_memory < 8:
    CONFIG['batch_size'] = 16
elif gpu_memory >= 16:
    CONFIG['batch_size'] = 64

print(f"\n⚙️  Configuration:")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# Estimation paramètres
def estimate_params(config):
    V, d, L = config['vocab_size'], config['embed_dim'], config['num_layers']
    embed = V * d + config['max_seq_len'] * d
    blocks = L * (4 * d * d + 8 * d * d)
    return embed + blocks

num_params_est = estimate_params(CONFIG)
print(f"\n📊 Paramètres estimés: {num_params_est:,} ({num_params_est/1e6:.1f}M)")

# ============================================
# TOKENIZER (avec cache)
# ============================================
print("\n🔤 Chargement tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", 
    trust_remote_code=True
)
CONFIG['vocab_size'] = len(tokenizer)
print(f"✓ Tokenizer chargé (vocab: {len(tokenizer)})")

# ============================================
# DATASET ULTRA-OPTIMISÉ
# ============================================
print("\n📥 Chargement dataset...")
os.makedirs("data", exist_ok=True)

# Essayer de charger depuis cache
cache_file = "data/oasst2_tokenized_cache.pt"

if os.path.exists(cache_file):
    print(f"✓ Chargement depuis cache: {cache_file}")
    cached_data = torch.load(cache_file)
    all_tokens = cached_data['tokens']
    print(f"✓ {len(all_tokens):,} tokens chargés depuis cache")
else:
    # Charger et tokenizer
    dataset = load_dataset("OpenAssistant/oasst2", split="train")
    texts = [item['text'] for item in dataset if item['text']][:CONFIG['max_texts']]
    full_text = "\n\n".join(texts)
    
    print(f"✓ Dataset: {len(texts)} messages, {len(full_text):,} chars")
    print("  Tokenization...")
    
    all_tokens = tokenizer.encode(full_text)
    
    # Sauvegarder cache
    torch.save({'tokens': all_tokens, 'config': CONFIG}, cache_file)
    print(f"✓ Cache sauvegardé: {cache_file}")
    print(f"✓ {len(all_tokens):,} tokens")

# ============================================
# DATASET PYTORCH OPTIMISÉ
# ============================================
class FastTokenDataset(torch.utils.data.Dataset):
    """Dataset optimisé - pré-calcule tous les indices"""
    def __init__(self, tokens, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.num_samples = len(tokens) - seq_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

train_dataset = FastTokenDataset(all_tokens, CONFIG['max_seq_len'])
print(f"✓ Dataset prêt: {len(train_dataset):,} séquences")

# DataLoader optimisé
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=2,  # ← IMPORTANT: charge les données en parallèle
    pin_memory=True,  # ← IMPORTANT: transfert GPU plus rapide
    persistent_workers=True  # ← Garde les workers en vie
)

print(f"✓ DataLoader: {len(train_loader)} batches/epoch")

# ============================================
# MODÈLE
# ============================================
print("\n🤖 Création modèle...")
model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout']
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modèle: {num_params:,} params ({num_params/1e6:.1f}M)")

# Compiler le modèle (PyTorch 2.0+) - ÉNORME gain de perf
# NOTE: 1er batch prend ~60s (compilation), puis 3-4x plus rapide
if hasattr(torch, 'compile') and device == 'cuda':
    print("⚡ Compilation du modèle (PyTorch 2.0)...")
    model = torch.compile(model, mode='reduce-overhead')  # Plus rapide que max-autotune
    print("✓ Modèle compilé (1er batch ~60s, puis accélération)")

torch.cuda.empty_cache()
print(f"✓ GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB utilisés")

# ============================================
# OPTIMIZER OPTIMISÉ
# ============================================
# Utiliser AdamW avec fused=True (plus rapide sur GPU)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True  # ← 10-20% plus rapide sur GPU
)

# Learning rate scheduler (optionnel mais recommandé)
total_steps = len(train_loader) * CONFIG['num_epochs']
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

# ============================================
# BOUCLE D'ENTRAÎNEMENT OPTIMISÉE
# ============================================
print("\n" + "="*60)
print(f"🚀 DÉBUT ENTRAÎNEMENT")
print("="*60)
print(f"Durée estimée: ~{len(train_loader) * 0.5:.0f} secondes")
print("="*60 + "\n")

os.makedirs("checkpoints", exist_ok=True)
start_time = time.time()

model.train()
train_losses = []

# Scaler pour mixed precision (FP16) - 2x plus rapide!
scaler = torch.cuda.amp.GradScaler()

for epoch in range(CONFIG['num_epochs']):
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)  # ← non_blocking pour async
        y = y.to(device, non_blocking=True)
        
        # Mixed Precision Training (FP16)
        with torch.cuda.amp.autocast():
            logits, loss = model(x, targets=y)
        
        # Backward avec gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # ← Plus rapide que zero_grad()
        scheduler.step()
        
        # Stats
        epoch_loss += loss.item()
        train_losses.append(loss.item())
        
        # Affichage toutes les 10 itérations
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"\n✓ Epoch {epoch+1} terminée | Loss: {avg_loss:.4f}")
    
    # Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': CONFIG
    }, f'./checkpoints/checkpoint_epoch_{epoch+1}.pt')

elapsed = time.time() - start_time

# ============================================
# TEST GÉNÉRATION
# ============================================
print("\n" + "="*60)
print("🎉 TEST GÉNÉRATION")
print("="*60)

model.eval()
prompt = "Hi"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)

text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
print(f"\nPrompt: {prompt}")
print(f"Généré: {text}\n")

# ============================================
# STATISTIQUES FINALES
# ============================================
print("="*60)
print("📊 STATISTIQUES")
print("="*60)
print(f"✓ GPU: {gpu_name}")
print(f"✓ Mémoire max: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
print(f"✓ Paramètres: {num_params:,} ({num_params/1e6:.1f}M)")
print(f"✓ Dataset: {len(train_dataset):,} séquences")
print(f"✓ Batch size: {CONFIG['batch_size']}")
print(f"✓ Seq length: {CONFIG['max_seq_len']}")
print(f"✓ Loss initiale: {train_losses[0]:.4f}")
print(f"✓ Loss finale: {train_losses[-1]:.4f}")
print(f"✓ Temps total: {elapsed/60:.1f} min ({elapsed:.0f}s)")
print(f"✓ Temps/batch: {elapsed/len(train_loader):.2f}s")
print("="*60)

# Sauvegarde finale
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'num_params': num_params,
    'train_losses': train_losses,
}, './checkpoints/hessgpt_final.pt')

torch.cuda.empty_cache()
print("\n✅ ENTRAÎNEMENT TERMINÉ!")
print(f"💾 Modèle sauvegardé: checkpoints/hessgpt_final.pt")
print("="*60)
