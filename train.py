#!/usr/bin/env python3
"""
Script d'entraînement HessGpt (~100M paramètres)
Optimisé pour GPU CUDA avec tokenizer Qwen 0.5B
Dataset: OASST2
Durée estimée: 
  - GPU T4 (Colab gratuit): ~1-2 heures
  - GPU A100: ~20-30 minutes
  - CPU: 8-12 heures
"""

import torch
import sys
import os
from pathlib import Path
import time
from datasets import load_dataset

# Ajout des chemins locaux (CORRECTS selon ta structure)
sys.path.append('./Core/Model')
sys.path.append('./Core/Training')

# Import du MODÈLE depuis Core/Model/HessGpt.py
from HessGpt import HessGPT

# Import des OUTILS depuis Core/Training/training.py
from training import GPT2Trainer

# Import du tokenizer Qwen
from transformers import AutoTokenizer

print("="*60)
print("🚀 ENTRAÎNEMENT HessGpt (≈100M paramètres)")
print("="*60)

# ============================================
# DÉTECTION GPU/CPU
# ============================================
if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU DÉTECTÉ: {gpu_name}")
    print(f"   Mémoire: {gpu_memory:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    device = 'cpu'
    print("\n⚠️  Aucun GPU détecté - Utilisation du CPU")
    print("   💡 Pour accélérer: Utilisez Google Colab avec GPU")

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'vocab_size': 151665,  # Tokenizer Qwen 0.5B
    'embed_dim': 64,      # Dimension standard GPT-2
    'num_heads': 8,       # Nombre de têtes d'attention
    'num_layers': 4,      # Nombre de blocs Transformer (12 pour ~100M)
    'max_seq_len': 1024,    # Longueur de séquence
    'batch_size': 2 if device == 'cuda' else 2,  # GPU: 8, CPU: 2
    'num_epochs': 1 if device == 'cuda' else 1,  # Plus d'epochs sur GPU
    'learning_rate': 3e-4,
    'dropout': 0.1,
}

# Ajuster batch_size selon la mémoire GPU disponible
if device == 'cuda':
    if gpu_memory < 8:  # Moins de 8GB (ex: Colab gratuit T4)
        CONFIG['batch_size'] = 4
        print("   📊 GPU < 8GB détecté → batch_size = 4")
    elif gpu_memory >= 16:  # GPU puissant (A100, etc.)
        CONFIG['batch_size'] = 16
        print("   📊 GPU >= 16GB détecté → batch_size = 16")

print(f"\n⚙️  Configuration ({device.upper()}):")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# ============================================
# ESTIMATION DES PARAMÈTRES
# ============================================
def estimate_params(config):
    """Estime le nombre de paramètres du modèle"""
    V = config['vocab_size']
    d = config['embed_dim']
    L = config['num_layers']
    
    # Embeddings (token + position)
    embed_params = V * d + config['max_seq_len'] * d
    
    # Chaque Transformer Block:
    # - MultiHeadAttention: 4*d*d (Q,K,V,O projections)
    # - FFN: 2 * (d * 4d) = 8*d*d
    block_params = (4 * d * d) + (8 * d * d)
    
    # Total
    total = embed_params + (L * block_params)
    return total

estimated = estimate_params(CONFIG)
print(f"\n📊 Paramètres estimés: {estimated:,} ({estimated/1e6:.1f}M)")

# Estimation mémoire GPU
if device == 'cuda':
    # ~4 bytes par paramètre (float32)
    # x2 pour gradients
    # x1.5 pour optimizer states (Adam)
    estimated_memory = (estimated * 4 * 3.5) / 1e9
    print(f"   Mémoire GPU estimée: ~{estimated_memory:.1f} GB")
    if estimated_memory > gpu_memory * 0.8:
        print(f"   ⚠️  ATTENTION: Risque de dépassement mémoire!")
        print(f"   💡 Réduisez batch_size ou num_layers si erreur OOM")

# ============================================
# 1. TOKENIZER
# ============================================
print("\n🔤 Chargement du tokenizer Qwen 0.5B...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", 
        trust_remote_code=True
    )
    print(f"✓ Tokenizer chargé (vocab_size: {len(tokenizer)})")
    
    # Vérifier que la taille du vocabulaire correspond
    actual_vocab = len(tokenizer)
    if actual_vocab != CONFIG['vocab_size']:
        print(f"⚠️  Ajustement vocab_size: {CONFIG['vocab_size']} → {actual_vocab}")
        CONFIG['vocab_size'] = actual_vocab
        
except Exception as e:
    print(f"❌ Erreur lors du chargement du tokenizer: {e}")
    print("💡 Installation requise: pip install transformers")
    sys.exit(1)

# ============================================
# 2. DATASET OASST2
# ============================================
print("\n📥 Chargement du dataset OASST2...")
os.makedirs("data", exist_ok=True)

try:
    # Charger OASST2 depuis HuggingFace
    print("Téléchargement de OASST2...")
    dataset = load_dataset("OpenAssistant/oasst2", split="train")
    print(f"✓ Dataset chargé: {len(dataset)} conversations")
    
    # Extraire et préparer le texte
    print("Préparation du texte...")
    texts = []
    
    for item in dataset:
        if item['text']:
            texts.append(item['text'])
    
    # Limiter selon device (GPU peut gérer plus)
    max_texts = 2000 if device == 'cuda' else 5000
    full_text = "\n\n".join(texts[:max_texts])
    
    # Sauvegarder pour réutilisation
    with open("data/oasst2_train.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"✓ Texte préparé: {len(full_text):,} caractères ({max_texts} messages)")
    print(f"✓ Sauvegardé dans: data/oasst2_train.txt")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement de OASST2: {e}")
    print("💡 Installation requise: pip install datasets")
    
    # Fallback: chercher le fichier local
    if os.path.exists("data/oasst2_train.txt"):
        print("✓ Utilisation du fichier local data/oasst2_train.txt")
        with open("data/oasst2_train.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        print("❌ Aucun dataset disponible. Sortie.")
        sys.exit(1)

# ============================================
# 3. DATASET ADAPTÉ POUR QWEN
# ============================================
print("\n📚 Préparation du dataset...")

class QwenTextDataset(torch.utils.data.Dataset):
    """
    Dataset adapté pour le tokenizer Qwen
    Compatible avec GPT2Trainer de Core/Training/training.py
    
    IMPORTANT: Cette classe est nécessaire car le tokenizer Qwen
    a une API différente du tokenizer BPE original dans training.py
    """
    def __init__(self, text, tokenizer, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Tokenizer tout le texte
        print("  Tokenization en cours...")
        self.tokens = tokenizer.encode(text)
        print(f"  ✓ {len(self.tokens):,} tokens")
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        # Extraire une séquence de tokens
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        
        # Input et target (décalé de 1 pour next-token prediction)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

train_dataset = QwenTextDataset(
    full_text, 
    tokenizer, 
    seq_len=CONFIG['max_seq_len']
)
print(f"✓ {len(train_dataset)} séquences prêtes")

# ============================================
# 4. MODÈLE HessGpt
# ============================================
print("\n🤖 Création du modèle HessGpt...")
model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout']
)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modèle initialisé: {num_params:,} paramètres ({num_params/1e6:.1f}M)")

if abs(num_params - estimated) / estimated > 0.1:
    print(f"⚠️  Différence avec estimation: {estimated:,} vs {num_params:,}")

# Déplacer le modèle sur GPU si disponible
if device == 'cuda':
    model = model.to(device)
    print(f"✓ Modèle déplacé sur GPU")
    
    # Vérifier la mémoire GPU utilisée
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1e9
    print(f"   Mémoire GPU utilisée: {allocated:.2f} GB")

# ============================================
# 5. DOSSIER DE SAUVEGARDE
# ============================================
os.makedirs("checkpoints", exist_ok=True)

# ============================================
# 6. TRAINER
# Utilise la classe GPT2Trainer de Core/Training/training.py
# ============================================
print("\n🏋️  Initialisation du Trainer...")

trainer = GPT2Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=None,  # Pas de validation pour économiser du temps
    learning_rate=CONFIG['learning_rate'],
    batch_size=CONFIG['batch_size'],
    num_epochs=CONFIG['num_epochs'],
    device=device,  # ← CUDA ou CPU automatique
    checkpoint_dir='./checkpoints'
)

# ============================================
# 7. ENTRAÎNEMENT
# ============================================
print("\n" + "="*60)
print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT ({device.upper()})")
print("="*60)

if device == 'cuda':
    print(f"⚡ Accélération GPU active!")
    print(f"   GPU: {gpu_name}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Durée estimée: 1-2 heures (T4) ou 20-30 min (A100)")
else:
    print(f"⏱️  Mode CPU - Durée estimée: 8-12 heures")
    print("💡 Astuce: Fermez tout ce qui consomme CPU.")

print("="*60 + "\n")

start = time.time()

try:
    trainer.train(save_every=1)
except KeyboardInterrupt:
    print("\n⚠️  Interrompu par l'utilisateur")
except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"\n❌ ERREUR: Mémoire GPU insuffisante!")
        print(f"💡 Solutions:")
        print(f"   1. Réduire batch_size (actuellement: {CONFIG['batch_size']})")
        print(f"   2. Réduire max_seq_len (actuellement: {CONFIG['max_seq_len']})")
        print(f"   3. Réduire num_layers (actuellement: {CONFIG['num_layers']})")
        print(f"   4. Utiliser un GPU plus puissant")
    else:
        print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

elapsed = time.time() - start

# ============================================
# 8. TEST DE GÉNÉRATION
# ============================================
print("\n" + "="*60)
print("🎉 TEST DE GÉNÉRATION")
print("="*60)

model.eval()
prompt = "Once upon a time"
print(f"\nPrompt: '{prompt}'")

input_ids = tokenizer.encode(prompt, return_tensors="pt")

if device == 'cuda':
    input_ids = input_ids.to(device)

with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=30, temperature=0.8)

text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
print(f"\nGénéré:\n{text}\n")

# ============================================
# 9. STATISTIQUES
# ============================================
print("="*60)
print("📊 STATISTIQUES")
print("="*60)
print(f"✓ Device: {device.upper()}")
if device == 'cuda':
    print(f"✓ GPU: {gpu_name}")
    max_memory = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"✓ Mémoire GPU max utilisée: {max_memory:.2f} GB")
print(f"✓ Paramètres: {num_params:,} ({num_params/1e6:.1f}M)")
print(f"✓ Vocabulaire: {CONFIG['vocab_size']:,}")
print(f"✓ Séquences: {len(train_dataset)}")
print(f"✓ Batch size: {CONFIG['batch_size']}")
print(f"✓ Epochs: {CONFIG['num_epochs']}")
if trainer.train_losses:
    print(f"✓ Loss initiale: {trainer.train_losses[0]:.4f}")
    print(f"✓ Loss finale: {trainer.train_losses[-1]:.4f}")
print(f"✓ Temps: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} heures)")

if device == 'cuda':
    speedup = 8 * 60 / (elapsed / 60)  # Comparé à 8h sur CPU
    print(f"✓ Accélération vs CPU: ~{speedup:.0f}x plus rapide!")

print("="*60)

# ============================================
# 10. SAUVEGARDE FINALE
# ============================================
final_path = './checkpoints/hessgpt_final.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'num_params': num_params,
    'train_losses': trainer.train_losses,
    'device': device,
}, final_path)
print(f"\n✓ Modèle sauvegardé: {final_path}")

# Nettoyage GPU
if device == 'cuda':
    torch.cuda.empty_cache()
    print("✓ Cache GPU nettoyé")

print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ!")
print("="*60)

if device == 'cpu':
    print("\n💡 Pour un entraînement plus rapide:")
    print("   → Utilisez Google Colab avec GPU (gratuit)")
    print("   → GPU T4 = 50-100× plus rapide qu'un CPU")
    print("   → Tuto: https://colab.research.google.com")

print("\n📁 Fichiers générés:")
print(f"   → Checkpoints: {CONFIG['num_epochs']} fichiers dans ./checkpoints/")
print(f"   → Modèle final: {final_path}")
print("="*60)
