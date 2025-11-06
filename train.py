#!/usr/bin/env python3
"""
Script d'entraînement HessGpt (~100M paramètres)
Optimisé pour CPU / Codespaces avec tokenizer Qwen 0.5B
Dataset: OASST2
Durée estimée: 8-12h sur CPU standard
"""

import torch
import sys
import os
from pathlib import Path
import time
from datasets import load_dataset

# Ajout des chemins locaux
sys.path.append('./Core/Model')
sys.path.append('./Training')

from HessGpt import GPT2Model
from training import TextDataset, GPT2Trainer

# Import du tokenizer Qwen
from transformers import AutoTokenizer

print("="*60)
print("🚀 ENTRAÎNEMENT HessGpt (≈100M paramètres) sur CPU")
print("="*60)

# Configuration pour ~100M paramètres
CONFIG = {
    'vocab_size': 151936,  # Tokenizer Qwen 0.5B (121k étendu)
    'embed_dim': 768,      # Dimension standard GPT-2
    'num_heads': 12,       # Nombre de têtes d'attention
    'num_layers': 12,      # Nombre de blocs Transformer (12 pour ~100M)
    'max_seq_len': 512,    # Longueur de séquence raisonnable
    'batch_size': 2,       # CPU: petit batch
    'num_epochs': 3,
    'learning_rate': 3e-4,
    'dropout': 0.1,
}

print("\n⚙️  Configuration:")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# Estimation des paramètres
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
    # - LayerNorms: ~2*d (négligeable)
    block_params = (4 * d * d) + (8 * d * d)
    
    # Total
    total = embed_params + (L * block_params)
    return total

estimated = estimate_params(CONFIG)
print(f"\n📊 Paramètres estimés: {estimated:,} ({estimated/1e6:.1f}M)")

# 1. Tokenizer Qwen 0.5B
print("\n🔤 Chargement du tokenizer Qwen 0.5B...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
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

# 2. Dataset OASST2
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
            # Ajouter le texte de chaque message
            texts.append(item['text'])
    
    # Joindre tous les textes
    full_text = "\n\n".join(texts[:50000])  # Limiter pour CPU (50k premiers messages)
    
    # Sauvegarder pour réutilisation
    with open("data/oasst2_train.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"✓ Texte préparé: {len(full_text):,} caractères")
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

# 3. Adapter le Dataset pour le tokenizer Qwen
print("\n📚 Préparation du dataset...")

class QwenTextDataset(torch.utils.data.Dataset):
    """Dataset adapté pour le tokenizer Qwen"""
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
        
        # Input et target
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

train_dataset = QwenTextDataset(full_text, tokenizer, seq_len=CONFIG['max_seq_len'])
print(f"✓ {len(train_dataset)} séquences prêtes")

# 4. Modèle HessGpt (~100M)
print("\n🤖 Création du modèle HessGpt...")
model = GPT2Model(
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

# 5. Sauvegardes
os.makedirs("checkpoints", exist_ok=True)

# 6. Trainer adapté
print("\n🏋️  Initialisation du Trainer...")

class HessGptTrainer:
    """Trainer simplifié pour HessGpt"""
    def __init__(self, model, train_dataset, learning_rate, batch_size, num_epochs, device='cpu', checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.train_losses = []
        
    def train(self, save_every=1):
        """Entraîne le modèle"""
        self.model.train()
        
        # DataLoader
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        total_steps = len(dataloader) * self.num_epochs
        step = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n📖 Epoch {epoch+1}/{self.num_epochs}")
            epoch_loss = 0
            
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward
                logits, loss = self.model(x, targets=y)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Stats
                epoch_loss += loss.item()
                self.train_losses.append(loss.item())
                step += 1
                
                # Affichage
                if batch_idx % 10 == 0:
                    progress = (step / total_steps) * 100
                    print(f"  [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} | Progress: {progress:.1f}%")
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"✓ Epoch {epoch+1} terminée | Loss moyenne: {avg_loss:.4f}")
            
            # Sauvegarde
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"💾 Checkpoint sauvegardé: {checkpoint_path}")

trainer = HessGptTrainer(
    model=model,
    train_dataset=train_dataset,
    learning_rate=CONFIG['learning_rate'],
    batch_size=CONFIG['batch_size'],
    num_epochs=CONFIG['num_epochs'],
    device='cpu',
    checkpoint_dir='./checkpoints'
)

# 7. Entraînement
print("\n" + "="*60)
print("🚀 DÉBUT DE L'ENTRAÎNEMENT (CPU)")
print("="*60)
print("⏱️  Estimé: 8-12 heures")
print("💡 Astuce: Fermez tout ce qui consomme CPU dans Codespaces.")
print("="*60 + "\n")

start = time.time()

try:
    trainer.train(save_every=1)
except KeyboardInterrupt:
    print("\n⚠️  Interrompu par l'utilisateur")
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

elapsed = time.time() - start

# 8. Test de génération
print("\n" + "="*60)
print("🎉 TEST DE GÉNÉRATION")
print("="*60)

model.eval()
prompt = "Once upon a time"
print(f"\nPrompt: '{prompt}'")

input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=30, temperature=0.8)

text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
print(f"\nGénéré:\n{text}\n")

# 9. Statistiques
print("="*60)
print("📊 STATISTIQUES")
print("="*60)
print(f"✓ Paramètres: {num_params:,} ({num_params/1e6:.1f}M)")
print(f"✓ Vocabulaire: {CONFIG['vocab_size']:,}")
print(f"✓ Séquences: {len(train_dataset)}")
print(f"✓ Epochs: {CONFIG['num_epochs']}")
if trainer.train_losses:
    print(f"✓ Loss initiale: {trainer.train_losses[0]:.4f}")
    print(f"✓ Loss finale: {trainer.train_losses[-1]:.4f}")
print(f"✓ Temps: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} heures)")
print("="*60)

# 10. Sauvegarde finale
final_path = './checkpoints/hessgpt_final.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'num_params': num_params,
    'train_losses': trainer.train_losses,
}, final_path)
print(f"\n✓ Modèle sauvegardé: {final_path}")

print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ!")
print("="*60)
print("\n💡 Pour un entraînement plus rapide:")
print("   → Utilisez Google Colab avec GPU (T4/A100)")
print("   → GPU = 50-100× plus rapide qu'un CPU")
print("\n📁 Fichiers générés:")
print(f"   → Checkpoints: {CONFIG['num_epochs']} fichiers dans ./checkpoints/")
print(f"   → Modèle final: {final_path}")
print("="*60)