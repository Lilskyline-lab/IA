#HessGpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransformerBlock.transformer_block import TransformerBlock

# ============================================
# MODÈLE HessGPT COMPLET
# ============================================

class HessGPT(nn.Module):
    """
    Modèle HessGPT - Architecture Transformer personnalisée
    
    Architecture :
    - Token Embeddings + Position Embeddings
    - N Transformer Blocks
    - Layer Norm finale
    - Output Head (projection vers vocabulaire)
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=1024,
        dropout=0.1
    ):
        """
        Args:
            vocab_size (int): Taille du vocabulaire
            embed_dim (int): Dimension des embeddings
            num_heads (int): Nombre de têtes d'attention
            num_layers (int): Nombre de Transformer Blocks
            max_seq_len (int): Longueur max de séquence
            dropout (float): Taux de dropout
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks (empiler N blocs)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer Norm finale
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Output Head (projection vers vocabulaire)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Partager les poids entre token_embeddings et output_head
        self.output_head.weight = self.token_embeddings.weight
        
        # Initialisation des poids
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialisation des poids"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - IDs des tokens
            targets: [batch_size, seq_len] - Targets pour calculer la loss (optionnel)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] - Prédictions
            loss: Scalar (si targets fourni)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Embeddings
        token_embeds = self.token_embeddings(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device)
        position_embeds = self.position_embeddings(positions)
        x = self.dropout(token_embeds + position_embeds)
        
        # 2. Créer le masque causal
        mask = self.create_causal_mask(seq_len, device=input_ids.device)
        
        # 3. Passer à travers tous les Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # 4. Layer Norm finale
        x = self.ln_final(x)
        
        # 5. Output Head (projection vers vocabulaire)
        logits = self.output_head(x)
        
        # 6. Calculer la loss si targets fourni
        loss = None
        if targets is not None:
            # Reshape pour calculer la cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
        
        return logits, loss
    
    def create_causal_mask(self, seq_len, device):
        """Crée un masque causal triangulaire"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Génération de texte (autoregressive)
        
        Args:
            input_ids: [batch_size, seq_len] - Prompt
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la randomness (1.0 = normal, <1 = plus déterministe)
            top_k: Si fourni, ne garde que les top-k tokens les plus probables
        
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Tronquer si trop long
                input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self.forward(input_ids_cond)
                
                # Prendre les logits du dernier token
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling (optionnel)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Softmax pour obtenir les probabilités
                probs = F.softmax(logits, dim=-1)
                
                # Sampler le prochain token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter à la séquence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ============================================
# TESTS
# ============================================

def test_hessgpt_model():
    """Test du modèle HessGPT complet"""
    print("\n" + "="*60)
    print("TEST 1: HessGPT Model - Forward Pass")
    print("="*60)
    
    # Paramètres
    vocab_size = 300
    batch_size = 2
    seq_len = 10
    
    # Créer le modèle (petit pour tester)
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=128
    )
    
    # Input aléatoire
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"✓ Input shape: {input_ids.shape}")
    
    # Forward pass
    logits, loss = model(input_ids)
    
    print(f"✓ Logits shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {vocab_size}]")
    
    # Vérifier les shapes
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"✓ Shape correcte!")
    
    # Nombre de paramètres
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Nombre de paramètres: {num_params:,}")


def test_with_loss():
    """Test avec calcul de la loss"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass avec Loss")
    print("="*60)
    
    vocab_size = 300
    batch_size = 2
    seq_len = 10
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4
    )
    
    # Input et targets
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Targets shape: {targets.shape}")
    
    # Forward avec loss
    logits, loss = model(input_ids, targets)
    
    print(f"\n✓ Logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"  (Loss aléatoire ~{math.log(vocab_size):.2f} au début)")


def test_generation():
    """Test de génération de texte"""
    print("\n" + "="*60)
    print("TEST 3: Génération de texte")
    print("="*60)
    
    vocab_size = 300
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2
    )
    
    # Prompt (quelques tokens)
    prompt = torch.randint(0, vocab_size, (1, 5))
    
    print(f"✓ Prompt shape: {prompt.shape}")
    print(f"✓ Prompt tokens: {prompt[0].tolist()}")
    
    # Générer 10 nouveaux tokens
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    
    print(f"\n✓ Generated shape: {generated.shape}")
    print(f"✓ Generated tokens: {generated[0].tolist()}")
    print(f"✓ Génération réussie! ({generated.shape[1] - prompt.shape[1]} nouveaux tokens)")


def test_hessgpt_20m():
    """Test avec configuration 20M paramètres"""
    print("\n" + "="*60)
    print("TEST 4: HessGPT 20M paramètres")
    print("="*60)
    
    # Configuration 20M
    model = HessGPT(
        vocab_size=20000,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=512
    )
    
    print(f"✓ Modèle créé avec succès!")
    print(f"  - Vocab size: {model.vocab_size}")
    print(f"  - Embed dim: {model.embed_dim}")
    print(f"  - Num heads: {model.num_heads}")
    print(f"  - Num layers: {model.num_layers}")
    print(f"  - Max seq len: {model.max_seq_len}")
    
    # Nombre de paramètres
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Nombre total de paramètres: {num_params:,}")
    
    # Détails
    embeddings_params = sum(p.numel() for p in model.token_embeddings.parameters())
    embeddings_params += sum(p.numel() for p in model.position_embeddings.parameters())
    
    blocks_params = sum(p.numel() for p in model.blocks.parameters())
    
    print(f"\n📊 Répartition:")
    print(f"  - Embeddings: {embeddings_params:,}")
    print(f"  - {model.num_layers} Transformer Blocks: {blocks_params:,}")
    print(f"  - Output partagé avec embeddings")
    
    # Test rapide
    input_ids = torch.randint(0, 20000, (1, 10))
    logits, _ = model(input_ids)
    print(f"\n✓ Test forward pass: {logits.shape}")


if __name__ == "__main__":
    print("\n🚀 TESTS DU MODÈLE HessGPT COMPLET\n")
    
    # Test 1: Forward basique
    test_hessgpt_model()
    
    # Test 2: Avec loss
    test_with_loss()
    
    # Test 3: Génération
    test_generation()
    
    # Test 4: 20M paramètres
    test_hessgpt_20m()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n🎉 FÉLICITATIONS! HessGPT est opérationnel!")
    print("\n📁 Modèle refactorisé avec imports depuis TransformerBlock/")
    print("🎯 Architecture optimisée pour ~20M paramètres")
    print("="*60 + "\n")
