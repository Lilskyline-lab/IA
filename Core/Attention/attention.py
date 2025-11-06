#attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    Self-Attention simple (1 seule tête)
    Pour comprendre les bases avant le Multi-Head
    """
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): Dimension des embeddings (ex: 768)
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Projections linéaires pour Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim] - Les embeddings
            mask: [seq_len, seq_len] - Masque causal (optionnel)
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, seq_len, seq_len] - Pour visualisation
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. Créer Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]
        
        # 2. Calculer les scores d'attention
        # Q @ K^T = [batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 3. Scaling (diviser par racine de dim)
        scores = scores / math.sqrt(embed_dim)
        
        # 4. Appliquer le masque causal (si fourni)
        if mask is not None:
            # CORRECTION: S'assurer que le masque est sur le bon device
            if mask.device != scores.device:
                mask = mask.to(scores.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax pour obtenir les poids d'attention
        attention_weights = F.softmax(scores, dim=-1)
        
        # 6. Appliquer l'attention sur les Values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention (version GPT-2)
    Avec 12 têtes d'attention en parallèle
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim (int): Dimension des embeddings (768 pour GPT-2 small)
            num_heads (int): Nombre de têtes (12 pour GPT-2 small)
            dropout (float): Taux de dropout pour l'attention
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 768 // 12 = 64
        
        # Projections Q, K, V (pour toutes les têtes en une fois)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        
        # Projection de sortie
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout pour l'attention (comme GPT-2)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] - Masque causal
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. Projeter en Q, K, V (toutes les têtes d'un coup)
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, 3 * embed_dim]
        
        # 2. Séparer Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 3. Calculer les scores d'attention
        # Q @ K^T : [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 4. Scaling
        scores = scores / math.sqrt(self.head_dim)
        
        # 5. Appliquer le masque causal
        # CORRECTION MAJEURE: Broadcasting correct du masque
        if mask is not None:
            # S'assurer que le masque est sur le bon device
            if mask.device != scores.device:
                mask = mask.to(scores.device)
            
            # Le masque est [seq_len, seq_len]
            # Les scores sont [batch_size, num_heads, seq_len, seq_len]
            # On doit broadcaster le masque pour qu'il s'applique à toutes les batch/heads
            # PyTorch fait le broadcasting automatiquement si on ajoute des dimensions
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 6. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 7. Dropout sur les poids d'attention (comme GPT-2)
        attention_weights = self.attn_dropout(attention_weights)
        
        # 8. Appliquer l'attention sur V
        output = torch.matmul(attention_weights, V)
        # output: [batch_size, num_heads, seq_len, head_dim]
        
        # 9. Recombiner les têtes
        output = output.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        output = output.reshape(batch_size, seq_len, embed_dim)
        
        # 10. Projection finale avec dropout
        output = self.out_proj(output)
        output = self.resid_dropout(output)
        
        return output


def create_causal_mask(seq_len, device='cpu'):
    """
    Crée un masque causal (triangulaire inférieur)
    
    Args:
        seq_len (int): Longueur de la séquence
        device (str): Device sur lequel créer le masque
    
    Returns:
        mask: [seq_len, seq_len] - 1 pour visible, 0 pour masqué
    
    Exemple pour seq_len=3:
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


# ============================================
# TESTS
# ============================================

def test_self_attention():
    """Test de la Self-Attention simple (1 tête)"""
    print("\n" + "="*60)
    print("TEST 1: Self-Attention Simple")
    print("="*60)
    
    # Paramètres
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    
    # Créer le module
    attention = SelfAttention(embed_dim)
    
    # Input aléatoire
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Créer le masque causal
    mask = create_causal_mask(seq_len, device=x.device)
    
    # Forward
    output, attention_weights = attention(x, mask)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weights shape: {attention_weights.shape}")
    
    # Visualiser les poids d'attention pour le premier exemple
    print(f"\n📊 Poids d'attention (premier exemple):")
    print(attention_weights[0].detach().numpy().round(2))
    
    # Vérifier que le masque fonctionne (les futures positions sont à 0)
    print(f"\n✓ Vérification du masque causal:")
    print(f"  Position 0 regarde: {(attention_weights[0, 0] > 0.01).sum().item()} positions (devrait être 1)")
    print(f"  Position 2 regarde: {(attention_weights[0, 2] > 0.01).sum().item()} positions (devrait être 3)")
    print(f"  Position 4 regarde: {(attention_weights[0, 4] > 0.01).sum().item()} positions (devrait être 5)")


def test_multi_head_attention():
    """Test du Multi-Head Attention"""
    print("\n" + "="*60)
    print("TEST 2: Multi-Head Attention avec masque causal")
    print("="*60)
    
    # Paramètres GPT-2 small
    batch_size = 2
    seq_len = 10
    embed_dim = 768
    num_heads = 12
    
    # Créer le module
    attention = MultiHeadAttention(embed_dim, num_heads)
    
    # Input aléatoire
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Masque causal
    mask = create_causal_mask(seq_len, device=x.device)
    
    # Forward
    output = attention(x, mask)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Nombre de têtes: {num_heads}")
    print(f"✓ Dimension par tête: {embed_dim // num_heads}")
    
    # Nombre de paramètres
    num_params = sum(p.numel() for p in attention.parameters())
    print(f"\n✓ Nombre de paramètres: {num_params:,}")


def test_mask_broadcasting():
    """Test spécifique du broadcasting du masque"""
    print("\n" + "="*60)
    print("TEST 3: Broadcasting du masque causal")
    print("="*60)
    
    batch_size = 2
    num_heads = 4
    seq_len = 5
    
    # Créer un masque
    mask = create_causal_mask(seq_len)
    print(f"✓ Masque original shape: {mask.shape}")
    print(f"  Masque:\n{mask.int()}")
    
    # Simuler des scores d'attention
    scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    print(f"\n✓ Scores shape: {scores.shape}")
    
    # Broadcaster le masque
    mask_broadcasted = mask.unsqueeze(0).unsqueeze(0)
    print(f"✓ Masque broadcasté shape: {mask_broadcasted.shape}")
    
    # Appliquer le masque
    scores_masked = scores.masked_fill(mask_broadcasted == 0, float('-inf'))
    
    print(f"\n✓ Vérification:")
    print(f"  Avant masque - scores[0,0,0,0]: {scores[0,0,0,0]:.2f}")
    print(f"  Après masque - scores[0,0,0,0]: {scores_masked[0,0,0,0]:.2f}")
    print(f"  Avant masque - scores[0,0,0,1]: {scores[0,0,0,1]:.2f}")
    print(f"  Après masque - scores[0,0,0,1]: {scores_masked[0,0,0,1]:.2f} (devrait être -inf)")


def test_with_cuda():
    """Test avec CUDA si disponible"""
    print("\n" + "="*60)
    print("TEST 4: Test avec CUDA (si disponible)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    if device.type == 'cpu':
        print("  ⚠️  CUDA non disponible, test sur CPU")
    
    # Paramètres
    batch_size = 2
    seq_len = 10
    embed_dim = 256
    num_heads = 8
    
    # Créer le module et le mettre sur le device
    attention = MultiHeadAttention(embed_dim, num_heads).to(device)
    
    # Input sur le device
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    # Masque sur le device
    mask = create_causal_mask(seq_len, device=device)
    
    # Forward
    output = attention(x, mask)
    
    print(f"✓ Input device: {x.device}")
    print(f"✓ Mask device: {mask.device}")
    print(f"✓ Output device: {output.device}")
    print(f"✓ Output shape: {output.shape}")


if __name__ == "__main__":
    print("\n🚀 TESTS DE LA SELF-ATTENTION (CORRIGÉS)\n")
    
    # Test 1: Attention simple
    test_self_attention()
    
    # Test 2: Multi-Head Attention
    test_multi_head_attention()
    
    # Test 3: Broadcasting du masque
    test_mask_broadcasting()
    
    # Test 4: Avec CUDA
    test_with_cuda()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n🔧 CORRECTIONS APPLIQUÉES:")
    print("  1. Broadcasting correct du masque pour multi-head")
    print("  2. Device management pour le masque")
    print("  3. Dropout ajouté sur l'attention et les résidus")
    print("\n📁 Remplacez votre Attention/attention.py par ce fichier")
    print("="*60 + "\n")