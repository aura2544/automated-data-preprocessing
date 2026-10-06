import re
import numpy as np
import pandas as pd

def preprocess_text_pipeline(text: str) -> dict:
    """
    Executes text preprocessing steps:
    1. Case Folding (lowercase)
    2. Remove Punctuation (alphanumeric & whitespace only)
    3. Tokenization (word tokens)
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Case Folding
    case_folded = text.lower()
    
    # 2. Remove Punctuation
    punct_removed = re.sub(r'[^\w\s]', '', case_folded)
    
    # 3. Tokenization
    tokens = [tok for tok in punct_removed.split() if tok]
    
    return {
        'original': text,
        'case_folded': case_folded,
        'punct_removed': punct_removed,
        'tokens': tokens
    }


def build_vocabulary(processed_texts: dict) -> list:
    """
    Collects unique tokens from all texts and sorts them alphabetically.
    """
    vocab = set()
    for item in processed_texts.values():
        vocab.update(item['tokens'])
    return sorted(list(vocab))


def build_binary_matrix(processed_texts: dict, vocab: list) -> pd.DataFrame:
    """
    Constructs Binary (1 / 0) representation matrix.
    1 = word appears in text
    0 = word does not appear in text
    STRICTLY NO TF-IDF.
    """
    matrix = []
    text_labels = list(processed_texts.keys())
    vocab_len = len(vocab)
    
    if vocab_len == 0:
        return pd.DataFrame(index=text_labels)

    vocab_index = {word: i for i, word in enumerate(vocab)}

    for label in text_labels:
        tokens_set = set(processed_texts[label]['tokens'])
        row = [0] * vocab_len
        for word in tokens_set:
            if word in vocab_index:
                row[vocab_index[word]] = 1
        matrix.append(row)
        
    df_binary = pd.DataFrame(matrix, index=text_labels, columns=vocab)
    return df_binary


def calculate_cosine_similarity_detail(vec_a: list, vec_b: list, vocab: list) -> dict:
    """
    Calculates binary cosine similarity with step-by-step mathematical breakdown.
    Formula: Cos(A, B) = (A · B) / (||A|| × ||B||)
    """
    va = np.array(vec_a, dtype=float)
    vb = np.array(vec_b, dtype=float)
    
    # Dot Product
    dot_product = float(np.dot(va, vb))
    
    # Magnitudes (Euclidean norm)
    mag_a = float(np.sqrt(np.sum(va ** 2)))
    mag_b = float(np.sqrt(np.sum(vb ** 2)))
    
    # Common terms
    common_terms = [vocab[i] for i in range(len(vocab)) if va[i] == 1 and vb[i] == 1]
    
    if mag_a == 0 or mag_b == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (mag_a * mag_b)
        
    percentage = similarity * 100.0
    
    # Categorization based on specification:
    # >= 0.70 : Sangat Mirip
    # 0.40 - 0.69 : Cukup Mirip
    # 0.20 - 0.39 : Kurang Mirip
    # < 0.20 : Tidak Mirip
    if similarity >= 0.70:
        kategori = "Sangat Mirip"
        color = "#10B981" # Green
    elif similarity >= 0.40:
        kategori = "Cukup Mirip"
        color = "#2563EB" # Blue
    elif similarity >= 0.20:
        kategori = "Kurang Mirip"
        color = "#F59E0B" # Yellow/Orange
    else:
        kategori = "Tidak Mirip"
        color = "#EF4444" # Red
        
    return {
        'dot_product': dot_product,
        'mag_a': mag_a,
        'mag_b': mag_b,
        'similarity': similarity,
        'percentage': percentage,
        'kategori': kategori,
        'color': color,
        'common_terms': common_terms,
        'vec_a': va.astype(int).tolist(),
        'vec_b': vb.astype(int).tolist()
    }


def compute_all_pairs_similarity(binary_df: pd.DataFrame, vocab: list) -> tuple[pd.DataFrame, dict]:
    """
    Computes pairwise Cosine Similarity for all distinct pairs (S_i, S_j) where i < j.
    Uses vectorized matrix operations for fast performance even with 50-500+ texts.
    Returns (summary_dataframe_sorted, details_cache).
    """
    labels = list(binary_df.index)
    n = len(labels)
    
    if n < 2:
        return pd.DataFrame(), {}
        
    X = binary_df.values.astype(np.float32)
    
    # Fast vectorized computation
    dot_matrix = np.dot(X, X.T)
    norms = np.sqrt(np.sum(X, axis=1))
    norm_matrix = np.outer(norms, norms)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sim_matrix = np.where(norm_matrix > 0, dot_matrix / norm_matrix, 0.0)
        
    # Extract upper triangle pairs (i < j)
    r_idx, c_idx = np.triu_indices(n, k=1)
    
    pairs = []
    details = {}
    
    include_irisan_for_all = (n <= 30) # Compute irisan text for all if reasonable count
    
    for i, j in zip(r_idx, c_idx):
        l1, l2 = labels[i], labels[j]
        pair_key = f"{l1} - {l2}"
        sim_val = float(sim_matrix[i, j])
        dot_val = int(dot_matrix[i, j])
        pct_val = sim_val * 100.0
        
        if sim_val >= 0.70:
            kat = "Sangat Mirip"
            col = "#10B981"
        elif sim_val >= 0.40:
            kat = "Cukup Mirip"
            col = "#2563EB"
        elif sim_val >= 0.20:
            kat = "Kurang Mirip"
            col = "#F59E0B"
        else:
            kat = "Tidak Mirip"
            col = "#EF4444"
            
        common_words = []
        if include_irisan_for_all:
            v1 = X[i]
            v2 = X[j]
            common_words = [vocab[k] for k in range(len(vocab)) if v1[k] == 1 and v2[k] == 1]
            details[pair_key] = {
                'label_a': l1,
                'label_b': l2,
                'dot_product': dot_val,
                'mag_a': float(norms[i]),
                'mag_b': float(norms[j]),
                'similarity': sim_val,
                'percentage': pct_val,
                'kategori': kat,
                'color': col,
                'common_terms': common_words,
                'vec_a': X[i].astype(int).tolist(),
                'vec_b': X[j].astype(int).tolist()
            }
            
        pairs.append({
            'Pasangan': pair_key,
            'Status 1': l1,
            'Status 2': l2,
            'Dot Product': dot_val,
            'Similarity': round(sim_val, 4),
            'Persentase': f"{pct_val:.2f}%",
            'Kategori': kat,
            'Kata Irisan': ", ".join(common_words) if common_words else ("-" if include_irisan_for_all else "(Lihat Detail)")
        })
        
    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        
    return df_pairs, details


def compute_single_pair_detail(label_a: str, label_b: str, binary_df: pd.DataFrame, vocab: list) -> dict:
    """
    Computes full step-by-step detail on-demand for any selected pair.
    """
    if label_a not in binary_df.index or label_b not in binary_df.index:
        return {}
    vec1 = binary_df.loc[label_a].values
    vec2 = binary_df.loc[label_b].values
    calc = calculate_cosine_similarity_detail(vec1, vec2, vocab)
    return {
        'label_a': label_a,
        'label_b': label_b,
        **calc
    }
