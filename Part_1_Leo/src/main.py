# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
"""
arXiv Embedding Pipeline
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
import time

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.run(["pip", "install", "-q", "sentence-transformers", "torch"])
    from sentence_transformers import SentenceTransformer
    import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Model selection
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  
ARXIV_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "arxiv_embeddings.npz"
METADATA_FILE = "arxiv_metadata_simple.json"

BATCH_SIZE = None 
SAVE_INTERVAL = 100000
MAX_ABSTRACT_LENGTH = 512

# ---------------------------------------------------------------------------
# GPU Detection & Auto-tuning
# ---------------------------------------------------------------------------

def detect_gpu_and_tune():
    """Detect GPU and set optimal batch size."""
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected!")
        return torch.device("cpu"), 32

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 70)
    print("GPU DETECTED")
    print("=" * 70)
    print(f"Name: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")

    # Auto-tune batch size based on GPU
    if "A100" in gpu_name:
        batch_size = 512
        tier = "🏆 EXCELLENT (A100)"
        speedup = "5-6x faster than T4"
    elif "V100" in gpu_name:
        batch_size = 256
        tier = "⭐ GREAT (V100)"
        speedup = "2-3x faster than T4"
    elif "L4" in gpu_name:
        batch_size = 256
        tier = "⭐ GREAT (L4)"
        speedup = "3-4x faster than T4"
    elif "T4" in gpu_name:
        batch_size = 128
        tier = "✅ GOOD (T4)"
        speedup = "baseline"
    else:
        batch_size = 128
        tier = "✅ UNKNOWN GPU"
        speedup = "unknown"

    print(f"Tier: {tier}")
    print(f"Speed: {speedup}")
    print(f"Auto-tuned batch size: {batch_size}")
    print("=" * 70)

    return device, batch_size


# ---------------------------------------------------------------------------
# Model recommendation based on GPU
# ---------------------------------------------------------------------------

def recommend_model(gpu_name):
    """Recommend model based on GPU capability."""
    print("\n" + "=" * 70)
    print("MODEL RECOMMENDATION")
    print("=" * 70)

    if "A100" in gpu_name or "V100" in gpu_name or "L4" in gpu_name:
        print("✨ You have a powerful GPU!")
        print("\nRecommendations:")
        print("  1. all-MiniLM-L6-v2 (current): ~1-2 hours, good quality")
        print("  2. SciBERT: ~2-3 hours, ⭐ BETTER for science papers")
        print("  3. SPECTER: ~2-3 hours, ⭐ BETTER for citation-aware")
        print("\n💡 Consider using SciBERT for better domain-specific quality!")
        print("   (Edit MODEL_NAME in script)")
    else:
        print("T4 GPU: Stick with all-MiniLM-L6-v2 (fast & good enough)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Time estimation
# ---------------------------------------------------------------------------

def estimate_time(total_papers, batch_size, gpu_name):
    """Estimate processing time based on GPU."""
    # Papers per second by GPU type (empirical)
    speeds = {
        "A100": 1200,
        "V100": 600,
        "L4": 700,
        "T4": 250,
    }

    papers_per_sec = speeds.get(gpu_name.split()[0], 250)
    estimated_seconds = total_papers / papers_per_sec
    estimated_hours = estimated_seconds / 3600

    print(f"\nEstimated processing time:")
    print(f"  Total papers: {total_papers:,}")
    print(f"  Speed: ~{papers_per_sec} papers/sec")
    print(f"  Time: ~{estimated_hours:.1f} hours")

    return estimated_hours


# ---------------------------------------------------------------------------
# Load and encode (same as before, but with auto-tuned batch size)
# ---------------------------------------------------------------------------

def load_arxiv_data(filepath, max_papers=None):
    """Load arXiv papers."""
    print(f"\nLoading arXiv data from {filepath}...")
    print("(This may take 2-3 minutes)")

    abstracts = []
    metadata = []
    skipped = 0
    start_time = time.time()

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_papers and i >= max_papers:
                break

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Processed: {i+1:,} papers ({rate:.0f} papers/sec)")

            try:
                paper = json.loads(line)
                abstract = paper.get('abstract', '').strip()

                if not abstract or len(abstract) < 50:
                    skipped += 1
                    continue

                if len(abstract) > MAX_ABSTRACT_LENGTH * 6:
                    abstract = abstract[:MAX_ABSTRACT_LENGTH * 6]

                abstracts.append(abstract)
                metadata.append({
                    'id': paper.get('id', ''),
                    'title': paper.get('title', '')[:200],
                    'categories': paper.get('categories', ''),
                    'update_date': paper.get('update_date', ''),
                })

            except:
                skipped += 1
                continue

    elapsed = time.time() - start_time
    print(f"\n✓ Loaded: {len(abstracts):,} papers ({elapsed/60:.1f} minutes)")
    print(f"  Skipped: {skipped:,} papers")

    return abstracts, metadata


def encode_abstracts(abstracts, model_name, device, batch_size, save_interval, output_file):
    """Encode abstracts with optimal batch size."""
    print(f"\n{'='*70}")
    print(f"Encoding {len(abstracts):,} abstracts")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    # Load model
    print(f"\nLoading model...")
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded: {embedding_dim}D embeddings")

    # Encode
    all_embeddings = []
    start_time = time.time()

    print(f"\nStarting encoding...")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for i in range(0, len(abstracts), batch_size):
        batch_abstracts = abstracts[i:i+batch_size]

        batch_embeddings = model.encode(
            batch_abstracts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        all_embeddings.append(batch_embeddings)

        processed = min(i + batch_size, len(abstracts))
        if processed % 10000 == 0 or processed == len(abstracts):
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (len(abstracts) - processed) / rate / 3600
            pct = processed / len(abstracts) * 100

            print(f"  [{processed:,}/{len(abstracts):,}] ({pct:.1f}%) | "
                  f"Rate: {rate:.0f} papers/sec | ETA: {remaining:.1f}h")

        if processed % save_interval == 0 and processed < len(abstracts):
            checkpoint_embeddings = np.vstack(all_embeddings)
            checkpoint_file = output_file.replace('.npz', f'_checkpoint_{processed}.npz')
            np.savez_compressed(checkpoint_file, embeddings=checkpoint_embeddings)
            print(f"  ✓ Checkpoint saved: {checkpoint_file}")

        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    print(f"\nCombining batches...")
    embeddings = np.vstack(all_embeddings)

    elapsed = time.time() - start_time
    print(f"\n✓ Encoding complete!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Time: {elapsed/3600:.2f} hours")
    print(f"  Rate: {len(abstracts) / elapsed:.0f} papers/sec")

    return embeddings


def save_embeddings(embeddings, metadata, output_file, metadata_file):
    """Save embeddings and metadata."""
    print(f"\nSaving outputs...")

    paper_ids = [m['id'] for m in metadata]
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        paper_ids=np.array(paper_ids, dtype=object)
    )

    file_size = Path(output_file).stat().st_size / 1024**3
    print(f"  ✓ Embeddings saved: {output_file} ({file_size:.2f} GB)")

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)

    meta_size = Path(metadata_file).stat().st_size / 1024**2
    print(f"  ✓ Metadata saved: {metadata_file} ({meta_size:.1f} MB)")


def validate_embeddings(embeddings):
    """Validation checks."""
    print(f"\n{'='*70}")
    print("Validation")
    print(f"{'='*70}")

    print(f"Shape: {embeddings.shape}")
    print(f"Memory: {embeddings.nbytes / 1024**3:.2f} GB")

    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print(f"NaN: {nan_count}, Inf: {inf_count}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nL2 norms: mean={norms.mean():.4f}, std={norms.std():.4f}")

    if nan_count == 0 and inf_count == 0 and 0.95 < norms.mean() < 1.05:
        print(f"\n✅ Validation passed!")
    else:
        print(f"\n⚠️ Validation warnings detected")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("arXiv Embedding Pipeline (Colab Pro Optimized)")
    print("=" * 70)

    # Auto-detect GPU and tune
    device, batch_size = detect_gpu_and_tune()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # Recommend model
    recommend_model(gpu_name)

    # Load data
    abstracts, metadata = load_arxiv_data(ARXIV_FILE)

    # Estimate time
    estimate_time(len(abstracts), batch_size, gpu_name)

    # Confirm
    if device.type == "cpu":
        response = input("\nNo GPU detected. Continue? (y/n): ")
        if response.lower() != 'y':
            return

    # Encode
    embeddings = encode_abstracts(
        abstracts, MODEL_NAME, device, batch_size,
        SAVE_INTERVAL, OUTPUT_FILE
    )

    # Save
    save_embeddings(embeddings, metadata, OUTPUT_FILE, METADATA_FILE)

    # Validate
    validate_embeddings(embeddings)

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

# %%
# ============================================
# KLab x arXiv Similarity Search
# Run directly from Google Drive (~5 mins on A100)
# ============================================

import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

print("=" * 70)
print("KLab x arXiv Similarity Search")
print("=" * 70)

# Parameters
TOP_K = 1000
SWEET_SPOT_LOW = 0.3
SWEET_SPOT_HIGH = 0.7
BATCH_SIZE = 1000

# Paths
DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
OUTPUT_PATH = DRIVE_PATH / 'candidates'
OUTPUT_PATH.mkdir(exist_ok=True)

# ============================================
# 1. Load Data
# ============================================

print("\n[1/4] Loading KLab embeddings...")
klab_data = np.load(DRIVE_PATH / 'klab_embeddings.npz')
klab_embeddings = klab_data['embeddings']
print(f"  Loaded: {klab_embeddings.shape}")

print("\n[2/4] Loading KLab metadata...")
with open(DRIVE_PATH / 'klab_metadata.pkl', 'rb') as f:
    klab_metadata = pickle.load(f)
print(f"  Loaded: {len(klab_metadata)} papers")

print("\n[3/4] Loading arXiv embeddings (4 GB, takes ~30 seconds)...")
arxiv_data = np.load(DRIVE_PATH / 'arxiv_embeddings.npz')
arxiv_embeddings = arxiv_data['embeddings']
print(f"  Loaded: {arxiv_embeddings.shape}")

print("\n[4/4] Loading arXiv metadata...")
with open(DRIVE_PATH / 'arxiv_metadata_simple.json', 'r') as f:
    arxiv_metadata = json.load(f)
print(f"  Loaded: {len(arxiv_metadata)} papers")

# ============================================
# 2. Calculate Similarity
# ============================================

print("\n" + "=" * 70)
print("Calculating KLab x arXiv Similarity")
print("=" * 70)
print(f"Matrix: {len(klab_embeddings)} x {len(arxiv_embeddings)}")
print(f"Total pairs: {len(klab_embeddings) * len(arxiv_embeddings):,}")

n_arxiv = len(arxiv_embeddings)
n_batches = int(np.ceil(n_arxiv / BATCH_SIZE))

similarities = np.zeros((len(klab_embeddings), n_arxiv), dtype=np.float32)

for i in tqdm(range(n_batches), desc="Calculating similarities"):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, n_arxiv)

    arxiv_batch = arxiv_embeddings[start_idx:end_idx]
    sim_batch = cosine_similarity(klab_embeddings, arxiv_batch)

    similarities[:, start_idx:end_idx] = sim_batch

print("\nSimilarity calculation complete")

# ============================================
# 3. Extract Candidates
# ============================================

print("\n" + "=" * 70)
print("Extracting candidate paper pairs")
print("=" * 70)
print(f"Top-{TOP_K} per KLab paper")
print(f"Sweet spot: [{SWEET_SPOT_LOW}, {SWEET_SPOT_HIGH}]")

candidates = []
stats = {
    'total_pairs': 0,
    'in_sweet_spot': 0,
    'above_sweet_spot': 0,
    'below_sweet_spot': 0,
}

for klab_idx in tqdm(range(len(klab_metadata)), desc="Extracting candidates"):
    klab_paper = klab_metadata[klab_idx]
    sims = similarities[klab_idx]

    top_indices = np.argsort(sims)[::-1][:TOP_K]
    top_sims = sims[top_indices]

    for arxiv_idx, sim in zip(top_indices, top_sims):
        stats['total_pairs'] += 1

        in_sweet_spot = (SWEET_SPOT_LOW <= sim <= SWEET_SPOT_HIGH)

        if in_sweet_spot:
            stats['in_sweet_spot'] += 1
        if sim > SWEET_SPOT_HIGH:
            stats['above_sweet_spot'] += 1
        elif sim < SWEET_SPOT_LOW:
            stats['below_sweet_spot'] += 1

        arxiv_paper = arxiv_metadata[int(arxiv_idx)]
        candidates.append({
            'klab_idx': klab_idx,
            'klab_id': klab_paper['openalex_id'],
            'klab_title': klab_paper['title'],
            'klab_year': klab_paper['year'],
            'klab_citations': klab_paper['cited_by_count'],
            'arxiv_idx': int(arxiv_idx),
            'arxiv_id': arxiv_paper['id'],
            'arxiv_title': arxiv_paper['title'],
            'arxiv_categories': arxiv_paper['categories'],
            'arxiv_year': int(arxiv_paper['update_date'].split('-')[0]) if arxiv_paper.get('update_date') else None,
            'similarity': float(sim),
            'in_sweet_spot': bool(in_sweet_spot),
        })

print(f"\nExtracted {len(candidates):,} candidate pairs")
# ============================================
# 4. Save Results
# ============================================

print("\n" + "=" * 70)
print("Saving results to Google Drive")
print("=" * 70)

# JSON
json_file = OUTPUT_PATH / 'klab_arxiv_candidates.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(candidates, f, indent=2, ensure_ascii=False)
print(f"  Saved JSON: {json_file}")

# CSV
csv_file = OUTPUT_PATH / 'klab_arxiv_candidates.csv'
df = pd.DataFrame(candidates)
df.to_csv(csv_file, index=False)
print(f"  Saved CSV: {csv_file}")

# Report
report_lines = [
    "=" * 70,
    "KLab x arXiv Similarity Search Report",
    "=" * 70,
    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "Parameters:",
    f"  Sweet spot: [{SWEET_SPOT_LOW}, {SWEET_SPOT_HIGH}]",
    f"  Top-K: {TOP_K}",
    "",
    "Statistics:",
    f"  Total candidate pairs: {len(candidates):,}",
    f"  In sweet spot: {stats['in_sweet_spot']:,} ({stats['in_sweet_spot']/stats['total_pairs']*100:.1f}%)",
    f"  Above sweet spot: {stats['above_sweet_spot']:,} ({stats['above_sweet_spot']/stats['total_pairs']*100:.1f}%)",
    f"  Below sweet spot: {stats['below_sweet_spot']:,} ({stats['below_sweet_spot']/stats['total_pairs']*100:.1f}%)",
    "",
    "Similarity distribution:",
    f"  Mean:   {df['similarity'].mean():.4f}",
    f"  Median: {df['similarity'].median():.4f}",
    f"  Min:    {df['similarity'].min():.4f}",
    f"  Max:    {df['similarity'].max():.4f}",
    "",
]

sweet_df = df[df['in_sweet_spot']]
if len(sweet_df) > 0:
    report_lines.extend([
        f"Sweet spot details:",
        f"  Candidate pairs: {len(sweet_df):,}",
        f"  Average similarity: {sweet_df['similarity'].mean():.4f}",
        "",
        "Top 10 KLab papers (by sweet spot candidate count):",
        "",
    ])

    top_klab = sweet_df.groupby('klab_title').size().sort_values(ascending=False).head(10)
    for i, (title, count) in enumerate(top_klab.items(), 1):
        report_lines.append(f"  {i:2d}. {count:4d} candidates | {title[:55]}...")

    report_lines.extend([
        "",
        "Random samples (sweet spot):",
        "",
    ])

    samples = sweet_df.sample(min(5, len(sweet_df)))
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        report_lines.extend([
            f"{i}. KLab:  [{row['klab_year']}] {row['klab_title'][:55]}",
            f"   arXiv: [{row['arxiv_year']}] {row['arxiv_title'][:55]}",
            f"   Similarity: {row['similarity']:.4f} | Categories: {row['arxiv_categories']}",
            "",
        ])

report_text = "\n".join(report_lines)

report_file = OUTPUT_PATH / 'similarity_search_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"  Saved Report: {report_file}")

# ============================================
# 5. Display Report
# ============================================

print("\n" + "=" * 70)
print(report_text)
print("=" * 70)

# Download
print("\nDownloading results...")
from google.colab import files
files.download(str(csv_file))
files.download(str(report_file))

print("\nSimilarity search complete!")
print(f"All results saved to: {OUTPUT_PATH}")

# %%
df = pd.read_csv('/content/drive/MyDrive/serendipity_agent/candidates/klab_arxiv_candidates.csv')

print(f"Before deduplication: {len(df):,} rows")

# Method 1: Deduplicate by klab_title (keep first)
df_dedup = df.drop_duplicates(subset=['klab_title', 'arxiv_id'])
print(f"Deduplicated by title: {len(df_dedup):,} rows")

# Method 2: Deduplicate by klab_id (keep first of all versions)
df_dedup2 = df.drop_duplicates(subset=['klab_id', 'arxiv_id'])
print(f"Deduplicated by ID: {len(df_dedup2):,} rows")

# Save deduplicated results
df_dedup.to_csv('/content/drive/MyDrive/serendipity_agent/candidates/klab_arxiv_candidates_dedup.csv', index=False)
print("\nDeduplicated results saved")

# Recalculate statistics
sweet_df = df_dedup[df_dedup['in_sweet_spot']]
print(f"\nSweet spot candidates after deduplication: {len(sweet_df):,}")

# %%
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# ==========================================
# Configuration
# ==========================================
HF_TOKEN = "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"
DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
INPUT_CSV = DRIVE_PATH / 'candidates/klab_arxiv_candidates_dedup.csv'
ARXIV_RAW_JSON = '/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json'

# Output paths
BATCH_GPT2 = DRIVE_PATH / 'candidates/batch_gpt2_2019_2023.csv'
BATCH_LLAMA3 = DRIVE_PATH / 'candidates/batch_llama3_2024_plus.csv'
BATCH_ARCHIVE = DRIVE_PATH / 'candidates/batch_archive_pre_2019.csv'

SCORED_GPT2 = DRIVE_PATH / 'candidates/scored_gpt2_2019_2023.csv'
SCORED_LLAMA3 = DRIVE_PATH / 'candidates/scored_llama3_2024_plus.csv'

MAX_LENGTH = 512

# ==========================================
# 1. Temporal Stratification
# ==========================================
def stratify_data():
    df = pd.read_csv(INPUT_CSV)
    df['arxiv_year'] = pd.to_numeric(df['arxiv_year'], errors='coerce')
    df = df.dropna(subset=['arxiv_year'])

    mask_gpt2 = (df['arxiv_year'] >= 2019) & (df['arxiv_year'] <= 2023)
    mask_llama3 = (df['arxiv_year'] >= 2024)
    mask_archive = (df['arxiv_year'] <= 2018)

    df_gpt2 = df[mask_gpt2]
    df_llama3 = df[mask_llama3]
    df_archive = df[mask_archive]

    df_gpt2.to_csv(BATCH_GPT2, index=False)
    df_llama3.to_csv(BATCH_LLAMA3, index=False)
    df_archive.to_csv(BATCH_ARCHIVE, index=False)

# ==========================================
# 2. Abstract Extraction Helper (Streaming)
# ==========================================
def load_abstracts(needed_ids, json_path):
    id_to_abstract = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            paper = json.loads(line)
            pid = str(paper.get('id', ''))
            if pid in needed_ids:
                abstract = paper.get('abstract', '').strip()
                if abstract:
                    id_to_abstract[pid] = abstract
            if len(id_to_abstract) == len(needed_ids):
                break
    return id_to_abstract

# ==========================================
# 3. Unified Perplexity Evaluation Engine
# ==========================================
def evaluate_perplexity(model_id, input_csv, output_csv, batch_size, is_llama=False):
    # Load inputs
    df_input = pd.read_csv(input_csv)
    needed_ids = set(df_input['arxiv_id'].astype(str))
    
    # Load and map abstracts
    id_to_abstract = load_abstracts(needed_ids, ARXIV_RAW_JSON)
    df_input['arxiv_abstract'] = df_input['arxiv_id'].astype(str).map(id_to_abstract).fillna("No abstract available.")

    # Checkpoint resumption
    processed_ids = set()
    if output_csv.exists():
        df_out = pd.read_csv(output_csv)
        if 'arxiv_id' in df_out.columns:
            processed_ids = set(df_out['arxiv_id'].astype(str))
            
    df_todo = df_input[~df_input['arxiv_id'].astype(str).isin(processed_ids)].copy()
    if len(df_todo) == 0:
        return

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_kwargs = {"token": HF_TOKEN} if is_llama else {}
    model_kwargs = {"token": HF_TOKEN, "torch_dtype": torch.bfloat16, "device_map": "auto"} if is_llama else {}
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not is_llama:
        model = model.to(device)
    model.eval()

    # Calculation logic
    def calculate_ppl_batch(texts):
        encodings = tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = model(encodings.input_ids, attention_mask=encodings.attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = encodings.input_ids[..., 1:].contiguous()
            shift_mask = encodings.attention_mask[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            masked_loss = loss * shift_mask
            seq_lengths = torch.clamp(shift_mask.sum(dim=1), min=1.0)
            seq_loss = masked_loss.sum(dim=1) / seq_lengths
            ppl = torch.exp(seq_loss)

        if is_llama:
            torch.cuda.empty_cache()
            
        return ppl.cpu().tolist()

    # Deduplicate for computation efficiency
    unique_papers = df_todo[['arxiv_id', 'arxiv_abstract']].drop_duplicates(subset='arxiv_id')
    abstracts = unique_papers['arxiv_abstract'].tolist()
    arxiv_ids = unique_papers['arxiv_id'].tolist()
    ppl_map = {}

    for i in tqdm(range(0, len(abstracts), batch_size), desc=f"Scoring {model_id}"):
        batch_texts = [t if len(t.strip()) > 10 else "No abstract available." for t in abstracts[i : i+batch_size]]
        batch_ids = arxiv_ids[i : i+batch_size]
        
        batch_ppls = calculate_ppl_batch(batch_texts)

        for aid, ppl in zip(batch_ids, batch_ppls):
            ppl_map[aid] = round(ppl, 4)

    # Map results back and save
    df_todo['perplexity'] = df_todo['arxiv_id'].astype(str).map(ppl_map)
    
    # Append to existing CSV or create new
    mode = 'a' if output_csv.exists() else 'w'
    header = not output_csv.exists()
    df_todo.to_csv(output_csv, mode=mode, header=header, index=False)
        
    # Free memory before next model
    del model
    del tokenizer
    torch.cuda.empty_cache()

# ==========================================
# Execution Pipeline
# ==========================================
if __name__ == "__main__":
    stratify_data()
    
    evaluate_perplexity(
        model_id="gpt2",
        input_csv=BATCH_GPT2,
        output_csv=SCORED_GPT2,
        batch_size=16,
        is_llama=False
    )
    
    login(token=HF_TOKEN)
    evaluate_perplexity(
        model_id="meta-llama/Meta-Llama-3-8B",
        input_csv=BATCH_LLAMA3,
        output_csv=SCORED_LLAMA3,
        batch_size=8,
        is_llama=True
    )

# %%
import numpy as np
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Configuration
# ==========================================
DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
EMBEDDINGS_FILE = DRIVE_PATH / 'arxiv_embeddings.npz'
METADATA_FILE = DRIVE_PATH / 'arxiv_metadata_simple.json'

OUTPUT_ALL = DRIVE_PATH / 'candidates/convergence_analysis.csv'
OUTPUT_MAJOR = DRIVE_PATH / 'candidates/convergence_major_domains.csv'

MAJOR_DOMAINS = [
    'astro-ph', 'cond-mat', 'cs', 'econ', 'eess', 'hep-ph', 'hep-th',
    'math', 'math-ph', 'nlin', 'nucl-th', 'physics', 'q-bio', 'q-fin',
    'quant-ph', 'stat', 'gr-qc'
]

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
def load_and_preprocess_data():
    print("=" * 70)
    print("Phase 3: Temporal Convergence Analysis")
    print("=" * 70)
    
    print("\n[1/4] Loading arXiv embeddings and metadata...")
    arxiv_data = np.load(EMBEDDINGS_FILE)
    arxiv_embeddings = arxiv_data['embeddings']
    print(f"  Embeddings shape: {arxiv_embeddings.shape}")

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        arxiv_meta = json.load(f)
    print(f"  Metadata records: {len(arxiv_meta):,}")

    print("\n[2/4] Extracting temporal and domain metadata...")
    records = []
    for i, m in enumerate(arxiv_meta):
        update_date = m.get('update_date')
        if not update_date:
            continue
            
        year = int(update_date.split('-')[0])
        cats = m.get('categories', '')
        primary_cat = cats.split()[0] if cats else None
        domain = primary_cat.split('.')[0] if primary_cat and '.' in primary_cat else primary_cat
        
        if year and domain and 2010 <= year <= 2025:
            records.append({'idx': i, 'year': year, 'domain': domain, 'category': primary_cat})

    df = pd.DataFrame(records)
    print(f"  Valid records: {len(df):,}")
    print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"  Unique domains: {df['domain'].nunique()}")
    
    return df, arxiv_embeddings

# ==========================================
# 2. Embedding Aggregation
# ==========================================
def compute_group_embeddings(df, embeddings):
    print("\n[3/4] Computing mean embeddings for (year, domain) groups...")
    group_embeddings = {}
    
    for (year, domain), group in df.groupby(['year', 'domain']):
        indices = group['idx'].values
        mean_emb = embeddings[indices].mean(axis=0)
        # L2 Normalization
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        group_embeddings[(year, domain)] = mean_emb

    print(f"  Computed {len(group_embeddings):,} group embeddings")
    return group_embeddings

# ==========================================
# 3. Convergence Trend Analysis
# ==========================================
def analyze_convergence(df, group_embeddings):
    print("\n[4/4] Analyzing temporal distance trends between domain pairs...")
    domains = sorted(df['domain'].unique())
    years = sorted(df['year'].unique())

    results = []
    for i, d1 in enumerate(domains):
        for d2 in domains[i+1:]:
            distances = []
            for y in years:
                if (y, d1) in group_embeddings and (y, d2) in group_embeddings:
                    sim = cosine_similarity(
                        [group_embeddings[(y, d1)]],
                        [group_embeddings[(y, d2)]]
                    )[0][0]
                    distances.append({'year': y, 'distance': 1 - sim})

            # Require at least 5 data points to establish a reliable trend
            if len(distances) >= 5:
                dist_df = pd.DataFrame(distances)
                x = dist_df['year'].values
                y_vals = dist_df['distance'].values
                slope = np.polyfit(x, y_vals, 1)[0]

                if slope < -0.001:
                    trend = 'Converging'
                elif slope > 0.001:
                    trend = 'Diverging'
                else:
                    trend = 'Stable'

                results.append({
                    'domain_1': d1,
                    'domain_2': d2,
                    'slope': round(slope, 6),
                    'trend': trend,
                    'mean_distance': round(y_vals.mean(), 4),
                    'n_years': len(distances),
                })

    conv_df = pd.DataFrame(results)
    return conv_df

# ==========================================
# 4. Reporting and Export
# ==========================================
def generate_reports(conv_df):
    print("\n" + "=" * 70)
    print("Convergence Analysis Results")
    print("=" * 70)
    
    # 1. Overall Report
    print(f"Total domain pairs: {len(conv_df):,}")
    print(f"  Converging: {(conv_df['trend'] == 'Converging').sum():,}")
    print(f"  Diverging:  {(conv_df['trend'] == 'Diverging').sum():,}")
    print(f"  Stable:     {(conv_df['trend'] == 'Stable').sum():,}")

    print("\n--- Top 10 Fastest Converging (All Domains) ---")
    print(conv_df.nsmallest(10, 'slope')[['domain_1', 'domain_2', 'slope', 'mean_distance', 'trend']].to_string(index=False))

    print("\n--- Top 10 Fastest Diverging (All Domains) ---")
    print(conv_df.nlargest(10, 'slope')[['domain_1', 'domain_2', 'slope', 'mean_distance', 'trend']].to_string(index=False))

    conv_df.to_csv(OUTPUT_ALL, index=False)
    print(f"\nSaved all results to: {OUTPUT_ALL.name}")

    # 2. Major Domains Report
    major_df = conv_df[
        conv_df['domain_1'].isin(MAJOR_DOMAINS) &
        conv_df['domain_2'].isin(MAJOR_DOMAINS)
    ]

    print("\n" + "=" * 70)
    print("Major Domains Subset")
    print("=" * 70)
    print(f"Major domain pairs: {len(major_df):,}")
    print(f"  Converging: {(major_df['trend'] == 'Converging').sum():,}")
    print(f"  Diverging:  {(major_df['trend'] == 'Diverging').sum():,}")
    print(f"  Stable:     {(major_df['trend'] == 'Stable').sum():,}")

    print("\n--- Top 15 Fastest Converging (Major Domains) ---")
    print(major_df.nsmallest(15, 'slope')[['domain_1', 'domain_2', 'slope', 'mean_distance', 'trend']].to_string(index=False))

    print("\n--- Top 15 Fastest Diverging (Major Domains) ---")
    print(major_df.nlargest(15, 'slope')[['domain_1', 'domain_2', 'slope', 'mean_distance', 'trend']].to_string(index=False))

    major_df.to_csv(OUTPUT_MAJOR, index=False)
    print(f"\nSaved major domains results to: {OUTPUT_MAJOR.name}")

# ==========================================
# Execution Pipeline
# ==========================================
if __name__ == "__main__":
    df_records, embeddings = load_and_preprocess_data()
    group_embs = compute_group_embeddings(df_records, embeddings)
    convergence_df = analyze_convergence(df_records, group_embs)
    generate_reports(convergence_df)
    
    print("\nPhase 3 execution completed.")

# %%
# ==========================================
# Phase 4: Steering Score Evaluation (Dot Product)
# ==========================================
import time
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
BATCH_SIZE = 16

# Note: This script assumes `tokenizer`, `model`, `device`, `best_layer`, 
# and `sv_gpu` are already initialized in your environment.

# Load previous results
df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')

# Deduplicate to score only unique papers
unique_df = df[['arxiv_id', 'arxiv_abstract']].drop_duplicates(subset='arxiv_id')
print(f"Unique papers to score: {len(unique_df):,}")

def score_batch_dot(texts):
    safe_texts = [str(t) if len(str(t).strip()) > 10 else "No abstract available." for t in texts]
    inputs = tokenizer(safe_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[best_layer + 1]
        mean_act = hidden.mean(dim=1).float()
        # Dot product computation
        scores = torch.matmul(mean_act, sv_gpu)
        
    return scores.cpu().tolist()

sv_scores = {}
abstracts = unique_df['arxiv_abstract'].tolist()
arxiv_ids = unique_df['arxiv_id'].tolist()

start_time = time.time()

# Process in clean chunks
for i in tqdm(range(0, len(abstracts), BATCH_SIZE), desc="Steering (Dot Product)"):
    batch_texts = abstracts[i : i + BATCH_SIZE]
    batch_ids = arxiv_ids[i : i + BATCH_SIZE]
    
    # Identify valid abstracts
    valid_mask = [bool(str(t).strip() and len(str(t).strip()) >= 10) for t in batch_texts]
    
    # Calculate scores for the batch
    scores = score_batch_dot(batch_texts)
    
    for idx, (bid, score) in enumerate(zip(batch_ids, scores)):
        # Assign 0.0 if the abstract was invalid/empty, otherwise keep the raw float
        sv_scores[bid] = score if valid_mask[idx] else 0.0
        
    torch.cuda.empty_cache()

elapsed = time.time() - start_time
print(f"\nExecution completed in {elapsed/60:.1f} minutes.")

# Map scores back to the main dataframe
df['steering_score'] = df['arxiv_id'].astype(str).map(sv_scores)

print(f"Unique steering scores: {df['steering_score'].nunique():,}")
print(df['steering_score'].describe())

# Save and overwrite
output_path = DRIVE_PATH / 'candidates/scored_all_4metrics.csv'
df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path.name}")


# %%
# ==========================================
# Backtesting Part 1: Metrics vs. KLab Citations
# ==========================================
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')

# 1. Standardize the three metrics (z-score)
for col in ['similarity', 'perplexity', 'steering_score']:
    df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()

# Composite score (equal weighting)
df['composite_score'] = (df['similarity_z'] + df['perplexity_z'] + df['steering_score_z']) / 3

# 2. Stratify KLab papers into High/Low citation groups
klab_stats = df.groupby('klab_idx').agg(
    citations=('klab_citations', 'first'),
    mean_composite=('composite_score', 'mean'),
    mean_similarity=('similarity', 'mean'),
    mean_perplexity=('perplexity', 'mean'),
    mean_steering=('steering_score', 'mean'),
    n_pairs=('arxiv_id', 'count')
).reset_index()

median_cit = klab_stats['citations'].median()
klab_stats['group'] = np.where(klab_stats['citations'] > median_cit, 'High Citation', 'Low Citation')

print("=" * 60)
print("Backtesting: High Citation vs. Low Citation KLab Papers")
print("=" * 60)
print(f"Median KLab citations: {median_cit}")
print(f"High Citation group: {(klab_stats['group'] == 'High Citation').sum()} papers")
print(f"Low Citation group:  {(klab_stats['group'] == 'Low Citation').sum()} papers")

# 3. Independent T-Tests across metrics
print(f"\n{'Metric':<20} {'High Mean':>12} {'Low Mean':>12} {'Difference':>12} {'p-value':>12}")
print("-" * 72)

for col in ['mean_composite', 'mean_similarity', 'mean_perplexity', 'mean_steering']:
    high = klab_stats[klab_stats['group'] == 'High Citation'][col]
    low = klab_stats[klab_stats['group'] == 'Low Citation'][col]
    t_stat, p_val = stats.ttest_ind(high, low, nan_policy='omit')
    label = col.replace('mean_', '')
    
    diff = high.mean() - low.mean()
    print(f"{label:<20} {high.mean():>12.4f} {low.mean():>12.4f} {diff:>12.4f} {p_val:>12.4f}")

# 4. Correlation: KLab Citations vs. Metrics
print(f"\n{'Metric':<20} {'Spearman r':>12} {'p-value':>12}")
print("-" * 46)

for col in ['mean_composite', 'mean_similarity', 'mean_perplexity', 'mean_steering']:
    r, p = stats.spearmanr(klab_stats['citations'], klab_stats[col])
    label = col.replace('mean_', '')
    print(f"{label:<20} {r:>12.4f} {p:>12.4f}")

print("\nComposite Score Distribution:")
print(df['composite_score'].describe())


# %%
# ==========================================
# Backtesting Part 2: Temporal Predictive Power
# ==========================================

# 1. Historical period (2019-2022) vs. Validation period (2023-2026)
df_historical = df[df['arxiv_year'] <= 2022].copy()
df_validation = df[df['arxiv_year'] >= 2023].copy()

print("=" * 75)
print("Temporal Backtesting: Predictive Power of Historical Scores (2019-2022)")
print("=" * 75)
print(f"Historical period (2019-2022): {len(df_historical):,} records")
print(f"Validation period (2023-2026): {len(df_validation):,} records")

# 2. Average scores per KLab paper in the historical period
historical_scores = df_historical.groupby('klab_idx').agg(
    citations=('klab_citations', 'first'),
    klab_title=('klab_title', 'first'),
    hist_similarity=('similarity', 'mean'),
    hist_perplexity=('perplexity', 'mean'),
    hist_steering=('steering_score', 'mean'),
    n_historical=('arxiv_id', 'count')
).reset_index()

# 3. Subsequent activity and metrics in the validation period
validation_scores = df_validation.groupby('klab_idx').agg(
    val_similarity=('similarity', 'mean'),
    val_perplexity=('perplexity', 'mean'),
    val_steering=('steering_score', 'mean'),
    n_validation=('arxiv_id', 'count')
).reset_index()

# 4. Merge periods
backtest = historical_scores.merge(validation_scores, on='klab_idx', how='inner')
print(f"\nKLab papers present in both periods: {len(backtest):,} papers")

# 5. Correlation: Historical Scores vs. Future Activity (n_validation)
print(f"\nHistorical Metric vs. Future Activity (n_validation):")
print(f"{'Historical Metric':<25} {'Spearman r':>12} {'p-value':>12}")
print("-" * 51)

for col in ['hist_similarity', 'hist_perplexity', 'hist_steering']:
    r, p = stats.spearmanr(backtest[col], backtest['n_validation'])
    label = col.replace('hist_', '')
    print(f"{label:<25} {r:>12.4f} {p:>12.4f}")

# 6. Correlation: Historical Scores vs. Future Similarity Change
backtest['similarity_change'] = backtest['val_similarity'] - backtest['hist_similarity']

print(f"\nHistorical Metric vs. Future Distance Change (similarity_change):")
print(f"{'Historical Metric':<25} {'Spearman r':>12} {'p-value':>12}")
print("-" * 51)

for col in ['hist_similarity', 'hist_perplexity', 'hist_steering']:
    r, p = stats.spearmanr(backtest[col], backtest['similarity_change'])
    label = col.replace('hist_', '')
    print(f"{label:<25} {r:>12.4f} {p:>12.4f}")

# 7. Extracting Top Predictive Cases
# Z-score normalize the historical perplexity for comparison
backtest['hist_composite'] = (
    (backtest['hist_perplexity'] - backtest['hist_perplexity'].mean()) / backtest['hist_perplexity'].std()
)

columns_to_show = ['klab_title', 'citations', 'hist_perplexity', 'n_historical', 'n_validation', 'similarity_change']
top5 = backtest.nlargest(5, 'hist_composite')[columns_to_show]

print(f"\nTop 5 KLab Directions by Historical Perplexity (Subsequent Outcomes):")
print(top5.to_string(index=False))

# %%
# %%
# ==========================================
# Setup and Global Data Loading
# ==========================================
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from sklearn.manifold import TSNE
from numpy.linalg import norm

warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
OUT_DIR = DRIVE_PATH / 'figures'
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')
df_conv = pd.read_csv(DRIVE_PATH / 'candidates/convergence_major_domains.csv')

klab_emb_data = np.load(DRIVE_PATH / 'klab_embeddings.npz')
klab_embeddings = klab_emb_data['embeddings']

with open(DRIVE_PATH / 'klab_metadata.pkl', 'rb') as f:
    klab_meta = pickle.load(f)

klab_full = None
for path in ['/content/klab_papers.json', DRIVE_PATH / 'klab_papers.json']:
    try:
        with open(path, 'r') as f:
            klab_full = json.load(f)
        break
    except FileNotFoundError:
        continue

# %%
# ==========================================
# Figure 1: KLab Research Landscape (t-SNE)
# ==========================================
def clean_html(text):
    return re.sub(r'<[^>]+>', '', str(text))

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(klab_embeddings)

domains = []
citations = []
for i, meta in enumerate(klab_meta):
    cit = meta.get('cited_by_count', 0)
    citations.append(cit if cit else 0)
    
    domain = 'Other'
    if klab_full:
        oa_id = meta.get('openalex_id', '')
        for p in klab_full:
            if p.get('openalex_id', '') == oa_id:
                concepts = p.get('concepts', [])
                top = [c for c in concepts if c.get('level', 99) == 0]
                if top:
                    domain = sorted(top, key=lambda x: x.get('score', 0), reverse=True)[0]['name']
                break
    domains.append(domain)

citations = np.array(citations)
domain_counts = Counter(domains)
top_domains = [d for d, _ in domain_counts.most_common(8)]
domains_clean = [d if d in top_domains else 'Other' for d in domains]
unique_domains = sorted(set(domains_clean))

color_map = {
    'Biology': '#27ae60', 'Chemistry': '#e67e22', 'Computer science': '#3498db',
    'Materials science': '#e91e63', 'Medicine': '#00bcd4', 'Other': '#bdc3c7',
    'Political science': '#f39c12', 'Psychology': '#9b59b6',
}

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor('#fafafa')

for domain in unique_domains:
    mask = np.array([d == domain for d in domains_clean])
    sizes = np.clip(citations[mask] / 3 + 20, 20, 400)
    c = color_map.get(domain, '#bdc3c7')
    ax.scatter(coords[mask, 0], coords[mask, 1],
               s=sizes, c=c, label=f'{domain} ({mask.sum()})',
               alpha=0.7, edgecolors='white', linewidth=0.6, zorder=2)

top_indices = np.argsort(citations)[-8:]
texts = []

try:
    from adjustText import adjust_text
    has_adjust = True
except ImportError:
    has_adjust = False

for idx in top_indices:
    if citations[idx] > 30:
        title = clean_html(klab_meta[idx].get('title', ''))
        title = title[:35] + '…' if len(title) > 35 else title

        if has_adjust:
            t = ax.annotate(title, (coords[idx, 0], coords[idx, 1]),
                           fontsize=7, alpha=0.9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
            texts.append(t)
        else:
            offset_x = 8 if idx % 2 == 0 else -8
            offset_y = 8 if idx % 3 == 0 else -8
            ax.annotate(title, (coords[idx, 0], coords[idx, 1]),
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=7, alpha=0.9,
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

if has_adjust and texts:
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

ax.set_title('KLab Research Landscape\n(Size = Citation Count, Color = Primary Domain)', pad=15)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='lower right', fontsize=9, framealpha=0.95, ncol=1, markerscale=0.8, title='Domain')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_klab_landscape.png')
plt.show()

# %%
# ==========================================
# Figure 2: Perplexity Distribution
# ==========================================
klab_info = df.groupby('klab_idx').agg(citations=('klab_citations', 'first')).reset_index()
median_cit = klab_info['citations'].median()

high_cit_klab = set(klab_info[klab_info['citations'] > median_cit]['klab_idx'])
ppl_high = df[df['klab_idx'].isin(high_cit_klab)]['perplexity']
ppl_low = df[~df['klab_idx'].isin(high_cit_klab)]['perplexity']

clip_max = np.percentile(df['perplexity'], 98)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(ppl_high.clip(upper=clip_max), bins=80, alpha=0.6, color='#e74c3c', label=f'High-citation (>{median_cit:.0f})', density=True)
ax.hist(ppl_low.clip(upper=clip_max), bins=80, alpha=0.6, color='#3498db', label=f'Low-citation (≤{median_cit:.0f})', density=True)

ax.axvline(ppl_high.median(), color='#e74c3c', linestyle='--', lw=1.5, label=f'High median: {ppl_high.median():.1f}')
ax.axvline(ppl_low.median(), color='#3498db', linestyle='--', lw=1.5, label=f'Low median: {ppl_low.median():.1f}')

_, p_val = stats.ttest_ind(ppl_high, ppl_low)
ax.text(0.97, 0.95, f't-test p = {p_val:.4f}', transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Perplexity Score')
ax.set_ylabel('Density')
ax.set_title('Perplexity Distribution: High-Citation vs Low-Citation KLab Papers')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_perplexity_distribution.png')
plt.show()

# %%
# ==========================================
# Figure 3: Temporal Trajectory (Data Prep & Plot)
# ==========================================
ARXIV_RAW = '/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json'
YEARLY_OUT = DRIVE_PATH / 'candidates/convergence_yearly.csv'

arxiv_emb = np.load(DRIVE_PATH / 'arxiv_embeddings.npz', allow_pickle=True)
embeddings = arxiv_emb['embeddings']
id_to_idx = {pid: i for i, pid in enumerate(arxiv_emb['paper_ids'])}

id_to_info = {}
with open(ARXIV_RAW, 'r') as f:
    for line in f:
        paper = json.loads(line)
        pid = paper.get('id', '')
        if pid in id_to_idx:
            cats = paper.get('categories', '')
            primary_domain = cats.split()[0].split('.')[0] if cats else ''
            update_date = paper.get('update_date', '')
            year = int(update_date[:4]) if update_date and len(update_date) >= 4 else 0
            if primary_domain and year >= 2019:
                id_to_info[pid] = (primary_domain, year)

group_embeddings = defaultdict(list)
for pid, (domain, year) in id_to_info.items():
    group_embeddings[(domain, year)].append(id_to_idx[pid])

centroids = {k: embeddings[v].mean(axis=0) for k, v in group_embeddings.items() if len(v) >= 10}
domains = sorted(set(d for d, y in centroids.keys()))
years = sorted(set(y for d, y in centroids.keys()))

results = []
for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        yearly_distances = {}
        for year in years:
            if (d1, year) in centroids and (d2, year) in centroids:
                v1, v2 = centroids[(d1, year)], centroids[(d2, year)]
                cos_sim = np.dot(v1, v2) / (norm(v1) * norm(v2))
                yearly_distances[year] = round(float(cos_sim), 6)
        if len(yearly_distances) >= 3:
            results.append({'domain_1': d1, 'domain_2': d2, 'yearly_distances': yearly_distances})

rows = []
for r in results:
    for year, dist in r['yearly_distances'].items():
        rows.append({'domain_1': r['domain_1'], 'domain_2': r['domain_2'], 'year': year, 'cosine_similarity': dist})

df_yearly = pd.DataFrame(rows)
df_yearly.to_csv(YEARLY_OUT, index=False)

pair_slopes = []
for (d1, d2), grp in df_yearly.groupby(['domain_1', 'domain_2']):
    if len(grp) >= 4:
        slope, _, _, _, _ = stats.linregress(grp['year'], grp['cosine_similarity'])
        pair_slopes.append({'d1': d1, 'd2': d2, 'slope': slope})

df_slopes = pd.DataFrame(pair_slopes)
selected = pd.concat([df_slopes.nlargest(3, 'slope'), df_slopes.nsmallest(2, 'slope')])

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#fafafa')
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
markers = ['o', 's', '^', 'D', 'v']

for i, (_, row) in enumerate(selected.iterrows()):
    d1, d2 = row['d1'], row['d2']
    data = df_yearly[(df_yearly['domain_1']==d1) & (df_yearly['domain_2']==d2)].sort_values('year')
    direction = "↗ Converging" if row['slope'] > 0 else "↘ Diverging"
    ax.plot(data['year'], data['cosine_similarity'], color=colors[i], marker=markers[i],
            linewidth=2.5, markersize=8, label=f"{d1} × {d2} ({direction}, slope={row['slope']:.4f})", alpha=0.85)

ax.set_xlabel('Year')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Temporal Trajectory: Inter-Domain Embedding Distance (2019–2026)')
ax.legend(fontsize=9, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(2019, 2027))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_temporal_trajectory.png')
plt.show()

# %%
# ==========================================
# Figure 4: Convergence Ranking
# ==========================================
top15_conv = df_conv.nsmallest(15, 'slope').copy()
top15_conv['pair_label'] = top15_conv['domain_1'] + ' × ' + top15_conv['domain_2']

fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = ['#e74c3c' if s < -0.01 else '#e67e22' if s < -0.005 else '#f1c40f' for s in top15_conv['slope']]
bars = ax.barh(range(len(top15_conv)), top15_conv['slope'].values, color=colors_bar, alpha=0.85)

ax.set_yticks(range(len(top15_conv)))
ax.set_yticklabels(top15_conv['pair_label'].values, fontsize=9)
ax.set_xlabel('Convergence Slope (More Negative = Faster Convergence)')
ax.set_title('Top 15 Fastest Converging Domain Pairs')
ax.axvline(0, color='black', linewidth=0.5)
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

for i, val in enumerate(top15_conv['slope'].values):
    ax.text(val - 0.001, i, f'{val:.4f}', va='center', ha='right', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_convergence_ranking.png')
plt.show()

# %%
# ==========================================
# Figure 5: Sweet Spot Map
# ==========================================
df_unique = df.drop_duplicates(subset='arxiv_id').copy()
sample = df_unique.sample(n=min(5000, len(df_unique)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 9))
scatter = ax.scatter(
    1 - sample['similarity'], 
    sample['perplexity'].clip(upper=np.percentile(sample['perplexity'], 98)),
    c=sample['steering_score'],
    cmap='RdYlGn', s=12, alpha=0.5, edgecolors='none'
)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Steering Score (Interdisciplinarity)')
ax.set_xlabel('Novelty (1 - Cosine Similarity)')
ax.set_ylabel('Perplexity (Language Surprise)')
ax.set_title('"Sweet Spot" Map: Research Opportunity Landscape')

ax.axhline(y=np.percentile(sample['perplexity'], 75), color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=np.percentile(1 - sample['similarity'], 75), color='gray', linestyle=':', alpha=0.5)
ax.text(0.97, 0.97, 'HIGH\nOPPORTUNITY', transform=ax.transAxes, ha='right', va='top', fontsize=12, color='darkgreen', alpha=0.4, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_sweet_spot_map.png')
plt.show()

# %%
# ==========================================
# Figure 6: Token-Level Steering Vector Activation
# ==========================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"

if 'model' not in globals():
    HF_TOKEN = "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"
    login(token=HF_TOKEN)
    MODEL_ID = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

if 'sv_gpu' not in globals():
    best_layer = 15
    steering_vector = torch.load(DRIVE_PATH / 'steering_vector_v2.pt', map_location=device, weights_only=False)
    sv = steering_vector.layer_activations[best_layer]
    sv = sv / sv.norm()
    sv_gpu = sv.to(device).float()

high_papers = df_unique.nlargest(3, 'steering_score')[['arxiv_title', 'arxiv_abstract', 'steering_score']]
low_papers = df_unique.nsmallest(3, 'steering_score')[['arxiv_title', 'arxiv_abstract', 'steering_score']]

def get_token_activations(text, max_length=128):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[best_layer + 1][0]
        activations = torch.matmul(hidden.float(), sv_gpu).cpu().numpy()
        
    clean_tokens = [t.replace('Ġ', ' ').replace('▁', ' ').replace('<s>', '[BOS]') for t in tokens]
    return clean_tokens, activations

all_results = []
for label, papers in [('HIGH', high_papers), ('LOW', low_papers)]:
    for _, row in papers.iterrows():
        tokens, activations = get_token_activations(str(row['arxiv_abstract'])[:500])
        all_results.append({
            'label': label, 'title': row['arxiv_title'][:50], 
            'score': row['steering_score'], 'tokens': tokens, 'activations': activations
        })

fig, axes = plt.subplots(6, 1, figsize=(20, 18))
all_acts = np.concatenate([r['activations'] for r in all_results])
vmin, vmax = np.percentile(all_acts, 5), np.percentile(all_acts, 95)

for i, result in enumerate(all_results):
    ax = axes[i]
    n_show = min(60, len(result['tokens']))
    im = ax.imshow(result['activations'][:n_show].reshape(1, -1), aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
    
    ax.set_yticks([0])
    ax.set_yticklabels([f"{result['label']} [{result['score']:.3f}]"])
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(result['tokens'][:n_show], rotation=90, fontsize=6, ha='center')
    ax.set_title(f"{result['title']}...", loc='left', pad=2)

cbar = fig.colorbar(im, ax=axes, shrink=0.3, location='right', pad=0.02)
cbar.set_label('Activation along steering vector')
fig.suptitle('Token-Level Activation Along "Interdisciplinarity" Steering Vector', y=1.01)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_token_activation_full.png', bbox_inches='tight')
plt.show()

# %%
# ==========================================
# Figure 6b: Top Tokens
# ==========================================
token_scores = defaultdict(list)
for _, papers in [('high', df_unique.nlargest(50, 'steering_score')), ('low', df_unique.nsmallest(50, 'steering_score'))]:
    for _, row in papers.iterrows():
        tokens, acts = get_token_activations(str(row['arxiv_abstract'])[:300], max_length=80)
        for t, a in zip(tokens, acts):
            clean = t.replace('Ġ', '').replace('▁', '').strip().lower()
            if len(clean) >= 3:
                token_scores[clean].append(float(a))

avg_scores = {w: np.mean(s) for w, s in token_scores.items() if len(s) >= 5}
sorted_tokens = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
combined = sorted_tokens[-25:] + sorted_tokens[:25]

fig, ax = plt.subplots(figsize=(12, 10))
words, scores = [w[0] for w in combined], [w[1] for w in combined]
colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in scores]

ax.barh(range(len(words)), scores, color=colors, alpha=0.85)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mean Activation')
ax.set_title('Top Tokens by Steering Vector Activation')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6b_token_ranking.png')
plt.show()

# %%
# ==========================================
# Figure 7: Correlation Matrix
# ==========================================
df_for_corr = df_unique[['similarity', 'perplexity', 'steering_score']].copy()
df_for_corr.columns = ['Embedding\nSimilarity', 'Perplexity', 'Steering\nScore']
corr_matrix = df_for_corr.corr(method='spearman')

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spearman Correlation')

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns)
ax.set_yticklabels(corr_matrix.columns)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=14, fontweight='bold', color=color)

ax.set_title('Correlation Matrix: Three Measurement Scales')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig7_correlation_matrix.png')
plt.show()

# %%
# ==========================================
# Figure 8: Backtest and Rankings
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Backtest
df_early = df[df['arxiv_year'] <= 2022]
df_late = df[df['arxiv_year'] >= 2023]

early_scores = df_early.groupby('klab_idx').agg(
    early_steering=('steering_score', 'mean'),
    early_similarity=('similarity', 'mean'),
    early_perplexity=('perplexity', 'mean'),
).reset_index()
late_scores = df_late.groupby('klab_idx').agg(n_late=('arxiv_id', 'count')).reset_index()

backtest = early_scores.merge(late_scores, on='klab_idx')
metrics = ['early_steering', 'early_similarity', 'early_perplexity']
labels = ['Steering\nScore', 'Embedding\nSimilarity', 'Perplexity']
r_values = [stats.spearmanr(backtest[m], backtest['n_late'])[0] for m in metrics]

colors_bt = ['#2ecc71' if r > 0.3 else '#f1c40f' if r > 0.1 else '#e74c3c' for r in r_values]
bars = axes[0].bar(labels, r_values, color=colors_bt, alpha=0.85, width=0.6)

for bar, r in zip(bars, r_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'r = {r:.3f}', ha='center', fontweight='bold')

axes[0].set_ylabel('Spearman Correlation with Future Activity')
axes[0].set_title('Predictive Power for Future Research Activity (2019-22 → 2023-26)')
axes[0].set_ylim(0, 0.7)
axes[0].grid(True, axis='y', alpha=0.3)
axes[0].axhline(0, color='black', linewidth=0.5)

# Subplot 2: Rankings
for col in ['similarity', 'perplexity', 'steering_score']:
    df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
df['composite'] = 0.5 * df['steering_score_z'] + 0.3 * df['perplexity_z'] + 0.2 * df['similarity_z']

top15 = df.nlargest(15, 'composite')[['klab_title', 'arxiv_title', 'composite']].copy()
top15['label'] = top15.apply(lambda r: r['klab_title'][:25] + '...\n× ' + r['arxiv_title'][:25] + '...', axis=1)

colors_rank = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 15))[::-1]
axes[1].barh(range(14, -1, -1), top15['composite'].values, color=colors_rank, alpha=0.85)
axes[1].set_yticks(range(14, -1, -1))
axes[1].set_yticklabels(top15['label'].values, fontsize=7)
axes[1].set_xlabel('Composite Score (Weighted)')
axes[1].set_title('Top 15 Research Opportunity Candidates')
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig8_backtest_and_ranking.png')
plt.show()