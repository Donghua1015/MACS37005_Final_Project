"""
arXiv Embedding Pipeline - Colab Pro Optimized

Optimizations for Colab Pro:
- Auto-detect GPU and adjust batch size
- Support for A100/V100 large batch processing
- Optional SciBERT for better quality (if A100 available)
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
# Config - AUTO-TUNED FOR YOUR GPU
# ---------------------------------------------------------------------------

# Model selection
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Default: fast
# MODEL_NAME = "allenai/scibert_scivocab_uncased"     # Better for science papers
# MODEL_NAME = "allenai/specter"                      # Citation-aware

# Paths
ARXIV_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "arxiv_embeddings.npz"
METADATA_FILE = "arxiv_metadata_simple.json"

# Auto-tuned parameters (will be adjusted based on GPU)
BATCH_SIZE = None  # Will be auto-set
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
