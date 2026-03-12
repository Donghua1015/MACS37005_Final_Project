# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# ===== 第一步：先挂载 Drive（在跑 pipeline 之前！）=====
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/serendipity_agent

# ===== 第二步：跑你的 embedding pipeline =====
!python arxiv_embedding_colab_pro.py

# ===== 第三步：跑完后立刻备份 =====
!cp arxiv_embeddings.npz /content/drive/MyDrive/serendipity_agent/
!cp arxiv_metadata_simple.json /content/drive/MyDrive/serendipity_agent/
!cp klab_embeddings.npz /content/drive/MyDrive/serendipity_agent/
!cp klab_metadata.pkl /content/drive/MyDrive/serendipity_agent/

# %%
from google.colab import drive
drive.mount('/content/drive')
!ls -lh /content/drive/MyDrive/serendipity_agent/

# %%
# ============================================
# KLab × arXiv 相似度搜索
# 直接从 Google Drive 运行（A100 约 5 分钟）
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
print("KLab × arXiv 相似度搜索")
print("=" * 70)

# 参数
TOP_K = 1000
SWEET_SPOT_LOW = 0.3
SWEET_SPOT_HIGH = 0.7
BATCH_SIZE = 1000

# 路径
DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
OUTPUT_PATH = DRIVE_PATH / 'candidates'
OUTPUT_PATH.mkdir(exist_ok=True)

# ============================================
# 1. 加载数据
# ============================================

print("\n[1/4] 加载 KLab embeddings...")
klab_data = np.load(DRIVE_PATH / 'klab_embeddings.npz')
klab_embeddings = klab_data['embeddings']
print(f"  ✓ {klab_embeddings.shape}")

print("\n[2/4] 加载 KLab metadata...")
with open(DRIVE_PATH / 'klab_metadata.pkl', 'rb') as f:
    klab_metadata = pickle.load(f)
print(f"  ✓ {len(klab_metadata)} papers")

print("\n[3/4] 加载 arXiv embeddings (4 GB，需要 30 秒)...")
arxiv_data = np.load(DRIVE_PATH / 'arxiv_embeddings.npz')
arxiv_embeddings = arxiv_data['embeddings']
print(f"  ✓ {arxiv_embeddings.shape}")

print("\n[4/4] 加载 arXiv metadata...")
with open(DRIVE_PATH / 'arxiv_metadata_simple.json', 'r') as f:
    arxiv_metadata = json.load(f)
print(f"  ✓ {len(arxiv_metadata)} papers")

# ============================================
# 2. 计算相似度
# ============================================

print("\n" + "=" * 70)
print("计算 KLab × arXiv 相似度")
print("=" * 70)
print(f"矩阵: {len(klab_embeddings)} × {len(arxiv_embeddings)}")
print(f"总对数: {len(klab_embeddings) * len(arxiv_embeddings):,}")

n_arxiv = len(arxiv_embeddings)
n_batches = int(np.ceil(n_arxiv / BATCH_SIZE))

similarities = np.zeros((len(klab_embeddings), n_arxiv), dtype=np.float32)

for i in tqdm(range(n_batches), desc="计算相似度"):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, n_arxiv)

    arxiv_batch = arxiv_embeddings[start_idx:end_idx]
    sim_batch = cosine_similarity(klab_embeddings, arxiv_batch)

    similarities[:, start_idx:end_idx] = sim_batch

print("\n✓ 相似度计算完成")

# ============================================
# 3. 提取候选
# ============================================

print("\n" + "=" * 70)
print("提取候选论文对")
print("=" * 70)
print(f"Top-{TOP_K} per KLab paper")
print(f"甜蜜区间: [{SWEET_SPOT_LOW}, {SWEET_SPOT_HIGH}]")

candidates = []
stats = {
    'total_pairs': 0,
    'in_sweet_spot': 0,
    'above_sweet_spot': 0,
    'below_sweet_spot': 0,
}

for klab_idx in tqdm(range(len(klab_metadata)), desc="提取候选"):
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

print(f"\n✓ 提取了 {len(candidates):,} 候选对")
# ============================================
# 4. 保存结果
# ============================================

print("\n" + "=" * 70)
print("保存结果到 Google Drive")
print("=" * 70)

# JSON
json_file = OUTPUT_PATH / 'klab_arxiv_candidates.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(candidates, f, indent=2, ensure_ascii=False)
print(f"  ✓ JSON: {json_file}")

# CSV
csv_file = OUTPUT_PATH / 'klab_arxiv_candidates.csv'
df = pd.DataFrame(candidates)
df.to_csv(csv_file, index=False)
print(f"  ✓ CSV: {csv_file}")

# 报告
report_lines = [
    "=" * 70,
    "KLab × arXiv 相似度搜索报告",
    "=" * 70,
    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "参数:",
    f"  甜蜜区间: [{SWEET_SPOT_LOW}, {SWEET_SPOT_HIGH}]",
    f"  Top-K: {TOP_K}",
    "",
    "统计:",
    f"  总候选对: {len(candidates):,}",
    f"  在甜蜜区间: {stats['in_sweet_spot']:,} ({stats['in_sweet_spot']/stats['total_pairs']*100:.1f}%)",
    f"  高于甜蜜区间: {stats['above_sweet_spot']:,} ({stats['above_sweet_spot']/stats['total_pairs']*100:.1f}%)",
    f"  低于甜蜜区间: {stats['below_sweet_spot']:,} ({stats['below_sweet_spot']/stats['total_pairs']*100:.1f}%)",
    "",
    "相似度分布:",
    f"  Mean:   {df['similarity'].mean():.4f}",
    f"  Median: {df['similarity'].median():.4f}",
    f"  Min:    {df['similarity'].min():.4f}",
    f"  Max:    {df['similarity'].max():.4f}",
    "",
]

sweet_df = df[df['in_sweet_spot']]
if len(sweet_df) > 0:
    report_lines.extend([
        f"甜蜜区间详细:",
        f"  候选对数: {len(sweet_df):,}",
        f"  平均相似度: {sweet_df['similarity'].mean():.4f}",
        "",
        "Top 10 KLab 论文（按甜蜜区间候选数）:",
        "",
    ])

    top_klab = sweet_df.groupby('klab_title').size().sort_values(ascending=False).head(10)
    for i, (title, count) in enumerate(top_klab.items(), 1):
        report_lines.append(f"  {i:2d}. {count:4d} 候选 | {title[:55]}...")

    report_lines.extend([
        "",
        "随机样本（甜蜜区间）:",
        "",
    ])

    samples = sweet_df.sample(min(5, len(sweet_df)))
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        report_lines.extend([
            f"{i}. KLab:  [{row['klab_year']}] {row['klab_title'][:55]}",
            f"   arXiv: [{row['arxiv_year']}] {row['arxiv_title'][:55]}",
            f"   相似度: {row['similarity']:.4f} | 分类: {row['arxiv_categories']}",
            "",
        ])

report_text = "\n".join(report_lines)

report_file = OUTPUT_PATH / 'similarity_search_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"  ✓ Report: {report_file}")

# ============================================
# 5. 显示报告
# ============================================

print("\n" + "=" * 70)
print(report_text)
print("=" * 70)

# 下载
print("\n下载结果...")
from google.colab import files
files.download(str(csv_file))
files.download(str(report_file))

print("\n✅ 相似度搜索完成！")
print(f"所有结果已保存到: {OUTPUT_PATH}")

# %%
import pandas as pd

# 读取结果
df = pd.read_csv('/content/drive/MyDrive/serendipity_agent/candidates/klab_arxiv_candidates.csv')

# 检查 KLab 论文数量
print("KLab 论文统计:")
print(f"  唯一 KLab 标题: {df['klab_title'].nunique()}")
print(f"  唯一 KLab ID: {df['klab_id'].nunique()}")
print(f"  总行数: {len(df)}")

# 查看重复最多的标题
print("\n重复最多的 KLab 论文:")
title_counts = df.groupby('klab_title')['klab_id'].nunique()
duplicates = title_counts[title_counts > 1].sort_values(ascending=False).head(10)
for title, count in duplicates.items():
    print(f"  {count} 个不同 ID | {title[:60]}")

# 查看 "A Simple Threshold..." 的详细信息
threshold_df = df[df['klab_title'].str.contains('A Simple Threshold', na=False)]
print(f"\n'A Simple Threshold...' 的记录:")
print(f"  总行数: {len(threshold_df)}")
print(f"  唯一 klab_idx: {threshold_df['klab_idx'].nunique()}")
print(f"  klab_idx 列表: {threshold_df['klab_idx'].unique()}")

# 查看这些 klab_idx 对应的原始数据
unique_idx = threshold_df['klab_idx'].unique()
for idx in unique_idx[:5]:  # 只看前 5 个
    rows = threshold_df[threshold_df['klab_idx'] == idx]
    print(f"\nklab_idx={idx}:")
    print(f"  klab_id: {rows.iloc[0]['klab_id']}")
    print(f"  klab_year: {rows.iloc[0]['klab_year']}")
    print(f"  候选数: {len(rows)}")

# %%
import pandas as pd

# 读取结果
df = pd.read_csv('/content/drive/MyDrive/serendipity_agent/candidates/klab_arxiv_candidates.csv')

print(f"去重前: {len(df):,} 行")

# 方法 1: 按 klab_title 去重（保留第一个）
df_dedup = df.drop_duplicates(subset=['klab_title', 'arxiv_id'])
print(f"按标题去重: {len(df_dedup):,} 行")

# 方法 2: 按 klab_id 去重（保留所有版本的第一个）
df_dedup2 = df.drop_duplicates(subset=['klab_id', 'arxiv_id'])
print(f"按 ID 去重: {len(df_dedup2):,} 行")

# 保存去重后的结果
df_dedup.to_csv('/content/drive/MyDrive/serendipity_agent/candidates/klab_arxiv_candidates_dedup.csv', index=False)
print("\n✓ 去重结果已保存")

# 重新统计
sweet_df = df_dedup[df_dedup['in_sweet_spot']]
print(f"\n去重后甜蜜区间候选: {len(sweet_df):,}")

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd
from pathlib import Path

print("=" * 70)
print("量尺 2 预处理：按年份分层候选数据 (Temporal Stratification)")
print("=" * 70)

# 配置路径
DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent/candidates')
INPUT_CSV = DRIVE_PATH / 'klab_arxiv_candidates_dedup.csv'

# 定义输出路径
OUTPUT_GPT2 = DRIVE_PATH / 'batch_gpt2_2019_2023.csv'
OUTPUT_LLAMA3 = DRIVE_PATH / 'batch_llama3_2024_plus.csv'
OUTPUT_ARCHIVE = DRIVE_PATH / 'batch_archive_pre_2019.csv'

print(f"\n[1/3] 正在读取候选数据: {INPUT_CSV.name}...")
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"  ✓ 成功读取: {len(df):,} 条记录")
except FileNotFoundError:
    print(f"  ❌ 找不到文件: {INPUT_CSV}")
    print("  请确保你的 Google Drive 已挂载且路径正确。")
    raise

# 确保 arxiv_year 是数字类型 (处理可能的异常值)
df['arxiv_year'] = pd.to_numeric(df['arxiv_year'], errors='coerce')

# 清理空值（如果有的话）
initial_len = len(df)
df = df.dropna(subset=['arxiv_year'])
if len(df) < initial_len:
    print(f"  ⚠️ 清理了 {initial_len - len(df):,} 条没有年份数据的记录")

print("\n[2/3] 正在按时间分桶 (Stratifying by arXiv Year)...")

# 1. GPT-2 批次 (2019 - 2023)
# 理由: GPT-2 训练数据截止于 2018，这批论文对它来说是"未知"的
mask_gpt2 = (df['arxiv_year'] >= 2019) & (df['arxiv_year'] <= 2023)
df_gpt2 = df[mask_gpt2]

# 2. Llama-3 / OLMo 批次 (2024 及以后)
# 理由: 较新的模型知识截止较晚，适合评估最新的研究
mask_llama3 = (df['arxiv_year'] >= 2024)
df_llama3 = df[mask_llama3]

# 3. 归档批次 (2018及以前)
# 理由: 大多数现代 LLM 都读过这些，测出来的 PPL 极低，建议跳过或使用极老的模型
mask_archive = (df['arxiv_year'] <= 2018)
df_archive = df[mask_archive]

print(f"  ├─ GPT-2 批次 (2019-2023): {len(df_gpt2):,} 条 ({len(df_gpt2)/len(df)*100:.1f}%)")
print(f"  ├─ Llama-3 批次 (2024+):   {len(df_llama3):,} 条 ({len(df_llama3)/len(df)*100:.1f}%)")
print(f"  └─ 归档批次 (≤ 2018):      {len(df_archive):,} 条 ({len(df_archive)/len(df)*100:.1f}%)")

# 验证数据没有遗漏
assert len(df_gpt2) + len(df_llama3) + len(df_archive) == len(df), "分桶数据总和不匹配！"

print("\n[3/3] 正在保存分层数据集...")
df_gpt2.to_csv(OUTPUT_GPT2, index=False)
df_llama3.to_csv(OUTPUT_LLAMA3, index=False)
df_archive.to_csv(OUTPUT_ARCHIVE, index=False)

print(f"  ✓ GPT-2 Batch:   {OUTPUT_GPT2.name}")
print(f"  ✓ Llama-3 Batch: {OUTPUT_LLAMA3.name}")
print(f"  ✓ Archive Batch: {OUTPUT_ARCHIVE.name}")

print("\n" + "=" * 70)
print("🎉 预处理完成！下一步: 为不同批次运行专属的 Perplexity 计算引擎。")
print("=" * 70)

# %%
import pandas as pd
import torch
import os
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("量尺 2: Perplexity 计算引擎 (GPT-2 终极版)")
print("=" * 70)

# ==========================================
# 1. 配置参数
# ==========================================
MODEL_ID = "gpt2"
BATCH_SIZE = 16       # 如果 OOM 提示显存不足，可调小至 8 或 4
SAVE_INTERVAL = 1000  # 每算 1000 条存一次档
MAX_LENGTH = 512      # 文本截断长度

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
INPUT_CSV = DRIVE_PATH / 'candidates/batch_gpt2_2019_2023.csv'
JSON_PATH = DRIVE_PATH / 'arxiv_metadata_simple.json'
OUTPUT_CSV = DRIVE_PATH / 'candidates/scored_gpt2_2019_2023.csv'

# ==========================================
# 2. 智能合并摘要 (防内存溢出设计)
# ==========================================
print(f"\n[1/5] 正在读取待处理候选对列表...")
df_input = pd.read_csv(INPUT_CSV)
needed_ids = set(df_input['arxiv_id'].astype(str))
print(f"  ✓ 找到 {len(df_input):,} 条候选对，涉及 {len(needed_ids):,} 篇独立的 arXiv 论文。")

print(f"\n[2/5] 正在从原始 JSON 中按需提取摘要 (这可能需要一两分钟)...")
id_to_abstract = {}

# 适配不同格式的 JSON 文件加载
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        # 尝试直接加载 (如果文件是标准 JSON 数组或字典)
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                arxiv_id = str(item.get('id', item.get('arxiv_id', '')))
                if arxiv_id in needed_ids:
                    id_to_abstract[arxiv_id] = item.get('abstract', '')
        elif isinstance(data, dict):
            for k, v in data.items():
                if str(k) in needed_ids:
                    # v 可能是字符串(只有摘要)或字典(包含多字段)
                    id_to_abstract[str(k)] = v.get('abstract', '') if isinstance(v, dict) else str(v)
except json.JSONDecodeError:
    # 如果是 JSONL (按行分割的 JSON) 格式
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            arxiv_id = str(item.get('id', item.get('arxiv_id', '')))
            if arxiv_id in needed_ids:
                id_to_abstract[arxiv_id] = item.get('abstract', '')

print(f"  ✓ 成功提取了 {len(id_to_abstract):,} 篇摘要！")

# 将摘要拼接到 DataFrame 中
df_input['arxiv_abstract'] = df_input['arxiv_id'].astype(str).map(id_to_abstract)
# 填补极少数找不到摘要的空值
df_input['arxiv_abstract'] = df_input['arxiv_abstract'].fillna("No abstract available.")

# ==========================================
# 3. 环境与模型初始化
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[3/5] 正在加载模型 {MODEL_ID} 到 {device.upper()}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
model.eval()
print("  ✓ 模型加载完成！")

# ==========================================
# 4. 断点续传机制
# ==========================================
print(f"\n[4/5] 正在检查断点续传进度...")
processed_ids = set()
if OUTPUT_CSV.exists():
    df_out = pd.read_csv(OUTPUT_CSV)
    if 'arxiv_id' in df_out.columns:
        processed_ids = set(df_out['arxiv_id'].astype(str))
        print(f"  ✓ 发现已有进度，已跳过 {len(processed_ids):,} 条已计算的记录。")

df_todo = df_input[~df_input['arxiv_id'].astype(str).isin(processed_ids)].copy()
print(f"  ▶ 本次需要计算 PPL: {len(df_todo):,} 条")

if len(df_todo) == 0:
    print("\n🎉 所有数据都已处理完毕！可以开始准备 Llama-3 批次了！")
    import sys
    sys.exit()

# ==========================================
# 5. 核心计算逻辑 (GPU 批处理)
# ==========================================
print(f"\n[5/5] 开始计算 Perplexity (Batch Size: {BATCH_SIZE})...")

def calculate_ppl_batch(texts):
    encodings = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH
    ).to(device)

    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        masked_loss = loss * shift_mask
        seq_lengths = torch.clamp(shift_mask.sum(dim=1), min=1.0)
        seq_loss = masked_loss.sum(dim=1) / seq_lengths
        ppl = torch.exp(seq_loss)

    return ppl.cpu().tolist()

results = []
abstracts = df_todo['arxiv_abstract'].tolist()
arxiv_ids = df_todo['arxiv_id'].tolist()

if not OUTPUT_CSV.exists():
    pd.DataFrame(columns=list(df_todo.columns) + ['perplexity']).to_csv(OUTPUT_CSV, index=False)

for i in tqdm(range(0, len(abstracts), BATCH_SIZE), desc="PPL Calculation"):
    batch_texts = abstracts[i : i+BATCH_SIZE]
    batch_texts = [t if len(t.strip()) > 10 else "No abstract available." for t in batch_texts]

    batch_ppls = calculate_ppl_batch(batch_texts)

    for j in range(len(batch_ppls)):
        row_idx = df_todo.index[i + j]
        row_data = df_todo.loc[row_idx].to_dict()
        row_data['perplexity'] = round(batch_ppls[j], 4)
        results.append(row_data)

    if len(results) >= SAVE_INTERVAL:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
        results = []

if len(results) > 0:
    pd.DataFrame(results).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

print(f"\n✓ 批次计算完成！结果已安全保存至: {OUTPUT_CSV.name}")
print("=" * 70)

# %%
import pandas as pd
import torch
import os
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

print("=" * 70)
print("量尺 2: Perplexity 计算引擎 (Llama-3 重装版)")
print("=" * 70)

# ==========================================
# 0. Hugging Face 登录 (必须)
# ==========================================
HF_TOKEN = "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"
login(token=HF_TOKEN)

# ==========================================
# 1. 配置参数
# ==========================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
BATCH_SIZE = 8        # Llama-3 很大，A100 建议 8，如果 OOM 就改 4
SAVE_INTERVAL = 500   # 存盘频率调高，防止意外
MAX_LENGTH = 512

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
INPUT_CSV = DRIVE_PATH / 'candidates/batch_llama3_2024_plus.csv'
JSON_PATH = DRIVE_PATH / 'arxiv_metadata_simple.json'
OUTPUT_CSV = DRIVE_PATH / 'candidates/scored_llama3_2024_plus.csv'

# ==========================================
# 2. 智能合并摘要
# ==========================================
print(f"\n[1/5] 正在读取待处理候选对列表...")
df_input = pd.read_csv(INPUT_CSV)
needed_ids = set(df_input['arxiv_id'].astype(str))
print(f"  ✓ 找到 {len(df_input):,} 条候选对。")

print(f"\n[2/5] 正在按需提取摘要...")
id_to_abstract = {}
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                arxiv_id = str(item.get('id', item.get('arxiv_id', '')))
                if arxiv_id in needed_ids:
                    id_to_abstract[arxiv_id] = item.get('abstract', '')
        elif isinstance(data, dict):
            for k, v in data.items():
                if str(k) in needed_ids:
                    id_to_abstract[str(k)] = v.get('abstract', '') if isinstance(v, dict) else str(v)
except json.JSONDecodeError:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            arxiv_id = str(item.get('id', item.get('arxiv_id', '')))
            if arxiv_id in needed_ids:
                id_to_abstract[arxiv_id] = item.get('abstract', '')

df_input['arxiv_abstract'] = df_input['arxiv_id'].astype(str).map(id_to_abstract).fillna("No abstract available.")
print(f"  ✓ 成功提取摘要！")

# ==========================================
# 3. 加载 Llama-3 (Bfloat16 防爆机制)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[3/5] 正在加载模型 {MODEL_ID} 到 {device.upper()} (需下载 ~16GB，请耐心等待)...")

# Llama-3 特殊配置
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# 使用 bfloat16 加载，极大减少显存占用
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("  ✓ 模型加载完成！")

# ==========================================
# 4. 断点续传检查
# ==========================================
print(f"\n[4/5] 正在检查进度...")
processed_ids = set()
if OUTPUT_CSV.exists():
    df_out = pd.read_csv(OUTPUT_CSV)
    if 'arxiv_id' in df_out.columns:
        processed_ids = set(df_out['arxiv_id'].astype(str))
        print(f"  ✓ 发现已有进度，已跳过 {len(processed_ids):,} 条记录。")

df_todo = df_input[~df_input['arxiv_id'].astype(str).isin(processed_ids)].copy()
print(f"  ▶ 本次需要计算 PPL: {len(df_todo):,} 条")

if len(df_todo) == 0:
    print("\n🎉 Llama-3 批次计算完毕！")
    import sys
    sys.exit()

# ==========================================
# 5. Llama-3 PPL 计算核心逻辑
# ==========================================
print(f"\n[5/5] 开始计算 Perplexity (Batch Size: {BATCH_SIZE})...")

def calculate_ppl_batch(texts):
    encodings = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH
    ).to(device)

    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        masked_loss = loss * shift_mask
        seq_lengths = torch.clamp(shift_mask.sum(dim=1), min=1.0)
        seq_loss = masked_loss.sum(dim=1) / seq_lengths
        ppl = torch.exp(seq_loss)

    # 释放缓存防 OOM
    torch.cuda.empty_cache()
    return ppl.cpu().tolist()

results = []
abstracts = df_todo['arxiv_abstract'].tolist()
arxiv_ids = df_todo['arxiv_id'].tolist()

if not OUTPUT_CSV.exists():
    pd.DataFrame(columns=list(df_todo.columns) + ['perplexity']).to_csv(OUTPUT_CSV, index=False)

for i in tqdm(range(0, len(abstracts), BATCH_SIZE), desc="Llama-3 PPL"):
    batch_texts = abstracts[i : i+BATCH_SIZE]
    batch_texts = [t if len(t.strip()) > 10 else "No abstract available." for t in batch_texts]

    batch_ppls = calculate_ppl_batch(batch_texts)

    for j in range(len(batch_ppls)):
        row_idx = df_todo.index[i + j]
        row_data = df_todo.loc[row_idx].to_dict()
        row_data['perplexity'] = round(batch_ppls[j], 4)
        results.append(row_data)

    if len(results) >= SAVE_INTERVAL:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
        results = []

if len(results) > 0:
    pd.DataFrame(results).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

print(f"\n✓ Llama-3 计算完成！结果已保存至: {OUTPUT_CSV.name}")
print("=" * 70)

# %%
# ====== 步骤 1：从原始 JSONL 提取需要的摘要 ======
import json
import pandas as pd
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
ARXIV_RAW = '/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json'
INPUT_CSV = DRIVE_PATH / 'candidates/batch_gpt2_2019_2023.csv'

df = pd.read_csv(INPUT_CSV)
needed_ids = set(df['arxiv_id'].astype(str))
print(f"需要 {len(needed_ids):,} 篇摘要")

id_to_abstract = {}
with open(ARXIV_RAW, 'r') as f:
    for i, line in enumerate(f):
        if (i + 1) % 500000 == 0:
            print(f"  已扫描 {i+1:,} 行，已找到 {len(id_to_abstract):,} 篇")
        paper = json.loads(line)
        pid = paper.get('id', '')
        if pid in needed_ids:
            abstract = paper.get('abstract', '').strip()
            if abstract:
                id_to_abstract[pid] = abstract
        if len(id_to_abstract) == len(needed_ids):
            print(f"  ✅ 全部找到！提前退出")
            break

found = len(id_to_abstract)
missing = len(needed_ids) - found
print(f"\n结果: 找到 {found:,} / {len(needed_ids):,}，缺失 {missing}")

# 写入 df
df['arxiv_abstract'] = df['arxiv_id'].astype(str).map(id_to_abstract)
empty_count = df['arxiv_abstract'].isna().sum()
print(f"合并后空摘要: {empty_count} 条")

# 验证
print(f"\n抽样检查:")
sample = df[df['arxiv_abstract'].notna()].head(3)
for _, row in sample.iterrows():
    print(f"  {row['arxiv_id']}: {row['arxiv_abstract'][:80]}...")

# %%
# ====== Llama-3 完整版：提取摘要 + 计算 PPL ======
import json
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

HF_TOKEN = "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"
login(token=HF_TOKEN)

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
ARXIV_RAW = '/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json'
INPUT_CSV = DRIVE_PATH / 'candidates/batch_llama3_2024_plus.csv'
OUTPUT_CSV = DRIVE_PATH / 'candidates/scored_llama3_2024_plus_v2.csv'

# 1. 提取摘要
df_llama = pd.read_csv(INPUT_CSV)
needed_ids = set(df_llama['arxiv_id'].astype(str))
print(f"需要 {len(needed_ids):,} 篇摘要")

id_to_abstract = {}
with open(ARXIV_RAW, 'r') as f:
    for i, line in enumerate(f):
        if (i + 1) % 500000 == 0:
            print(f"  已扫描 {i+1:,} 行，已找到 {len(id_to_abstract):,} 篇")
        paper = json.loads(line)
        pid = paper.get('id', '')
        if pid in needed_ids:
            abstract = paper.get('abstract', '').strip()
            if abstract:
                id_to_abstract[pid] = abstract
        if len(id_to_abstract) == len(needed_ids):
            print("  ✅ 全部找到！")
            break

df_llama['arxiv_abstract'] = df_llama['arxiv_id'].astype(str).map(id_to_abstract)
print(f"空摘要: {df_llama['arxiv_abstract'].isna().sum()} 条")

# 2. 加载 Llama-3
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
BATCH_SIZE = 16
MAX_LENGTH = 512

print(f"\n加载 {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print("✓ Llama-3 就绪")

# 3. 计算 PPL
def calculate_ppl_batch(texts):
    encodings = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH
    ).to(device)
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        masked_loss = loss * shift_mask
        seq_lengths = torch.clamp(shift_mask.sum(dim=1), min=1.0)
        seq_loss = masked_loss.sum(dim=1) / seq_lengths
        ppl = torch.exp(seq_loss)
    return ppl.cpu().tolist()

unique_llama = df_llama[['arxiv_id', 'arxiv_abstract']].drop_duplicates(subset='arxiv_id')
print(f"独立论文: {len(unique_llama):,} 篇")

abstracts = unique_llama['arxiv_abstract'].tolist()
arxiv_ids = unique_llama['arxiv_id'].tolist()
ppl_map = {}

for i in tqdm(range(0, len(abstracts), BATCH_SIZE), desc="Llama-3 PPL"):
    batch = abstracts[i:i+BATCH_SIZE]
    batch_ids = arxiv_ids[i:i+BATCH_SIZE]
    ppls = calculate_ppl_batch(batch)
    for aid, p in zip(batch_ids, ppls):
        ppl_map[aid] = round(p, 4)

df_llama['perplexity'] = df_llama['arxiv_id'].map(ppl_map)

print(f"\n✓ 完成！")
print(f"唯一 PPL 值: {df_llama['perplexity'].nunique():,}")
print(df_llama['perplexity'].describe())

df_llama.to_csv(OUTPUT_CSV, index=False)
print(f"✓ 保存至: {OUTPUT_CSV}")

# %%
# ====== 量尺 3：时序收敛分析 ======
import numpy as np
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')

# 1. 加载 arXiv embeddings + metadata
print("加载 arXiv embeddings...")
arxiv_data = np.load(DRIVE_PATH / 'arxiv_embeddings.npz')
arxiv_embeddings = arxiv_data['embeddings']
print(f"  Shape: {arxiv_embeddings.shape}")

print("加载 arXiv metadata...")
with open(DRIVE_PATH / 'arxiv_metadata_simple.json', 'r') as f:
    arxiv_meta = json.load(f)
print(f"  论文数: {len(arxiv_meta):,}")

# 2. 提取年份和主分类
records = []
for i, m in enumerate(arxiv_meta):
    year = int(m['update_date'].split('-')[0]) if m.get('update_date') else None
    cats = m.get('categories', '')
    primary_cat = cats.split()[0] if cats else None  # 取第一个分类
    # 取大类（如 cs.AI → cs）
    domain = primary_cat.split('.')[0] if primary_cat and '.' in primary_cat else primary_cat
    if year and domain and 2010 <= year <= 2025:
        records.append({'idx': i, 'year': year, 'domain': domain, 'category': primary_cat})

df = pd.DataFrame(records)
print(f"\n有效记录: {len(df):,}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"大类数量: {df['domain'].nunique()} ({', '.join(sorted(df['domain'].unique()))})")

# 3. 计算每个 (year, domain) 的平均 embedding
print("\n计算各 (年份, 大类) 的平均 embedding...")
group_embeddings = {}
for (year, domain), group in df.groupby(['year', 'domain']):
    indices = group['idx'].values
    mean_emb = arxiv_embeddings[indices].mean(axis=0)
    # L2 归一化
    mean_emb = mean_emb / np.linalg.norm(mean_emb)
    group_embeddings[(year, domain)] = mean_emb

print(f"  计算了 {len(group_embeddings)} 个组")

# 4. 计算所有学科对之间的距离随时间变化
print("\n计算学科对的时序距离...")
domains = sorted(df['domain'].unique())
years = sorted(df['year'].unique())

convergence_results = []
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

        if len(distances) >= 5:  # 至少 5 个数据点才算趋势
            dist_df = pd.DataFrame(distances)
            # 线性回归算斜率
            x = dist_df['year'].values
            y_vals = dist_df['distance'].values
            slope = np.polyfit(x, y_vals, 1)[0]

            convergence_results.append({
                'domain_1': d1,
                'domain_2': d2,
                'slope': round(slope, 6),
                'trend': '收敛' if slope < -0.001 else ('发散' if slope > 0.001 else '稳定'),
                'mean_distance': round(y_vals.mean(), 4),
                'n_years': len(distances),
            })

conv_df = pd.DataFrame(convergence_results)
print(f"\n学科对总数: {len(conv_df)}")
print(f"收敛: {(conv_df['trend']=='收敛').sum()}")
print(f"发散: {(conv_df['trend']=='发散').sum()}")
print(f"稳定: {(conv_df['trend']=='稳定').sum()}")

# 5. Top 收敛/发散对
print("\n=== Top 10 最快收敛 ===")
print(conv_df.nsmallest(10, 'slope')[['domain_1','domain_2','slope','mean_distance','trend']].to_string())

print("\n=== Top 10 最快发散 ===")
print(conv_df.nlargest(10, 'slope')[['domain_1','domain_2','slope','mean_distance','trend']].to_string())

# 6. 保存
conv_df.to_csv(DRIVE_PATH / 'candidates/convergence_analysis.csv', index=False)
print(f"\n✓ 保存至: convergence_analysis.csv")

# %%
# 过滤：只看主流大类
major_domains = ['astro-ph', 'cond-mat', 'cs', 'econ', 'eess', 'hep-ph', 'hep-th',
                 'math', 'math-ph', 'nlin', 'nucl-th', 'physics', 'q-bio', 'q-fin',
                 'quant-ph', 'stat', 'gr-qc']

major_df = conv_df[
    conv_df['domain_1'].isin(major_domains) &
    conv_df['domain_2'].isin(major_domains)
]

print(f"主流学科对: {len(major_df)}")
print(f"收敛: {(major_df['trend']=='收敛').sum()}")
print(f"发散: {(major_df['trend']=='发散').sum()}")
print(f"稳定: {(major_df['trend']=='稳定').sum()}")

print("\n=== Top 15 最快收敛（主流学科）===")
print(major_df.nsmallest(15, 'slope')[['domain_1','domain_2','slope','mean_distance','trend']].to_string())

print("\n=== Top 15 最快发散（主流学科）===")
print(major_df.nlargest(15, 'slope')[['domain_1','domain_2','slope','mean_distance','trend']].to_string())

# 保存主流学科版本
major_df.to_csv(DRIVE_PATH / 'candidates/convergence_major_domains.csv', index=False)
print(f"\n✓ 保存至: convergence_major_domains.csv")

# %%
import time
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
BATCH_SIZE = 16

# 读取之前的结果（拿 arxiv_id 和 abstract 的映射）
df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')

# 去重，只给独立论文打分
unique_df = df[['arxiv_id', 'arxiv_abstract']].drop_duplicates(subset='arxiv_id')
print(f"独立论文: {len(unique_df):,}")

# 打分函数：dot product，不 round
def score_batch_dot(texts):
    safe_texts = [str(t) if len(str(t).strip()) > 10 else "No abstract available." for t in texts]
    inputs = tokenizer(safe_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[best_layer + 1]
        mean_act = hidden.mean(dim=1).float()
        # dot product 而不是 cosine similarity
        scores = torch.matmul(mean_act, sv_gpu)
    return scores.cpu().tolist()

# 跑全量
sv_scores = {}
batch_texts, batch_ids = [], []
start = time.time()

for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="Steering (dot product)"):
    uid = str(row['arxiv_id'])
    abstract = str(row['arxiv_abstract'])
    if not abstract or len(abstract.strip()) < 10:
        sv_scores[uid] = 0.0
        continue
    batch_texts.append(abstract[:500])
    batch_ids.append(uid)

    if len(batch_texts) == BATCH_SIZE:
        scores = score_batch_dot(batch_texts)
        for bid, s in zip(batch_ids, scores):
            sv_scores[bid] = s  # 不 round！
        batch_texts, batch_ids = [], []
        torch.cuda.empty_cache()

if batch_texts:
    scores = score_batch_dot(batch_texts)
    for bid, s in zip(batch_ids, scores):
        sv_scores[bid] = s

elapsed = time.time() - start
print(f"\n✓ 完成! 耗时 {elapsed/60:.1f} 分钟")

# 替换原来的 steering_score
df['steering_score'] = df['arxiv_id'].astype(str).map(sv_scores)

print(f"唯一分数: {df['steering_score'].nunique():,}")
print(df['steering_score'].describe())

# 保存
df.to_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv', index=False)
print("✓ 已覆盖保存")

# %%
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')

# ==========================================
# 回测：三量尺 vs KLab 论文引用数
# ==========================================

# 1. 标准化三个量尺（z-score）
for col in ['similarity', 'perplexity', 'steering_score']:
    df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()

# 综合分数（等权重）
# similarity 越高越好，perplexity 越高=越意外=越好，steering 越高越好
df['composite_score'] = (df['similarity_z'] + df['perplexity_z'] + df['steering_score_z']) / 3

# 2. 把 KLab 论文分成高引/低引两组
klab_stats = df.groupby('klab_idx').agg(
    citations=('klab_citations', 'first'),
    mean_composite=('composite_score', 'mean'),
    mean_similarity=('similarity', 'mean'),
    mean_perplexity=('perplexity', 'mean'),
    mean_steering=('steering_score', 'mean'),
    n_pairs=('arxiv_id', 'count')
).reset_index()

median_cit = klab_stats['citations'].median()
klab_stats['group'] = np.where(klab_stats['citations'] > median_cit, '高引用', '低引用')

print("=" * 50)
print("回测：高引用 vs 低引用 KLab 论文")
print("=" * 50)
print(f"KLab 引用数中位数: {median_cit}")
print(f"高引用组: {(klab_stats['group']=='高引用').sum()} 篇")
print(f"低引用组: {(klab_stats['group']=='低引用').sum()} 篇")

# 3. 对比两组在各量尺上的差异
print(f"\n{'量尺':<20} {'高引用均值':>10} {'低引用均值':>10} {'差异':>10} {'p-value':>10}")
print("-" * 65)

for col in ['mean_composite', 'mean_similarity', 'mean_perplexity', 'mean_steering']:
    high = klab_stats[klab_stats['group']=='高引用'][col]
    low = klab_stats[klab_stats['group']=='低引用'][col]
    t_stat, p_val = stats.ttest_ind(high, low)
    label = col.replace('mean_', '')
    print(f"{label:<20} {high.mean():>10.4f} {low.mean():>10.4f} {high.mean()-low.mean():>+10.4f} {p_val:>10.4f}")

# 4. 相关性：KLab 引用数 vs 各量尺
print(f"\n{'量尺':<20} {'Spearman r':>10} {'p-value':>10}")
print("-" * 45)

for col in ['mean_composite', 'mean_similarity', 'mean_perplexity', 'mean_steering']:
    r, p = stats.spearmanr(klab_stats['citations'], klab_stats[col])
    label = col.replace('mean_', '')
    print(f"{label:<20} {r:>10.4f} {p:>10.4f}")

print(f"\n综合分数分布:")
print(df['composite_score'].describe())

# %%
# ==========================================
# 时间回测：2019-2022 的分数能否预测 2023-2026 的成功
# ==========================================

# 1. 只看 2019-2022 年的 arXiv 论文作为"历史评分"
df_early = df[df['arxiv_year'] <= 2022].copy()
# 2. 只看 2023-2026 年的 arXiv 论文作为"未来验证"
df_late = df[df['arxiv_year'] >= 2023].copy()

print(f"历史期 (2019-2022): {len(df_early):,} 行")
print(f"验证期 (2023-2026): {len(df_late):,} 行")

# 3. 每个 KLab 论文在历史期的平均分数
early_scores = df_early.groupby('klab_idx').agg(
    citations=('klab_citations', 'first'),
    klab_title=('klab_title', 'first'),
    early_similarity=('similarity', 'mean'),
    early_perplexity=('perplexity', 'mean'),
    early_steering=('steering_score', 'mean'),
    n_early=('arxiv_id', 'count')
).reset_index()

# 4. 同一个 KLab 论文在验证期有没有配对到更多/更相似的论文
late_scores = df_late.groupby('klab_idx').agg(
    late_similarity=('similarity', 'mean'),
    late_perplexity=('perplexity', 'mean'),
    late_steering=('steering_score', 'mean'),
    n_late=('arxiv_id', 'count')
).reset_index()

# 5. 合并
backtest = early_scores.merge(late_scores, on='klab_idx', how='inner')
print(f"\n两期都有数据的 KLab 论文: {len(backtest)} 篇")

# 6. 核心问题：历史期 perplexity 高的方向，验证期是否出现了更多论文？
# n_late = 验证期配对论文数量，可以理解为"这个方向后来有多活跃"
from scipy import stats

print(f"\n历史期分数 vs 验证期活跃度 (n_late):")
print(f"{'历史指标':<25} {'Spearman r':>10} {'p-value':>10}")
print("-" * 50)

for col in ['early_similarity', 'early_perplexity', 'early_steering']:
    r, p = stats.spearmanr(backtest[col], backtest['n_late'])
    label = col.replace('early_', '')
    print(f"{label:<25} {r:>10.4f} {p:>10.4f}")

# 7. 历史期分数 vs 验证期距离变化（方向是否在靠近）
backtest['similarity_change'] = backtest['late_similarity'] - backtest['early_similarity']

print(f"\n历史期分数 vs 验证期距离变化 (similarity_change):")
print(f"{'历史指标':<25} {'Spearman r':>10} {'p-value':>10}")
print("-" * 50)

for col in ['early_similarity', 'early_perplexity', 'early_steering']:
    r, p = stats.spearmanr(backtest[col], backtest['similarity_change'])
    label = col.replace('early_', '')
    print(f"{label:<25} {r:>10.4f} {p:>10.4f}")

# 8. 展示最有预测力的案例
backtest['early_composite'] = (
    (backtest['early_perplexity'] - backtest['early_perplexity'].mean()) / backtest['early_perplexity'].std()
)
top5 = backtest.nlargest(5, 'early_composite')[['klab_title', 'citations', 'early_perplexity', 'n_early', 'n_late', 'similarity_change']]
print(f"\n历史期 Perplexity 最高的 5 个 KLab 方向（后来发生了什么）:")
print(top5.to_string(index=False))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sklearn.manifold import TSNE
from pathlib import Path
import pickle

matplotlib.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')

# ==========================================
# 加载所有数据
# ==========================================
df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')
df_conv = pd.read_csv(DRIVE_PATH / 'candidates/convergence_major_domains.csv')

klab_emb = np.load(DRIVE_PATH / 'klab_embeddings.npz')
with open(DRIVE_PATH / 'klab_metadata.pkl', 'rb') as f:
    klab_meta = pickle.load(f)

print("数据加载完成")
print(f"主表: {df.shape}")
print(f"KLab embeddings keys: {list(klab_emb.keys())}")
print(f"KLab metadata type: {type(klab_meta)}")
if isinstance(klab_meta, list):
    print(f"KLab metadata 数量: {len(klab_meta)}")
    print(f"第一条 keys: {list(klab_meta[0].keys()) if isinstance(klab_meta[0], dict) else 'not dict'}")
elif isinstance(klab_meta, dict):
    print(f"KLab metadata keys: {list(klab_meta.keys())[:10]}")

# %%
from google.colab import files
uploaded = files.upload()

# %%
"""
KLab 研究机会四量尺可视化 - 完整 8 图
在 Colab 中运行，确保已挂载 Google Drive
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.manifold import TSNE
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 配置
# ==========================================
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

# ==========================================
# 加载数据
# ==========================================
print("加载数据...")
df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')
df_conv = pd.read_csv(DRIVE_PATH / 'candidates/convergence_major_domains.csv')

klab_emb_data = np.load(DRIVE_PATH / 'klab_embeddings.npz')
klab_embeddings = klab_emb_data['embeddings']
klab_ids = klab_emb_data['paper_ids']

with open(DRIVE_PATH / 'klab_metadata.pkl', 'rb') as f:
    klab_meta = pickle.load(f)

# 尝试加载 klab_papers.json（有完整 concepts）
try:
    with open('/content/klab_papers.json', 'r') as f:
        klab_full = json.load(f)
except:
    try:
        with open(DRIVE_PATH / 'klab_papers.json', 'r') as f:
            klab_full = json.load(f)
    except:
        klab_full = None
        print("⚠️ klab_papers.json 未找到，UMAP 将不按领域着色")

print(f"主表: {df.shape}, 收敛表: {df_conv.shape}, KLab embeddings: {klab_embeddings.shape}")


# ==========================================
# 图 1: KLab 研究主题 UMAP/t-SNE 散点图
# ==========================================
print("\n[1/8] KLab 研究空间散点图...")

# 用 t-SNE 降维（比 UMAP 更容易安装）
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(klab_embeddings)

# 获取每篇论文的主领域和引用数
domains = []
citations = []
for i, meta in enumerate(klab_meta):
    cit = meta.get('cited_by_count', 0)
    citations.append(cit if cit else 0)

    # 从 klab_full 获取 level=0 的 concept 作为领域
    domain = 'Other'
    if klab_full:
        # 通过 openalex_id 匹配
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

# 只保留前 8 大领域，其余归 Other
from collections import Counter
domain_counts = Counter(domains)
top_domains = [d for d, _ in domain_counts.most_common(8)]
domains_clean = [d if d in top_domains else 'Other' for d in domains]

# 颜色映射
unique_domains = sorted(set(domains_clean))
colors_palette = plt.cm.Set2(np.linspace(0, 1, len(unique_domains)))
domain_to_color = {d: colors_palette[i] for i, d in enumerate(unique_domains)}

fig, ax = plt.subplots(figsize=(12, 9))

for domain in unique_domains:
    mask = np.array([d == domain for d in domains_clean])
    sizes = np.clip(citations[mask] / 5 + 15, 15, 300)
    ax.scatter(coords[mask, 0], coords[mask, 1],
               s=sizes, c=[domain_to_color[domain]],
               label=domain, alpha=0.7, edgecolors='white', linewidth=0.5)

ax.set_title('KLab Research Landscape\n(size = citation count, color = primary domain)', fontsize=15)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
ax.set_xticks([])
ax.set_yticks([])

# 标注几篇高引论文
top_indices = np.argsort(citations)[-5:]
for idx in top_indices:
    if citations[idx] > 50:
        title = klab_meta[idx].get('title', '')[:40] + '...'
        ax.annotate(title, (coords[idx, 0], coords[idx, 1]),
                    fontsize=6, alpha=0.8,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_klab_landscape.png')
plt.show()
print("✓ 图1 已保存")


# ==========================================
# 图 2: Perplexity 分数分布图（高引 vs 普通）
# ==========================================
print("\n[2/8] Perplexity 分布图...")

# 按 KLab 引用数分组
klab_info = df.groupby('klab_idx').agg(
    citations=('klab_citations', 'first')
).reset_index()

median_cit = klab_info['citations'].median()

df_merged = df.merge(klab_info[['klab_idx']], on='klab_idx')
high_cit_klab = set(klab_info[klab_info['citations'] > median_cit]['klab_idx'])

ppl_high = df[df['klab_idx'].isin(high_cit_klab)]['perplexity']
ppl_low = df[~df['klab_idx'].isin(high_cit_klab)]['perplexity']

fig, ax = plt.subplots(figsize=(10, 6))

# 截断极端值方便可视化
clip_max = np.percentile(df['perplexity'], 98)

ax.hist(ppl_high.clip(upper=clip_max), bins=80, alpha=0.6, color='#e74c3c',
        label=f'High-citation KLab (>{median_cit:.0f} cites)', density=True)
ax.hist(ppl_low.clip(upper=clip_max), bins=80, alpha=0.6, color='#3498db',
        label=f'Low-citation KLab (≤{median_cit:.0f} cites)', density=True)

ax.axvline(ppl_high.median(), color='#e74c3c', linestyle='--', lw=1.5, label=f'High median: {ppl_high.median():.1f}')
ax.axvline(ppl_low.median(), color='#3498db', linestyle='--', lw=1.5, label=f'Low median: {ppl_low.median():.1f}')

t_stat, p_val = stats.ttest_ind(ppl_high, ppl_low)
ax.text(0.97, 0.95, f't-test p = {p_val:.4f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Perplexity Score')
ax.set_ylabel('Density')
ax.set_title('Perplexity Distribution: High-Citation vs Low-Citation KLab Papers')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_perplexity_distribution.png')
plt.show()
print("✓ 图2 已保存")


# ==========================================
# 图 3: 时序轨迹图（5 对主题的距离变化）
# ==========================================
print("\n[3/8] 时序轨迹图...")

# 从主表按年份和主领域计算距离
df['arxiv_domain'] = df['arxiv_categories'].apply(
    lambda x: str(x).split()[0].split('.')[0] if pd.notna(x) else 'unknown'
)

# 选 5 对有趣的领域组合（收敛最快 + 发散最快 + 稳定）
top_converging = df_conv[df_conv['trend'] == '收敛'].nsmallest(3, 'slope')
top_diverging = df_conv[df_conv['trend'] == '发散'].nlargest(2, 'slope')
selected_pairs = pd.concat([top_converging, top_diverging])

fig, ax = plt.subplots(figsize=(12, 7))

colors_line = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
markers = ['o', 's', '^', 'D', 'v']

for i, (_, row) in enumerate(selected_pairs.iterrows()):
    d1, d2 = row['domain_1'], row['domain_2']

    # 从主表计算这对领域每年的平均距离
    mask = df['arxiv_domain'].isin([d1, d2])
    yearly = df[mask].groupby('arxiv_year')['similarity'].mean()

    if len(yearly) >= 3:
        label = f"{d1} × {d2} (slope={row['slope']:.4f})"
        ax.plot(yearly.index, yearly.values,
                color=colors_line[i], marker=markers[i],
                linewidth=2, markersize=6, label=label, alpha=0.85)

ax.set_xlabel('Year')
ax.set_ylabel('Mean Cosine Similarity')
ax.set_title('Temporal Trajectory: Domain Pair Distance Over Time\n(declining = converging, rising = diverging)')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_temporal_trajectory.png')
plt.show()
print("✓ 图3 已保存")


# ==========================================
# 图 4: 收敛速度排行榜 Top 15
# ==========================================
print("\n[4/8] 收敛速度排行榜...")

top15_conv = df_conv.nsmallest(15, 'slope').copy()
top15_conv['pair_label'] = top15_conv['domain_1'] + ' × ' + top15_conv['domain_2']

fig, ax = plt.subplots(figsize=(10, 8))

colors_bar = ['#e74c3c' if s < -0.01 else '#e67e22' if s < -0.005 else '#f1c40f'
              for s in top15_conv['slope']]

bars = ax.barh(range(len(top15_conv)), top15_conv['slope'].values, color=colors_bar, alpha=0.85)
ax.set_yticks(range(len(top15_conv)))
ax.set_yticklabels(top15_conv['pair_label'].values, fontsize=9)
ax.set_xlabel('Convergence Slope (more negative = faster convergence)')
ax.set_title('Top 15 Fastest Converging Domain Pairs')
ax.axvline(0, color='black', linewidth=0.5)
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

# 标注数值
for i, (bar, val) in enumerate(zip(bars, top15_conv['slope'].values)):
    ax.text(val - 0.001, i, f'{val:.4f}', va='center', ha='right', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_convergence_ranking.png')
plt.show()
print("✓ 图4 已保存")


# ==========================================
# 图 5: 甜蜜点地图（Perplexity × Similarity × Steering 三维）
# ==========================================
print("\n[5/8] 甜蜜点地图...")

# 每篇 arXiv 取一行（去重）
df_unique = df.drop_duplicates(subset='arxiv_id').copy()

# 采样（太多点画不了）
sample = df_unique.sample(n=min(5000, len(df_unique)), random_state=42)

fig, ax = plt.subplots(figsize=(12, 9))

# x = similarity (反转: 低相似度 = 更新颖), y = perplexity, color = steering
scatter = ax.scatter(
    1 - sample['similarity'],  # 反转：novelty = 1 - similarity
    sample['perplexity'].clip(upper=np.percentile(sample['perplexity'], 98)),
    c=sample['steering_score'],
    cmap='RdYlGn',
    s=12, alpha=0.5,
    edgecolors='none'
)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Steering Score (interdisciplinarity)', fontsize=10)

ax.set_xlabel('Novelty (1 - cosine similarity)')
ax.set_ylabel('Perplexity (language surprise)')
ax.set_title('"Sweet Spot" Map: Research Opportunity Landscape\n(top-right + green = highest opportunity)')

# 标注甜蜜区域
ax.axhline(y=np.percentile(sample['perplexity'], 75), color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=np.percentile(1 - sample['similarity'], 75), color='gray', linestyle=':', alpha=0.5)
ax.text(0.97, 0.97, 'HIGH\nOPPORTUNITY', transform=ax.transAxes,
        ha='right', va='top', fontsize=12, color='darkgreen', alpha=0.4, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_sweet_spot_map.png')
plt.show()
print("✓ 图5 已保存")


# ==========================================
# 图 6: Token 激活热力图（占位版 - 真正版需要 GPU）
# ==========================================
print("\n[6/8] Token 激活分析...")

# 这里用 steering score 按词频分析作为替代
# 真正的 token-level 激活需要 GPU 上的模型
# 我们用高 steering score 论文 vs 低 steering score 论文的关键词差异

from collections import Counter
import re

def get_words(text):
    return re.findall(r'[a-z]{3,}', str(text).lower())

# 高 steering 论文的词频
high_steer = df_unique.nlargest(500, 'steering_score')['arxiv_abstract']
low_steer = df_unique.nsmallest(500, 'steering_score')['arxiv_abstract']

high_words = Counter()
for text in high_steer:
    high_words.update(get_words(text))

low_words = Counter()
for text in low_steer:
    low_words.update(get_words(text))

# 停用词
stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'were', 'have', 'has',
             'from', 'been', 'which', 'their', 'our', 'can', 'also', 'these', 'than', 'its',
             'not', 'but', 'more', 'between', 'such', 'other', 'using', 'used', 'based',
             'may', 'however', 'both', 'each', 'about', 'into', 'does', 'will', 'how',
             'show', 'study', 'results', 'paper', 'method', 'data', 'model', 'approach'}

# 计算差异分数：在高 steering 中高频但在低 steering 中低频的词
diff_scores = {}
for word in set(list(high_words.keys())[:2000]):
    if word in stopwords:
        continue
    h = high_words.get(word, 0) / len(high_steer)
    l = low_words.get(word, 0) / len(low_steer)
    if h + l > 0.01:  # 至少出现一定次数
        diff_scores[word] = h - l

# Top 正向（跨学科激活词）和负向（单领域词）
sorted_words = sorted(diff_scores.items(), key=lambda x: x[1], reverse=True)
top_positive = sorted_words[:20]
top_negative = sorted_words[-20:]
all_words = top_negative + top_positive

fig, ax = plt.subplots(figsize=(12, 8))

words = [w[0] for w in all_words]
scores = [w[1] for w in all_words]
colors_hm = ['#e74c3c' if s < 0 else '#2ecc71' for s in scores]

bars = ax.barh(range(len(words)), scores, color=colors_hm, alpha=0.8)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Frequency Difference (high steering − low steering)')
ax.set_title('Words Associated with "Interdisciplinarity" Direction\n(green = activates interdisciplinary, red = activates single-domain)')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_token_activation.png')
plt.show()
print("✓ 图6 已保存")


# ==========================================
# 图 7: 四量尺相关矩阵
# ==========================================
print("\n[7/8] 四量尺相关矩阵...")

# 每篇 arXiv 论文一行
df_for_corr = df_unique[['similarity', 'perplexity', 'steering_score']].copy()
df_for_corr.columns = ['Embedding\nSimilarity', 'Perplexity', 'Steering\nScore']

# Spearman 相关
corr_matrix = df_for_corr.corr(method='spearman')

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spearman Correlation', fontsize=10)

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, fontsize=11)
ax.set_yticklabels(corr_matrix.columns, fontsize=11)

# 标注数值
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=14,
                fontweight='bold', color=color)

ax.set_title('Correlation Matrix: Three Measurement Scales\n(Spearman rank correlation)', fontsize=13)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig7_correlation_matrix.png')
plt.show()
print("✓ 图7 已保存")


# ==========================================
# 图 8: 回测结果 + 最终候选排行榜
# ==========================================
print("\n[8/8] 回测 + 候选排行榜...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- 左图: 回测结果 ---
ax1 = axes[0]

# 时间回测数据
df_early = df[df['arxiv_year'] <= 2022]
df_late = df[df['arxiv_year'] >= 2023]

early_scores = df_early.groupby('klab_idx').agg(
    early_steering=('steering_score', 'mean'),
    early_similarity=('similarity', 'mean'),
    early_perplexity=('perplexity', 'mean'),
).reset_index()

late_scores = df_late.groupby('klab_idx').agg(
    n_late=('arxiv_id', 'count'),
).reset_index()

backtest = early_scores.merge(late_scores, on='klab_idx')

metrics = ['early_steering', 'early_similarity', 'early_perplexity']
labels = ['Steering\nScore', 'Embedding\nSimilarity', 'Perplexity']
r_values = []

for m in metrics:
    r, p = stats.spearmanr(backtest[m], backtest['n_late'])
    r_values.append(r)

colors_bt = ['#2ecc71' if r > 0.3 else '#f1c40f' if r > 0.1 else '#e74c3c' for r in r_values]
bars = ax1.bar(labels, r_values, color=colors_bt, alpha=0.85, width=0.6)

for bar, r in zip(bars, r_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'r = {r:.3f}', ha='center', fontsize=11, fontweight='bold')

ax1.set_ylabel('Spearman Correlation with Future Activity')
ax1.set_title('Backtest: Which Scale Best Predicts\nFuture Research Activity? (2019-22 → 2023-26)')
ax1.set_ylim(0, 0.7)
ax1.grid(True, axis='y', alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

# --- 右图: Top 15 候选排行榜 ---
ax2 = axes[1]

# 综合分数（加权：steering 权重最高因为预测力最强）
for col in ['similarity', 'perplexity', 'steering_score']:
    df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()

df['composite'] = 0.5 * df['steering_score_z'] + 0.3 * df['perplexity_z'] + 0.2 * df['similarity_z']

# 每个 KLab-arXiv 配对取最高分
top15 = df.nlargest(15, 'composite')[['klab_title', 'arxiv_title', 'composite']].copy()
top15['label'] = top15.apply(
    lambda r: r['klab_title'][:25] + '...\n× ' + r['arxiv_title'][:25] + '...', axis=1
)

colors_rank = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 15))[::-1]
bars2 = ax2.barh(range(14, -1, -1), top15['composite'].values, color=colors_rank, alpha=0.85)
ax2.set_yticks(range(14, -1, -1))
ax2.set_yticklabels(top15['label'].values, fontsize=7)
ax2.set_xlabel('Composite Score (weighted)')
ax2.set_title('Top 15 Research Opportunity Candidates')
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig8_backtest_and_ranking.png')
plt.show()
print("✓ 图8 已保存")


# ==========================================
# 完成
# ==========================================
print("\n" + "=" * 50)
print("🎉 全部 8 张图已保存至:")
print(f"   {OUT_DIR}")
print("=" * 50)
for f in sorted(OUT_DIR.glob('fig*.png')):
    print(f"   ✓ {f.name}")

# %%
"""
图 6 完整版：Token-Level Steering Vector 激活热力图
需要 GPU + 模型和 steering vector 还在内存中
"""
!pip install --quiet steering-vectors
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
OUT_DIR = DRIVE_PATH / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. 确认模型和 sv 还在内存，否则重新加载
# ==========================================
try:
    _ = model.config
    print("✓ 模型已在内存")
except:
    print("重新加载模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login

    HF_TOKEN = "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"
    login(token=HF_TOKEN)
    MODEL_ID = "meta-llama/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

try:
    _ = sv_gpu.shape
    print(f"✓ Steering vector 已在内存: {sv_gpu.shape}")
except:
    print("重新加载 steering vector...")
    steering_vector = torch.load(DRIVE_PATH / 'steering_vector_v2.pt', map_location='cuda', weights_only=False)
    best_layer = 15
    sv = steering_vector.layer_activations[best_layer]
    sv = sv / sv.norm()
    sv_gpu = sv.to('cuda').float()

device = "cuda"

# ==========================================
# 2. 选取代表性论文（高/中/低 steering score）
# ==========================================
import pandas as pd

df = pd.read_csv(DRIVE_PATH / 'candidates/scored_all_4metrics.csv')
df_unique = df.drop_duplicates(subset='arxiv_id')

# 选 3 篇高 steering + 3 篇低 steering
high_papers = df_unique.nlargest(3, 'steering_score')[['arxiv_id', 'arxiv_title', 'arxiv_abstract', 'steering_score']]
low_papers = df_unique.nsmallest(3, 'steering_score')[['arxiv_id', 'arxiv_title', 'arxiv_abstract', 'steering_score']]

print(f"\n高 Steering 论文:")
for _, r in high_papers.iterrows():
    print(f"  [{r['steering_score']:.4f}] {r['arxiv_title'][:60]}")

print(f"\n低 Steering 论文:")
for _, r in low_papers.iterrows():
    print(f"  [{r['steering_score']:.4f}] {r['arxiv_title'][:60]}")


# ==========================================
# 3. 逐 token 计算激活投影
# ==========================================
def get_token_activations(text, max_length=128):
    """对每个 token 计算它在 steering vector 方向上的激活值"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # layer 15 的 hidden states（+1 因为 index 0 是 embedding 层）
        hidden = outputs.hidden_states[best_layer + 1][0]  # shape: (seq_len, 4096)

        # 每个 token 在 steering vector 方向上的投影（dot product）
        activations = torch.matmul(hidden.float(), sv_gpu).cpu().numpy()

    # 清理 token 显示（去掉 Llama tokenizer 的特殊字符）
    clean_tokens = []
    for t in tokens:
        t = t.replace('Ġ', ' ').replace('▁', ' ').replace('<s>', '[BOS]')
        clean_tokens.append(t)

    return clean_tokens, activations

print("\n正在计算 token 激活...")

all_results = []

for label, papers in [('HIGH', high_papers), ('LOW', low_papers)]:
    for _, row in papers.iterrows():
        abstract = str(row['arxiv_abstract'])[:500]
        tokens, activations = get_token_activations(abstract)
        all_results.append({
            'label': label,
            'title': row['arxiv_title'][:50],
            'score': row['steering_score'],
            'tokens': tokens,
            'activations': activations
        })
        print(f"  ✓ {label} | {row['arxiv_title'][:40]}... | {len(tokens)} tokens")


# ==========================================
# 4. 画热力图：每篇论文一行
# ==========================================
print("\n画热力图...")

fig, axes = plt.subplots(6, 1, figsize=(20, 18))

# 统一颜色范围
all_acts = np.concatenate([r['activations'] for r in all_results])
vmin, vmax = np.percentile(all_acts, 5), np.percentile(all_acts, 95)

for i, result in enumerate(all_results):
    ax = axes[i]
    tokens = result['tokens']
    acts = result['activations']

    # 只显示前 60 个 token（太多显示不下）
    n_show = min(60, len(tokens))
    tokens_show = tokens[:n_show]
    acts_show = acts[:n_show]

    # 画热力条
    im = ax.imshow(acts_show.reshape(1, -1), aspect='auto',
                   cmap='RdYlGn', vmin=vmin, vmax=vmax)

    ax.set_yticks([0])
    prefix = "🟢 HIGH" if result['label'] == 'HIGH' else "🔴 LOW"
    ax.set_yticklabels([f"{prefix} [{result['score']:.3f}]"], fontsize=9)

    ax.set_xticks(range(n_show))
    ax.set_xticklabels(tokens_show, rotation=90, fontsize=6, ha='center')

    ax.set_title(f"{result['title']}...", fontsize=9, loc='left', pad=2)

# 颜色条
cbar = fig.colorbar(im, ax=axes, shrink=0.3, location='right', pad=0.02)
cbar.set_label('Activation along steering vector\n(green = interdisciplinary, red = single-domain)', fontsize=10)

fig.suptitle('Token-Level Activation Along "Interdisciplinarity" Steering Vector\n(Layer 15, Llama-3-8B)',
             fontsize=14, y=1.01)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_token_activation_full.png', bbox_inches='tight')
plt.show()
print("✓ 图6 完整版已保存")


# ==========================================
# 5. 额外：Top 激活词汇统计
# ==========================================
print("\n统计最高激活词汇...")

token_scores = {}
token_counts = {}

# 对高 steering 和低 steering 各取 50 篇做统计
for group, papers in [('high', df_unique.nlargest(50, 'steering_score')),
                       ('low', df_unique.nsmallest(50, 'steering_score'))]:
    for _, row in papers.iterrows():
        abstract = str(row['arxiv_abstract'])[:300]
        tokens, activations = get_token_activations(abstract, max_length=80)

        for t, a in zip(tokens, activations):
            clean = t.replace('Ġ', '').replace('▁', '').strip().lower()
            if len(clean) < 3:
                continue
            if clean not in token_scores:
                token_scores[clean] = []
            token_scores[clean].append(float(a))

print("  ✓ 统计完成")

# 计算每个词的平均激活
avg_scores = {word: np.mean(scores) for word, scores in token_scores.items()
              if len(scores) >= 5}  # 至少出现 5 次

sorted_tokens = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
top_positive = sorted_tokens[:25]
top_negative = sorted_tokens[-25:]
combined = top_negative + top_positive

fig, ax = plt.subplots(figsize=(12, 10))

words = [w[0] for w in combined]
scores = [w[1] for w in combined]
colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in scores]

ax.barh(range(len(words)), scores, color=colors, alpha=0.85)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mean Activation Along Steering Vector')
ax.set_title('Top Tokens by Steering Vector Activation\n(green = interdisciplinary direction, red = single-domain direction)')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6b_token_ranking.png')
plt.show()
print("✓ 图6b 词汇排名已保存")

print("\n" + "=" * 50)
print("🎉 Token 激活分析完成！")
print(f"  fig6_token_activation_full.png - 6 篇论文的逐 token 热力图")
print(f"  fig6b_token_ranking.png - 最高/最低激活词汇排行")
print("=" * 50)

# %%
# ==========================================
# 图 1 修复版: KLab 研究空间散点图
# ==========================================
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import pickle, json
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
OUT_DIR = DRIVE_PATH / 'figures'

klab_emb_data = np.load(DRIVE_PATH / 'klab_embeddings.npz')
klab_embeddings = klab_emb_data['embeddings']

with open(DRIVE_PATH / 'klab_metadata.pkl', 'rb') as f:
    klab_meta = pickle.load(f)

try:
    with open('/content/klab_papers.json', 'r') as f:
        klab_full = json.load(f)
except:
    with open(DRIVE_PATH / 'klab_papers.json', 'r') as f:
        klab_full = json.load(f)

# 清除 HTML 标签
def clean_html(text):
    return re.sub(r'<[^>]+>', '', str(text))

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(klab_embeddings)

# 领域和引用数
domains = []
citations = []
for i, meta in enumerate(klab_meta):
    cit = meta.get('cited_by_count', 0)
    citations.append(cit if cit else 0)
    domain = 'Other'
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

# Top 8 领域
domain_counts = Counter(domains)
top_domains = [d for d, _ in domain_counts.most_common(8)]
domains_clean = [d if d in top_domains else 'Other' for d in domains]
unique_domains = sorted(set(domains_clean))

# 手选高对比度颜色
color_map = {
    'Biology': '#27ae60',
    'Chemistry': '#e67e22',
    'Computer science': '#3498db',
    'Materials science': '#e91e63',
    'Medicine': '#00bcd4',
    'Other': '#bdc3c7',
    'Political science': '#f39c12',
    'Psychology': '#9b59b6',
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

# 只标注 Top 8 高引论文，用 adjustText 防重叠
top_indices = np.argsort(citations)[-8:]
texts = []

try:
    from adjustText import adjust_text
    has_adjust = True
except:
    has_adjust = False

for idx in top_indices:
    if citations[idx] > 30:
        title = clean_html(klab_meta[idx].get('title', ''))
        # 截短标题
        if len(title) > 35:
            title = title[:35] + '…'

        if has_adjust:
            t = ax.annotate(title, (coords[idx, 0], coords[idx, 1]),
                           fontsize=7, alpha=0.9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor='gray', alpha=0.8))
            texts.append(t)
        else:
            # 手动偏移防重叠
            offset_x = 8 if idx % 2 == 0 else -8
            offset_y = 8 if idx % 3 == 0 else -8
            ax.annotate(title, (coords[idx, 0], coords[idx, 1]),
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=7, alpha=0.9,
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor='gray', alpha=0.8))

ax.set_title('KLab Research Landscape\n(size = citation count, color = primary domain)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('t-SNE 1', fontsize=11)
ax.set_ylabel('t-SNE 2', fontsize=11)
ax.set_xticks([])
ax.set_yticks([])

# 图例放右下角，不遮挡数据
ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
          ncol=1, markerscale=0.8, title='Domain', title_fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_klab_landscape.png', dpi=200)
plt.show()
print("✓ 图1 修复版已保存")

# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
ARXIV_RAW = '/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json'

# 1. 建立 paper_id → embedding index 的映射
arxiv_emb = np.load(DRIVE_PATH / 'arxiv_embeddings.npz', allow_pickle=True)
embeddings = arxiv_emb['embeddings']
paper_ids = arxiv_emb['paper_ids']

id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
print(f"Embedding 索引: {len(id_to_idx):,} 篇")

# 2. 从 arXiv JSON 提取每篇论文的 category 和 year
print("从 JSON 提取 category + year（需要几分钟）...")
id_to_info = {}

with open(ARXIV_RAW, 'r') as f:
    for i, line in enumerate(f):
        if (i+1) % 500000 == 0:
            print(f"  扫描 {i+1:,}...")
        paper = json.loads(line)
        pid = paper.get('id', '')
        if pid in id_to_idx:
            cats = paper.get('categories', '')
            # 提取大类
            primary_domain = cats.split()[0].split('.')[0] if cats else ''
            # 提取年份（从 update_date）
            update_date = paper.get('update_date', '')
            year = int(update_date[:4]) if update_date and len(update_date) >= 4 else 0
            if primary_domain and year >= 2019:
                id_to_info[pid] = (primary_domain, year)

print(f"✓ 提取 {len(id_to_info):,} 篇 (2019+)")

# 3. 按 (domain, year) 分组计算平均 embedding
print("\n按 (domain, year) 分组...")
group_embeddings = defaultdict(list)

for pid, (domain, year) in id_to_info.items():
    idx = id_to_idx[pid]
    group_embeddings[(domain, year)].append(idx)

# 计算每组的 centroid
centroids = {}
for (domain, year), indices in group_embeddings.items():
    if len(indices) >= 10:  # 至少 10 篇才算
        centroids[(domain, year)] = embeddings[indices].mean(axis=0)

print(f"✓ {len(centroids)} 个 (domain, year) centroid")

# 4. 计算每对领域每年的余弦距离
from numpy.linalg import norm

domains = sorted(set(d for d, y in centroids.keys()))
years = sorted(set(y for d, y in centroids.keys()))
print(f"领域: {len(domains)} 个, 年份: {years}")

results = []
for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        yearly_distances = {}
        for year in years:
            if (d1, year) in centroids and (d2, year) in centroids:
                v1 = centroids[(d1, year)]
                v2 = centroids[(d2, year)]
                cos_sim = np.dot(v1, v2) / (norm(v1) * norm(v2))
                yearly_distances[year] = round(float(cos_sim), 6)

        if len(yearly_distances) >= 3:  # 至少 3 年数据
            results.append({
                'domain_1': d1,
                'domain_2': d2,
                'yearly_distances': yearly_distances
            })

print(f"✓ {len(results)} 对领域有逐年数据")

# 5. 保存逐年数据
rows = []
for r in results:
    for year, dist in r['yearly_distances'].items():
        rows.append({
            'domain_1': r['domain_1'],
            'domain_2': r['domain_2'],
            'year': year,
            'cosine_similarity': dist
        })

df_yearly = pd.DataFrame(rows)
df_yearly.to_csv(DRIVE_PATH / 'candidates/convergence_yearly.csv', index=False)
print(f"\n✓ 保存 convergence_yearly.csv: {len(df_yearly)} 行")
print(df_yearly.head(10))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/serendipity_agent')
OUT_DIR = DRIVE_PATH / 'figures'

df_yearly = pd.read_csv(DRIVE_PATH / 'candidates/convergence_yearly.csv')

# 计算每对领域的 slope
pair_slopes = []
for (d1, d2), grp in df_yearly.groupby(['domain_1', 'domain_2']):
    if len(grp) >= 4:
        slope, _, _, _, _ = stats.linregress(grp['year'], grp['cosine_similarity'])
        pair_slopes.append({'d1': d1, 'd2': d2, 'slope': slope})

df_slopes = pd.DataFrame(pair_slopes)

# 选 5 对：3 个收敛最快 + 2 个发散最快
top_conv = df_slopes.nlargest(3, 'slope')  # similarity 升高 = 收敛
top_div = df_slopes.nsmallest(2, 'slope')   # similarity 降低 = 发散
selected = pd.concat([top_conv, top_div])

print("选中的 5 对领域:")
for _, r in selected.iterrows():
    direction = "收敛" if r['slope'] > 0 else "发散"
    print(f"  {r['d1']} × {r['d2']}: slope={r['slope']:.5f} ({direction})")

# 画图
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#fafafa')

colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
markers = ['o', 's', '^', 'D', 'v']

for i, (_, row) in enumerate(selected.iterrows()):
    d1, d2 = row['d1'], row['d2']
    data = df_yearly[(df_yearly['domain_1']==d1) & (df_yearly['domain_2']==d2)].sort_values('year')

    direction = "↗ converging" if row['slope'] > 0 else "↘ diverging"
    label = f"{d1} × {d2} ({direction}, slope={row['slope']:.4f})"

    ax.plot(data['year'], data['cosine_similarity'],
            color=colors[i], marker=markers[i],
            linewidth=2.5, markersize=8, label=label, alpha=0.85)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cosine Similarity (higher = more similar)', fontsize=12)
ax.set_title('Temporal Trajectory: Inter-Domain Embedding Distance (2019–2026)\n'
             'Based on arXiv domain centroid embeddings', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(2019, 2027))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_temporal_trajectory.png', dpi=200)
plt.show()
print("✓ 图3 修复版已保存")

# %%



