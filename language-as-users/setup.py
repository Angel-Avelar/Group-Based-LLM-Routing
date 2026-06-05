from sentence_transformers import SentenceTransformer
import numpy as np
import json
from util import get_data

split = get_data()

prompts =  ["\n".join([turn['user']['content'][0]['text'] for turn in x]) for x in split['full_conversation']]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = model.encode(
    prompts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print(f"Embeddings shape: {embeddings.shape}")
np.save("prompt_embeddings_no_ties.npy", embeddings)

models = sorted(set(split["model_a"]) | set(split["model_b"]))
model_to_idx = {model: idx for idx, model in enumerate(models)}
idx_to_model = {idx: model for model, idx in model_to_idx.items()}

df_pairs = split.select_columns(["model_a", "model_b"]).to_pandas()

comparison_pairs = np.array([
    [model_to_idx[a], model_to_idx[b]]
    for a, b in zip(df_pairs["model_a"], df_pairs["model_b"])
])

df_results = split.select_columns(["winner"]).to_pandas()

winner_map = {"model_b": 1, "model_a": 0}
comparison_results = df_results["winner"].map(winner_map).to_numpy()

np.save("comparison_pairs_no_ties.npy", comparison_pairs)
np.save("comparison_results_no_ties.npy", comparison_results)

with open("idx_to_model.json", "w") as f:
    json.dump(idx_to_model, f)

with open("model_to_idx.json", "w") as f:
    json.dump(model_to_idx, f)