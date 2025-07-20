import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Load your JSON metadata
with open("../data/project_gutenberg/gold_pg-1900_meta_full.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract genres
genres = [entry['gold_source']['gutenberg'].get('genre', 'Unknown') for entry in data]

# Count occurrences
genre_counts = Counter(genres)

# Get top 9 + "Other"
top_n = 5
top_genres = genre_counts.most_common(top_n)
other_count = sum(count for genre, count in genre_counts.items() if (genre, count) not in top_genres)

labels, counts = zip(*top_genres)
labels += ('Other',)
counts += (other_count,)

# Convert to numpy array
counts = np.array(counts)

# Radar chart setup
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
counts = np.concatenate((counts, [counts[0]]))  # close the loop
angles += angles[:1]  # close the loop

# Create plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, counts, color='tab:blue', linewidth=2)
ax.fill(angles, counts, color='skyblue', alpha=0.4)

# Fix axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_title('Genre Distribution (Radar Chart)', size=14, pad=20)

# Optional: radial labels and limits
ax.set_rlabel_position(0)
max_count = max(counts)
ax.set_ylim(0, max_count + max_count * 0.1)

plt.tight_layout()
plt.savefig("genre_distribution_spider.png")