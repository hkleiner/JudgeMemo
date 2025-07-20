import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

# Load your JSON file
with open("../data/project_gutenberg/gold_pg-1900_meta_full.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

genres = [entry['gold_source']['gutenberg'].get('genre', 'Unknown') for entry in data]

# Count genre occurrences
genre_counts = Counter(genres)

# Get top 10 genres (top 9 + 'Other')
top_n = 9
top_genres = genre_counts.most_common(top_n)
other_count = sum(count for genre, count in genre_counts.items() if (genre, count) not in top_genres)

# Prepare labels and counts
labels, counts = zip(*top_genres)
labels += ('Other',)
counts += (other_count,)

# Generate unique colors using updated Matplotlib colormap API
num_bars = len(labels)
cmap = plt.colormaps.get_cmap('tab10')  # Modern way
colors = [cmap(i / num_bars) for i in range(num_bars)]  # Normalize index to [0,1]

# Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(labels, counts, color=colors)

# Aesthetics
plt.xlabel('Number of Documents')
plt.title('Genre Distribution in CoFluEval-LC')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()

plt.savefig("genre_distribution_bar.pdf")
