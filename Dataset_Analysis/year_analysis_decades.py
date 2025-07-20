import json
import matplotlib.pyplot as plt
from collections import Counter

path = "../data/project_gutenberg/gold_pg-1900_meta_full.json"
with open(path, 'r', encoding='utf-8') as f:
    content = json.load(f)

    # Filter years from 1900 onward
    years = [entry["gold_source"]["year"] for entry in content if entry["gold_source"]["year"] >= 1900]
    ids = [entry["id"] for entry in content if entry["gold_source"]["year"] >= 1900]

    entries_after_1900 = [entry for entry in content if entry['id'] in ids]

    # Save filtered entries back (optional)
    with open("../data/project_gutenberg/final_1900_pg_gold_meta.json", 'w', encoding='utf-8') as f:
        json.dump(entries_after_1900, f, indent=4)

    # Group years into decades
    decades = [(year // 10) * 10 for year in years]
    decade_counts = Counter(decades)

    # Sort decades for ordered labels
    sorted_decades = sorted(decade_counts.keys())
    sorted_counts = [decade_counts[decade] for decade in sorted_decades]

    # Calculate average year
    average_year = sum(years) / len(years)
    print(f"Average Year: {average_year:.2f}")

    # Create labels for decades, e.g. "1900-1909"
    decade_labels = [f"{decade}-{decade+9}" for decade in sorted_decades]

    # Plot pie chart for decades
    plt.figure(figsize=(9, 9), dpi=600)
    wedges, texts = plt.pie(sorted_counts, labels=decade_labels, startangle=140, labeldistance=1.15)

    # Adjust label font size
    for text in texts:
        text.set_horizontalalignment('center')
        text.set_fontsize(8)  # Adjust font size as needed

    plt.title(f"Distribution of publication decades (Ø {average_year:.2f})", fontsize=14)
    plt.savefig("./decade_distribution_1900.pdf", dpi=300, bbox_inches='tight')
    # plt.show()

    # (Optional) Continue with century grouping or other visualizations as before
