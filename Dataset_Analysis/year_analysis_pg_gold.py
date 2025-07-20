import json
import matplotlib.pyplot as plt
from collections import Counter


path = "../data/project_gutenberg/gold_pg-1900_meta_full.json"
with open(path, 'r', encoding='utf-8') as f:
    content = json.load(f)

    years = [entry["gold_source"]["year"] for entry in content if entry["gold_source"]["year"] >= 1900]
    ids = [entry["id"] for entry in content if entry["gold_source"]["year"] >= 1900]

    entries_after_1900 = list()
    for entry in content:
        if entry['id'] in ids:
            entries_after_1900.append(entry)

    with open("../data/project_gutenberg/final_1900_pg_gold_meta.json", 'w', encoding='utf-8') as f:
        json.dump(entries_after_1900, f, indent=4)

    # Count occurrences of each year
    year_counts = Counter(years)

    # Sort the years for ordered labels
    sorted_years = sorted(year_counts.keys())
    sorted_counts = [year_counts[year] for year in sorted_years]

    # Calculate average year
    average_year = sum(years) / len(years)
    print(f"Average Year: {average_year:.2f}")

    # Create a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sorted_counts, labels=sorted_years, startangle=140)
    plt.title(f"Distribution of years of publication from 1900 (Ø {average_year:.2f})")
    #plt.show()

    plt.figure(figsize=(9, 9), dpi=600)
    wedges, texts = plt.pie(sorted_counts, labels=sorted_years, startangle=140, labeldistance=1.15)

    # Adjust label font size
    for text in texts:
        text.set_horizontalalignment('center')
        text.set_fontsize(4)  # Set label font size

    plt.title(f"Distribution of publication years (Ø {average_year:.2f})", fontsize=14)
    plt.savefig("./year_distribution_1900.png")
    # plt.show()

    # Centuries
    # Group years by century
    centuries = [(year // 100 + 1) * 100 for year in years]
    century_counts = Counter(centuries)

    # Sort centuries
    sorted_centuries = sorted(century_counts.keys())
    sorted_century_counts = [century_counts[century] for century in sorted_centuries]

    # Create a pie chart for centuries
    plt.figure(figsize=(9, 9), dpi=600)
    wedges, texts = plt.pie(
        sorted_century_counts,
        labels=[f"{c - 100}-{c - 1}" for c in sorted_centuries],
        startangle=140,
        pctdistance=0.85,
        labeldistance=1.2
    )

    # Adjust label font size
    for text in texts:
        text.set_horizontalalignment('center')
        text.set_fontsize(10)

    plt.title("Distribution of the publication centuries", fontsize=14)
    plt.savefig("./century_distribution_1900.png", dpi=300, bbox_inches='tight')
    # plt.show()
