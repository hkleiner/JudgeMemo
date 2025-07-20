def _get_char_ranges(ranges: list[str], delimiter: str, current_pos: int = 0) -> list[list[int]]:
    """
    Calculates the character index ranges for a list of text segments, accounting for delimiters.

    This function iterates over a list of strings (e.g., paragraphs, tokens, sections), and returns
    their character start and end positions in the full text. It assumes that the segments are joined
    by a consistent delimiter, which is included in the position calculation.

    Args:
        ranges (list[str]):
            The list of text segments (e.g., paragraphs or tokens) for which to compute character ranges.
        delimiter (str):
            The delimiter string used to separate the segments in the original text (e.g., "\n\n").
        current_pos (int, optional):
            The starting character index of the first segment. Defaults to 0.

    Returns:
        list[list[int]]:
            A list of [start, end] pairs representing the character ranges for each segment.
            The `start` index is inclusive, and the `end` index is exclusive.
    """
    char_ranges = []

    for range_content in ranges:
        start_pos = current_pos
        end_pos = start_pos + len(range_content)
        char_ranges.append([start_pos, end_pos])

        # update the current position to include the delimiter
        current_pos = end_pos + len(delimiter)

    # start = inclusive; end = exclusive
    return char_ranges


def _find_valid_paragraph_sections(paragraphs: list[str], char_ranges: list[list[int]], min_chars: int) -> list[tuple[list[str], dict[tuple[int, int], list[list[int]]]]]:
    """
    Finds all minimal-length sequences of consecutive paragraphs whose combined character count
    meets or exceeds a given threshold.

    The function returns only the minimal valid sequence from each starting paragraph — that is,
    it stops expanding a sequence as soon as the threshold is reached, avoiding overlong combinations.

    Args:
        paragraphs (list[str]):
            List of paragraph strings to evaluate.
        char_ranges (list[list[int]]):
            Character index ranges for each paragraph, where each range is [start, end].
            These must align with the `paragraphs` list.
        min_chars (int):
            Minimum total character count required for a paragraph sequence to be considered valid.

    Returns:
        list[tuple[list[str], dict[tuple[int, int], list[list[int]]]]]:
            A list of tuples, each representing a valid section:
            - list of paragraph texts in the section,
            - dictionary with a single key `(start_idx, end_idx)` (paragraph indices),
              and a list of corresponding character ranges for the section.
              The `start_idx` is inclusive and `end_idx` is exclusive.
    """
    valid_sections = []
    n = len(paragraphs)

    for start in range(n):
        char_count = 0
        for end in range(start, n):
            char_count += len(paragraphs[end].strip())
            if char_count >= min_chars:
                section = paragraphs[start:end + 1]
                section_ranges = char_ranges[start:end + 1]
                valid_sections.append(
                    (section, {(start, end + 1): section_ranges})
                )
                break  # Only take minimal valid sequence from this start

    return valid_sections


if __name__ == "__main__":
    text = (
        "This is the first paragraph. It's short.\n\n"
        "This is the second paragraph. It's a bit longer than the first one and should count more.\n\n"
        "Here comes the third. Also not too short, not too long.\n\n"
        "Finally, the fourth paragraph is here, which adds some extra length just in case we need it."
    )

    # Split into paragraphs
    delimiter = "\n\n"
    paragraphs = text.split(delimiter)

    # Get character ranges for each paragraph (assuming a utility function you have)
    char_ranges = _get_char_ranges(paragraphs, delimiter)
    print(char_ranges)

    # Define minimum number of characters a section must sum up to
    min_chars = 100  # Adjust this value to test different thresholds

    # Get valid sections
    valid_sections = _find_valid_paragraph_sections(paragraphs, char_ranges, min_chars)

    # Display result
    for section, meta in valid_sections:
        (start, end), ranges = list(meta.items())[0]
        print(f"Paragraphs {start} to {end - 1} (char range {ranges[0][0]} to {ranges[-1][-1]}):")
        for p in section:
            print("   ", p.strip())
        print("---")
