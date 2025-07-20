import json
import requests
from bs4 import BeautifulSoup


def get_subject_from_gutenberg(url):
    """
    Extracts the assigned subjects for a given PG instance from the according instance website.
    :param url: URL to PG instance website
    :return: subjects assigned to the PG instance
    """
    try:
        # Send an HTTP request to the given URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all <td> tags with property="dcterms:subject"
        subject_elements = soup.find_all("td", {"property": "dcterms:subject"})

        # Extract the text content of the <a> tags inside each <td>
        subjects = [element.find("a").get_text(strip=True) for element in subject_elements if element.find("a")]

        return subjects
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_LoC_from_gutenberg(url):
    """
    Extracts the assigned LoC categories for a given PG instance from the according instance website.
    :param url: URL to PG instance website
    :return: LoCs assigned to the PG instance
    """
    try:
        # Send an HTTP request to the given URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the "LoC" metadata in the page
        loc_elements = soup.find_all("tr", {"property": "dcterms:subject"})

        # Extract the text content of the Subjects
        locs = [loc_element.find("td").get_text(strip=True) for loc_element in loc_elements]

        return locs
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_wikipedia_links(url):
    """
    Extracts related Wikipedia link for a given Project Gutenberg instance from according website.
    :param url: URL to PG instance website
    :return: Links to Wikipedia page from PG
    """
    try:
        # Send an HTTP request to the given URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        wikipedia_links = []
        # Find all <a> tags with href containing 'wikipedia.org'
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "wikipedia.org" in href:
                # Clean the link by adding 'https:' if it's missing
                full_link = "https:" + href if href.startswith("//") else href
                wikipedia_links.append(full_link)
        return wikipedia_links
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def save_metadata(data, final_gold_dataset_path):
    """
    Saves meta data entries in a JSOn file to use it as a reference.
    :param data: dataset meta data
    :param final_gold_dataset_path: path to save meta data
    :return: None
    """
    with open(final_gold_dataset_path, 'w', encoding='utf-8') as out:
        json.dump(data, out, indent=4)
    print(f"Saved dataset to {final_gold_dataset_path}")


def get_formatted_idx(idx):
    """
    Formats a given ID, so that it can be used to reference a Project Gutenberg instance.
    :param idx: ID from PG dataset
    :return: formatted ID
    """
    return idx.split('-')[0]


def get_pg_url(idx):
    """
    Builds URL to Project Gutenberg website for the given book ID
    :param idx: ID for PG instance
    :return: URL for given ID
    """
    return f"https://www.gutenberg.org/ebooks/{idx}"
