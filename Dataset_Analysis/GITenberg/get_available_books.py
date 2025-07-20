import requests
import webbrowser

GITENBERG_API = "https://api.github.com/orgs/GITenberg/repos"
PER_PAGE = 100  # Max allowed per request


def list_books():
    books = []
    page = 1

    while True:
        response = requests.get(GITENBERG_API, params={"per_page": PER_PAGE, "page": page})
        if response.status_code != 200:
            print(f"Failed to fetch books: {response.status_code}")
            break

        data = response.json()
        if not data:  # Stop if no more results
            break

        books.extend(data)
        page += 1  # Move to next page

    return books


def open_gutenberg(book_id):
    url = f"https://www.gutenberg.org/ebooks/{book_id}"
    webbrowser.open(url)


if __name__ == "__main__":
    books = list_books()
    print(f"Total books retrieved: {len(books)}")

    [print(book["name"], sep='\n') for book in books]
    # open_gutenberg('41947')
