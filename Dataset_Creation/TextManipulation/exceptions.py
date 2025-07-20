class NoChapterException(Exception):
    """
    Exception raised when a document is expected to have chapters,
    but none are found during preprocessing or analysis.

    Attributes:
        message (str): Explanation of the error.
    """
    def __init__(self):
        self.message = "No Chapters available for this document!"
        super().__init__(self.message)