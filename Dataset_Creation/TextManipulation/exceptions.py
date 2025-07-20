class NoChapterException(Exception):
    def __init__(self):
        self.message = "No Chapters available for this document!"
        super().__init__(self.message)