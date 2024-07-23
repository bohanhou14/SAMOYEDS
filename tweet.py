import hashlib

class Tweet:
    def __init__(self, text, time, author_id):
        self.text = text
        self.time = time
        self.seed = 42
        self.author = author_id
        self.hash_id()

    def hash_id(self):
        combined = self.text + str(self.time) + str(self.seed) + str(self.author)
        self.id = hashlib.sha256(combined.encode()).hexdigest()

