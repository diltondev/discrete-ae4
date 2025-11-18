import random

class WordGenerator:
    def __init__(self):
        self.table = {}
        self.start = []
    
    def add_word(self, word):
        size = len(word)
        if size < 3:
            self.start.append((word[0], word[1]))
            return
        for i in range(size - 2):
            key = (word[i], word[i+1])
            next_char = word[i+2] if i+2 < size else None
            self.table.setdefault(key, []).append(next_char)
        self.start.append((word[0], word[1]))
    
    def get_word(self, max_len=12):
        c1, c2 = random.choice(self.start)
        word = [c1, c2]
        while len(word) < max_len:
            key = (c1, c2)
            next_chars = self.table.get(key)
            if not next_chars:
                break
            next_char = random.choice(next_chars)
            if next_char is None:
                break
            word.append(next_char)
            c1, c2 = c2, next_char
        return ''.join(word)

# Example usage
generator = WordGenerator()
words = [
    "Bakerloo",
    "Central",
    "Circle",
    "District",
    "Hammersmith and City",
    "Jubilee",
    "Metropolitan",
    "Northern",
    "Piccadilly",
    "Victoria",
    "Waterloo and City",
    "Lioness",
    "Mildmay",
    "Windrush",
    "Weaver",
    "Suffragette",
    "Liberty",
    "Thameslink",
    "Embankment",
    "Wibbly",
    "Quintessence",
    "Elizabeth",
    "Embankment",
    "Highbury",
    "Islington",
    "Farringdon",
    "Hampstead",
    "Tottenham"
]

for w in words:
    generator.add_word(w)

for _ in range(11):
    print(generator.get_word())