"""
    from word_lists import WORD_LISTS
    WORDS = WORD_LISTS["numbers"]
"""

WORD_LISTS = {
    "original_paper": [
        "apple", "bird", "car", "egg",
        "house", "milk", "plane", "opera",
        "box", "sand", "sun", "mango",
        "rock", "math", "code", "phone",
    ],

    # 8x8 = 64 single-token (with leading space) common nouns, for a larger grid.
    "grid_64": [
        "apple", "bird", "car", "egg", "house", "milk", "plane", "opera",
        "box", "sand", "sun", "mango", "rock", "math", "code", "phone",
        "tree", "river", "mountain", "cloud", "star", "moon", "fire", "water",
        "leaf", "stone", "grass", "field", "forest", "island", "ocean", "desert",
        "snow", "rain", "wind", "storm", "bridge", "road", "train", "ship",
        "boat", "wheel", "engine", "wing", "door", "window", "wall", "floor",
        "chair", "table", "lamp", "clock", "mirror", "knife", "spoon", "plate",
        "cup", "bowl", "pillow", "blanket", "shirt", "shoe", "hat", "glove",
    ],

    # 6x6 = 36 single-token (with leading space) common nouns; first 36 of grid_64.
    "grid_36": [
        "apple", "bird", "car", "egg", "house", "milk",
        "plane", "opera", "box", "sand", "sun", "mango",
        "rock", "math", "code", "phone", "tree", "river",
        "mountain", "cloud", "star", "moon", "fire", "water",
        "leaf", "stone", "grass", "field", "forest", "island",
        "ocean", "desert", "snow", "rain", "wind", "storm",
    ],

    "geographic_grid": [
        "Cairo", "Sydney", "Santiago", "Vancouver",
        "Tokyo", "Jakarta", "Oslo", "Miami", 
        "Lagos", "Manila", "Lima", "Boston",
        "Lisbon", "Delhi", "Nairobi", "Vienna",
    ],

    "world_cities": [
        "Paris", "London", "Berlin", "Rome",
        "Madrid", "Tokyo", "Beijing", "Seoul",
        "Cairo", "Lima", "Austin", "Boston",
        "Denver", "Miami", "Phoenix", "Chicago",
    ],

    "text_numbers": [
        "one", "two", "three", "four",
        "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen",
    ],

    "two_digit_numbers": [
        "11", "12", "13", "14",
        "21", "22", "23", "24",
        "31", "32", "33", "34",
        "41", "42", "43", "44"
    ],

    "two_digit_numbers_permuted" : 
    [
        "11", "22", "33", "44",
        "23", "34", "41", "12",
        "31", "42", "13", "24",
        "43", "14", "21", "32"
    ],

    "european_cities": [
        "Paris", "London", "Berlin", "Rome",
        "Madrid", "Vienna", "Dublin", "Prague",
        "Athens", "Brussels", "Lisbon", "Warsaw",
        "Milan", "Munich", "Zurich", "Geneva",
    ],

    "us_presidents_surnames": [
        "Washington", "Jefferson", "Madison", "Jackson",
        "Lincoln", "Grant", "Truman", "Kennedy",
        "Nixon", "Ford", "Carter", "Reagan",
        "Bush", "Clinton", "Obama", "Trump",
    ],

    "misc_words": [
        "and", "stone", "run", "yellow",
        "carefully", "circle", "heavy", "computer",
        "ancient", "stomach", "ocean", "music",
        "ghost", "oxygen", "market", "building",
    ],

    "us_presidents_firstnames": [
        "George", "John", "Thomas", "James",
        "Andrew", "Martin", "William", "Franklin",
        "Harry", "Richard", "Gerald", "Jimmy",
        "Ronald", "Bill", "Donald", "Joe",
    ],

    "grammar_vs_concrete_nouns": [
        # Group 1: Heavy grammar / structural tokens
        "the", "and", "of", "to",
        "with", "it", "that", "is",

        # Group 2: Ultra-specific concrete nouns
        "dinosaur", "galaxy", "concrete", "submarine",
        "microscope", "volcano", "oxygen", "glacier",
    ],

    "synonyms_big": [
        "big", "large", "huge", "massive",
        "enormous", "giant", "vast", "immense",
        "gigantic", "colossal", "sizable", "substantial",
        "hefty", "bulky", "tremendous", "monumental",
    ],

    "synonyms_happy": [
        # elated/joyous/jubilant/gleeful/blissful (originally here) don't
        # tokenize to a single Llama token (with or without leading space),
        # so they're swapped for single-token synonyms at the same positions.
        "happy", "glad", "joyful", "pleased",
        "delighted", "content", "cheerful", "ecstatic",
        "radiant", "merry", "satisfied", "thrilled",
        "sunny", "playful", "smiling", "upbeat",
    ],

    "synonyms_walk": [
        # trudge/plod/prowl/hobble/wade (originally here) don't tokenize to a
        # single Llama token (with or without leading space), so they're
        # swapped for single-token synonyms at the same positions.
        "walk", "shuffle", "wander", "stroll",
        "stumble", "creep", "sneak", "pace",
        "lumber", "march", "trot", "roam",
        "parade", "strut", "limp", "stride"
    ],

    "multilingual_father": [
        "Apa", "Padre", "Father", "Otec",
        "Athair", "Tėvas", "Père", "Vater",
        "Ojciec", "Tad", "Baba", "Pare",
        "Vader", "Missier", "Pai", "Πατέρας"
    ],

    "judging_nearest_neighbors": [
        " judged", " Judge", " assessing", " judges",
        "judge", " judgment", " Jud", " judgments",
        " judgement", "Jud", " measuring", " Judges",
        "ging", " Judgment", " judge", "judging"
    ],
}
