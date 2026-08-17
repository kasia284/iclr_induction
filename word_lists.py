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

    # Diagonal-Latin-square shuffle of "original_paper" (see two_digit_numbers_permuted
    # for the algorithm), so grid-adjacent words were never grid-adjacent originally.
    "original_paper_permuted": [
        "apple", "milk", "sun", "phone",
        "plane", "mango", "rock", "bird",
        "box", "math", "car", "opera",
        "code", "egg", "house", "sand",
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

    # Diagonal-Latin-square shuffle of "grid_64" (8x8 grid).
    "grid_64_permuted": [
        "apple", "sand", "mountain", "field", "bridge", "window", "spoon", "glove",
        "sun", "cloud", "forest", "road", "wall", "plate", "cup", "bird",
        "star", "island", "train", "floor", "chair", "bowl", "car", "mango",
        "ocean", "ship", "boat", "table", "pillow", "egg", "rock", "moon",
        "snow", "wheel", "lamp", "blanket", "house", "math", "fire", "desert",
        "engine", "clock", "shirt", "milk", "code", "water", "leaf", "rain",
        "mirror", "shoe", "plane", "phone", "tree", "stone", "wind", "wing",
        "hat", "opera", "box", "river", "grass", "storm", "door", "knife",
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

    # Diagonal-Latin-square shuffle of "grid_36" (6x6 grid).
    "grid_36_permuted": [
        "apple", "opera", "code", "moon", "forest", "storm",
        "box", "phone", "fire", "island", "ocean", "bird",
        "tree", "water", "leaf", "desert", "car", "sand",
        "mountain", "stone", "snow", "egg", "sun", "river",
        "grass", "rain", "house", "mango", "rock", "cloud",
        "wind", "milk", "plane", "math", "star", "field",
    ],

    "geographic_grid": [
        "Cairo", "Sydney", "Santiago", "Vancouver",
        "Tokyo", "Jakarta", "Oslo", "Miami",
        "Lagos", "Manila", "Lima", "Boston",
        "Lisbon", "Delhi", "Nairobi", "Vienna",
    ],

    # Diagonal-Latin-square shuffle of "geographic_grid".
    "geographic_grid_permuted": [
        "Cairo", "Jakarta", "Lima", "Vienna",
        "Oslo", "Boston", "Lisbon", "Sydney",
        "Lagos", "Delhi", "Santiago", "Miami",
        "Nairobi", "Vancouver", "Tokyo", "Manila",
    ],

    "world_cities": [
        "Paris", "London", "Berlin", "Rome",
        "Madrid", "Tokyo", "Beijing", "Seoul",
        "Cairo", "Lima", "Austin", "Boston",
        "Denver", "Miami", "Phoenix", "Chicago",
    ],

    # Diagonal-Latin-square shuffle of "world_cities".
    "world_cities_permuted": [
        "Paris", "Tokyo", "Austin", "Chicago",
        "Beijing", "Boston", "Denver", "London",
        "Cairo", "Miami", "Berlin", "Seoul",
        "Phoenix", "Rome", "Madrid", "Lima",
    ],

    "text_numbers": [
        "one", "two", "three", "four",
        "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen",
    ],

    # Diagonal-Latin-square shuffle of "text_numbers".
    "text_numbers_permuted": [
        "one", "six", "eleven", "sixteen",
        "seven", "twelve", "thirteen", "two",
        "nine", "fourteen", "three", "eight",
        "fifteen", "four", "five", "ten",
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

    "two_digit_numbers_4to7": [
        "44", "45", "46", "47",
        "54", "55", "56", "57",
        "64", "65", "66", "67",
        "74", "75", "76", "77"
    ],

    "two_digit_numbers_2468": [
        "22", "24", "26", "28",
        "42", "44", "46", "48",
        "62", "64", "66", "68",
        "82", "84", "86", "88"
    ],

    "european_cities": [
        "Paris", "London", "Berlin", "Rome",
        "Madrid", "Vienna", "Dublin", "Prague",
        "Athens", "Brussels", "Lisbon", "Warsaw",
        "Milan", "Munich", "Zurich", "Geneva",
    ],

    # Diagonal-Latin-square shuffle of "european_cities".
    "european_cities_permuted": [
        "Paris", "Vienna", "Lisbon", "Geneva",
        "Dublin", "Warsaw", "Milan", "London",
        "Athens", "Munich", "Berlin", "Prague",
        "Zurich", "Rome", "Madrid", "Brussels",
    ],

    "us_presidents_surnames": [
        "Washington", "Jefferson", "Madison", "Jackson",
        "Lincoln", "Grant", "Truman", "Kennedy",
        "Nixon", "Ford", "Carter", "Reagan",
        "Bush", "Clinton", "Obama", "Trump",
    ],

    # Diagonal-Latin-square shuffle of "us_presidents_surnames", so
    # chronological neighbors are never grid-adjacent.
    "us_presidents_surnames_permuted": [
        "Washington", "Grant", "Carter", "Trump",
        "Truman", "Reagan", "Bush", "Jefferson",
        "Nixon", "Clinton", "Madison", "Kennedy",
        "Obama", "Jackson", "Lincoln", "Ford",
    ],

    "misc_words": [
        "and", "stone", "run", "yellow",
        "carefully", "circle", "heavy", "computer",
        "ancient", "stomach", "ocean", "music",
        "ghost", "oxygen", "market", "building",
    ],

    # Diagonal-Latin-square shuffle of "misc_words".
    "misc_words_permuted": [
        "and", "circle", "ocean", "building",
        "heavy", "music", "ghost", "stone",
        "ancient", "oxygen", "run", "computer",
        "market", "yellow", "carefully", "stomach",
    ],

    "us_presidents_firstnames": [
        "George", "John", "Thomas", "James",
        "Andrew", "Martin", "William", "Franklin",
        "Harry", "Richard", "Gerald", "Jimmy",
        "Ronald", "Bill", "Donald", "Joe",
    ],

    # Diagonal-Latin-square shuffle of "us_presidents_firstnames".
    "us_presidents_firstnames_permuted": [
        "George", "Martin", "Gerald", "Joe",
        "William", "Jimmy", "Ronald", "John",
        "Harry", "Bill", "Thomas", "Franklin",
        "Donald", "James", "Andrew", "Richard",
    ],

    "grammar_vs_concrete_nouns": [
        # Group 1: Heavy grammar / structural tokens
        "the", "and", "of", "to",
        "with", "it", "that", "is",

        # Group 2: Ultra-specific concrete nouns
        "dinosaur", "galaxy", "concrete", "submarine",
        "microscope", "volcano", "oxygen", "glacier",
    ],

    "grammar_vs_concrete_nouns_permuted": [
        # Checkerboard layout: grammar tokens on one grid-parity, concrete
        # nouns on the other, so same-category words are never grid-adjacent.
        "the", "dinosaur", "and", "galaxy",
        "concrete", "of", "submarine", "to",
        "with", "microscope", "it", "volcano",
        "oxygen", "that", "glacier", "is",
    ],

    "large_vs_small": [
        # Group 1: synonyms of "large"
        "big", "large", "huge", "massive",
        "enormous", "giant", "vast", "immense",

        # Group 2: synonyms of "small"
        "small", "little", "tiny", "mini",
        "petite", "miniature", "compact", "wee",
    ],

    "large_vs_small_permuted": [
        # Checkerboard layout: "large" synonyms on one grid-parity, "small"
        # synonyms on the other, so same-category (similar-embedding) words
        # are never grid-adjacent -- every neighbor is cross-category.
        "big", "small", "large", "little",
        "tiny", "huge", "mini", "massive",
        "enormous", "petite", "giant", "miniature",
        "compact", "vast", "wee", "immense",
    ],

    "synonyms_big": [
        "big", "large", "huge", "massive",
        "enormous", "giant", "vast", "immense",
        "gigantic", "colossal", "sizable", "substantial",
        "hefty", "bulky", "tremendous", "monumental",
    ],

    # Diagonal-Latin-square shuffle of "synonyms_big".
    "synonyms_big_permuted": [
        "big", "giant", "sizable", "monumental",
        "vast", "substantial", "hefty", "large",
        "gigantic", "bulky", "huge", "immense",
        "tremendous", "massive", "enormous", "colossal",
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

    # Diagonal-Latin-square shuffle of "synonyms_happy".
    "synonyms_happy_permuted": [
        "happy", "content", "satisfied", "upbeat",
        "cheerful", "thrilled", "sunny", "glad",
        "radiant", "playful", "joyful", "ecstatic",
        "smiling", "pleased", "delighted", "merry",
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

    # Diagonal-Latin-square shuffle of "synonyms_walk".
    "synonyms_walk_permuted": [
        "walk", "creep", "trot", "stride",
        "sneak", "roam", "parade", "shuffle",
        "lumber", "strut", "wander", "pace",
        "limp", "stroll", "stumble", "march",
    ],

    "multilingual_father": [
        "Apa", "Padre", "Father", "Otec",
        "Athair", "Tėvas", "Père", "Vater",
        "Ojciec", "Tad", "Baba", "Pare",
        "Vader", "Missier", "Pai", "Πατέρας"
    ],

    # Diagonal-Latin-square shuffle of "multilingual_father".
    "multilingual_father_permuted": [
        "Apa", "Tėvas", "Baba", "Πατέρας",
        "Père", "Pare", "Vader", "Padre",
        "Ojciec", "Missier", "Father", "Vater",
        "Pai", "Otec", "Athair", "Tad",
    ],

    "judging_nearest_neighbors": [
        " judged", " Judge", " assessing", " judges",
        "judge", " judgment", " Jud", " judgments",
        " judgement", "Jud", " measuring", " Judges",
        "ging", " Judgment", " judge", "judging"
    ],

    # Diagonal-Latin-square shuffle of "judging_nearest_neighbors".
    "judging_nearest_neighbors_permuted": [
        " judged", " judgment", " measuring", "judging",
        " Jud", " Judges", "ging", " Judge",
        " judgement", " Judgment", " assessing", " judgments",
        " judge", " judges", "judge", "Jud",
    ],


    "chemical_elements": [
        "hydrogen", "helium", "lithium", "beryllium",
        "boron", "carbon", "nitrogen", "oxygen",
        "fluorine", "neon", "sodium", "magnesium",
        "aluminum", "silicon", "phosphorus", "sulfur"
    ],

    # Diagonal-Latin-square shuffle of "chemical_elements".
    "chemical_elements_permuted": [
        "hydrogen", "carbon", "sodium", "sulfur",
        "nitrogen", "magnesium", "aluminum", "helium",
        "fluorine", "silicon", "lithium", "oxygen",
        "phosphorus", "beryllium", "boron", "neon"
    ],  

    # "I" was replaced with "Q", then "Q" with "S" -- as the first-person
    # pronoun, "I" was far more semantically loaded than the other
    # single-letter tokens (see
    # results/gram-matrix/embeddings_similarity_capital_letters.png, where it
    # was the one outlier with near-zero similarity to every other letter).
    # "Q" fixed the semantics but turned out to be an embedding-norm/rare-
    # token outlier instead (mean cosine similarity to the other 15 letters
    # = 0.238 vs. the cluster's own internal average of 0.381); of the
    # remaining unused letters, "S" sits closest to the rest of the cluster
    # (mean cosine similarity = 0.393, slightly above the cluster average).
    # "L" was replaced with "R" -- it was a persistent PC1/PC2 outlier across
    # every swept layer (see the pca_across_layers_Grid_capital_letters_2d
    # plots), separating from the rest of the grid cluster from layer 6 on.
    # Grid layout re-arranged (letters no longer in A-P order) to maximize
    # total cosine similarity of embeddings across grid edges, i.e. so
    # embedding-close letters sit graph-adjacent -- found via multi-restart
    # pairwise-swap local search over the quadratic assignment problem
    # maximize sum_{i,j in grid edges} cos_sim(letter[i], letter[j]).
    # Total edge-weighted similarity: 19.71 vs. 18.74 for plain A-P order.
    "capital_letters": [
        "J", "H", "K", "N",
        "R", "M", "G", "S",
        "P", "C", "D", "E",
        "F", "B", "A", "O"
    ],

    # Diagonal-Latin-square shuffle of "capital_letters".
    "capital_letters_permuted": [
        "J", "M", "D", "O",
        "G", "E", "F", "H",
        "P", "B", "K", "S",
        "A", "N", "R", "C"
    ],

    "semantic_analogy": [
        "king", "queen", "man", "woman",
        "boy", "girl", "prince", "princess",
        "paris", "france", "rome", "germany",
        "apple", "banana", "car", "train",
    ],

    # Diagonal-Latin-square shuffle of "semantic_analogy".
    "semantic_analogy_permuted": [
        "king", "girl", "rome", "train",
        "prince", "germany", "apple", "queen",
        "paris", "banana", "man", "princess",
        "car", "woman", "boy", "france",
    ],

    # "morphology": [
    #     "run", "running", "runner", "walk",
    #     "walking", "walker", "drive", "driving",
    #     "driver", "teach", "teaching", "teacher",
    #     "quick", "quickly", "slow", "slowly",
    # ],

    # # Diagonal-Latin-square shuffle of "morphology".
    # "morphology_permuted": [
    #     "run", "walker", "teaching", "slowly",
    #     "drive", "teacher", "quick", "running",
    #     "driver", "quickly", "runner", "driving",
    #     "slow", "walk", "walking", "teach",
    # ],

    "morphology": [
        # play
        "play", "plays", "played", "playing",

        # walk
        "walk", "walks", "walked", "walking",

        # teach
        "teach", "teaches", "taught", "teaching",

        # drive
        "drive", "drives", "drove", "driving",
    ],

    # Diagonal-Latin-square shuffle of "morphology". 0% of grid edges connect
    # same-lemma words (vs. 50% for "morphology").
    "morphology_permuted": [
        "play", "walks", "taught", "driving",
        "walk", "teaches", "drive", "played",
        "teach", "drives", "playing", "walked",
        "drove", "plays", "walking", "teaching",
    ],

    # Random (non-symmetry) permutations of "morphology"'s 16 words onto the
    # same 4x4 grid, spanning a range of same-lemma grid-edge fractions
    # between "morphology" (50%) and "morphology_permuted" (0%) -- for
    # sweeping distance correlation / accuracy against degree of alignment
    # between grid adjacency and the model's static lexical/morphological
    # prior. Verified to not be rotations/reflections of morphology or
    # morphology_permuted.
    "morphology_rand1": [  # 17% same-lemma grid edges
        "plays", "walked", "teach", "teaches",
        "drives", "walk", "played", "drove",
        "taught", "walking", "driving", "teaching",
        "playing", "play", "walks", "drive",
    ],
    "morphology_rand2": [  # 29% same-lemma grid edges
        "played", "drove", "drive", "teaches",
        "play", "playing", "walk", "taught",
        "teaching", "walks", "driving", "teach",
        "drives", "walking", "walked", "plays",
    ],
    "morphology_rand3": [  # 38% same-lemma grid edges
        "walks", "drives", "walking", "walk",
        "taught", "drove", "walked", "teaching",
        "teaches", "drive", "plays", "teach",
        "playing", "played", "play", "driving",
    ],

    # Each lemma's 4 forms fill one 2x2 corner block of the 4x4 grid (play:
    # top-left, walk: top-right, teach: bottom-left, drive: bottom-right).
    # 67% of grid edges connect same-lemma words -- higher than "morphology"
    # (50%, lemma families as full rows) since a 2x2 block has same-lemma
    # edges in both directions, not just horizontally.
    "morphology_corners": [
        "play", "plays", "walk", "walks",
        "played", "playing", "walked", "walking",
        "teach", "teaches", "drive", "drives",
        "taught", "teaching", "drove", "driving",
    ],

    # ── Additional random grid-position permutations ─────────────────────
    # Generated by generate_extra_grid_permutations.py: 2 new plain random
    # shuffles per existing word-set family (same words, new grid layout),
    # for more (distance correlation, accuracy) points on the gridness-vs-
    # accuracy scatter. Requires 01_reproduce.py to be run for each new key
    # (GPU) before it can be plotted.
    "capital_letters_rand1": [
        "F", "O", "E", "R",
        "A", "D", "J", "G",
        "S", "C", "B", "K",
        "N", "M", "H", "P"
    ],

    "capital_letters_rand2": [
        "N", "S", "C", "B",
        "F", "D", "G", "K",
        "H", "M", "J", "P",
        "E", "O", "R", "A"
    ],

    "chemical_elements_rand1": [
        "aluminum", "lithium", "sodium", "helium",
        "carbon", "hydrogen", "magnesium", "nitrogen",
        "sulfur", "beryllium", "oxygen", "neon",
        "silicon", "phosphorus", "boron", "fluorine"
    ],

    "chemical_elements_rand2": [
        "silicon", "fluorine", "boron", "sulfur",
        "lithium", "sodium", "hydrogen", "nitrogen",
        "phosphorus", "magnesium", "carbon", "helium",
        "beryllium", "neon", "oxygen", "aluminum"
    ],

    "european_cities_rand1": [
        "Dublin", "Berlin", "London", "Milan",
        "Prague", "Geneva", "Paris", "Munich",
        "Zurich", "Warsaw", "Vienna", "Lisbon",
        "Madrid", "Brussels", "Athens", "Rome"
    ],

    "european_cities_rand2": [
        "Munich", "Madrid", "London", "Dublin",
        "Vienna", "Brussels", "Paris", "Warsaw",
        "Prague", "Berlin", "Milan", "Zurich",
        "Rome", "Lisbon", "Athens", "Geneva"
    ],

    "geographic_grid_rand1": [
        "Oslo", "Nairobi", "Boston", "Vancouver",
        "Jakarta", "Tokyo", "Cairo", "Santiago",
        "Lima", "Lagos", "Sydney", "Manila",
        "Delhi", "Vienna", "Miami", "Lisbon"
    ],

    "geographic_grid_rand2": [
        "Vancouver", "Vienna", "Lagos", "Delhi",
        "Tokyo", "Cairo", "Miami", "Boston",
        "Nairobi", "Sydney", "Jakarta", "Lima",
        "Manila", "Oslo", "Lisbon", "Santiago"
    ],

    "grammar_vs_concrete_nouns_rand1": [
        "glacier", "dinosaur", "volcano", "oxygen",
        "to", "that", "of", "it",
        "is", "microscope", "galaxy", "submarine",
        "concrete", "with", "the", "and"
    ],

    "grammar_vs_concrete_nouns_rand2": [
        "it", "the", "dinosaur", "with",
        "galaxy", "volcano", "concrete", "is",
        "and", "oxygen", "of", "microscope",
        "submarine", "that", "glacier", "to"
    ],

    "grid_36_rand1": [
        "egg", "mountain", "tree", "plane", "car", "river",
        "rock", "grass", "storm", "desert", "bird", "sand",
        "field", "wind", "rain", "water", "apple", "forest",
        "leaf", "box", "house", "code", "mango", "opera",
        "star", "snow", "island", "fire", "stone", "cloud",
        "milk", "math", "moon", "sun", "phone", "ocean"
    ],

    "grid_36_rand2": [
        "river", "apple", "ocean", "forest", "island", "sun",
        "bird", "rock", "phone", "house", "sand", "box",
        "grass", "tree", "moon", "milk", "math", "fire",
        "cloud", "mountain", "field", "opera", "car", "stone",
        "star", "plane", "leaf", "water", "wind", "snow",
        "code", "mango", "storm", "egg", "rain", "desert"
    ],

    "grid_64_rand1": [
        "egg", "mountain", "apple", "river", "water", "sun", "boat", "wing",
        "house", "milk", "rock", "plane", "glove", "shirt", "phone", "tree",
        "cup", "rain", "island", "knife", "clock", "hat", "engine", "code",
        "math", "box", "desert", "storm", "grass", "train", "road", "field",
        "wheel", "plate", "leaf", "lamp", "table", "sand", "forest", "ocean",
        "stone", "mango", "fire", "chair", "spoon", "shoe", "mirror", "snow",
        "car", "wind", "cloud", "moon", "wall", "floor", "bird", "bridge",
        "blanket", "window", "opera", "star", "ship", "bowl", "pillow", "door"
    ],

    "grid_64_rand2": [
        "storm", "water", "tree", "field", "floor", "bird", "hat", "road",
        "pillow", "stone", "bowl", "star", "code", "shoe", "boat", "river",
        "grass", "spoon", "lamp", "door", "bridge", "rock", "window", "wall",
        "clock", "forest", "milk", "snow", "car", "math", "mango", "phone",
        "house", "mirror", "leaf", "chair", "wing", "box", "cup", "knife",
        "egg", "engine", "island", "fire", "plate", "blanket", "plane", "ocean",
        "sand", "train", "shirt", "cloud", "opera", "table", "mountain", "apple",
        "wind", "wheel", "rain", "sun", "ship", "desert", "moon", "glove"
    ],

    "judging_nearest_neighbors_rand1": [
        " judgments", " judgment", "ging", "Jud",
        " Judgment", " measuring", "judge", " judged",
        " Judge", " judge", " judges", " Jud",
        " Judges", " judgement", " assessing", "judging"
    ],

    "judging_nearest_neighbors_rand2": [
        "judge", " measuring", " judged", "judging",
        "ging", " Judges", " judgement", " judgments",
        " Judge", " Jud", " Judgment", " judge",
        " assessing", " judgment", " judges", "Jud"
    ],

    "large_vs_small_rand1": [
        "big", "large", "enormous", "massive",
        "compact", "giant", "miniature", "vast",
        "wee", "small", "immense", "tiny",
        "mini", "huge", "petite", "little"
    ],

    "large_vs_small_rand2": [
        "little", "vast", "miniature", "enormous",
        "giant", "petite", "small", "big",
        "wee", "large", "massive", "mini",
        "huge", "tiny", "immense", "compact"
    ],

    "misc_words_rand1": [
        "run", "stone", "carefully", "building",
        "computer", "and", "ghost", "oxygen",
        "circle", "yellow", "stomach", "heavy",
        "ocean", "ancient", "market", "music"
    ],

    "misc_words_rand2": [
        "heavy", "carefully", "run", "building",
        "stone", "music", "oxygen", "yellow",
        "ancient", "market", "circle", "stomach",
        "ocean", "computer", "ghost", "and"
    ],

    "morphology_rand4": [
        "drove", "play", "teaches", "driving",
        "played", "walks", "drives", "teach",
        "drive", "teaching", "playing", "walking",
        "walk", "plays", "taught", "walked"
    ],

    "morphology_rand5": [
        "drove", "taught", "walk", "teaching",
        "walks", "drives", "play", "played",
        "teaches", "walking", "walked", "plays",
        "teach", "playing", "driving", "drive"
    ],

    "multilingual_father_rand1": [
        "Athair", "Padre", "Vater", "Père",
        "Pare", "Father", "Tėvas", "Missier",
        "Πατέρας", "Tad", "Otec", "Apa",
        "Pai", "Ojciec", "Vader", "Baba"
    ],

    "multilingual_father_rand2": [
        "Tėvas", "Padre", "Πατέρας", "Otec",
        "Apa", "Vater", "Tad", "Missier",
        "Father", "Père", "Baba", "Vader",
        "Pare", "Athair", "Ojciec", "Pai"
    ],

    "original_paper_rand1": [
        "egg", "box", "apple", "plane",
        "phone", "math", "car", "opera",
        "code", "house", "mango", "bird",
        "sun", "milk", "sand", "rock"
    ],

    "original_paper_rand2": [
        "sun", "bird", "apple", "rock",
        "plane", "sand", "mango", "house",
        "math", "opera", "code", "phone",
        "milk", "box", "car", "egg"
    ],

    "semantic_analogy_rand1": [
        "paris", "banana", "queen", "boy",
        "girl", "apple", "woman", "king",
        "train", "car", "germany", "prince",
        "princess", "man", "france", "rome"
    ],

    "semantic_analogy_rand2": [
        "germany", "paris", "france", "woman",
        "apple", "prince", "banana", "king",
        "rome", "queen", "girl", "man",
        "princess", "car", "train", "boy"
    ],

    "synonyms_big_rand1": [
        "big", "bulky", "large", "colossal",
        "giant", "massive", "monumental", "substantial",
        "huge", "enormous", "sizable", "gigantic",
        "vast", "immense", "hefty", "tremendous"
    ],

    "synonyms_big_rand2": [
        "huge", "vast", "massive", "sizable",
        "enormous", "colossal", "giant", "monumental",
        "large", "big", "gigantic", "tremendous",
        "hefty", "substantial", "immense", "bulky"
    ],

    "synonyms_happy_rand1": [
        "joyful", "upbeat", "merry", "glad",
        "radiant", "playful", "sunny", "happy",
        "pleased", "satisfied", "content", "cheerful",
        "delighted", "smiling", "thrilled", "ecstatic"
    ],

    "synonyms_happy_rand2": [
        "smiling", "delighted", "sunny", "joyful",
        "satisfied", "glad", "happy", "cheerful",
        "playful", "pleased", "content", "radiant",
        "merry", "thrilled", "ecstatic", "upbeat"
    ],

    "synonyms_walk_rand1": [
        "stride", "wander", "pace", "roam",
        "strut", "lumber", "march", "shuffle",
        "stroll", "limp", "parade", "walk",
        "creep", "sneak", "trot", "stumble"
    ],

    "synonyms_walk_rand2": [
        "march", "stride", "stumble", "parade",
        "lumber", "creep", "strut", "limp",
        "trot", "roam", "pace", "stroll",
        "wander", "sneak", "shuffle", "walk"
    ],

    "text_numbers_rand1": [
        "fifteen", "five", "three", "nine",
        "eight", "ten", "eleven", "sixteen",
        "twelve", "six", "fourteen", "one",
        "two", "seven", "thirteen", "four"
    ],

    "text_numbers_rand2": [
        "six", "four", "three", "seven",
        "ten", "twelve", "eleven", "fifteen",
        "sixteen", "one", "nine", "two",
        "thirteen", "five", "fourteen", "eight"
    ],

    "text_numbers_rand3": [
        "thirteen", "fifteen", "two", "one",
        "nine", "three", "fourteen", "ten",
        "four", "eight", "six", "seven",
        "five", "eleven", "sixteen", "twelve"
    ],

    "text_numbers_rand4": [
        "thirteen", "one", "fourteen", "eleven",
        "seven", "twelve", "ten", "five",
        "four", "nine", "three", "eight",
        "six", "two", "sixteen", "fifteen"
    ],

    "text_numbers_rand5": [
        "seven", "thirteen", "eight", "fourteen",
        "three", "ten", "one", "four",
        "nine", "six", "fifteen", "five",
        "twelve", "eleven", "two", "sixteen"
    ],

    "two_digit_numbers_rand1": [
        "24", "21", "32", "31",
        "23", "14", "41", "34",
        "42", "33", "44", "43",
        "13", "12", "22", "11"
    ],

    "two_digit_numbers_rand2": [
        "22", "33", "41", "42",
        "32", "11", "21", "12",
        "34", "23", "13", "44",
        "43", "31", "14", "24"
    ],

    "two_digit_numbers_rand3": [
        "32", "13", "42", "14",
        "23", "44", "34", "24",
        "21", "41", "22", "33",
        "12", "11", "31", "43"
    ],

    "two_digit_numbers_rand4": [
        "11", "23", "31", "42",
        "34", "41", "32", "24",
        "43", "44", "12", "13",
        "21", "14", "22", "33"
    ],

    "two_digit_numbers_rand5": [
        "43", "33", "32", "42",
        "12", "41", "31", "23",
        "22", "44", "14", "11",
        "24", "21", "34", "13"
    ],

    "two_digit_numbers_2468_rand1": [
        "82", "68", "44", "24",
        "64", "42", "84", "26",
        "62", "88", "46", "66",
        "22", "86", "28", "48"
    ],

    "two_digit_numbers_2468_rand2": [
        "46", "48", "62", "68",
        "64", "26", "44", "86",
        "22", "84", "66", "88",
        "24", "28", "82", "42"
    ],

    "two_digit_numbers_4to7_rand1": [
        "56", "46", "66", "77",
        "54", "55", "44", "47",
        "64", "76", "74", "45",
        "67", "75", "65", "57"
    ],

    "two_digit_numbers_4to7_rand2": [
        "47", "77", "46", "55",
        "65", "74", "66", "57",
        "54", "56", "76", "45",
        "64", "67", "44", "75"
    ],

    "us_presidents_firstnames_rand1": [
        "Gerald", "George", "Andrew", "Joe",
        "James", "Franklin", "Richard", "Jimmy",
        "Ronald", "William", "Thomas", "Harry",
        "Bill", "John", "Martin", "Donald"
    ],

    "us_presidents_firstnames_rand2": [
        "Ronald", "Joe", "James", "William",
        "Harry", "Andrew", "Jimmy", "Thomas",
        "Gerald", "Martin", "Richard", "Franklin",
        "Donald", "Bill", "George", "John"
    ],

    "us_presidents_surnames_rand1": [
        "Jackson", "Clinton", "Carter", "Kennedy",
        "Washington", "Truman", "Lincoln", "Bush",
        "Madison", "Reagan", "Jefferson", "Nixon",
        "Obama", "Ford", "Trump", "Grant"
    ],

    "us_presidents_surnames_rand2": [
        "Nixon", "Lincoln", "Truman", "Trump",
        "Carter", "Grant", "Jefferson", "Kennedy",
        "Obama", "Jackson", "Bush", "Clinton",
        "Ford", "Madison", "Reagan", "Washington"
    ],

    "world_cities_rand1": [
        "Paris", "Miami", "Berlin", "Phoenix",
        "Chicago", "Beijing", "Cairo", "Austin",
        "Denver", "Tokyo", "London", "Seoul",
        "Rome", "Madrid", "Lima", "Boston"
    ],

    "world_cities_rand2": [
        "Lima", "London", "Chicago", "Madrid",
        "Paris", "Miami", "Rome", "Tokyo",
        "Phoenix", "Beijing", "Cairo", "Boston",
        "Austin", "Berlin", "Seoul", "Denver"
    ],

    # ── Days of the week -- for the RING graph (7 nodes in a cycle, not a
    # square grid). "days_of_week" is the natural Mon-Sun cyclic order, i.e.
    # ring adjacency == actual weekday adjacency.
    "days_of_week": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ],

    # Step-2 cyclic shuffle (gcd(2,7)=1, so this visits all 7 days in a new
    # order) -- the ring analog of the grid word lists' diagonal-Latin-
    # square "_permuted" shuffle: a systematic, non-random scramble of ring
    # adjacency rather than a random one.
    "days_of_week_permuted": [
        "Monday", "Wednesday", "Friday", "Sunday", "Tuesday", "Thursday", "Saturday"
    ],

    "days_of_week_rand1": [
        "Friday", "Wednesday", "Tuesday", "Thursday", "Sunday", "Monday", "Saturday"
    ],

    "days_of_week_rand2": [
        "Thursday", "Wednesday", "Tuesday", "Friday", "Monday", "Saturday", "Sunday"
    ],

    "days_of_week_rand3": [
        "Monday", "Friday", "Thursday", "Tuesday", "Sunday", "Saturday", "Wednesday"
    ],

    # ── Months of the year -- for the RING graph (12 nodes in a cycle).
    # "months_of_year" is the natural Jan-Dec cyclic order, i.e. ring
    # adjacency == actual month adjacency.
    "months_of_year": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ],

    # Step-5 cyclic shuffle (gcd(5,12)=1, so this visits all 12 months in a
    # new order) -- the ring analog of the grid word lists' diagonal-Latin-
    # square "_permuted" shuffle.
    "months_of_year_permuted": [
        "January", "June", "November", "April", "September", "February",
        "July", "December", "May", "October", "March", "August"
    ],

    "months_of_year_rand1": [
        "May", "August", "February", "December", "April", "October",
        "June", "January", "July", "September", "November", "March"
    ],

    "months_of_year_rand2": [
        "January", "February", "September", "December", "October", "June",
        "November", "May", "July", "August", "March", "April"
    ],

    "months_of_year_rand3": [
        "September", "March", "July", "November", "May", "April",
        "January", "August", "June", "February", "October", "December"
    ],


    # ── Additional random grid-position permutations of original_paper ───
    # Generated by a one-off scoped variant of
    # generate_extra_grid_permutations.py (same plain-random-shuffle
    # approach as morphology_rand1/2/3), for the distance-correlation-
    # accuracy-phase-plane sweep. Requires 01_reproduce.py and
    # morphology-grid-evolution-layers-vs-seqlen.py to be run for each new
    # key (GPU) before they can be plotted.
    "original_paper_rand3": [
        "mango", "plane", "sun", "rock",
        "milk", "opera", "code", "sand",
        "phone", "bird", "apple", "egg",
        "house", "car", "math", "box"
    ],

    "original_paper_rand4": [
        "phone", "opera", "plane", "car",
        "rock", "mango", "box", "apple",
        "milk", "code", "math", "egg",
        "house", "bird", "sun", "sand"
    ],

    "original_paper_rand5": [
        "apple", "opera", "egg", "phone",
        "box", "car", "house", "math",
        "sun", "code", "rock", "plane",
        "bird", "mango", "sand", "milk"
    ],

    "original_paper_rand6": [
        "egg", "sand", "mango", "math",
        "sun", "car", "milk", "bird",
        "plane", "opera", "phone", "code",
        "house", "apple", "rock", "box"
    ],

    "original_paper_rand7": [
        "bird", "rock", "phone", "house",
        "milk", "sand", "apple", "mango",
        "math", "sun", "code", "egg",
        "box", "car", "plane", "opera"
    ],

}
