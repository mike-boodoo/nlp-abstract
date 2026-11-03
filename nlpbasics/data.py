"""Small built-in datasets, so the whole course runs offline in seconds.

Everything here is a teaching toy, and scores on it say little about real-world
performance. Each dataset is made to expose one specific concept:

  STORY             original hand-written prose       n-gram models, Zipf
  DOCUMENTS         hand-written mini encyclopedia    search and ranking
  REVIEWS           hand-written, labeled sentiment   classification
  pos_corpus()      grammar-generated tagged text     sequence labeling
  embedding_corpus  grammar-generated, topic-structured   word vectors
  agreement_corpus  grammar-generated long-range task     neural context
"""
from __future__ import annotations

import random

STORY = """
The old keeper climbed the stairs of the lighthouse every evening. He lit the lamp and watched the sea.
The sea was dark and the wind was cold. A small boat came toward the rocks that night.
The keeper saw the boat and rang the bell. The sailors heard the bell and turned the boat away from the rocks.
In the morning the sailors came to the lighthouse. They thanked the old keeper and gave him a basket of fish.
The keeper shared the fish with his cat. The cat slept by the fire and the keeper read a book.
Every day the keeper cleaned the lamp and wrote in his book. He wrote about the weather and the ships.
He wrote about the birds that came in spring. Some days the sea was calm and the sky was bright.
Other days the storm came and the waves struck the rocks. The lighthouse stood through every storm.
A young girl from the village came to visit the keeper. She asked him about the lamp and the bell.
The keeper showed her the stairs and the great lamp. She asked why he stayed alone on the rocks.
He told her that the ships needed the light. She came back the next day with bread from the village.
The girl learned to clean the lamp and to read the weather. She learned the names of the birds and the ships.
The winter came and the snow covered the village. The keeper grew old and his steps grew slow.
The girl climbed the stairs and lit the lamp for him. The sailors saw the light and came home safe.
When the spring came the keeper sat by the fire with his cat. He gave the girl his book and his key.
She wrote in the book every evening. She wrote about the weather and the ships and the birds.
The village was proud of the young keeper. The light shone over the dark sea every night.
A great storm came in the autumn and the boats stayed in the harbor. One boat was lost in the wind.
The young keeper rang the bell through the night. The lost boat followed the sound and found the harbor.
The sailors told the story in the village for many years. The old keeper smiled and the cat slept by the fire.
"""

DOCUMENTS = [
    "Photosynthesis is the process by which green plants use sunlight to make sugar from water and carbon dioxide.",
    "The mitochondrion produces most of the energy of a cell by burning sugar with oxygen.",
    "A volcano erupts when molten rock called magma rises through the crust of the earth.",
    "Earthquakes happen when tectonic plates slip suddenly along a fault in the crust of the earth.",
    "The stock market lets investors buy and sell shares of public companies at prices set by supply and demand.",
    "A central bank raises interest rates to slow inflation and lowers rates to encourage borrowing.",
    "Bonds pay investors interest, and bond prices usually fall when interest rates rise.",
    "A neural network learns by adjusting weights to reduce a loss on training examples.",
    "Gradient descent moves the weights of a model a small step against the gradient of the loss.",
    "A compiler translates source code into machine code that a processor can execute.",
    "The Roman Empire built roads across Europe, and its army used them to move troops quickly.",
    "The printing press spread books across Europe and made reading far more common.",
    "Bread rises because yeast eats sugar in the dough and releases carbon dioxide gas.",
    "Coffee beans are roasted to develop flavor, and darker roasts taste more bitter.",
    "Honey bees collect nectar from flowers and turn it into honey inside the hive.",
    "Whales are mammals that breathe air and feed their young with milk in the sea.",
]

# Hand-written for this course; none are real customer reviews. Hard cases such
# as negation and mixed sentiment are included on purpose.
_POSITIVE = """an absolute delight from start to finish
the best meal i have had all year
the staff were friendly and the room was spotless
this phone is fast and the battery lasts for days
i loved every minute of this film
great sound quality for the price
wonderful book with characters i truly cared about
the service was quick and very helpful
works perfectly and setup took two minutes
a beautiful and moving story
highly recommend this to anyone
the food was fresh and full of flavor
i would happily buy this again
it exceeded all of my expectations
comfortable shoes that look great too
not bad at all and actually quite fun
i was not disappointed by this one
never had a problem with it and it still works great
the plot was slow at first but the ending was brilliant
it is small but the picture is stunning
the kids loved it and asked to go back
excellent value and fast delivery
a clever and funny show with a great cast
my new favorite coffee shop in town
the hotel was quiet and the bed was very comfortable
solid build and it feels premium
this game is addictive and a joy to play
the guide was knowledgeable and kind
such a charming little restaurant
easy to use and looks fantastic
the acting was superb and the music was lovely
i cannot recommend this place enough
nothing to complain about and everything arrived on time
exactly what i needed and it works well
the instructions were clear and assembly was easy
a thrilling ride that kept me hooked
really happy with this purchase
fresh ingredients and generous portions
the camera takes gorgeous photos in low light
a pleasant surprise and worth every penny
customer support solved my issue right away
the album is warm and beautifully recorded
i enjoyed it far more than i expected
lovely view and a very relaxing stay
smart design and great attention to detail
the soup was delicious and the bread was perfect
this vacuum is powerful and surprisingly quiet
an inspiring film that made me smile
it was cheap but it has been very reliable
the teacher was patient and the class was fun"""

_NEGATIVE = """a complete waste of time and money
the worst meal i have had all year
the staff were rude and the room was filthy
this phone is slow and the battery dies by noon
i hated every minute of this film
terrible sound quality even for the price
boring book with characters i could not stand
the service was slow and very unhelpful
stopped working after two days
a dull and predictable story
do not buy this
the food was cold and had no flavor
i would never buy this again
it fell far short of my expectations
uncomfortable shoes that look cheap too
not good at all and honestly quite boring
i was not impressed by this one
always had problems with it and now it is broken
the plot started well but the ending was awful
it is big but the picture is blurry
the kids hated it and asked to leave
poor value and late delivery
a lazy and unfunny show with a weak cast
my least favorite coffee shop in town
the hotel was noisy and the bed was very hard
flimsy build and it feels cheap
this game is buggy and a chore to play
the guide was clueless and rude
such a disappointing little restaurant
hard to use and looks ugly
the acting was wooden and the music was grating
i cannot believe how bad this place is
everything arrived damaged and nobody replied
not what i needed and it barely works
the instructions were confusing and assembly was a nightmare
a tedious film that put me to sleep
really unhappy with this purchase
stale ingredients and tiny portions
the camera takes grainy photos in low light
an unpleasant surprise and not worth the money
customer support ignored my issue for weeks
the album is thin and badly recorded
i enjoyed it far less than i expected
terrible view and a very stressful stay
stupid design and no attention to detail
the soup was bland and the bread was stale
this vacuum is weak and surprisingly loud
a depressing film that made me angry
it was expensive and it has been very unreliable
the teacher was impatient and the class was a mess"""

REVIEWS = [(line, "pos") for line in _POSITIVE.splitlines()] + \
          [(line, "neg") for line in _NEGATIVE.splitlines()]

# Line i of the positive list and line i of the negative list are near-twins
# ("the best meal..." / "the worst meal..."). Same group id = same twin pair.
# Lesson 6 shows why a split must keep each pair together.
REVIEW_GROUPS = list(range(len(REVIEWS) // 2)) * 2


_POS_LEXICON = {
    "DET": ["the", "a", "this", "that", "every"],
    "NOUN": ["dog", "cat", "book", "plan", "watch", "train", "park", "teacher", "child",
             "river", "letter", "house", "garden", "light", "ring", "walk", "run", "play"],
    "ADJ": ["old", "small", "bright", "quiet", "happy", "green", "light", "new"],
    "PRON": ["she", "he", "they", "we", "i"],
    "ADV": ["quickly", "slowly", "often", "quietly", "today"],
    "ADP": ["in", "near", "with", "over", "under"],
    "CONJ": ["and", "but"],
    "AUX": ["will", "can", "should", "must"],
    "BE": ["is", "was", "were"],
    "NUM": ["two", "three", "seven", "12", "40"],
}
_VERBS_BASE = ["book", "plan", "watch", "train", "park", "ring", "walk", "run", "play",
               "open", "follow", "paint", "light", "read"]
_VERBS_PAST = ["booked", "planned", "watched", "trained", "parked", "walked", "played",
               "opened", "followed", "painted"]
_VERBS_ING = ["booking", "planning", "watching", "training", "walking", "playing",
              "opening", "following", "painting"]
_VERBS_S = ["books", "plans", "watches", "trains", "rings", "walks", "plays", "opens"]

# Words appearing ONLY at test time, to measure unknown-word handling. Their
# endings are the sort a shape feature can use.
_OOV = {
    "NOUN": ["zebra", "lawyer", "curtain", "bicycle", "window"],
    "VERB_PAST": ["cleaned", "carried", "visited"],
    "VERB_ING": ["cleaning", "carrying", "visiting"],
    "ADV": ["gently", "rarely", "boldly"],
    "ADJ": ["purple", "tiny"],
    "NUM": ["75", "1999"],
}


_NONCE_SUFFIX = {"VERB_PAST": "ed", "VERB_ING": "ing", "VERB_S": "s", "ADV": "ly",
                 "NOUN": "", "ADJ": ""}


def _nonce_word(rng: random.Random, tag: str) -> str:
    """Invent a word that occurs about once, imitating natural language's long
    tail. Real corpora are full of such hapax legomena (Zipf, lesson 1), and they
    are how a tagger learns what an unknown word tends to look like."""
    if tag == "NUM":
        return str(rng.randrange(100, 100000))
    syllables = ["bl", "dr", "gr", "fl", "sk", "tr", "pl", "kr"]
    vowels = ["a", "e", "i", "o", "u"]
    stem = "".join(rng.choice(syllables) + rng.choice(vowels) for _ in range(3)) + rng.choice("mnpt")
    return stem + _NONCE_SUFFIX[tag]


def _pick(rng: random.Random, tag: str, oov: float, rare: float = 0.0) -> tuple[str, str]:
    """Choose a word for a tag. VERB_* slots collapse to the single VERB tag."""
    label = "VERB" if tag.startswith("VERB") else "AUX" if tag == "BE" else tag
    if rare and tag in {*_NONCE_SUFFIX, "NUM"} and rng.random() < rare:
        return _nonce_word(rng, tag), label
    if oov and rng.random() < oov and tag in _OOV:
        return rng.choice(_OOV[tag]), label
    pools = {"VERB": _VERBS_BASE, "VERB_PAST": _VERBS_PAST, "VERB_ING": _VERBS_ING,
             "VERB_S": _VERBS_S}
    return rng.choice(pools.get(tag) or _POS_LEXICON[tag]), label


_NP = [["DET", "NOUN"], ["DET", "ADJ", "NOUN"], ["NUM", "NOUN"]]
_TEMPLATES = [
    ["NP", "VERB_PAST", "NP"],
    ["NP", "VERB_PAST", "NP", "ADP", "NP"],
    ["NP", "VERB_S", "NP"],
    ["NP", "VERB_PAST", "ADV"],
    ["PRON", "AUX", "VERB", "NP"],
    ["PRON", "AUX", "VERB", "NP", "ADP", "NP"],
    ["PRON", "VERB_PAST", "NP", "CONJ", "VERB_PAST", "NP"],
    ["NP", "AUX", "VERB", "ADV"],
    ["PRON", "AUX", "VERB", "NP", "CONJ", "AUX", "VERB", "NP"],
    ["ADV", "PRON", "VERB_PAST", "NP"],
    ["PRON", "BE", "VERB_ING", "NP"],
    ["NP", "BE", "VERB_ING", "ADV"],
    ["NP", "ADP", "NP", "VERB_PAST", "NP"],
]


def pos_corpus(n: int, seed: int = 0, oov: float = 0.0,
               rare: float = 0.0) -> list[list[tuple[str, str]]]:
    """Tagged sentences containing deliberately ambiguous words: "book" and "run"
    are nouns or verbs, and "light" is a noun, verb or adjective.

    oov:  chance that a slot takes a word reserved for testing
    rare: chance that a slot takes a freshly invented one-off word
    """
    rng = random.Random(seed)
    sentences = []
    for _ in range(n):
        tokens: list[tuple[str, str]] = []
        for slot in rng.choice(_TEMPLATES):
            for tag in (rng.choice(_NP) if slot == "NP" else [slot]):
                tokens.append(_pick(rng, tag, oov, rare))
        tokens.append((".", "PUNCT"))
        sentences.append(tokens)
    return sentences


def embedding_corpus(n: int = 6000, seed: int = 0) -> list[list[str]]:
    """Sentences engineered with three independent axes, gender, royalty and age,
    plus separate animal, food and place topics. The structure is what makes word
    analogies solvable here, by construction. Treat this as a clean demonstration
    of the mechanism, not evidence about embeddings on natural text."""
    rng = random.Random(seed)
    pairs = [("king", "man", "he", "his"), ("queen", "woman", "she", "her"),
             ("prince", "boy", "he", "his"), ("princess", "girl", "she", "her")]
    animals, foods = ["cat", "dog", "horse", "cow"], ["bread", "cheese", "apple", "soup"]
    places = ["paris", "rome", "london", "madrid"]
    royal_templates = [
        "the {r} ruled the kingdom from the palace", "the {r} wore the golden crown",
        "the {r} sat upon the royal throne", "the {r} addressed the royal court",
        "{p} crown rested on the {r}", "the loyal knights served the {r}",
    ]
    common_templates = [
        "the {c} walked to the market", "the {c} worked on the small farm",
        "the {c} went home to the village", "the {c} talked with a neighbor",
    ]
    gender_templates = [
        "{s} said that {p} family was the {x}", "the {x} said {s} loved {p} home",
        "{s} was the {x} and {s} smiled", "the {x} combed {p} hair",
    ]
    adult_templates = ["the grown {x} paid the old debts", "the elder {x} raised the children"]
    young_templates = ["the young {x} played in the yard", "the little {x} went to school"]
    topic_templates = [
        ("animal", "the {a} chased the {b} across the field"),
        ("animal", "the farmer fed the {a} and the {b}"),
        ("animal", "a {a} slept in the barn near a {b}"),
        ("food", "they ate {a} with {b} for dinner"),
        ("food", "the cook served fresh {a} and warm {b}"),
        ("food", "she bought {a} and {b} to eat"),
        ("place", "the train from {a} arrived in {b}"),
        ("place", "the city of {a} is far from {b}"),
        ("place", "travelers flew from {a} to the city of {b}"),
    ]
    topics = {"animal": animals, "food": foods, "place": places}
    sentences = []
    for _ in range(n):
        royal, common, s, p = rng.choice(pairs)
        roll = rng.random()
        if roll < 0.25:
            line = rng.choice(royal_templates).format(r=royal, p=p)
        elif roll < 0.45:
            line = rng.choice(common_templates).format(c=common)
        elif roll < 0.60:
            line = rng.choice(gender_templates).format(s=s, p=p, x=rng.choice((royal, common)))
        elif roll < 0.75:
            templates = adult_templates if royal in ("king", "queen") else young_templates
            line = rng.choice(templates).format(x=rng.choice((royal, common)))
        else:
            topic, template = rng.choice(topic_templates)
            a, b = rng.sample(topics[topic], 2)
            line = template.format(a=a, b=b)
        sentences.append(line.split())
    return sentences


MALE, FEMALE = ["king", "prince", "father", "boy"], ["queen", "princess", "mother", "girl"]
_CLAUSES = [
    "who lived by the river", "who walked past the old mill",
    "who sang near the stone bridge", "who rode across the wide valley",
    "who slept under the tall oak", "who hid behind the broken wall",
]
_OBJECTS = ["crown", "ring", "map", "hat", "horse"]
_VERBS = ["lost", "found", "sold", "kept"]


def agreement_corpus(n: int = 4000, seed: int = 0, holdout: bool = False,
                     extra_clauses: int = 0) -> list[list[str]]:
    """Long-distance dependency: the pronoun depends on a subject 7+ words back.

        the queen who lived by the river lost her crown .

    Half of the sentences add a distractor noun of the opposite gender:

        the queen who saw the boy by the river lost her crown .

    A short context never contains the subject, and "nearest gendered noun" gives
    the wrong answer. With holdout=True we draw only (subject, clause) pairs that
    training never uses, so a model cannot succeed by memorizing whole sentences.

    Note a deliberate flaw: the distractor is always the opposite gender, so it is
    itself a clue. Lesson 9 shows a model exploiting this shortcut.

    extra_clauses appends "and who ..." clauses, stretching the dependency.
    """
    rng = random.Random(seed)
    sentences = []
    while len(sentences) < n:
        is_male = rng.random() < 0.5
        subject = rng.choice(MALE if is_male else FEMALE)
        clause_id = rng.randrange(len(_CLAUSES))
        # Hold out pairs by a fixed rule on (subject, clause) identity.
        pair_held_out = (MALE + FEMALE).index(subject) % 4 == clause_id % 4
        if pair_held_out != holdout:
            continue
        clause = _CLAUSES[clause_id]
        if rng.random() < 0.5:
            distractor = rng.choice(FEMALE if is_male else MALE)
            tail = " ".join(clause.split()[-3:])
            clause = f"who saw the {distractor} {tail}"
        for _ in range(extra_clauses):
            clause += " and " + rng.choice(_CLAUSES)
        pronoun = "his" if is_male else "her"
        line = f"the {subject} {clause} {rng.choice(_VERBS)} {pronoun} {rng.choice(_OBJECTS)} ."
        sentences.append(line.split())
    return sentences
