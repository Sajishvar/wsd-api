# wsd_core.py
from typing import Dict, List, Tuple
from collections import defaultdict

import stanza
from pyiwn import IndoWordNet, Language
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load heavy resources once (process-wide singletons) ---
# Stanza: make sure Tamil pipeline is present; download if missing.
# On Render, we call stanza.download('ta') during build (render.yaml),
# but this fallback helps on first cold-starts.
try:
    stanza.Pipeline('ta')
except:
    stanza.download('ta')
nlp = stanza.Pipeline('ta', processors='tokenize,pos,lemma')

# IndoWordNet
iwn = IndoWordNet(lang=Language.TAMIL)

def fetch_tamil_glosses(word: str) -> List[Dict[str, List[str]]]:
    synsets = iwn.synsets(word)
    gloss_data = []
    for syn in synsets:
        gloss_data.append({"gloss": syn.gloss(), "examples": syn.examples})
    return gloss_data

def tokenize_ta(text: str) -> List[str]:
    doc = nlp(text)
    return [t.text for s in doc.sentences for t in s.tokens]

def build_bags_for_sentence(sentence_ta: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
      bag_of_words[word] = {
          "overall": [...],
          "sense_1": [...],
          "sense_2": [...],
          ...
      }
    """
    doc = nlp(sentence_ta)
    bag_of_words: Dict[str, Dict[str, List[str]]] = {}

    # choose content words; you can tune this
    for w in doc.sentences[0].words:
        if w.upos in {"NOUN", "VERB", "NUM"}:
            lemma = w.lemma
            synsets = iwn.synsets(lemma)
            if not synsets:
                continue

            word_bow = {"overall": []}
            for idx, syn in enumerate(synsets, start=1):
                gloss = syn.gloss()
                toks = tokenize_ta(gloss)
                key = f"sense_{idx}"
                word_bow[key] = toks
                word_bow["overall"].extend(toks)

            # dedupe
            word_bow["overall"] = list(dict.fromkeys(word_bow["overall"]))
            bag_of_words[lemma] = word_bow

    return bag_of_words

def _avg_intra_cos_sim(tokens: List[str]) -> float:
    # If fewer than 2 tokens, similarity doesn't make sense
    if len(tokens) < 2:
        return 0.0
    # vectorize each token as a "document"
    vectorizer = CountVectorizer().fit(tokens)
    mat = vectorizer.transform(tokens).toarray()
    sims = cosine_similarity(mat)
    # exclude diagonal
    n = len(tokens)
    return (sims.sum() - n) / (n * (n - 1))

def pick_target_word(bag_of_words: Dict[str, Dict[str, List[str]]]) -> str:
    """
    Choose the 'least consistent' word (lowest average intra-token similarity in its overall bag)
    """
    best_word = None
    best_score = None
    for word, bows in bag_of_words.items():
        overall = bows.get("overall", [])
        score = _avg_intra_cos_sim(overall) if overall else 0.0
        if best_score is None or score < best_score:
            best_word, best_score = word, score
    return best_word

def cosine_between_bags(b1: List[str], b2: List[str]) -> float:
    if not b1 or not b2:
        return 0.0
    v = CountVectorizer().fit_transform([' '.join(b1), ' '.join(b2)]).toarray()
    return float(cosine_similarity(v)[0, 1])

def score_senses(target_word: str, context_words: List[str],
                 bag_of_words: Dict[str, Dict[str, List[str]]]) -> Tuple[str, Dict[str, float]]:
    target = bag_of_words.get(target_word, {}).copy()
    target.pop("overall", None)
    combined_scores: Dict[str, float] = defaultdict(float)

    for sense_key, target_bow in target.items():
        total = 0.0
        for cw in context_words:
            cs = bag_of_words.get(cw, {}).copy()
            cs.pop("overall", None)
            for _, cbow in cs.items():
                total += cosine_between_bags(target_bow, cbow)
        combined_scores[sense_key] = total

    if not combined_scores:
        return "", {}

    best_sense = max(combined_scores, key=combined_scores.get)
    return best_sense, dict(combined_scores)

def get_tamil_gloss_for_sense(word: str, sense_key: str) -> str:
    # map sense_# to the #th synset
    try:
        idx = int(sense_key.split("_")[1]) - 1
    except:
        return ""
    synsets = iwn.synsets(word)
    if 0 <= idx < len(synsets):
        return synsets[idx].gloss()
    return ""

def disambiguate(sentence_ta: str):
    """
    Main entry: returns the chosen target word, best sense, gloss, and all scores.
    """
    bag = build_bags_for_sentence(sentence_ta)
    if not bag:
        return {
            "target_word": None,
            "best_sense": None,
            "gloss_ta": None,
            "scores": {},
            "bag_words": list(bag.keys())
        }

    target = pick_target_word(bag)
    context = [w for w in bag.keys() if w != target]
    best_sense, scores = score_senses(target, context, bag)
    gloss_ta = get_tamil_gloss_for_sense(target, best_sense) if best_sense else None

    return {
        "target_word": target,
        "best_sense": best_sense,
        "gloss_ta": gloss_ta,
        "scores": scores,
        "bag_words": list(bag.keys())
    }
