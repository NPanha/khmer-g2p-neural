"""Khmer phonological inventory and series-sensitive vowel realization tables.

References:
    - Huffman, F. (1970). *Cambodian System of Writing and Beginning Reader*.
    - Open Dictionary / SEAlang Khmer phonology notes.
    - Unicode 15.1, Khmer block (U+1780–U+17FF).

Khmer consonants belong to one of two *series* (1st or 2nd) — a.k.a.
"light"/"heavy" or "a-series"/"o-series". The series determines how the
following vowel letter is realized phonetically.
"""

# --- Consonants --------------------------------------------------------------

# Series 1 (ɑ-series, "light")
SERIES_1 = {
    "ក": "k",    # ka
    "ខ": "kʰ",   # kha
    "ច": "c",    # ca (palatal stop, IPA [c] ~ [tɕ])
    "ឆ": "cʰ",   # cha
    "ដ": "ɗ",    # da (implosive)
    "ឋ": "tʰ",   # ṭha
    "ណ": "n",    # ṇa
    "ត": "t",    # ta
    "ថ": "tʰ",   # tha
    "ប": "ɓ",    # ba (implosive)
    "ផ": "pʰ",   # pha
    "ស": "s",    # sa
    "ហ": "h",    # ha
    "ឡ": "l",    # la
    "អ": "ʔ",    # qa (glottal stop)
}

# Series 2 (oː-series, "heavy")
SERIES_2 = {
    "គ": "k",    # ko
    "ឃ": "kʰ",   # kho
    "ង": "ŋ",    # ngo
    "ជ": "c",    # co
    "ឈ": "cʰ",   # cho
    "ញ": "ɲ",    # nyo
    "ឌ": "ɗ",    # do
    "ឍ": "tʰ",   # ḍho
    "ទ": "t",    # to
    "ធ": "tʰ",   # tho
    "ន": "n",    # no
    "ព": "p",    # po
    "ភ": "pʰ",   # pho
    "ម": "m",    # mo
    "យ": "j",    # yo
    "រ": "r",    # ro
    "ល": "l",    # lo
    "វ": "ʋ",    # vo
}
# NOTE: ហ is a Series-1 consonant only. To express a Series-2 /h/, Khmer
# uses the TRIISAP diacritic (៊) on ហ rather than a distinct letter.

# Convenience maps
CONSONANT_SERIES = {c: 1 for c in SERIES_1}
CONSONANT_SERIES.update({c: 2 for c in SERIES_2 if c not in CONSONANT_SERIES})

CONSONANT_IPA = {}
CONSONANT_IPA.update(SERIES_1)
# Add Series-2-only characters; setdefault keeps the Series-1 mapping for
# any character that happens to appear in both tables (shouldn't occur).
for c, ipa in SERIES_2.items():
    CONSONANT_IPA.setdefault(c, ipa)

# --- Subscript (COENG) -------------------------------------------------------

COENG = "\u17D2"  # ្  — the subscript marker

# --- Dependent vowels --------------------------------------------------------
# Each dependent vowel has two realizations depending on the series of its
# preceding (base) consonant.
# (series_1_ipa, series_2_ipa)
DEPENDENT_VOWELS = {
    "\u17B6": ("aː", "iə"),   # ា
    "\u17B7": ("e",  "i"),    # ិ
    "\u17B8": ("əj", "iː"),   # ី
    "\u17B9": ("ə",  "ɨ"),    # ឹ
    "\u17BA": ("əɨ", "ɨː"),   # ឺ  (S1 is a distinct centralized diphthong, not /əj/)
    "\u17BB": ("o",  "u"),    # ុ
    "\u17BC": ("ou", "uː"),   # ូ
    "\u17BD": ("uə", "uə"),   # ួ  (neutral — same realization in both series)
    "\u17BE": ("aə", "əː"),   # ើ
    "\u17BF": ("ɨə", "ɨə"),   # ឿ  (high-central diphthong, contrasts with ៀ = /iə/)
    "\u17C0": ("iə", "iə"),   # ៀ
    "\u17C1": ("e",  "eː"),   # េ
    "\u17C2": ("ae", "ɛː"),   # ែ
    "\u17C3": ("aj", "ej"),   # ៃ
    "\u17C4": ("ao", "oː"),   # ោ
    "\u17C5": ("aw", "ɨw"),   # ៅ
    # Vowel+final combos encoded as single codepoints
    "\u17C6": ("om", "um"),   # ំ (nikahit)
    "\u17C7": ("ah", "eəh"),  # ះ
    "\u17C8": ("aʔ", "eəʔ"),  # ៈ
}

# --- Independent vowels ------------------------------------------------------
INDEPENDENT_VOWELS = {
    "\u17A5": "ʔa",     # ឥ
    "\u17A6": "ʔəj",    # ឦ
    "\u17A7": "ʔo",     # ឧ
    "\u17A9": "ʔou",    # ឩ
    "\u17AB": "ʔrɨ",    # ឫ
    "\u17AC": "ʔrɨː",   # ឬ
    "\u17AD": "ʔlɨ",    # ឭ
    "\u17AE": "ʔlɨː",   # ឮ
    "\u17AF": "ʔae",    # ឯ
    "\u17B0": "ʔaj",    # ឰ
    "\u17B1": "ʔao",    # ឱ
    "\u17B2": "ʔao",    # ឲ
    "\u17B3": "ʔaw",    # ឳ
}

# --- Diacritics (series shifters, etc.) --------------------------------------
MUUSIKATOAN = "\u17C9"   # ៉ — shifts series 2 → 1
TRIISAP = "\u17CA"       # ៊ — shifts series 1 → 2
BANTOC = "\u17CB"        # ់ — shortens vowel
ROBAT = "\u17CC"         # ៌ — silent r (historical)
TOANDAKHIAT = "\u17CD"   # ៍ — silences final consonant
KAKABAT = "\u17CE"       # ៎
AHSDA = "\u17CF"         # ៏
SAMYOK_SANNYA = "\u17D0" # ័
VIRIAM = "\u17D1"        # ៑ — kills vowel

# --- Digits ------------------------------------------------------------------
KHMER_DIGITS = {
    "\u17E0": "0", "\u17E1": "1", "\u17E2": "2", "\u17E3": "3", "\u17E4": "4",
    "\u17E5": "5", "\u17E6": "6", "\u17E7": "7", "\u17E8": "8", "\u17E9": "9",
}

# --- Final consonant simplification -----------------------------------------
# Khmer final consonants undergo heavy neutralization.
FINAL_CONSONANT_IPA = {
    "ក": "k", "ខ": "k", "គ": "k", "ឃ": "k", "ង": "ŋ",
    "ច": "c", "ឆ": "c", "ជ": "c", "ឈ": "c", "ញ": "ɲ",
    "ដ": "t", "ឋ": "t", "ឌ": "t", "ឍ": "t", "ណ": "n",
    "ត": "t", "ថ": "t", "ទ": "t", "ធ": "t", "ន": "n",
    "ប": "p", "ផ": "p", "ព": "p", "ភ": "p", "ម": "m",
    "យ": "j", "រ": "",  "ល": "l", "វ": "w",
    "ស": "h", "ហ": "h", "ឡ": "l", "អ": "ʔ",
}

# --- Inherent vowel (when a consonant has no explicit vowel) -----------------
# Series 1 inherent vowel ≈ "ɑː" (often realized short before a final)
# Series 2 inherent vowel ≈ "ɔː"
INHERENT_VOWEL = {1: "ɑː", 2: "ɔː"}
# Short variant when a final consonant follows
INHERENT_VOWEL_SHORT = {1: "ɑ", 2: "ɔ"}


def all_consonants() -> set:
    """Return the set of all Khmer base consonant characters."""
    return set(CONSONANT_SERIES.keys())


def series_of(consonant: str) -> int:
    """Return 1 or 2 for a base consonant; default to 1 for unknown."""
    return CONSONANT_SERIES.get(consonant, 1)
