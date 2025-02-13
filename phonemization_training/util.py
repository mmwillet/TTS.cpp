from copy import deepcopy
import re

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
BASIC_ENGLISH_PHONEMES = {
    'b': ['b', 'bb'],
    'd': ['d', 'dd', 'ed'],
    'f': ['f', 'ff', 'ph', 'gh', 'lf', 'ft'],
    'g': ['g', 'gg', 'gh', 'gu', 'gue'],
    'h': ['h', 'wh'],
    'dʒ': ['j', 'ge', 'g', 'dge', 'di', 'gg'],
    'k': ['k', 'c', 'ch', 'cc', 'lk', 'qu', 'q', 'ck', 'x'],
    'l': ['l', 'll'],
    'm': ['m', 'mm', 'mb', 'mn', 'lm'],
    'n': ['n', 'nn', 'kn', 'gn', 'pn', 'mn'],
    'p': ['p', 'pp'],
    'r': ['r', 'rr', 'wr', 'rh'],
    'ɹ': ['r', 'rr', 'ar', 'rh'],
    's': ['s', 'ss', 'c', 'sc', 'ps', 'st', 'ce', 'se'],
    't': ['t', 'tt', 'th', 'ed'],
    'v': ['v', 'f', 'ph', 've'],
    'w': ['w', 'wh', 'u', 'o'],
    'ɡ': ['g', 'gg', 'gh', 'gu', 'gue'],
    'z': ['z', 'zz', 's', 'ss', 'x', 'ze', 'se'],
    'ʒ': ['s', 'si', 'z'],
    'tʃ': ['ch', 'tch', 'tu', 'ti', 'te'],
    'ʃ': ['sh', 'ce', 's', 'ci', 'si', 'ch', 'sci', 'ti', "sch"],
    'θ': ['th'],
    'ɾ': ['r', 'rr', 'rt', 't'],
    'ɛ': ['e'],
    'ᵻ': ['i', 'e'],
    'ð': ['th'],
    'ŋ': ['ng', 'n', 'ngue'],
    'j': ['y', 'i', 'j'],
    'æ': ['a', 'ai', 'au'],
    'ksɬ': ['x'],
    'ɬ': ['ll', 'l'],
    'eɪ': ['a', 'ai', 'eigh', 'aigh', 'ay', 'er', 'et', 'ei', 'au', 'a_e', 'ea', 'ey'],
    'e': ['e', 'ea', 'u', 'ie', 'ai', 'a', 'eo', 'ei', 'ae'],
    'i:': ['e', 'ee', 'ea', 'y', 'ey', 'oe', 'ie', 'i', 'ei', 'eo', 'ay', 'ae'],
    'ɪ': ['i', 'e', 'o', 'u', 'ui', 'y', 'ie'],
    'aɪ': ['i', 'y', 'igh', 'ie', 'uy', 'ye', 'ai', 'is', 'eigh', 'ie'],
    'ɒ': ['a', 'ho', 'au', 'aw', 'ough'],
    'ɐ': ['a', 'ah', 'e', 'aw', 'augh'],
    'x': ['ha', 'ch', 'h', 'cht'],
    'oʊ': ['o', 'oa', 'o_e', 'oe', 'ow', 'ough', 'eau', 'oo', 'ew'],
    'ʊ': ['o', 'oo', 'u', 'ou'],
    'ʌ': ['u', 'o', 'oo', 'ou'],
    'u:': ['o', 'oo', 'ew', 'ue', 'u_e', 'oe', 'ough', 'ui', 'oew', 'ou'],
    'ɔɪ': ['oi', 'oy', 'uoy'], 'aʊ': ['ow', 'ou', 'ough'],
    'ə': ['a', 'er', 'i', 'ar', 'our', 'ur'],
    'eɚ': ['air', 'are', 'ear', 'ere', 'eir', 'ayer'],
    'ɑ:': ['a'],
    'ɜ:': ['ir', 'er', 'ur', 'ear', 'or', 'our', 'yr', 'r'],
    'ɔ:': ['aw', 'a', 'or', 'oor', 'ore', 'oar', 'our', 'augh', 'ar', 'ough', 'au'],
    'ɪɚ': ['ear', 'eer', 'ere', 'ier'],
    'ʊə': ['ure', 'ou', 'oo', 'ur'],
    'ç': ['sh', 'ch', 'she', 'che', 'sha', 'cha'],
    'ɚ': ['er', 'ur', 'ir', 'ere', 'eir', 'ayer', 'eer', 'ear', 'ier'],
}


def get_phonemes_by_grapheme(data):
    pbyg = {}
    for phoneme, graphemes in data.items():
        for grapheme in graphemes:
            if grapheme not in pbyg:
                pbyg[grapheme] = []
            pbyg[grapheme].append(phoneme)
    return pbyg

BASIC_ENGLISH_GRAPHEMES = get_phonemes_by_grapheme(BASIC_ENGLISH_PHONEMES)


def get_all_possible_phonetic_components(part):
    data = []
    _get_all_possible_phonetic_components(part, data)
    return data


def _get_all_possible_phonetic_components(part, output_list, current_list = None):
    if current_list is None:
        current_list = []
    if len(part) == 0:
        output_list.append(deepcopy(current_list))
        return
    possibles = get_all_next_graphemes(part)
    if len(possibles) < 1:
        return
    for possible in possibles:
        current_list.append(possible)
        _get_all_possible_phonetic_components(part[len(possible):], output_list, current_list)
        current_list.pop()


def get_all_next_graphemes(part):
    possibles = []
    for grapheme in BASIC_ENGLISH_GRAPHEMES:
        if is_prefix(grapheme, part):
            possibles.append(grapheme)
    return possibles


def build_word_regexes(word):
    regexes = []
    last_regexes = []
    for parts in get_all_possible_phonetic_components(word):
        regex_string = "^[ˌˈʔ]?"
        for part in parts:
            regex_string += "(" +"|".join([phoneme for phoneme in BASIC_ENGLISH_GRAPHEMES[part]]) + ")[ˌˈʔ]?"
        regex_string += "$"
        regexes.append(re.compile(regex_string))
        last_match = "(" +"|".join([phoneme for phoneme in BASIC_ENGLISH_GRAPHEMES[parts[-1]]]) + ")[ˌˈʔ]?"
        last_regexes.append(re.compile(last_match + "$"))
    return regexes, last_regexes


def one_or_more_match(regexes, word):
    for regex in regexes:
        if regex.match(word):
            return True
    return False


def is_prefix(w1, w2):
    if w1 is None or w2 is None:
        return False
    w1 = w1.replace("ˌ", "ˈ")
    w2 = w2.replace("ˌ", "ˈ")
    if w1 == "":
        return True
    if len(w1) > len(w2):
        return False
    return w2[:len(w1)] == w1


def is_suffix(w1, w2):
    if w1 is None or w2 is None:
        return False
    w1 = w1.replace("ˌ", "ˈ")
    w2 = w2.replace("ˌ", "ˈ")
    if w1 == "":
        return True
    if len(w1) > len(w2):
        return False
    return w2[len(w1)*-1:] == w1


def get_largest_shared_prefix(*words):
    if len(words) == 0 or None in words:
        return ""
    smallest_size = min([len(word) for word in words])
    for i in range(smallest_size):
        if any([words[0][:i+1] != word[:i+1] for word in words]):
            return words[0][:i]
    return words[0][:smallest_size]


def get_largest_shared_suffix(*words):
    if len(words) == 0 or None in words:
        return ""
    smallest_size = min([len(word) for word in words])
    for i in range(smallest_size):
        if any([words[0][-1*(i+1):] != word[-1*(i+1)] for word in words]):
            return words[0][len(words[0])-i:]
    return words[0][-1*smallest_size:]


def overlap_between(w1, w2):
    if w1 == None or w2 == None:
        return ""
    if w1 == "" or w2 == "":
        return ""
    if w1 == w2:
        return w1
    for i in range(len(w2)):
        if is_suffix(w2[:len(w2)-i], w1):
            return w2[:len(w2)-i]
    return ""


def get_possible_prefixes(word):
    for i in range(len(word)):
        yield word[:len(word)-i]
