from nltk.corpus import words
from os.path import isfile, join
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from util import *
import gguf
import gzip
import json
import logging
import nltk
import phonemizer
import re

ACRONYM_REGEX = re.compile(r'(\w\.|[A-Z]{2,}|\d+)+')
WORD_BREAKS_REGEX = re.compile(r'\'’ -/')


class PhonemizationTrainer:
    """
    This is a tool to iteratively compile a set of rules for converting graphemes into english IPA phonemes based on a
    set of known words (specifically english words that exist in the nltk corpus) and their respective phonemes. It
    does not use any complicated statistical analysis, but rather attempts to formulaically determine simple look forward
    and backward rules for phoneme generation.

    Parameters
    ----------
    dir : str
        The directory path in which the tokenizer and rule set will be persisted
    espeak_path : str
        The local path of the espeak-ng dylib.

    Examples
    --------
    The following will iteratively train the rule set. See #train for more detail on specific customization

    >>> trainer = PhonemizationTrainer(save_directory="/some/existing/directory")
    >>> trainer.train()
    """
    def __init__(self, save_directory=".", espeak_path='/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.1.dylib'):
        EspeakWrapper.set_library(espeak_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        output_handler = logging.StreamHandler()
        self.logger.addHandler(output_handler)
        self.dir = save_directory
        self.rule_set = RuleSet()
        self.tokenizer = SimpleTokenizer()
        self.phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        self.load()

    @property
    def tokenizer_path(self):
        return join(self.dir, "tokenizer.json")

    @property
    def rules_path(self):
        return join(self.dir, "rules.json")

    def get_unpronounceable_from_dictionary(self, dictionary):
        # there are a lot of words in some of these lists that can effectively be left out as they are common words
        # parsable via the ruleset. E.G. the word 'Island' in 'Norfolk Island' doesn't need to be stored in our list.
        for word_type, words_list in dictionary.items():
            if word_type in ["brands", "countries", "cities", "us_cities"]:
                for word in words_list:
                    for part in re.split(r'[ -&]', word):
                        if part != "" and ACRONYM_REGEX.search(part) is None:
                            actual_phoneme = self.phonemizer.phonemize([part])[0].strip()
                            try:
                                if actual_phoneme != self.phonemize_word(re.sub(r'[àá]', "a", re.sub(r'[èêé]', "e", part.lower())).replace("ç", "c")):
                                    yield part.lower(), actual_phoneme
                            except:
                                yield word.lower(), actual_phoneme
            elif word_type == "abbr":
                for word, translated_word in words_list.items():
                    yield word, self.phonemizer.phonemize([translated_word])[0].strip()
            else:
                for word in words_list:
                    if len(word) == 0:
                        continue
                    if word_type == "non_english" and word[-1] in ".!?":
                        # many of these are common expressions which often contain punctuation at the end
                        # make sure to trim that off
                        word = word[:-1]
                    actual_phoneme = self.phonemizer.phonemize([word])[0].strip()
                    try:
                        attempted_phoneme = " ".join(
                            [
                                self.phonemize_word(re.sub(r'[àá]', "a", re.sub(r'[èêé]', "e", part.lower())).replace("ç", "c").replace("’", ""))
                                for part in word.split(" ")
                            ]
                        )
                    except:
                        yield word.lower(), actual_phoneme
                        continue
                    if actual_phoneme != attempted_phoneme:
                        yield word.lower(), actual_phoneme
        # make sure to also yield all words that cannot be pronounced from our english dictionary
        count = 0
        total = 0
        for word in words.words():
            total += 1
            word = word.lower().strip()
            if " " in word or "-" in word:
                continue
            phoneme = self.phonemizer.phonemize([word])[0].strip()
            success, _ = self.attempt_word(word, phoneme)
            if not success:
                count += 1
                yield word, phoneme
        self.logger.info(f"Added {count} words ({(count / total)*100.0}% of all known words) not supported by the rule set.")

    def export_to_gguf(self, path, dictionary_path=None, stress_and_composites_path=None):
        stressed_and_composites = None
        if stress_and_composites_path is not None:
            with open(stress_and_composites_path, "r+") as f:
                stressed_and_composites = json.load(f);
        gguf_writer = gguf.GGUFWriter(path=None, arch="tts-phonemizer")
        self.logger.info("Distilling trained rules.")
        distilled_rules = {k: v for k, v in self.rule_set.distill_rules()}
        self.logger.info("Saving rules to gguf file.")
        gguf_writer.add_uint32("phonemizer.type", 0) # enum type for tts-phonemizer
        gguf_writer.add_array("phonemizer.graphemes", list(self.tokenizer.frequent_groups))
        gguf_writer.add_array("phonemizer.rules.keys", list(distilled_rules.keys()))
        gguf_writer.add_array("phonemizer.rules.phonemes", list(distilled_rules.values()))
        if dictionary_path is not None:
            self.logger.info(f"Opening dictionary at path '{dictionary_path}'.")
            with gzip.open(dictionary_path, "rb") as f:
                dictionary = json.load(f)
                out_dict = {}
                self.logger.info(f"Finding excpetions in dictionary.")
                for word, phoneme in self.get_unpronounceable_from_dictionary(dictionary):
                    dict_key = word
                    found_sub_key = False
                    if WORD_BREAKS_REGEX.search(word) is not None:
                        found_sub_key = True
                        dict_key = re.split(r'[\'’ -/]', word)[0]
                    dict_val = phoneme
                    if found_sub_key:
                        dict_val = f"{dict_val}:{key[len(dict_key):]}"
                    if dict_key not in out_dict:
                        out_dict[dict_key] = [dict_val]
                    elif dict_val not in out_dict[dict_key]:
                        out_dict[dict_key].append(dict_val)
                if stressed_and_composites is not None:
                    for word in stressed_and_composites:
                        found_sub_key = " " in word
                        expected = self.phonemizer.phonemize([word])[0].strip()
                        # determine the stress changes when the word is not the start or end of the clause.
                        e_s_word = self.phonemizer.phonemize(["blah " + word + " blah"])[0].strip()[6:-6]
                        s_word = self.phonemizer.phonemize(["blah " + word])[0].strip()[6:]
                        e_word = self.phonemizer.phonemize([word + " blah"])[0].strip()[:-6]
                        dict_key = word
                        if found_sub_key:
                            dict_key = word.split(" ")[0]              
                        if e_s_word != s_word and e_s_word != e_word and e_s_word != expected:
                            eskey = "#"+dict_key+"#"
                            val = e_s_word if not found_sub_key else f"{e_s_word}:{word[len(dict_key):]}"
                            if eskey not in out_dict:
                                out_dict[eskey] = [val]
                            else:
                                out_dict[eskey].append(val)
                        if s_word != expected:
                            skey = "#" + dict_key
                            val = s_word if not found_sub_key else f"{s_word}:{word[len(dict_key):]}"
                            if skey not in out_dict:
                                out_dict[skey] = [val]
                            else:
                                out_dict[skey].append(val)
                        if e_word != expected:
                            ekey = dict_key + "#"
                            val = e_word if not found_sub_key else f"{e_word}:{word[len(dict_key):]}"
                            if ekey not in out_dict:
                                out_dict[ekey] = [val]
                            else:
                                out_dict[ekey].append(val)
                        if found_sub_key:
                            # python dictionaries respect the order of added items such that these keys are always after stressed cases. 
                            # This is actually important to functionality here as end stressed and forward stressed cases need to be checked first
                            # via the phonemizer.
                            if dict_key not in out_dict:
                                out_dict[dict_key] = [f"{expected}:{word[len(dict_key):]}"]
                            else:
                                out_dict[dict_key].append(f"{expected}:{word[len(dict_key):]}")
                self.logger.info(f"Saving dictionary exceptions to gguf file.")
                gguf_writer.add_array("phonemizer.dictionary.keys", list(out_dict.keys()))
                gguf_writer.add_array("phonemizer.dictionary.values", [",".join(vals) for vals in out_dict.values()])
        gguf_writer.write_header_to_file(path=path)
        self.logger.info(f"Writing gguf data to disk.")
        gguf_writer.write_kv_data_to_file()
        gguf_writer.close()

    def save(self):
        self.tokenizer.save(self.tokenizer_path)
        self.rule_set.save(self.rules_path)

    def load(self):
        if isfile(self.tokenizer_path):
            self.tokenizer.load(self.tokenizer_path)
            if isfile(self.rules_path):
                self.rule_set = RuleSet.from_file(self.rules_path)

    def phonemize_word(self, word):
        """
        Parameters
        ----------
        word : str
            An unhyphenated single word for which to build a phoneme.

        Returns
        ----------
        str
            the IPA phoneme as determined by the trained tokenizer and rule set.
        """
        word = word.lower().strip()
        assert " " not in word and "-" not in word, f"An single unhyphenated word must be passed. {word} is invalid."
        chunks = self.tokenizer.tokenize(word)
        built = ""
        for i, part in enumerate(chunks):
            before = "^" if i == 0 else chunks[i-1]
            after = "$" if i + 1 >= len(chunks) else chunks[i+1]
            ph, _ = self.rule_set.get_phoneme(part, before, after, word)
            built += ph
        return built

    def train(self, stop_threshold=0.999, persist=True, initial_steps=5, plateau_granularity=0.001,
              plateau_count_to_allow_unique_cases=5, plateau_count_to_train_unique_cases=7):
        """
        End to end Training protocol. First it trains the tokenizer, then it iteratively trains the rule set. The rule
        set is effectively trained by picking the most common phoneme to grapheme relationships seen over the entire
        corpus of words with a preference for those phonemes that match the phonetic expectations of each grapheme
        (as defined by BASIC_ENGLISH_PHONEMES). Typically, this results in incomplete coverage of the entire corpus, and,
        thus, upon failing to improve coverage, mistakes are collected and word specific corrections are applied that
        seek to address issues with overlapping phonemes, overfitted phoneme predictions, and word specific pronunciations.

        Parameters
        ----------
        stop_threshold : float
            The coverage rate between 0.0 and 1.0 at which to stop training. defaults to 1.0
        persist : bool
            Whether to persist trained rule set and tokenizer
        initial_steps : int
            the number of initial training iterations to perform before any corrections
        plateau_granularity : float
            the granularity at which to determine that a plateau has been hit
        plateau_count_to_allow_unique_cases : int
            the number of plateaus before allowing for word specific rules to be added in corrections
        plateau_count_to_train_unique_cases : int
            the number of plateaus before allowing for word specific rules to be learned
        """
        try:
            # make sure that we download the word's corpus
            for _ in words.words():
                break
        except:
            nltk.download("words")
        if not self.tokenizer.trained:
            self.logger.info("Training the tokenizer on known words.")
            self.tokenizer.train()
            self.logger.info("Tokenizer training complete.")
        rate = 0.0
        best_rate = 0.0
        plateau_count = 0
        counter = 0
        errors = None
        while rate < stop_threshold:
            rate = 0.0
            last_rate = -1.0
            while rate > last_rate or counter < initial_steps:
                counter += 1
                last_rate = rate
                rate, errors = self.train_chunk()
                self.logger.info(f"At iteration {counter}, with success rate of {rate}.")
                self.rule_set.coalesce_count_data_to_rules()
                self.logger.info(f"Finished iteration {counter}.")
            if rate < stop_threshold:
                initial_error_length = len(errors)
                if rate - best_rate  < plateau_granularity:
                    plateau_count += 1
                if rate > best_rate:
                    if persist:
                        self.save()
                    best_rate = rate
                if plateau_count > plateau_count_to_allow_unique_cases:
                    self.rule_set.allow_unique_cases = True
                if plateau_count > plateau_count_to_train_unique_cases:
                    self.rule_set.learn_unique_cases = True
                after_corrections_count = len(self.fix_errors(errors))
                self.rule_set.clear_count_data()
                self.logger.info(f"Found {initial_error_length-after_corrections_count} of {initial_error_length} fixes to determinable errors.")

    def train_chunk(self):
        total = 0
        completed = 0
        errors = []
        for word in words.words():
            word = word.lower().strip()
            if " " in word or "-" in word:
                continue
            total += 1
            phoneme = self.phonemizer.phonemize([word])[0].strip()
            success, error = self.attempt_word(word, phoneme)
            if success:
                completed += 1
            elif error is not None:
                errors.append(error)
        return completed / total, errors

    def fix_errors(self, errors):
        remaining_errors = []
        corrected_steps = set([])
        for i, (word, current, before, after, built, predicted, remaining) in enumerate(errors):
            if self.rule_set.allow_unique_cases and after == "$":
                self.rule_set.update_rule_at(current, before, after, word, remaining)
                corrected_steps.add(",".join([current, before, after]))
                continue
            steps = self.get_phoneme_steps(word)
            # generally the cause of errors is either that an earlier phoneme is too large or there isn't enough
            # distinguishing information in particular cases.
            built = built.replace("ˌ", "ˈ")
            predicted = predicted.replace("ˌ", "ˈ")
            overlap = overlap_between(built, predicted)
            if len(built) > 0 and (predicted in built or len(overlap) > 1):
                if len(overlap) > 1 and len(overlap) < len(predicted):
                    predicted = overlap
                bsplit = built.split(predicted)
                before_chunk = bsplit[-2]
                if len(before_chunk) > 0:
                    if predicted is None:
                        remaining_errors.append((word, current, before, after, built, predicted, remaining))
                        continue
                    if not self.correct_overflow_case(word, current, before, after, len(bsplit) - 2 , before_chunk, predicted, corrected_steps):
                        if len(overlap) == 0 or not self.correct_overlap_with_defaults(current, before, after, overlap, steps, corrected_steps):
                            if not self.attempt_phonetic_correction(steps, current, before, after, built, remaining, corrected_steps):
                                remaining_errors.append((word, current, before, after, built, predicted, remaining))
            elif len(overlap) > 0:
                if not self.correct_overlap_with_defaults(current, before, after, predicted, steps, corrected_steps):
                    if not self.attempt_phonetic_correction(steps, current, before, after, built, remaining, corrected_steps):
                        remaining_errors.append((word, current, before, after, built, predicted, remaining))
            elif not self.attempt_phonetic_correction(steps, current, before, after, built, remaining, corrected_steps):
                remaining_errors.append((word, current, before, after, built, predicted, remaining))
        existing_errors = []
        for (word, current, before, after, built, predicted, remaining) in remaining_errors:
            if before != "^" and len(current) == 1 or len(before) == 1 and len(before+current) <= 3:
                self.tokenizer.add_token(before + current)
            elif after != "$" and len(current) == 1 or len(after) == 1 and len(current+after) <= 3:
                self.tokenizer.add_token(current + after)
            else:
                existing_errors.append((word, current, before, after, built, predicted, remaining))
                self.rule_set.remove_rule_at(current, before, after)
        return existing_errors

    def get_phoneme_steps(self, word):
        chunks = self.tokenizer.tokenize(word)
        steps = []
        for i, current in enumerate(chunks):
            before = chunks[i - 1] if i - 1 >= 0 else "^"
            after = chunks[i + 1] if i + 1 < len(chunks) else "$"
            phoneme, used = self.rule_set.get_phoneme(current, before, after, word)
            steps.append([current, before, after, phoneme, word if used else None])
        return steps

    def correct_overlap_with_defaults(self, current, before, after, overlap, steps, corrected_steps):
        """
            It is possible in nonphonetic cases for overflow to exist such that there is no space for the appropriate phoneme
            and for the overlap to be incompatible with simple splitting.

            E.G. imagine we have letters 1234 that spell the pronounced word abcd, if the grapheme 1 occupies ab, grapheme 2
            occupies c, and grapheme 3 occupies d then there will be no remaining phoneme parts for the letter d.

            In order to fix this issue we have to traverse backwards over our generated word to find the most likely culprit
            which in our example would be the phoneme 1.
        """
        relevant_steps = []
        for step in steps:
            if step[3] is None or ",".join([step[0], step[1], step[2]]) in corrected_steps:
                return True
            if step[0] == current and step[1] == before and step[2] == after:
                break
            relevant_steps.append(step)
        last = current
        cleaned_step = None
        for step in relevant_steps[::-1]:
            if len(step[3]) > len(overlap):
                # first see if we can chunk off the last items default
                if self.rule_set.in_rules(last, "$default"):
                    default_val = self.rule_set.get_phoneme(last, "$default", "")[0]
                    default_overlap = overlap_between(step[3], default_val or "")
                    if len(default_overlap) > 0 and len(default_overlap) >= len(overlap) and len(default_overlap) < len(step[3]):
                        cleaned_step = step
                        self.rule_set.update_rule_at(step[0], step[1], step[2], step[4], step[3][:len(default_overlap)])
                        corrected_steps.add(",".join([step[0], step[1], step[2]]))
                        break
                    elif self.rule_set.get_phoneme(last, "$default", "")[0] in step[3]:
                        chunks = step[3].split(self.rule_set.get_phoneme(last, "$default", "")[0])
                        if len(chunks[-2]) > 0:
                            cleaned_step = step
                            self.rule_set.update_rule_at(step[0], step[1], step[2], step[4], chunks[-2])
                            break
                if self.rule_set.in_rules(step[0], "$default") and self.rule_set.get_phoneme(step[0], "$default", "")[0] in step[3]:
                    # if the default for the current step is a stem of its predicted try to trim to that point.
                    chunks = step[3].split(self.rule_set.get_phoneme(step[0], "$default", "")[0])
                    if len(chunks[-1]) >= len(overlap):
                        cleaned_step = step
                        self.rule_set.update_rule_at(step[0], step[1], step[2], step[4], chunks[-2]+ self.rule_set.get_phoneme(step[0], "$default", "")[0])
                        corrected_steps.add(",".join([step[0], step[1], step[2]]))
                        break
            last = step[0]
        if cleaned_step is None:
            return False
        # finally clean up all rules after the corrected step
        should_delete = False
        for step in steps:
            if step == cleaned_step:
                should_delete = True
            elif should_delete:
                self.rule_set.remove_rule_at(step[0], step[1], step[2], step[4])
                corrected_steps.add(",".join([step[0], step[1], step[2]]))
            if step[0] == current and step[1] == before and step[2] == after:
                break
        return True

    def all_predictions(self, word):
        predictions = []
        if word not in self.rule_set.rule_map:
            return predictions
        for adata in self.rule_set.rule_map[word].values():
            if isinstance(adata, str):
                predictions.append(adata)
                continue
            for phoneme in adata.values():
                if isinstance(phoneme, str):
                    predictions.append(phoneme)
        return predictions

    def attempt_phonetic_correction(self, steps, current, before, after, built, remaining, corrected_steps):
        """
            The most common errors that we see after reaching a high rate of success tend to be overlapping
            or off by one or two character cases in which one of more predicted phoneme is too large or small
            this can sometime be corrected by preferring alternative phonetic options in our decision tree.
        """
        relevant_steps = []
        for step in steps:
            if ",".join([step[0], step[1], step[2]]) in corrected_steps:
                return True
            if step[0] == current and step[1] == before and step[2] == after:
                relevant_steps.append(step)
                break
            relevant_steps.append(step)
        corrected = False
        for i, step in enumerate(relevant_steps[::-1]):
            erroring_step = i == 0
            predicted = step[3]
            if predicted is None:
                return True
            regexes, last_regexes = build_word_regexes(step[0])
            overlap = overlap_between(built, predicted)
            if len(overlap) > 0:
                pmatch = one_or_more_match(regexes, predicted)
                plmatch = one_or_more_match(last_regexes, predicted)
                if erroring_step:
                    match = one_or_more_match(regexes, predicted[len(overlap):])
                    lmatch = one_or_more_match(last_regexes, predicted[len(overlap):])
                else:
                    match = one_or_more_match(regexes, overlap)
                    lmatch = one_or_more_match(last_regexes, overlap)
                pdist = abs(len(step[0]) - len(predicted))
                dist = abs(len(step[0]) - len(predicted[len(overlap):])) if erroring_step else abs(len(step[0]) - len(overlap))
                predicted_preferred = (pmatch and not match) or (not pmatch and not match and plmatch and not lmatch) or \
                                       (not pmatch and not match and not plmatch and not lmatch and pdist <= dist)
                if not predicted_preferred:
                    if erroring_step:
                        self.rule_set.update_rule_at(step[0], step[1], step[2], step[4], predicted[len(overlap):])
                    else:
                        self.rule_set.update_rule_at(step[0], step[1], step[2], step[4], overlap)
                    corrected_steps.add(",".join([step[0], step[1], step[2]]))
                    corrected = True
                    break
                else:
                    remaining = built[len(built)-len(overlap):] + remaining
                    built = built[:len(built)-len(overlap)]
                    continue
            if erroring_step:
                biggest_stem = get_largest_shared_prefix(predicted, remaining)
                if len(biggest_stem) > 0 and (one_or_more_match(regexes, biggest_stem) or (not one_or_more_match(regexes, predicted) and one_or_more_match(last_regexes, biggest_stem))):
                    self.rule_set.update_rule_at(step[0], step[1], step[2], step[4], biggest_stem)
                    corrected_steps.add(",".join([step[0], step[1], step[2]]))
                    corrected = True
                    break
                elif predicted in built and predicted not in remaining and one_or_more_match(regexes, predicted):
                    split = built.split(predicted)
                    built = predicted.join(split[:-1])
                    remaining = predicted + split[-1] + remaining
                elif predicted in remaining and one_or_more_match(regexes, predicted):
                    split = remaining.split(predicted)
                    built = built + split[0]
                    remaining = predicted + split[1]
                else:
                    break
            else:
                if is_suffix(predicted, built):
                    corrected = True
                    break
                possibilities = [
                    possible for possible in self.all_predictions(step[0])
                    if len(possible) > 0 and is_suffix(possible, built) and (step[1] == "^" or len(possible) < len(built))
                ]
                should_match_l = False
                should_fully_match = False
                closest_score = 100
                best_fit = None
                for possible in possibilities:
                    lmatch = one_or_more_match(last_regexes, possible)
                    match = one_or_more_match(regexes, possible)
                    dist = abs(len(step[0]) - len(possible))
                    if match and not should_fully_match:
                        best_fit = possible
                        closest_score = dist
                        should_fully_match = True
                        should_match_l = True
                        continue
                    elif lmatch and not should_match_l:
                        best_fit = possible
                        should_match_l = True
                        continue
                    if should_fully_match and not match:
                        continue
                    elif should_match_l and not lmatch:
                        continue
                    if dist < closest_score:
                        closest_score = dist
                        best_fit = possible
                        continue
                if best_fit is None:
                    self.rule_set.remove_rule_at(step[0], step[1], step[2], step[4])
                    corrected_steps.add(",".join([step[0], step[1], step[2]]))
                    break
                else:
                    self.rule_set.update_rule_at(step[0], step[1], step[2], step[4], best_fit)
                    corrected = True
                    corrected_steps.add(",".join([step[0], step[1], step[2]]))
                    remaining = built[len(built)-len(best_fit):] + remaining
                    built = built[:len(built)-len(best_fit)]
        return corrected

    def correct_overflow_case(self, word, lpart, lbefore, lafter, repeat_count, goal, predicted, corrected_steps):
        """
            Because our core approach learns phonemes initially from the start of each word and tends to prefer longer
            phonemes to shorter ones, the most common error correction we need to apply is to shorten predicted phonemes
            which stand in for multiple tokens. This can be detected in error cases when there is overlap between the
            generated phoneme string and the predicted phoneme.
        """
        chunks = self.tokenizer.tokenize(word)
        fixable_chunks = []
        for i, current in enumerate(chunks):
            before = chunks[i-1] if i-1 > 0 else "^"
            after = chunks[i+1] if i+1 < len(chunks) else "$"
            if current == lpart and before == lbefore and after == lafter:
                break
            fixable_chunks.append(current)
        built = ""
        remaining = goal
        found_issue = False
        i = 0
        added_chunks = []
        corrected_for_multi_step_overflow = False
        while i < len(fixable_chunks):
            current = chunks[i]
            before = chunks[i - 1] if i - 1 > 0 else "^"
            after = chunks[i + 1] if i + 1 < len(chunks) else "$"
            if not found_issue:
                correctable_phoneme, used = self.rule_set.get_phoneme(current, before, after, word)
                if correctable_phoneme is None or ",".join([current, before, after]) in corrected_steps:
                    return True
                if (i + 1) == len(fixable_chunks) and is_suffix(predicted, correctable_phoneme) and len(correctable_phoneme) < len(predicted):
                    if repeat_count > 0:
                        repeat_count -= 1
                    else:
                        correctable_phoneme = correctable_phoneme[:len(correctable_phoneme)-len(predicted)]
                        self.rule_set.update_rule_at(current, before, after, word if used else None, correctable_phoneme)
                        corrected_steps.add(",".join([current, before, after]))
                        found_issue = True
                elif not is_prefix(predicted, correctable_phoneme) and predicted in correctable_phoneme:
                    if repeat_count > 0:
                        repeat_count -= 1
                    else:
                        cpsplit = correctable_phoneme.split(predicted)
                        correctable_phoneme = cpsplit[-2]
                        remaining = cpsplit[-1]
                        if len(remaining) >= len(fixable_chunks) - (i + 1):
                            self.rule_set.update_rule_at(current, before, after, word if used else None, correctable_phoneme)
                            corrected_steps.add(",".join([current, before, after]))
                            found_issue = True
                if built != "" and not found_issue and not corrected_for_multi_step_overflow:
                    overlap_before = overlap_between(built, predicted)
                    overlap_after = overlap_between(predicted, correctable_phoneme)
                    if len(overlap_before) > 0 and len(overlap_after) > 0:
                        if repeat_count > 0:
                            repeat_count -= 1
                        else:
                            while(len(overlap_before) > 0):
                                predicted = overlap_before
                                i -= 1
                                built = built[:len(built)-len(added_chunks[-1])]
                                added_chunks = added_chunks[:-1]
                                overlap_before = overlap_between(built, predicted)
                            corrected_for_multi_step_overflow = True
                            continue
                added_chunks.append(correctable_phoneme)
                built += correctable_phoneme
                remaining = remaining[len(correctable_phoneme):]
            else:
                self.rule_set.remove_rule_at(current, before, after)
                corrected_steps.add(",".join([current, before, after]))
            i+=1
        return found_issue

    def attempt_word(self, word, phoneme):
        remaining = self.tokenizer.tokenize(word)
        remaining_phoneme = phoneme
        built_phoneme = ""
        before = "^"
        while len(remaining) > 0:
            current = remaining[0]
            after = remaining[1] if len(remaining) > 1 else "$"
            predicted, word_used = self.rule_set.get_phoneme(current, before, after, word)
            tpredicted = predicted
            if predicted is None:
                for possible_phoneme in get_possible_prefixes(remaining_phoneme[:len(remaining_phoneme) - (len(remaining[1:]))]):
                    # never resort to complicated rules when we don't yet have a value set
                    self.rule_set.increment_possibilities(word, current, before, after, possible_phoneme, False)
                return False, None
            if not is_prefix(predicted, remaining_phoneme):
                for possible_phoneme in get_possible_prefixes(remaining_phoneme[:len(remaining_phoneme) - (len(remaining[1:]))]):
                    self.rule_set.increment_possibilities(word, current, before, after, possible_phoneme, True)
                return False, (word, current, before, after, built_phoneme, predicted, remaining_phoneme)
            self.rule_set.increment_possibilities(word, current, before, after, tpredicted, word_used)
            remaining_phoneme = remaining_phoneme[len(predicted):]
            built_phoneme += predicted
            remaining = remaining[1:]
            before = current
        return True, None


class RuleSet:
    def __init__(self, rule_map=None, word_mapping=None, allow_unique_cases=False, learn_unique_cases = False):
        self.rule_map = rule_map if rule_map is not None else {}
        self.word_mapping = word_mapping if word_mapping is not None else {}
        self.count_data = {}
        self.allow_unique_cases = allow_unique_cases
        self.learn_unique_cases = learn_unique_cases

    @classmethod
    def from_json(cls, json):
        return cls(json[0], json[1], json[2], json[3])

    @classmethod
    def from_file(cls, file):
        with open(file, "r+") as f:
            return cls.from_json(json.load(f))

    def save(self, file):
        with open(file, "w+") as f:
            json.dump([self.rule_map, self.word_mapping, self.allow_unique_cases, self.learn_unique_cases], f)

    def clear_count_data(self):
        self.count_data = {}

    def update_rule_at(self, part, before, after, word, phoneme):
        if part not in self.rule_map:
            self.rule_map[part] = {}
        if before not in self.rule_map[part]:
            self.rule_map[part][before] = {}
        if word is None or word == "" or after not in self.rule_map[part][before]:
            self.rule_map[part][before][after] = phoneme
        else:
            if "$extensions" not in self.rule_map[part][before]:
                self.rule_map[part][before]["$extensions"] = {}
            if after not in self.rule_map[part][before]["$extensions"]:
                self.rule_map[part][before]["$extensions"][after] = {}
            self.rule_map[part][before]["$extensions"][after][word] = phoneme

    def remove_rule_at(self, part, before, after, word = None):
        if part not in self.rule_map:
            return
        elif before not in self.rule_map[part] and "$default" in self.rule_map[part]:
            del self.rule_map[part]["$default"]
        elif before in self.rule_map[part] and after not in self.rule_map[part][before] and "$default" in self.rule_map[part][before]:
            del self.rule_map[part][before]["$default"]
        elif word is not None and "$extensions" in self.rule_map[part][before] and after in self.rule_map[part][before]["$extensions"] and word in self.rule_map[part][before]["$extensions"][after]:
            del self.rule_map[part][before]["$extensions"][after][word]
        elif before in self.rule_map[part] and after in self.rule_map[part][before]:
            del self.rule_map[part][before][after]

    def in_rules(self, part, before=None, after=None, word=None):
        if part not in self.rule_map:
            return False
        if before is None:
            return True
        if before in self.rule_map[part]:
            if after is None:
                return True
            if after in self.rule_map[part][before]:
                if word is None:
                    return True
                return "$extensions" in self.rule_map[part][before] and after in self.rule_map[part][before]["$extensions"] and word in self.rule_map[part][before]["$extensions"][after]
        return False

    def get_phoneme(self, part, before, after, word=None):
        if part not in self.rule_map:
            return None, False
        if before not in self.rule_map[part]:
            if "$default" in self.rule_map[part]:
                return self.rule_map[part]["$default"], False
            return None, False
        if before == "$default":
            return self.rule_map[part]["$default"], False
        if after not in self.rule_map[part][before]:
            if "$default" in self.rule_map[part][before]:
                return self.rule_map[part][before]["$default"], False
            return None, False
        if after == "$default":
            return self.rule_map[part][before]["$default"], False
        if not self.allow_unique_cases or word is None or word == "" or "$extensions" not in self.rule_map[part][before] or after not in self.rule_map[part][before]["$extensions"] or word not in self.rule_map[part][before]["$extensions"][after]:
            return self.rule_map[part][before][after], False
        return self.rule_map[part][before]["$extensions"][after][word], True

    def increment_possibilities(self, word, part, before, after, phoneme, is_unique=False):
        if part not in self.count_data:
            self.count_data[part] = {}
        if before not in self.count_data[part]:
            self.count_data[part][before] = {}
        if after not in self.count_data[part][before]:
            self.count_data[part][before][after] = {}
        if phoneme not in self.count_data[part][before][after]:
            self.count_data[part][before][after][phoneme] = [set([]), 0]
        self.count_data[part][before][after][phoneme][1] += 1
        self.count_data[part][before][after][phoneme][0].add(word)
        if self.allow_unique_cases and is_unique and word is not None and word != "":
            if "$extensions" not in self.count_data[part][before][after]:
                self.count_data[part][before][after]["$extensions"] = {}
            if word not in self.count_data[part][before][after]["$extensions"]:
                self.count_data[part][before][after]["$extensions"][word] = {}
            if phoneme not in self.count_data[part][before][after]["$extensions"][word]:
                self.count_data[part][before][after]["$extensions"][word][phoneme] =  [set([]), 0]
            self.count_data[part][before][after]["$extensions"][word][phoneme][1] += 1
            self.count_data[part][before][after]["$extensions"][word][phoneme][0].add(word)
        if part not in self.word_mapping:
            self.word_mapping[part] = {}
        if before not in self.word_mapping[part]:
            self.word_mapping[part][before] = {}
        if after not in self.word_mapping[part][before]:
            self.word_mapping[part][before][after] = []
        if word not in self.word_mapping[part][before][after]:
            self.word_mapping[part][before][after].append(word)

    def words_at(self, part, before, after):
        if part not in self.word_mapping or before not in self.word_mapping[part] or after not in self.word_mapping[part][before]:
            return []
        return self.word_mapping[part][before][after]

    def determine_defaults(self):
        for part, rules_data in self.rule_map.items():
            overall_word_counts = {}
            for before, after_data in rules_data.items():
                if not isinstance(after_data, dict):
                    continue
                counts_by_phoneme_before = {}
                default_phoneme = None
                for after, phoneme in after_data.items():
                    if not isinstance(phoneme, str):
                        continue
                    if phoneme not in counts_by_phoneme_before:
                        counts_by_phoneme_before[phoneme] = 0
                    if phoneme not in overall_word_counts:
                        overall_word_counts[phoneme] = 0
                    counts_by_phoneme_before[phoneme] +=1
                    overall_word_counts[phoneme] += 1
                highest_count = 0
                for phoneme, count in counts_by_phoneme_before.items():
                    if count > highest_count:
                        highest_count = count
                        default_phoneme = phoneme
                self.rule_map[part][before]["$default"] = default_phoneme
            highest_count = 0
            default_phoneme = None
            for phoneme, count in overall_word_counts.items():
                if count > highest_count:
                    highest_count = count
                    default_phoneme = phoneme
            self.rule_map[part]["$default"] = default_phoneme

    def flatten_rule_map(self, data):
        flattened_map = {}
        for key, bdata in data.items():
            for before_key, adata in bdata.items():
                if before_key == "$default":
                    yield key, adata
                else:
                    for after_key, edata in adata.items():
                        if after_key == "$default":
                            yield ".".join([key, before_key]), edata
                        else:
                            for word, phoneme in edata.items():
                                if word == "$default":
                                    yield ".".join([key, before_key, after_key]), phoneme
                                else:
                                    yield ".".join([key, before_key, after_key, word]), phoneme

    def distill_rules(self):
        # If we have trained a rule set that has a high success rate it should also have a ton of unnecessary redundancy
        # that we are yet to trim. The largest proportion of assigned phonemes should be defaultable for all rules
        # and word specific logic may also be assignable to unique shared word stems.
        data = {}
        for part, before_data in self.rule_map.items():
            default = before_data["$default"]
            data[part] = {}
            data[part]["$default"] = default 
            for before, after_data in before_data.items():
                if before == "$default":
                    continue
                data[part][before] = {}
                before_default = after_data["$default"] if "$default" in after_data and after_data["$default"] is not None else default
                data[part][before]["$default"] = before_default
                for after, phoneme in after_data.items():
                    if after == "$extensions" or after == "$default":
                        continue
                    data[part][before][after] = {"$default": phoneme}
                if "$extensions" in after_data:
                    self.distill_unique_cases(part, before, after_data["$extensions"], data, before_default)
        yield from self.flatten_rule_map(data)

    def distill_unique_cases(self, current, before, extensions, out_data, before_default):
        for after, after_data in extensions.items():
            if after not in out_data:
                out_data[current][before][after] = {"$default": before_default}
            cases = {}
            for word, phoneme in after_data.items():
                if phoneme not in cases:
                    cases[phoneme] = []
                cases[phoneme].append(word)
            for shared_phoneme, words in cases.items():
                if len(words) > 1:
                    largest_prefix = get_largest_shared_prefix(*words)
                    largest_suffix = get_largest_shared_suffix(*words)
                    prefix_works = True
                    suffix_works = True
                    for word in self.word_mapping[current][before][after]:
                        if prefix_works and is_prefix(largest_prefix, word) and word not in words:
                            prefix_works = False
                        if suffix_works and is_suffix(largest_suffix, word) and word not in words:
                            suffix_works = False
                        if not suffix_works and not prefix_works:
                            break
                    if prefix_works:
                        out_data[current][before][after][largest_prefix + "*"] = shared_phoneme
                    elif suffix_works:
                        out_data[current][before][after]["*" + largest_suffix] = shared_phoneme
                    else:
                        for word in words:
                            out_data[current][before][after][word] = shared_phoneme
                else:
                    for shared_phoneme, words in cases.items():
                        for word in words:
                            out_data[current][before][after][word] = shared_phoneme

    def coalesce_count_data_to_rules(self):
        for part, rules_data in self.count_data.items():
            word_regexes, last_matches = build_word_regexes(part)
            for before, after_data in rules_data.items():
                for after, phoneme_data in after_data.items():
                    max_counter = 0
                    possibles = []
                    for phoneme, counter in phoneme_data.items():
                        if phoneme == "$extensions" and self.allow_unique_cases and self.learn_unique_cases:
                            for word, count_data in counter.items():
                                if self.in_rules(part, before, after, word):
                                    continue
                                epossibles = []
                                for ph, c in count_data.items():
                                    if one_or_more_match(word_regexes, ph):
                                        epossibles.append((ph, c[1]))
                                if len(epossibles) < 1:
                                    for ph, c in count_data.items():
                                        # fall back to just matching the last phoneme
                                        if one_or_more_match(last_matches, phoneme):
                                            epossibles.append((ph, c[1]))
                                emax_counter = 0
                                epossible_data = epossibles if len(epossibles) > 0 else list(count_data.items())
                                for ph, c in epossible_data:
                                    if isinstance(c, list):
                                        c = c[1]
                                    if c > emax_counter:
                                        emax_counter = c
                                emax_length = 0
                                best_fit = None
                                for ph, c in epossible_data:
                                    if isinstance(c, list):
                                        c = c[1]
                                    if c == emax_counter and len(ph) > emax_length:
                                        emax_length = len(ph)
                                        best_fit = ph
                                if best_fit is not None:
                                    self.update_rule_at(part, before, after, word, best_fit)
                            continue
                        if one_or_more_match(word_regexes, phoneme):
                            possibles.append((phoneme, counter[1]))
                    if self.in_rules(part, before, after):
                        continue
                    if len(possibles) < 1:
                        for phoneme, counter in phoneme_data.items():
                            if phoneme == "$extensions":
                                continue
                            # fall back to just matching the last phoneme
                            if one_or_more_match(last_matches, phoneme):
                                possibles.append((phoneme, counter[1]))
                    possible_data = possibles if len(possibles) > 0 else list(phoneme_data.items())
                    for phoneme, counter, in possible_data:
                        if phoneme == "$extensions":
                            continue
                        if isinstance(counter, list):
                            counter = counter[1]
                        if counter > max_counter:
                            max_counter = counter
                    if max_counter <= 3 and after != "$":
                        continue # probably not definitive
                    max_length = 0
                    inclusion_rate = 0.0
                    best_fit = None
                    words_seen = len(self.words_at(part, before, after))
                    for phoneme, counter in possible_data:
                        if phoneme == "$extensions":
                            continue
                        if isinstance(counter, list):
                            counter = counter[1]
                        calculated_rate = (len(phoneme_data[phoneme][0]) / words_seen) if words_seen > 0 else 0.0
                        if counter == max_counter and (len(phoneme) > max_length or (len(phoneme) >= max_length and calculated_rate > inclusion_rate)):
                            inclusion_rate = calculated_rate
                            max_length = len(phoneme)
                            best_fit = phoneme
                    if best_fit is not None:
                        self.update_rule_at(part, before, after, None, best_fit)
        self.determine_defaults()


class SimpleTokenizer:
    """
        This is a simple greedy single pass tokenizer used for the purpose of identifying rough phoneme tokens in a corpus of
        words.
    """
    def __init__(self, groups = None, top_n = 1500):
        self.frequent_groups = set(groups if groups is not None else self.build_default_groups())
        self.top_n = top_n
        self.letter_groups = {}
        self.trained = False

    def build_default_groups(self):
        basic_groups = []
        for graphemes in BASIC_ENGLISH_PHONEMES.values():
            basic_groups += graphemes
        for letter in ALPHABET:
            basic_groups.append(letter)
        return basic_groups

    def save(self, fp):
        with open(fp, "w+") as f:
            json.dump(list(self.frequent_groups), f)

    def load(self, fp):
        with open(fp, "r+") as f:
            self.frequent_groups = set(json.load(f))
        self.trained = True

    def add_token(self, token):
        self.frequent_groups.add(token)

    def train(self):
        self.train_word_chunks()
        self.post_process_word_chunks()
        del self.letter_groups
        self.trained = True

    def tokenize(self, word):
        chunks = []
        running = ""
        for letter in word:
            if len(running) == 0:
                running += letter
                continue
            if running + letter not in self.frequent_groups:
                chunks.append(running)
                running = letter
            else:
                running += letter
        chunks.append(running)
        return chunks

    def split_word_into_frequent_groups(self, word):
        chunks = []
        remaining = word
        while len(remaining) > 0:
            length_found = 0
            best_fit = None
            for grapheme in self.frequent_groups:
                if len(grapheme) > length_found and is_prefix(grapheme, remaining):
                    best_fit = grapheme
                    length_found = len(grapheme)
            chunks.append(best_fit)
            remaining = remaining[len(best_fit):]
        return chunks

    def parse_combos(self, word):
        chunks = self.split_word_into_frequent_groups(word)
        for i in range(len(chunks)):
            if i >= len(chunks) - 1:
                continue
            for ii in range(i+1, len(chunks)):
                k = chunks[i] + "".join(chunks[i+1:ii]) if ii != i+1 else chunks[ii]
                if k not in self.letter_groups:
                    self.letter_groups[k] = 1
                else:
                    self.letter_groups[k] += 1

    def train_word_chunks(self):
        for word in words.words():
            word = word.lower().strip()
            if " " in word or "-" in word or len(word) == 0:
                continue
            self.parse_combos(word)

    def post_process_word_chunks(self):
        for part in [k for (k, _) in sorted([(k, v) for k, v in self.letter_groups.items()], reverse=True, key=lambda a: a[1])][:self.top_n]:
            self.frequent_groups.add(part)
