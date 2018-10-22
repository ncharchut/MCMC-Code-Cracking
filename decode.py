import numpy as np
import random
import csv
from pprint import pprint
import matplotlib.pyplot as plt


CIPHER_LEN = 28

class MetropolisHastings(object):
    """
    A class representing the Metropolis-Hastings MCMC sampling algorithm.
    """

    def __init__(self, alphabet, M, letter_probs, unigram_p, iters=2000, num_cipher_text_chars=None):
        """
        Initializes the MH object.

        Args:
            alphabet (str) : string containing all letters of the alphabet of decoded text.
            M (str)        : path to file containing letter-transition probabilities.
            unigram_p (str): path to file containing unigram probabilities.
            iters (int)    : limit of iterations in the case of no convergence.
        """
        self.iters = iters
        self.alphabet = alphabet
        self.M = self.load_transition_matrix(M)
        self.letter_probabilities = self.load_letter_probabilities(letter_probs)
        self.unigram_p = self.load_unigram(unigram_p)


    def load_unigram(self, file):
        """ Loads the unigram frequency file. """

        unigrams = np.loadtxt(file, delimiter=',')
        total = np.sum(unigrams, axis=0)[1]
        new_unigrams = []
        for letter, pair in unigrams:
            new_unigrams.append([int(letter), pair / total])
        new_unigrams =  sorted(new_unigrams, key=lambda x: x[0])
        new_unigrams = np.array([prob for i, prob in new_unigrams])

        return new_unigrams


    def label_ngram_freqs(self, freq):
        """
        Associates a letter with its corresponding ngrams in the text.

        Args:
            freq (dict) : mapping of ngram to its frequency.

        Returns:
            Mapping of character to every transition involved in it.
        """

        res = {el: [] for el in range(len(self.alphabet))}
        for ngram in freq:
            for el in ngram:
                res[el].append(ngram)
        return res


    def compute_ngram_frequency(self, text, n=2):
        """ Computes frequency of ngrams from text. """

        freq = {}

        for i, after in enumerate(text):
            if i < n - 1:
                continue
            ngram = tuple([c for c in text[i - n + 1: i + 1]])
            if ngram not in freq:
                freq[ngram] = 1
            else:
                freq[ngram] += 1

        return freq


    def compute_letter_frequency(self, text, known):
        """ Computes frequency of letters in ciphertext. """

        freq = [0 for _ in xrange(CIPHER_LEN)]

        for i, char in enumerate(text):
            freq[char] += 1

        test = list(enumerate(freq))
        return [i for i in sorted(test, key=lambda x: -x[1]) if i[0] not in known]


    def init_cipher(self, encoded_text):
        """ 
        Smart initialization for cipher: 
            1. Obtains space and period.
            2. Aligns remaining letters by frequency (unigram attack).
        """ 

        # 1. Find period and space
        check_dict = {i: [None, 0] for i in range(CIPHER_LEN)}
        prev = encoded_text[0]

        for i, cur in enumerate(encoded_text):
            if i == 0:
                continue

            if prev not in check_dict:
                prev = cur

            elif check_dict[prev][0] is None:
                check_dict[prev] = [cur, 1]
                prev = cur

            elif check_dict[prev][0] != cur:
                del check_dict[prev]
            else:
                check_dict[prev][1] += 1

            prev = cur

        period = max(check_dict.iterkeys(), key=(lambda x: check_dict[x][1]))
        space = check_dict[period][0]

        known = {space: 26, period: 27}
        known_vals = [26, 27]

        del check_dict[period]

        ## FOR COMPARISON WITH NO INITIALIZATION
        # init_cipher = range(CIPHER_LEN)
        # random.shuffle(init_cipher)
        # return init_cipher, []

        # 2. Align remaining letters by frequency.
        ordered_frequencies = self.compute_letter_frequency(encoded_text, known)
        letter_probs = [i for i in sorted(enumerate(self.unigram_p), key=lambda x: -x[1]) if i[0] not in known_vals]
        init_cipher = [0 for _ in xrange(CIPHER_LEN - 2)]

        for i, pair in enumerate(ordered_frequencies):
            init_cipher[letter_probs[i][0]] = pair[0]

        # FOR COMPARISON WITH NO ALIGNMENT
        # init_cipher = [i for i in range(CIPHER_LEN) if i not in (space, period)]
        # random.shuffle(init_cipher)

        init_cipher.append(space)
        init_cipher.append(period)
        init_cipher = [init_cipher.index(i) for i in xrange(len(self.alphabet))]

        return init_cipher, known


    def load_letter_probabilities(self, file):
        """ Loads letter probabilities. """
        return np.log2(np.loadtxt(file, delimiter=',', dtype = np.float64))


    def generate_next_cipher(self, cipher, undefined_alphabet):
        """ Generates next cipher by swapping two indices at random. """

        i1, i2 = random.sample(undefined_alphabet, 2)

        new_cipher = list(cipher)
        new_cipher[i1], new_cipher[i2] = new_cipher[i2], new_cipher[i1]

        return new_cipher, i1, i2

            
    def load_transition_matrix(self, file):
        """ Loads transition matrix. """

        matrix = np.loadtxt(file, delimiter=',', dtype=np.float64)
        matrix[matrix == 0] = 0.001
        return np.log2(matrix)


    def debug_likelihood(self, encoded_text, f):

        res = self.letter_probabilities[f[encoded_text[0]]]

        for i, after in enumerate(encoded_text):
            if i == 0: continue

            before = encoded_text[i - 1]
            res += self.M[f[after]][f[before]]

        return res


    def easy_decode(self, cipher, encoded_text, pprint=False):
        """ Given a decoding cipher, returns decoded result. pprint indicates legibility. """
        decoded = []
        translate = {i: letter for i, letter in enumerate(self.alphabet)}

        if pprint:
            decoded = [translate[cipher[char]] for char in encoded_text]
            decoded.append('\n')
        else:
            decoded = [cipher[char] for char in encoded_text]

        if pprint:
            return ''.join(decoded)

        return decoded


    def init_likelihood(self, f, encoded_text, ngram_freq):
        """ Initializer for likelihood. """
        total = 0
        p_x = self.letter_probabilities[f[encoded_text[0]]]

        for ngram in ngram_freq:
            before, after = ngram
            prob = self.M[f[after]][f[before]]
            total += ngram_freq[ngram] * prob

        return total + p_x, p_x


    def easy_calculate_likelihood(self, f, f_prime, ngram_freq, label_ngrams, a1, a2):
        """ Calculates different in LL of two candidate cipher functions. """
        diff = 0

        for ngram in label_ngrams[a1]:
            before, after = ngram

            old_prob = self.M[f[after]][f[before]]
            new_prob = self.M[f_prime[after]][f_prime[before]]
            diff += ngram_freq[ngram] * (new_prob - old_prob)

        for ngram in label_ngrams[a2]:
            before, after = ngram

            old_prob = self.M[f[after]][f[before]]
            new_prob = self.M[f_prime[after]][f_prime[before]]
            diff += ngram_freq[ngram] * (new_prob - old_prob)

        return diff


    def estimate_cipher(self, ciphertext, repetitions):
        """ 
            Estimates decoding cipher, given a ciphertext.

            Speed optimizations include:
                1. Fixed number of iterations if no early break.
                2. Multiple runs, take maximum outcome.
                3. Early break if no viable swaps.
                4. Early break if no recent update to max LL.
                5. Early break from repetitions if majority early break themselves.

        """

        # I. Setup.
        candidate_f_LL = []
        break_early_count = 0
        last_changes = []

        # II. Convert encoding to numerical representation.
        translate = {alpha: i for i, alpha in enumerate(self.alphabet)}
        ciphertext = [translate[char] for char in ciphertext]

        # III. Repeat the following process the specified number of times.
        for rep in xrange(repetitions):
            encoded_text = ciphertext

            # 0. Initializating values.
            since_last_change = 0
            last_change_limit = 1000

            # 1. Initialize first cipher and undefined alphabet.
            f, known = self.init_cipher(encoded_text)
            undefined_alphabet = [i for i in xrange(CIPHER_LEN) if i not in known]
            swaps = self.generate_available_swaps(undefined_alphabet)
            NUM_SWAPS = len(swaps)
            swap_gen = self.sample_gen(NUM_SWAPS)
            invalid = []

            # 2. Gather all transitions associated with a given letter.
            ngram_freq = self.compute_ngram_frequency(encoded_text, n=2)
            label_ngrams = self.label_ngram_freqs(ngram_freq)

            # 3. Initialize likelihood.
            L_f, p_x = self.init_likelihood(f, encoded_text, ngram_freq)

            # 0b. More initialized values.
            max_ll = L_f
            best_cipher = f
            last_change = 0

            # 5. Iterate to find candidate cipher.
            for i in xrange(self.iters):
                ## FOLLOWING USED FOR COMPARISON WITH NO SWAPPING, WILL NEED TO KEEP TRACK OF LAST UPDATED ITERATION.
                # f_prime, a1, a2 = self.generate_next_cipher(f, undefined_alphabet)
                swap = None
                try:
                    swap = swap_gen.next()
                    a1, a2 = swaps[swap]
                    f_prime = list(f)
                    f_prime[a1], f_prime[a2] = f_prime[a2], f_prime[a1]

                except StopIteration:
                    break_early_count += 1
                    break  # tried all possible swaps

                diff = self.easy_calculate_likelihood(f, f_prime, ngram_freq, label_ngrams, a1, a2)
                flip = np.random.random()
                threshold = diff

                if threshold >= 0:
                    r = 1
                elif threshold < -15:
                    invalid.append(swap) # too improbable
                    swap_gen = self.sample_gen(NUM_SWAPS, invalid)
                    r = 0
                else:
                    r = np.exp(threshold)

                if flip < r:
                    f = f_prime
                    L_f_prime = self.init_likelihood(f, encoded_text, ngram_freq)[0]
                    L_f = L_f_prime
                    invalid = []
                    last_change = i
                    swap_gen = self.sample_gen(NUM_SWAPS, invalid)

                    since_last_change += 1

                    if L_f > max_ll:
                        max_ll = L_f
                        best_cipher = f
                        since_last_change = 0
                        last_change = i
                else:
                    since_last_change += 1

                if since_last_change >= last_change_limit:
                    break_early_count += 1
                    break

            # Keep track of candidate cipher functions and their likelihoods
            L_f = self.init_likelihood(best_cipher, ciphertext, ngram_freq)
            candidate_f_LL.append((best_cipher, L_f))

            last_changes.append(last_change)
            
            if break_early_count >= repetitions / 2.0:
                break

        ## USED FOR IDENTIFYING EFFECTIVENESS OF EARLY SWAP
        last_avg_iter = sum(last_changes) / float(len(last_changes))
        # print "Last iteration of change: {0}".format(sum(last_changes) / float(len(last_changes)))

        # IV. Identify best function w.r.t whole ciphertext.
        best_cipher = max(candidate_f_LL, key=lambda x: x[1])[0]
        result = self.easy_decode(best_cipher, ciphertext, pprint=True)

        return result, best_cipher, ciphertext, last_avg_iter


    def generate_available_swaps(self, undefined_alphabet):
        """ Given an alphabet of undefined mappings, determines possibles swaps. """

        swaps = []
        for i in xrange(len(undefined_alphabet)):
            first_el = undefined_alphabet[i]

            for j in xrange(i, len(undefined_alphabet)):
                if i == j:
                    continue

                second_el = undefined_alphabet[j]
                swaps.append((first_el, second_el))

        swaps = {i: swap for i, swap in enumerate(swaps)}

        return swaps


    def sample_gen(self, n, forbid=[]):
        """ Sample generator without replacement. This code was found in a StackOverflow post online."""
        state = dict()
        track = dict()
        for (i, o) in enumerate(forbid):
            x = track.get(o, o)
            t = state.get(n-i-1, n-i-1)
            state[x] = t
            track[t] = x
            state.pop(n-i-1, None)
            track.pop(o, None)
        del track
        for remaining in xrange(n-len(forbid), 0, -1):
            i = random.randrange(remaining)
            yield state.get(i, i)
            state[i] = state.get(remaining - 1, remaining - 1)
            state.pop(remaining - 1, None)


    @staticmethod
    def calculate_accuracy(alphabet, cipher, encoded_text, decoded_text):
        """ Calculate accuracy of the cipher with respec to the original plaintext. """
        alphabet = {i: char for i, char in enumerate(alphabet)}

        total_correct = sum([1 for i, char in enumerate(encoded_text) if cipher[char] == decoded_text[i]])
        incorrect_mappings = set()
        for i, char in enumerate(encoded_text):
            if cipher[char] != decoded_text[i]:
                incorrect_mappings.add((alphabet[cipher[char]], alphabet[decoded_text[i]]))

        assert len(encoded_text) == len(decoded_text)
        return float(total_correct) / len(encoded_text) , incorrect_mappings


def decode(ciphertext, output_file_name):
    """ Decodes a given ciphertext and saves it to the given filename. """

    letters = "abcdefghijklmnopqrstuvwxyz ."
    M = "./probs/letter_transition_matrix.csv"
    iters = 10000
    unigram_p = "./probs/unigram_p.csv"
    letter_probs = "./probs/letter_probabilities.csv"
    num_cipher_text_chars = float('inf')

    MH = MetropolisHastings(letters, M, letter_probs, unigram_p, iters=iters)
    repetitions = [3, 3, 10]

    long_text, medium_text, short_text = repetitions
    encoded_text_length = len(ciphertext)

    if encoded_text_length >= 7000:
        repetition = long_text
    elif encoded_text_length >= 2500:
        repetition = medium_text
    else:
        repetition = short_text

    decoded, cipher, encoded_text, last_avg_iter = MH.estimate_cipher(ciphertext, repetition)

    with open(output_file_name, 'wb') as writer:
        writer.write(decoded)

    return decoded, last_avg_iter

