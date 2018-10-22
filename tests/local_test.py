import random
from decode import decode
from time import time


def generate_ciphertext(text_file):
    ALPHABET = "abcdefghijklmnopqrstuvwxyz ."
    CIPHER_LEN = len(ALPHABET)

    alpha = {element: i for i, element in enumerate(ALPHABET)}

    # 1. Convert to integer mapping.
    with open(text_file, 'rb') as reader:
        chars = [alpha[i] for i in reader.read() if i != '\n']
        

    # 2. Generate random cipher function.
    cipher = range(CIPHER_LEN)
    random.shuffle(cipher)

    # 3. Convert text.
    ciphertext = [cipher[char] for char in  chars]

    translate = {i: element for i, element in enumerate(ALPHABET)}
    ciphertext = [translate[char] for char in ciphertext]

    return ciphertext

def calculate_accuracy(text_file, decoded_text):
    # 1. Convert to integer mapping.
    with open(text_file, 'rb') as reader:
        chars = [i for i in reader.read() if i != '\n']

    chars = ''.join(chars)

    incorrect = 0

    for i, char in enumerate(chars):
        if char != decoded_text[i]:
            incorrect += 1

    return float(len(chars) - incorrect) / len(chars)


def test(text_files, num_tests_per_file=250):
    number_correct = 0
    total_tests = len(text_files) * num_tests_per_file

    for file in text_files:
        print "Testing: {0}".format(file)
        total_accuracy = 0
        total_iter = 0
        start = time()
        for test in xrange(num_tests_per_file):
            output_filename = "test_output.txt"
            ciphertext = generate_ciphertext(file)
            decoded, last_avg_iter = decode(ciphertext, output_filename)

            accuracy = calculate_accuracy(file, decoded)
            total_accuracy += accuracy
            total_iter += last_avg_iter
            if accuracy == 1.0:
                print "\tTest: {0} | PASSED".format(test)
                number_correct += 1
            else:
                print "\tTest: {0} | FAILED | Accuracy: {1}".format(test, accuracy)
        end = time()

        print "Average accuracy: {0}, Time / Test: {1}, Average Last Iteration: {2}".format(total_accuracy / float(num_tests_per_file), (end - start) / num_tests_per_file, total_iter / num_tests_per_file)
        print '------------------------------------------'

    print "RESULTS:\n\t{0}/{1} PASSED".format(number_correct, total_tests)
    return


if __name__ == "__main__":
    text_files = ['plaintext.txt',
                  'plaintext_paradiselost.txt',
                  'plaintext_warandpeace.txt',
                  'plaintext_feynman.txt'
                 ]

    # text_files = ['plaintext_warandpeace.txt']

    test(text_files, num_tests_per_file=25)













