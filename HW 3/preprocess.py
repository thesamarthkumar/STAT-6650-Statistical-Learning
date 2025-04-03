"Provided code for preproessing the data."

import os

from glob import glob

from email.parser import BytesParser

from email import policy

from sklearn.model_selection import train_test_split

import chardet

import re

from nltk.stem import PorterStemmer

def load_emails_from_dir(folder: str, label: int):

    data = []

    files = glob(os.path.join(folder, '*'))

    for filepath in files:

        with open(filepath, 'rb') as f:

            raw_bytes = f.read()

        

        try:

            msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

        except Exception as e:

            print(f"Warning: failed to parse {filepath}: {e}")

            continue

        subject = msg['subject'] or ''

        if msg.is_multipart():

            body_parts = []

            for part in msg.walk():

                if part.get_content_type() == 'text/plain':

                    payload_bytes = part.get_payload(decode=True)

                    if payload_bytes is None:

                        continue

                    try:

                        detected = chardet.detect(payload_bytes) 

                        encoding = detected.get('encoding') or 'utf-8'

                        text_part = payload_bytes.decode(encoding, errors='replace')

                    except (UnicodeDecodeError, LookupError):

                        text_part = payload_bytes.decode('utf-8', errors='ignore')

                    

                    body_parts.append(text_part)

            body_text = "\n".join(body_parts)

        else:

            payload_bytes = msg.get_payload(decode=True)

            if payload_bytes:

                try:

                    detected = chardet.detect(payload_bytes)

                    encoding = detected.get('encoding') or 'utf-8'

                    body_text = payload_bytes.decode(encoding, errors='replace')

                except (UnicodeDecodeError, LookupError):

                    body_text = payload_bytes.decode('utf-8', errors='ignore')

            else:

                body_text = ''

        full_text = subject + "\n" + body_text

        data.append((full_text, label))

    return data

def load_all_emails(ham_dir, spam_dir):

    ham_data = load_emails_from_dir(ham_dir, label=-1)

    spam_data = load_emails_from_dir(spam_dir, label=+1)

    return ham_data + spam_data

ham_dir = r'Your directory path\ham'

spam_dir = r'Your directory path\spam'

all_data = load_all_emails(ham_dir, spam_dir)

stemmer = PorterStemmer()

def preprocess_text(raw_text: str) -> str:

    # 1. Lowercase

    text = raw_text.lower()

    # 2. Remove HTML tags

    text = re.sub(r'<.*?>', '', text)

    # 3. Normalize URLs, emails, numbers

    text = re.sub(r'(http|https)://[^\s]+', '__URL__', text)

    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '__EMAIL__', text)

    text = re.sub(r'\b\d+\b', '__NUMBER__', text)

    # 4. Remove punctuation / keep letters, digits, underscores, plus whitespace

    text = re.sub(r'[^a-zA-Z0-9_]+', ' ', text)

    # 5. Tokenize

    tokens = text.split()

    # 6. Stem each token

    tokens = [stemmer.stem(t) for t in tokens]

    # Return a single string or keep it as a list

    return " ".join(tokens)

def preprocess_dataset(data):

    cleaned_data = []

    for (txt, lbl) in data:

        cleaned_txt = preprocess_text(txt)

        cleaned_data.append((cleaned_txt, lbl))

    return cleaned_data

all_data_cleaned = preprocess_dataset(all_data)

print("Example cleaned text:\n", all_data_cleaned[0][0])

print("Label:", all_data_cleaned[0][1])

train_data, test_data = train_test_split(all_data_cleaned, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_data)}, Test set size: {len(test_data)}")

from collections import defaultdict

def build_vocabulary(train_data, min_email_count=30):

    """

    train_data: list of (cleaned_text, label)

    Returns a list of words that appear in >= min_email_count distinct emails.

    """

    word_in_email_count = defaultdict(int)

    for (cleaned_txt, lbl) in train_data:

        unique_words = set(cleaned_txt.split())

        for w in unique_words:

            word_in_email_count[w] += 1

    vocab = [w for w, c in word_in_email_count.items() if c >= min_email_count]

    vocab.sort()

    return vocab

vocabulary = build_vocabulary(train_data, min_email_count=30)

print(f"Vocabulary size: {len(vocabulary)}")

def email_to_feature_vector(cleaned_text, vocab_list):

    tokens = set(cleaned_text.split())

    return [1 if word in tokens else 0 for word in vocab_list]

X_train = []

y_train = []

for (cleaned_txt, lbl) in train_data:

    X_train.append(email_to_feature_vector(cleaned_txt, vocabulary))

    y_train.append(lbl)

X_test = []

y_test = []

for (cleaned_txt, lbl) in test_data:

    X_test.append(email_to_feature_vector(cleaned_txt, vocabulary))

    y_test.append(lbl)

# Just a Check
print("Example feature vector length:", len(X_train[0]))
print("First 30 features of the first vector:", X_train[0][:30])
