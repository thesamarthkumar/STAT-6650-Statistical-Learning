# Provided script for downloading the data.

from os import makedirs, path, remove, rename, rmdir
from tarfile import open as open_tar
from urllib import request, parse

def download_corpus(dataset_dir: str = r'C:\Users\skuma\OneDrive\Documents\STAT 6650\HW 3\data'):
    base_url = 'https://spamassassin.apache.org'
    corpus_path = 'old/publiccorpus'
    files = {
        '20021010_easy_ham.tar.bz2': 'ham',
        '20021010_hard_ham.tar.bz2': 'ham',
        '20021010_spam.tar.bz2': 'spam',
        '20030228_easy_ham.tar.bz2': 'ham',
        '20030228_easy_ham_2.tar.bz2': 'ham',
        '20030228_hard_ham.tar.bz2': 'ham',
        '20030228_spam.tar.bz2': 'spam',
        '20030228_spam_2.tar.bz2': 'spam',
        '20050311_spam_2.tar.bz2': 'spam' 
    }

    # Prepare directory structure
    downloads_dir = path.join(dataset_dir, 'downloads')
    ham_dir = path.join(dataset_dir, 'ham')
    spam_dir = path.join(dataset_dir, 'spam')

    makedirs(downloads_dir, exist_ok=True)
    makedirs(ham_dir, exist_ok=True)
    makedirs(spam_dir, exist_ok=True)

    for file, spam_or_ham in files.items():
        # Construct download URL and local path
        url = parse.urljoin(base_url, f'{corpus_path}/{file}')
        tar_filename = path.join(downloads_dir, file)

        # Download the tar.bz2 file
        request.urlretrieve(url, tar_filename)
        print(f"Downloaded {file} -> {tar_filename}")

        with open_tar(tar_filename) as tar:
            tar.extractall(path=downloads_dir)
            emails = []
            for tarinfo in tar:
                if tarinfo.isreg():
                    parts = tarinfo.name.split('/')
                    if len(parts) == 2:
                        directory, filename = parts
                        if not filename.startswith('cmds') and '.' in filename:
                            emails.append((directory, filename))

        for (directory, filename) in emails:
            source_folder = path.join(downloads_dir, directory)
            source_file = path.join(source_folder, filename)

            dest_folder = ham_dir if spam_or_ham == 'ham' else spam_dir
            dest_file = path.join(dest_folder, filename)

            if path.exists(dest_file):
                print(f"Skipping existing file: {dest_file}")
                continue

            rename(source_file, dest_file)
            print(f"Moved: {source_file} -> {dest_file}")

        # Remove the now-empty extracted folder
        folder_to_remove = path.join(downloads_dir, directory)
        if path.isdir(folder_to_remove):
            rmdir(folder_to_remove)

    print("All downloads/extractions complete!")

download_corpus(r'C:\Users\skuma\OneDrive\Documents\STAT 6650\HW 3\data')