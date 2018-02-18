#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Script that trains doc2vec model
"""

import os

# gensim modules
from gensim import utils
from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# random
from random import shuffle

from utils import pprint, timeit, init_directory


data_dir            = "./maildir/"
should_create_data  = False
num_dimensions      = 300


@timeit
def create_data():
    # scrape inbox/sent directories from each email user

    users = os.listdir(data_dir)
    users.sort()

    for user in users:
        sub_dirs = os.listdir(data_dir + user)

        blacklist = []

        # we only concern ourselves with 'sent' email for now (aside from two edge cases below)
        try:
            sent_folder = sub_dirs[sub_dirs.index('sent')]
        except:
            try:
                sent_folder = sub_dirs[sub_dirs.index('_sent_mail')]
            except:
                try:
                    sent_folder = sub_dirs[sub_dirs.index('sent_items')]
                except:
                    # if user_dir == 'stokley-c': # edge case whose 'sent' box is burried
                    #     sent_dir = 'chris_stokley/sent'
                    # elif user_dir == 'harris-s': # edge case with no 'sent' box so we just use 'inbox'
                    #     sent_dir = 'inbox'
                    # else:
                    #     print("user edge case: {}".format(user_dir))
                    pprint("user edge case: {}".format(user))
                    blacklist.append(user)

        if user not in blacklist:
            with open(init_directory('parsed_data/') + 'train-{}.txt'.format(user), 'a') as train_file:
                with open(init_directory('parsed_data/') + 'test-{}.txt'.format(user), 'a') as test_file:
                    emails_dir = data_dir + user + '/' + sent_folder

                    emails = os.listdir(emails_dir)
                    emails.sort()

                    pprint(user)

                    # try to grab just the immediate body of the email ignoring forwarded or previous emails in a chain
                    start_flag = 'X-FileName:'
                    end_flag = 'Original Message'
                    blacklist_flags = ['---------------------- Forwarded', 'To:', 'Original Message']
                    count = 0
                    for email in emails:
                        with open(emails_dir + '/' + email, 'r') as f:
                            lines = []
                            found_start = False
                            found_end = False
                            for idx, line in enumerate(f):
                                # print(found_start, found_end, '\n')
                                if found_start and not found_end and line != '\r\n' and line != '\n':
                                    garabage = False
                                    for blacklist_flag in blacklist_flags:
                                        if line.find(blacklist_flag) >= 0:
                                            garabage = True
                                            found_end = True
                                    if not garabage:
                                        lines.append(line.replace('\n', ''))
                                        count += 1
                                elif line.find(start_flag) >= 0:
                                    found_start = True
                                elif line.find(end_flag) >= 0:
                                    found_end = True
                                else:
                                    continue

                            pprint(lines)
                            body = " ".join(lines)

                            if len(body) > 0:
                                pprint('------------------- parsed body ---------------------')
                                pprint(body)
                                pprint('\n')

                                # write to train/test db files
                                if count % 4 == 0:  # 25% to be used as test
                                    test_file.write(body + '\n')
                                else:
                                    train_file.write(body + '\n')

                    pprint(emails_dir)

    pprint(users + '\n')


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


if __name__ == "__main__":
    # create data sources
    if should_create_data:
        create_data()

    # gather data sources
    sources = {}
    parsed_data_files = os.listdir('parsed_data')
    total_examples = 0
    for file in parsed_data_files:
        with open('parsed_data/' + file, 'r') as f:
            for idx, line in enumerate(f):
                pass
        total_examples += idx + 1
        sources['parsed_data/' + file] = file.replace('-', '_').upper()[:-4] # ignore '.txt'

    pprint(sources)
    sentences = LabeledLineSentence(sources)

    pprint('total examples: {}'.format(total_examples))

    # build model
    model = Doc2Vec(min_count=1, window=10, size=num_dimensions, sample=1e-4, negative=5, workers=8)
    model.build_vocab(sentences.to_array())

    # train model
    model.train(sentences.sentences_perm(), total_examples=total_examples, epochs=10)

    # save model
    model.save('./model.d2v')