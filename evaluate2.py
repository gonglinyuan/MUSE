# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate.py --crosslingual --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-es.es.vec

import argparse

import torch

from src.evaluation.word_translation import get_word_translation_accuracy2

# main
parser = argparse.ArgumentParser(description='Evaluation')
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings
parser.add_argument("--path", type=str, default="", help="Reload source embeddings")

# parse parameters
params = parser.parse_args()

model = torch.load(params.path)
get_word_translation_accuracy2(params.src_lang, model["dic0"].word2id, params.tgt_lang, model["dic1"].word2id,
                               model["out"], params.dico_eval)
