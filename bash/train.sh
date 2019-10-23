#!/bin/bash

# examples of training commands

# DM
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/dm-train.mrp \
DEV_PATH=data/dm-dev.mrp \
BERT_PATH=bert/wwm_cased_L-24_H-1024_A-16 \
WORD_DIM=1024 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
allennlp train \
-s checkpoints/dm_bert \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_bert_sdp_conll.jsonnet


# PSD
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/psd-train.mrp \
DEV_PATH=data/psd-dev.mrp \
BERT_PATH=bert/wwm_cased_L-24_H-1024_A-16 \
WORD_DIM=1024 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
allennlp train \
-s checkpoints/psd_bert \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_bert_sdp_conll.jsonnet


# EDS
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/eds-train.mrp \
DEV_PATH=data/eds-dev.mrp \
BERT_PATH=bert/wwm_cased_L-24_H-1024_A-16 \
WORD_DIM=1024 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
allennlp train \
-s checkpoints/eds_bert \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_bert_eds.jsonnet


# UCCA
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/ucca-train.mrp \
DEV_PATH=data/ucca-dev.mrp \
BERT_PATH=bert/wwm_cased_L-24_H-1024_A-16 \
WORD_DIM=1024 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
allennlp train \
-s checkpoints/ucca_bert \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_bert_ucca.jsonnet


# AMR
# !!! AMR parser accepts input of augmented amr format instead of mrp format !!!
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/amr-train.mrp.actions.aug.txt \
DEV_PATH=data/amr-dev.mrp.actions.aug.txt \
BERT_PATH=bert/wwm_cased_L-24_H-1024_A-16 \
WORD_DIM=1024 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
allennlp train \
-s checkpoints/amr_bert \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_bert_amr.jsonnet
