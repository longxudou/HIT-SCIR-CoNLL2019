#!/bin/bash

# examples of training commands

# DM
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/dm-train.mrp \
DEV_PATH=data/dm-dev.mrp \
PRETRAINED_FILE=glove/glove.6B.100d.txt\
BATCH_SIZE=32 \
allennlp train \
-s checkpoints/dm_glove \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_glove_sdp.jsonnet


# PSD
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/psd-train.mrp \
DEV_PATH=data/psd-dev.mrp \
PRETRAINED_FILE=glove/glove.6B.100d.txt\
BATCH_SIZE=32 \
allennlp train \
-s checkpoints/psd_glove \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_glove_sdp.jsonnet


# EDS
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/eds-train.mrp \
DEV_PATH=data/eds-dev.mrp \
PRETRAINED_FILE=glove/glove.6B.100d.txt\
BATCH_SIZE=32 \
allennlp train \
-s checkpoints/eds_glove \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_glove_eds.jsonnet


# UCCA
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/ucca-train.mrp \
DEV_PATH=data/ucca-dev.mrp \
PRETRAINED_FILE=glove/glove.6B.100d.txt\
BATCH_SIZE=32 \
allennlp train \
-s checkpoints/ucca_glove \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_glove_ucca.jsonnet


# AMR
# !!! AMR parser accepts input of augmented amr format instead of mrp format !!!
CUDA_VISIBLE_DEVICES=0 \
TRAIN_PATH=data/amr-train.mrp \
DEV_PATH=data/amr-dev.mrp \
PRETRAINED_FILE=glove/glove.6B.100d.txt\
BATCH_SIZE=32 \
allennlp train \
-s checkpoints/amr_glove \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_glove_amr.jsonnet
