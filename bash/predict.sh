#!/bin/bash

# examples of predicting commands

# DM
CUDA_VISIBLE_DEVICES=0 \
allennlp predict \
--cuda-device 0 \
--output-file dm-output.mrp \
--predictor transition_predictor_sdp \
--include-package utils \
--include-package modules \
--use-dataset-reader \
--batch-size 32 \
--silent \
checkpoints/dm_bert \
data/dm-test.mrp


# PSD
CUDA_VISIBLE_DEVICES=0 \
allennlp predict \
--cuda-device 0 \
--output-file psd-output.mrp \
--predictor transition_predictor_sdp \
--include-package utils \
--include-package modules \
--use-dataset-reader \
--batch-size 32 \
--silent \
checkpoints/psd_bert \
data/psd-test.mrp


# EDS
CUDA_VISIBLE_DEVICES=0 \
allennlp predict \
--cuda-device 0 \
--output-file eds-output.mrp \
--predictor transition_predictor_eds \
--include-package utils \
--include-package modules \
--use-dataset-reader \
--batch-size 32 \
--silent \
checkpoints/eds_bert \
data/eds-test.mrp


# UCCA
CUDA_VISIBLE_DEVICES=0 \
allennlp predict \
--cuda-device 0 \
--output-file ucca-output.mrp \
--predictor transition_predictor_ucca \
--include-package utils \
--include-package modules \
--use-dataset-reader \
--batch-size 32 \
--silent \
checkpoints/ucca_bert \
data/ucca-test.mrp


# AMR
# !!! AMR parser accepts input of augmented amr format instead of mrp format !!!
CUDA_VISIBLE_DEVICES=0 \
allennlp predict \
--cuda-device 0 \
--output-file amr-output.mrp \
--predictor transition_predictor_amr \
--include-package utils \
--include-package modules \
--use-dataset-reader \
--batch-size 32 \
--silent \
checkpoints/amr_bert \
data/amr-test.txt
