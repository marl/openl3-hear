
"""Test the model using the HEAR2021 API"""

import math

import openl3_hear as module
import numpy

# TODO
# test maximum length, 20 minutes
# test temporal resolution. <50 ms
# check shapes of outputs wrt inputs
# sanity check with audio from real datasets
# SpeechCommands 

TEST_WEIGHTS_PATH = 'unused'

def whitenoise_audio(sr=16000, duration=1.0, amplitude=1.0):
    n_samples = math.ceil(sr * duration)
    samples = numpy.random.uniform(low=-amplitude, high=amplitude, size=n_samples)
    return samples


def test_timestamp_embedding_basic():
    model = module.load_model(TEST_WEIGHTS_PATH)
    audio = numpy.array([whitenoise_audio(duration=1.5) for i in range(4)])
    emb, ts = module.get_timestamp_embeddings(audio=audio, model=model)

def test_scene_embedding_basic():
    model = module.load_model(TEST_WEIGHTS_PATH)
    audio = numpy.array([whitenoise_audio(duration=1.2) for i in range(3)])
    emb = module.get_scene_embeddings(audio=audio, model=model)


