
"""
HEAR2021 API implementation

As per specifications in
https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html
"""

HOP_SIZE_TIMESTAMPS = 0.050 # <50 ms recommended
HOP_SIZE_SCENE = 0.5

import openl3
import numpy

#import tensorflow_datasets
#from tensorflow_datasets.typing import Tensor
#from tensorflow.types.experimental import Tensor
from typing import NewType, Tuple
Tensor = NewType('Tensor', object)

# FIXME: inherit from tf.Module ?
class Model():
    def __init__(self, model, sample_rate=16000, embedding_size=512):
        self.sample_rate = sample_rate
        self.scene_embedding_size = embedding_size
        self.timestamp_embedding_size = embedding_size

        self.openl3_model = model # the OpenL3 model instance    


def load_model(model_file_path: str) -> Model:
    # FIXME: respect model_file_path

    embedding_size = 512

    openl3_model = openl3.models.load_audio_embedding_model(input_repr="mel256",
                            content_type="music",
                            embedding_size=embedding_size,
    )

    model = Model(model=openl3_model, embedding_size=embedding_size)
    return model

TimestampedEmbeddings = Tuple[Tensor, Tensor]

def get_timestamp_embeddings(
    audio: Tensor,
    model: Model,
    hop_size=HOP_SIZE_TIMESTAMPS,
) -> TimestampedEmbeddings:
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]
    model: Loaded Model. 

    Returns:

        embedding: A float32 Tensor with shape (n_sounds, n_timestamp, model.timestamp_embedding_size).
        timestamps: Tensor. Centered timestamps in milliseconds corresponding to each embedding in the output.
     """
    # pre-conditions
    assert len(audio.shape) == 2

    # get embeddings for a single audio clip
    def get_embedding(samples):
        emb, ts = openl3.get_audio_embedding(samples,
            sr=model.sample_rate,
            model=model.openl3_model,
            hop_size=hop_size,
            center=True,
            verbose=0,
        )

        return emb, ts

    # Compute embeddings for each clip
    embeddings = []
    ts = None
    for sound_no in range(audio.shape[0]):
        samples = audio[sound_no, :]
        emb, ts = get_embedding(samples)
        embeddings.append(emb)
    emb = numpy.stack(embeddings)

    # post-conditions
    assert len(ts.shape) == 1 
    assert len(ts) >= 1
    assert emb.shape[0] == audio.shape[0]
    assert len(emb.shape) == 3, emb.shape
    assert emb.shape[1] == len(ts), (emb.shape, ts.shape)
    if len(ts) >= 2:
        assert ts[1] == ts[0] + hop_size

    # XXX: are timestampes centered?
    # first results seems to be 0.0, which would indicate that window
    # starts at -window/2 ?
    #assert ts[0] > 0.0 and ts[0] < hop_size, ts
    return (emb, ts)


def get_scene_embeddings(
    audio: Tensor,
    model: Model,
    hop_size=HOP_SIZE_SCENE,
) -> Tensor:

    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1].
    model: Loaded Model.

    Returns:

        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    assert len(audio.shape) == 2 

    emb, ts = get_timestamp_embeddings(audio, model, hop_size=hop_size)

    # FIXME: use TensorFlow Tensor instead. Using tf.constant ?
    scene_embedding = numpy.mean(emb)
    return scene_embedding


