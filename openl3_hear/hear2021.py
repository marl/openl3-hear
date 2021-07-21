
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

    emb, ts = openl3.get_audio_embedding(audio, sr=model.sample_rate, model=model.openl3_model, hop_size=hop_size)
    # FIXME: check if timestampes are centered
    return emb, ts


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

    emb, ts = get_timestamp_embeddings(audio, model, hop_size=hop_size)

    # FIXME: use TensorFlow Tensor instead. Using tf.constant ?
    scene_embedding = numpy.mean(emb)
    return scene_embedding


