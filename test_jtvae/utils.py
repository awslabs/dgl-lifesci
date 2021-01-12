from dgllife.data import JTVAEDataset
from dgllife.utils import JTVAEVocab

def test_data():
    vocab = JTVAEVocab()
    dataset = JTVAEDataset('test.txt', vocab, training=True)

    return vocab, dataset
