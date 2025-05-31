import h5py
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for loading precomputed audio embeddings.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 (.h5) file containing the dataset.
    medium : str, optional
        The top-level group in the HDF5 file under which samples are stored. Default is 'audio_embeddings'.
    emb_type : str, optional
        The type of embedding to retrieve from each sample (e.g., 'average' or 'sequence'). Default is 'average'.
    label : str
        The key under each sample group that identifies the label (e.g., 'style', 'drummerId').
    """
    def __init__(self, h5_path, medium='audio_embeddings', emb_type='average', label=None):
        self.h5_path = h5_path
        self.medium = medium
        self.emb_type = emb_type
        self.label = label

        with h5py.File(h5_path, 'r') as f:
            self.sample_keys = list(f[self.medium].keys())
            self.sample_fields = list(f[self.medium][self.sample_keys[0]].keys())

            assert self.medium in ['audio_embeddings', 'symbolic_embeddings'], \
                f"Medium '{self.medium}' must be either 'audio_embeddings' or 'symbolic_embeddings'."

            assert self.emb_type in ['average', 'sequence'], \
                f"Embedding type '{self.emb_type}' must be either 'average' or 'sequence'."

            assert self.label in self.sample_fields, \
                f"Label '{self.label}' not found in sample fields: {self.sample_fields}"

            # Collect all label values for encoding
            label_values = []
            for key in self.sample_keys:
                val = f[self.medium][key][self.label][()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                label_values.append(val)

        # Fit label encoder
        self.encoder = LabelEncoder()
        self.encoder.fit(label_values)

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        """
        Retrieves the embedding and encoded label for a given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the embedding tensor and the encoded label tensor.
        """
        with h5py.File(self.h5_path, 'r') as f:
            # Retrieve the group for specific record
            grp = f[self.medium][self.sample_keys[idx]]

            # Get embedding
            emb = torch.tensor(grp[self.emb_type][:], dtype=torch.float32)

            # Get and encode label
            label_val = grp[self.label][()]
            if isinstance(label_val, bytes):
                label_val = label_val.decode("utf-8")
            label = self.encoder.transform([label_val])[0]
            label = torch.tensor(label, dtype=torch.long)

        return emb, label

    def get_classes(self):
        """Returns the list of class names in encoded order."""
        return list(self.encoder.classes_)

    def decode_label(self, index):
        """Returns the original label from encoded index."""
        return self.encoder.inverse_transform([index])[0] 