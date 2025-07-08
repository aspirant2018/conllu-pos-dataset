from sklearn.preprocessing import LabelEncoder
from itertools import chain

from datasets import Dataset
import pandas as pd
import conllu



class ConlluPosDataset:

    """
    A dataset class for converting CoNLL-U formatted part-of-speech (PoS) tagging data
    into a tokenized and aligned format suitable for training transformer-based models.

    This class performs the following steps:
    - Parses a CoNLL-U file to extract tokens and UPOS tags.
    - Updates token and tag sequences to handle UD-specific notations (e.g., '_' and merged tags).
    - Tokenizes sentences using a provided tokenizer.
    - Aligns word-level tags with subword tokenization using offset mappings.
    - Encodes the tags using a shared LabelEncoder across instances.
    - Provides utility methods to retrieve label encodings and to convert the processed data 
      into a Hugging Face `datasets.Dataset` object.

    Attributes:
        all_updated_tokens (list[list[str]]): Cleaned and updated token sequences.
        all_updated_tags (list[list[str]]): Corresponding UPOS tag sequences.
        encoded_inputs (dict): Tokenizer output containing input_ids and offset mappings.
        all_aligned_tags (list[list[str]]): Tags aligned to subword tokens.
        shared_label_encoder (LabelEncoder): Class-level label encoder shared across instances.

    Methods:
        get_labels_by_index(index, integer_labels=False):
            Returns the aligned tags for a specific sentence, optionally as integer labels.

        build_dataset():
            Builds a Hugging Face `Dataset` object with input_ids, attention_mask, and aligned labels.

        number_of_classes():
            Returns the total number of unique tags (classes) encoded by the LabelEncoder.
    """

    shared_label_encoder = None  # Class attribute
    language = None


    def __init__(self, filename: str,tokenizer=None, LabelEncoder=None,language=None):
        
        if tokenizer is None:
            raise ValueError("Please provide a tokenizer")

        if not isinstance(filename,str):
            raise ValueError("Please provide a valide filename")
            
        corpus = list(self._load_conllu(filename))
        df = self._conllu2df(corpus)
        self.all_updated_tokens, self.all_updated_tags = self._update_all_tags_tokens(df)
        
        self.encoded_inputs = tokenizer(
            self.all_updated_tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        self.all_aligned_tags = self.align_tags_with_subtokens(self.all_updated_tags,self.encoded_inputs)
        

        if (ConlluPosDataset.shared_label_encoder is None) or (language != ConlluPosDataset.language) :
            print('first Fitting')
            
            LabelEncoder.fit(list(chain.from_iterable(self.all_aligned_tags)))
            ConlluPosDataset.shared_label_encoder = LabelEncoder
            ConlluPosDataset.language = language
            

        

    def _load_conllu(self, filename:str):
        """
        Loads a CoNLL-U formatted file and extracts tokens and their UPOS tags.
    
        Args:
            filename (str): Path to the CoNLL-U file.
    
        Yields:
            tuple: A pair (tokens, tags), where:
                - tokens (list[str]): List of word forms in the sentence.
                - tags (list[str]): List of corresponding UPOS tags.
        """
        with open(filename, "r", encoding="utf-8") as f:
            sentences = conllu.parse(f.read())
            
        for sentence in sentences:
            tokenized_words = [token["form"] for token in sentence]
            gold_tags = [token["upos"] for token in sentence]
            yield tokenized_words, gold_tags        

    def _conllu2df(self, corpus: list) -> pd.DataFrame:
        """
        description
        """
        df = pd.DataFrame(corpus, columns=["Words", "Tags"])
        df["SentenceID"] = df.index
        df = df.explode(["Words", "Tags"], ignore_index=True)
        df = df[["SentenceID", "Words", "Tags"]]
        df.columns = ["sentenceID", "token", "tag"]
        df['token'] = df['token'].str.replace(' ', '', regex=False)
        return df

    def _update_UD_tokenization(self, tokens, tags):
        """
        description
        """
        updated_tokens = []
        updated_tags = []
        i = 0
        while i < len(tags):
            if tags[i] == '_':
                merged_tag = tags[i + 1] + '+' + tags[i + 2]
                updated_tags.append(merged_tag)
                updated_tokens.append(tokens[i])
                i += 3
            else:
                updated_tags.append(tags[i])
                updated_tokens.append(tokens[i])
                i += 1
        assert len(updated_tags) == len(updated_tokens)
        return updated_tokens, updated_tags

    def _update_all_tags_tokens(self, corpus_df):
        all_updated_tags = []
        all_updated_tokens = []

        for i in corpus_df['sentenceID'].unique():
            tokens = corpus_df[corpus_df['sentenceID'] == i]['token'].to_list()
            tags = corpus_df[corpus_df['sentenceID'] == i]['tag'].to_list()
            updated_tokens, updated_tags = self._update_UD_tokenization(tokens, tags)
            all_updated_tokens.append(updated_tokens)
            all_updated_tags.append(updated_tags)

        return all_updated_tokens, all_updated_tags


    def align_tags(self,tags: list[str], offset_mapping: list[tuple[int, int]]) -> list[str]:
        """
        Aligns word-level tags with subword tokens using offset mapping.
    
        Parameters:
            tags (list[str]): A list of word-level tags (e.g., for NER or PoS).
            offset_mapping (list[tuple[int, int]]): The list of (start, end) character positions
                                                    for each token produced by the tokenizer.
    
        Returns:
            list[str]: A list of aligned tags, where subword tokens are labeled as '<pad>'
        """
        tag_iter = iter(tags)
        aligned_tags = [next(tag_iter)]  # First tag is aligned with the first token
        prev_end = offset_mapping[0][1]
    
        for start, end in offset_mapping[1:]:
            if start == prev_end:
                aligned_tags.append('<pad>')  # Subword token
            else:
                aligned_tags.append(next(tag_iter, '<pad>'))  # Next word tag or pad
            prev_end = end
    
        assert len(aligned_tags) == len(offset_mapping)
    
    
        return aligned_tags

    def align_tags_with_subtokens(self,all_updated_tags,encoded_inputs):
        all_aligned_tags = []
        for index in range(len(all_updated_tags)):
            tags = all_updated_tags[index]
            offset_mapping = encoded_inputs['offset_mapping'][index]
            aligned_tags   = self.align_tags(tags,offset_mapping)
            all_aligned_tags.append(aligned_tags)
        return all_aligned_tags

    def get_labels_by_index(self,index, integer_labels = False):
        if not integer_labels:
            return self.all_aligned_tags[index]
        else:
            return ConlluPosDataset.shared_label_encoder.transform(self.all_aligned_tags[index])

    
    def build_dataset(self):
        """
        Create a Hugging Face Dataset from aligned tags and tokenized inputs.
        
        Returns:
            datasets.Dataset: A Hugging Face dataset with input_ids, attention_mask, and labels.
        """

        #self.all_aligned_tags
        #self.encoded_inputs
        data = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": [-100 if tag == ConlluPosDataset.shared_label_encoder.transform(["<pad>"]).item() else tag for tag in ConlluPosDataset.shared_label_encoder.transform(tags).tolist()]
            }
            for input_ids, attention_mask, tags in zip(
                self.encoded_inputs["input_ids"],
                self.encoded_inputs["attention_mask"],
                self.all_aligned_tags
            )
        ]
    
        return Dataset.from_list(data)


    def number_of_classes(self):
        return len(ConlluPosDataset.shared_label_encoder.classes_)

    @property
    def labels(self):
        return self.all_aligned_tags
