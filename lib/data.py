# inbuilt lib imports:
from typing import List, Dict, Tuple, Any, NamedTuple
import math

# external lib imports:
import numpy as np
from tqdm import tqdm

# project imports
from lib.dependency_tree import DependencyTree
from lib.parsing_system import ParsingSystem
from lib.configuration import Configuration
from lib.vocabulary import Vocabulary

class Token(NamedTuple):

    word: str = None
    pos: str = None
    head: int = None
    dep_type: str = None

Sentence = List[Token]


def read_conll_data(data_file_path: str) -> Tuple[List[Sentence], List[DependencyTree]]:
    """
    Reads Sentences and Trees from a CONLL formatted data file.

    Parameters
    ----------
    data_file_path : ``str``
        Path to data to be read.
    """
    sentences: List[Sentence] = []
    trees: List[DependencyTree] = []

    with open(data_file_path, 'r') as file:
        sentence_tokens = []
        tree = DependencyTree()
        for line in tqdm(file):
            line = line.strip()
            array = line.split('\t')
            if len(array) < 10:
                if sentence_tokens:
                    trees.append(tree)
                    sentences.append(sentence_tokens)
                    tree = DependencyTree()
                    sentence_tokens = []
            else:
                word = array[1]
                pos = array[4]
                head = int(array[6])
                dep_type = array[7]
                token = Token(word=word, pos=pos,
                              head=head, dep_type=dep_type)
                sentence_tokens.append(token)
                tree.add(head, dep_type)

    if not sentences:
        raise Exception(f"No sentences read from {data_file_path}. "
                        f"Make sure you have not replaced tabs with spaces "
                        f"in conll formatted file by mistake.")

    return sentences, trees


def write_conll_data(output_file: str,
                     sentences: List[Sentence],
                     trees: List[DependencyTree]) -> None:
    """
    Writes Sentences and Trees into a CONLL formatted data file.
    """
    with open(output_file, 'w') as fout:
        for i in range(len(sentences)):
            sent = sentences[i]
            tree = trees[i]
            for j in range(len(sent)):
                fout.write("%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t_\n"
                           % (j+1, sent[j].word, sent[j].pos,
                              sent[j].pos, tree.get_head(j+1), tree.get_label(j+1)))
            fout.write("\n")


def generate_training_instances(parsing_system: ParsingSystem,
                                sentences: List[List[str]],
                                vocabulary: Vocabulary,
                                trees: List[DependencyTree]) -> List[Dict]:
    """
    Generates training instances of configuration and transition labels
    from the sentences and the corresponding dependency trees.
    """
    num_transitions = parsing_system.num_transitions()
    instances: Dict[str, List] = []
    for i in tqdm(range(len(sentences))):
        if trees[i].is_projective():
            c = parsing_system.initial_configuration(sentences[i])
            while not parsing_system.is_terminal(c):
                oracle = parsing_system.get_oracle(c, trees[i])
                feature = get_configuration_features(c, vocabulary)
                label = []
                for j in range(num_transitions):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.can_apply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)
                if 1.0 not in label:
                    print(i, label)
                instances.append({"input": feature, "label": label})
                c = parsing_system.apply(c, oracle)
    return instances


def get_configuration_features(configuration: Configuration,
                               vocabulary: Vocabulary) -> List[List[int]]:
    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    # TODO(Students) Start

    features = []  # Store list of all features (initially only word features)
    pos_features = []  # Store list of pos tag features
    label_features = []  # Store list of label features

    top3Stack = [configuration.get_stack(i) for i in range(3)]  # top 3 elements of stack
    top3Buffer = [configuration.get_buffer(i) for i in range(3)]  # top 3 elements of buffer

    for token_index in top3Stack + top3Buffer:  # Iterate over top 3 words in stack and top 3 words in buffer
        # Add word to the features
        features.append(vocabulary.get_word_id(configuration.get_word(token_index)))
        # Add pos tag of corresponding word to the pos_features
        pos_features.append(vocabulary.get_pos_id(configuration.get_pos(token_index)))

    for token_index in top3Stack[:2]:  # Iterate over top 2 words in stack
        # Iterate over 1 and 2 to get 1st leftmost, 1st rightmost, 2nd leftmost and 2nd rightmost child
        # of corresponding word in stack.
        for i in range(1, 3):
            ith_left_child = configuration.get_left_child(token_index, i)  # Get ith_leftmost_child of word in stack
            # Add child to the features
            features.append(vocabulary.get_word_id(configuration.get_word(ith_left_child)))
            # Add pos tag of corresponding child to the pos_features
            pos_features.append(vocabulary.get_pos_id(configuration.get_pos(ith_left_child)))
            # Add label of corresponding child to the label_features
            label_features.append(vocabulary.get_label_id(configuration.get_label(ith_left_child)))

            # Similarly for rightmost child add child word, pos tag and label to respective features list
            ith_right_child = configuration.get_right_child(token_index, i)
            features.append(vocabulary.get_word_id(configuration.get_word(ith_right_child)))
            pos_features.append(vocabulary.get_pos_id(configuration.get_pos(ith_right_child)))
            label_features.append(vocabulary.get_label_id(configuration.get_label(ith_right_child)))

    for token_index in top3Stack[:2]:  # Iterate over top 2 words in stack
        # Get leftmost child of leftmost child of word in stack
        left_left_child = configuration.get_left_child(configuration.get_left_child(token_index, 1), 1)
        # Add the corresponding child word, pos tag and label to respective features list
        features.append(vocabulary.get_word_id(configuration.get_word(left_left_child)))
        pos_features.append(vocabulary.get_pos_id(configuration.get_pos(left_left_child)))
        label_features.append(vocabulary.get_label_id(configuration.get_label(left_left_child)))

        # Similarly for rightmost child of rightmost child add child word, pos tag and label to respective features list
        right_right_child = configuration.get_right_child(configuration.get_right_child(token_index, 1), 1)
        features.append(vocabulary.get_word_id(configuration.get_word(right_right_child)))
        pos_features.append(vocabulary.get_pos_id(configuration.get_pos(right_right_child)))
        label_features.append(vocabulary.get_label_id(configuration.get_label(right_right_child)))

    features += pos_features + label_features  # Append the pos and label features to the word features

    # TODO(Students) End

    assert len(features) == 48
    return features


def generate_batches(instances: List[Dict],
                     batch_size: int) -> List[Dict[str, np.ndarray]]:
    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """

    def chunk(items: List[Any], num: int) -> List[Any]:
        return [items[index:index+num] for index in range(0, len(items), num)]
    batches_of_instances = chunk(instances, batch_size)

    batches = []
    for batch_of_instances in tqdm(batches_of_instances):
        count = min(batch_size, len(batch_of_instances))
        features_count = len(batch_of_instances[0]["input"])

        batch = {"inputs": np.zeros((count, features_count), dtype=np.int32)}
        if "label" in  batch_of_instances[0]:
            labels_count = len(batch_of_instances[0]["label"])
            batch["labels"] = np.zeros((count, labels_count), dtype=np.int32)

        for batch_index, instance in enumerate(batch_of_instances):
            batch["inputs"][batch_index] = np.array(instance["input"])
            if "label" in instance:
                batch["labels"][batch_index] = np.array(instance["label"])

        batches.append(batch)

    return batches


def load_embeddings(embeddings_txt_file: str,
                    vocabulary: Vocabulary,
                    embedding_dim: int) -> np.ndarray:

    vocab_id_to_token = vocabulary.id_to_token
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings: Dict[str, np.ndarray] = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file) as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim),
                                        scale=1./math.sqrt(embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix
