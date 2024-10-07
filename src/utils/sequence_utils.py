import re
import torch
import random
import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY' # 20 canonical amino acids
AA_TO_IDX = {aa:idx for idx, aa in enumerate(AMINO_ACIDS)}
EPS = 1e-10

def generate_single_mutants(ref_seq, sites_to_exclude=[]):
    mutants = []
    L = len(ref_seq)  # Length of the reference sequence

    for i, original_aa in enumerate(ref_seq):
        site = i + 1  # 1-based site index
        if site in sites_to_exclude:
            continue  # Skip sites that are in the exclusion list

        for aa in AMINO_ACIDS:
            if aa != original_aa:  # Only consider mutations
                mutant = f"{original_aa}{site}{aa}"
                mutants.append(mutant)
    return mutants

def pseudolikelihood_ratio(rel_seq, ref_seq, logits):
    # rel_seq is a dictionary {site: aa}, and ref_seq is an absolute sequence in a string. Logits are (L, 20).
    # Using logprobs gives the same answer.
    score = 0.0
    for site, mutated_aa in rel_seq.items():
        site_idx = int(site)  # Already 0-based index
        mutated_aa_idx = AA_TO_IDX[mutated_aa]
        reference_aa_idx = AA_TO_IDX[ref_seq[site_idx]]
        # Compute score differences for the mutated sequence
        logprob_diff_mutant = logits[site_idx, mutated_aa_idx] - logits[site_idx, reference_aa_idx]
        score += logprob_diff_mutant.item()
    return score

def pseudolikelihood_ratio_from_tensor(rel_seq_tensor, ref_seq, logits):
    # rel_seq_tensor is a (N, 2) tensor where the first column is the one-based site index
    # and the second column is the amino acid index. N is number of mutations, with padding of nan's at the end.
    # ref_seq is an absolute sequence string. Logits are (L, 20).
    score = 0.0
    for mutation in rel_seq_tensor: # this loops over the rows
        if torch.isnan(mutation[0]):  # Stop processing when encountering NaN
            break
        site_idx = int(mutation[0].item()) - 1  # Convert to 0-based index
        mutated_aa_idx = int(mutation[1].item())
        reference_aa_idx = AA_TO_IDX[ref_seq[site_idx]]
        # Compute score differences for the mutated sequence
        logprob_diff_mutant = logits[site_idx, mutated_aa_idx] - logits[site_idx, reference_aa_idx]
        score += logprob_diff_mutant.item()
    return score

def compute_num_mutations_from_padded_rel_seqs_tensors(rel_seqs_tensors):
    num_mutations = []
    for rel_seq_tensor in rel_seqs_tensors:
        count = 0
        for mutation in rel_seq_tensor:
            if torch.isnan(mutation[0]):
                break
            count += 1
        num_mutations.append(count)
    return torch.tensor(num_mutations, dtype=torch.float32).unsqueeze(-1)

def pad_rel_seq_tensors_with_nan(rel_seqs_tensors):
    # pad rel seq tensors so that they all have the same size for DataParallel to work correctly
    # Determine the maximum length of sequences in the batch
    max_length = max(tensor.size(0) for tensor in rel_seqs_tensors)

    padded_tensors = []
    for tensor in rel_seqs_tensors:
        pad_size = max_length - tensor.size(0)
        if pad_size > 0:
            # Create padding tensor with NaN values
            padding = torch.full((pad_size, 2), torch.nan, dtype=torch.float32)
            # Concatenate the original tensor with the padding tensor
            padded_tensor = torch.cat((tensor.float(), padding), dim=0)
        else:
            padded_tensor = tensor.float()
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors into a single tensor
    padded_tensors = torch.stack(padded_tensors)

    return padded_tensors

def convert_rel_seqs_dicts_to_abs_tensors(ref_seq_tensor, rel_seqs_dicts):
    abs_seqs_tensors = []
    for rel_seq_dict in rel_seqs_dicts:
        abs_seq_tensor = ref_seq_tensor.clone()
        for pos, aa in rel_seq_dict.items():
            abs_seq_tensor[pos - 1] = AA_TO_IDX[aa]  # Adjust for zero-based indexing
        abs_seqs_tensors.append(abs_seq_tensor)
    return abs_seqs_tensors

def split_by_sequence(sequences, split_ratio=0.8, ref_seq='NA', return_sequences=False):
    """
    Splits a given list of sequences into training and validation sets, ensuring that the reference sequence
    (identified as `ref_seq`) and a proportion of unique non-reference sequences are included in the training set
    according to the specified split ratio.

    Arguments:
    - sequences: A list of sequences (strings) where the reference sequence name is identified by `ref_seq`.
    - split_ratio (float, optional): The ratio of the dataset to be included in the training set.
      Defaults to 0.8, meaning 80% training and 20% validation.
    - ref_seq (str, optional): The reference sequence name. Defaults to 'NA'.

    Outputs:
    - train_sequences: A list of sequences intended for training, containing a mix of reference and non-reference
      sequences according to the split ratio.
    - val_sequences: A list of sequences intended for validation, containing the remaining sequences not included in
      the training set.

    # Example usage:
    sequences = ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC", "DDD", "AAA"]
    ref_seq = "AAA"
    train_sequences, val_sequences = split_sequences_by_reference(sequences, split_ratio=0.8, ref_seq=ref_seq)

    print("Train sequences:", train_sequences)
    print("Validation sequences:", val_sequences)

    """
    # Step 1: Isolate reference sequence measurements
    ref_seq_indices = [idx for idx, seq in enumerate(sequences) if seq == ref_seq]
    non_ref_seq_indices = [idx for idx, seq in enumerate(sequences) if seq != ref_seq]
    print(f"Found {len(ref_seq_indices)} wildtype sequences in dataset.")
    # Prepare for counting frequencies of non-reference sequences
    sequence_to_indices = {}
    for idx in non_ref_seq_indices:
        seq = sequences[idx]
        if seq not in sequence_to_indices:
            sequence_to_indices[seq] = []
        sequence_to_indices[seq].append(idx)

    n_unique = len(sequence_to_indices) + 1

    # Step 2: Shuffle the list of non-reference unique sequences
    unique_non_ref_sequences = list(sequence_to_indices.keys())
    random.shuffle(unique_non_ref_sequences)

    # Step 3: Allocate sequences to training set based on frequency, starting with the reference sequence
    train_indices = ref_seq_indices
    cnt = 1
    for seq in unique_non_ref_sequences:
        if len(train_indices) / len(sequences) < split_ratio:
            train_indices.extend(sequence_to_indices[seq])
            cnt += 1
        else:
            break

    # Remaining sequences go to validation set
    val_indices = [idx for idx in non_ref_seq_indices if idx not in train_indices]

    print(f"train has {cnt} unique sequences out of {n_unique}.")
    if return_sequences:
        # Extract the sequences using the indices
        train_sequences = [sequences[idx] for idx in train_indices]
        val_sequences = [sequences[idx] for idx in val_indices]
        return train_sequences, val_sequences

    return train_indices, val_indices

def convert_rel_seqs_to_tensors(rel_seqs_dicts):
    # Assumes rel_seqs_dicts has one-based sites, e.g., [{1: F, 87: G}, ...]
    rel_seqs_tensors = []
    for rel_seq_dict in rel_seqs_dicts:
        if rel_seq_dict == {}:
            seq_tensor = torch.empty((0, 2), dtype=torch.long)
        else:
            seq_tensor = torch.tensor([[pos, AA_TO_IDX[aa]] for pos, aa in rel_seq_dict.items()], dtype=torch.long)
        rel_seqs_tensors.append(seq_tensor)
    return rel_seqs_tensors

def list_of_dicts_from_list_of_sequence_strings(sequences, ref_seq):
    # Inputs: list of full amino acid sequences, and reference sequence.
    # Output: list of dictionaries with {site: mutant_amino_acid} for each sequence.
    ref_seq_len = len(ref_seq)
    num_mutations = []
    rel_seqs = []

    for seq in sequences:
        mutations = 0
        rel_seq = {}

        for i, (a, b) in enumerate(zip(seq, ref_seq)):
            if a != b:
                mutations += 1
                rel_seq[i+1] = a

        # Handle sequences longer or shorter than the reference sequence
        if len(seq) > ref_seq_len:
            mutations += len(seq) - ref_seq_len
            for i in range(ref_seq_len, len(seq)):
                rel_seq[i+1] = seq[i]
        elif len(seq) < ref_seq_len:
            mutations += ref_seq_len - len(seq)
            for i in range(len(seq), ref_seq_len):
                rel_seq[i+1] = '-'

        num_mutations.append(mutations)
        rel_seqs.append(rel_seq)

def rel_sequences_to_dict(rel_sequences, sep='-'):
    """
    Converts a list of relative sequence strings to a nested dictionary.
    :param rel_sequences: List of strings, each containing concatenated mutations.
    :param sep: Separator used between mutations in the relative sequences.
    :return: A nested dictionary where the first key is the zero-based index of the sequences,
             and the second level key is the one-based mutation site, with the value being the mutated amino acid.
    """
    sequences_dict = {}
    for idx, rel_seq in enumerate(rel_sequences):
        sequence_mutations = {}
        mutations = rel_seq.split(sep)
        for mutation in mutations:
            if mutation:  # Non-empty mutation string
                #print(mutation)
                original_aa, site, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
                sequence_mutations[site] = mutated_aa
        sequences_dict[idx] = sequence_mutations
    return sequences_dict

def rel_seq_dict_to_list(rel_seq_dict, ref_seq, sep='-'):
    """
    Converts a nested dictionary of sequence mutations back into a list of relative sequence strings.
    :param rel_seq_dict: Nested dictionary from sequence indices to site-mutation pairs.
    :param ref_seq: The reference sequence string.
    :param sep: Separator used between mutations in the relative sequences.
    :return: A list of strings, each containing concatenated mutations.
    """
    rel_sequences = []
    for idx in sorted(rel_seq_dict.keys()):  # Ensure the order is maintained
        mutations_list = []
        for site, mutated_aa in sorted(rel_seq_dict[idx].items(), key=lambda x: x[0]):  # Sort by site
            original_aa = ref_seq[site - 1]  # Convert one-based site to zero-based index for ref_seq
            mutation_str = f"{original_aa}{site}{mutated_aa}"
            mutations_list.append(mutation_str)
        rel_sequence = sep.join(mutations_list)
        rel_sequences.append(rel_sequence)
    return rel_sequences

def rel_seq_list_to_strings(rel_seq_list, ref_seq, sep='-'):
    """
    Converts a list of dictionaries of sequence mutations into a list of relative sequence strings.
    :param rel_seq_list: List of dictionaries, each representing site-mutation pairs for a sequence.
    :param ref_seq: The reference sequence string.
    :param sep: Separator used between mutations in the relative sequences.
    :return: A list of strings, each containing concatenated mutations.
    """
    rel_sequences = []
    for mutation_dict in rel_seq_list:  # Iterate directly over the list of dictionaries
        mutations_list = []
        for site, mutated_aa in sorted(mutation_dict.items(), key=lambda x: x[0]):  # Sort by site
            original_aa = ref_seq[site - 1]  # Convert one-based site to zero-based index for ref_seq
            mutation_str = f"{original_aa}{site}{mutated_aa}"
            mutations_list.append(mutation_str)
        rel_sequence = sep.join(mutations_list)
        rel_sequences.append(rel_sequence)
    return rel_sequences

def count_mutations_in_sequences(sequence_dict):
    """
    Counts the number of mutations in each sequence represented by a nested dictionary.

    :param sequence_dict: A nested dictionary where the first key is the zero-based index of the sequences,
                          and the second level key is the one-based mutation site, with the value being the mutated amino acid.
    :return: A dictionary mapping each sequence index to the number of mutations in that sequence.
    """
    mutation_counts = {}
    for seq_idx, mutations in sequence_dict.items():
        mutation_counts[seq_idx] = len(mutations)  # The number of mutations is simply the size of the inner dictionary
    return mutation_counts

def build_dict_of_mutation_counts(sequences, sep = '-'):
    """
    Processes a list of sequence strings and produces a dictionary of mutation counts.

    :param sequences: List of sequence strings, where each string represents a sequence of mutations.
                      Example format: ["A1B-C2D", "A1B", ...]
    :return: Dictionary with structure mutations['site']['aa'] = cnt.
    """
    mutations = {}
    mutation_pattern = re.compile(r"([A-Z])(\d+)([A-Z])")

    for seq in sequences:
        mutations_list = seq.split('-')
        for mutation in mutations_list:
            match = mutation_pattern.match(mutation)
            if match:
                original_aa, site, mutated_aa = match.groups()
                site = int(site)  # Convert site to integer for consistency
                if site not in mutations:
                    mutations[site] = {}
                if mutated_aa not in mutations[site]:
                    mutations[site][mutated_aa] = 0
                mutations[site][mutated_aa] += 1

    return mutations

def count_unique_mutations(mutations):
    """
    Counts the number of unique mutations in the dictionary.

    :param mutations: Dictionary with structure mutations['site']['aa'] = cnt.
    :return: The count of unique mutations.
    To be used on the output of the function above. So mutations[site][aa] has counts.
    This just looks at total number of site, aa combos.
    """
    unique_mutation_count = 0
    for site, aa_dict in mutations.items():
        #print(f"Site: {site}, Mutations: {aa_dict}")  # Debugging line
        unique_mutation_count += len(aa_dict)
    return unique_mutation_count

def dedupe_and_map_sites_to_seqs(seq_names):
    """
    Creates a mapping from each site to sequences (deduped) that have mutations at that site,
    and a mapping from deduped sequence names to original sequence indices in the dataset.
    """
    # Deduplicate sequences while preserving their order
    deduped_seqs = list(dict.fromkeys(seq_names))
    site_to_seqs = {}
    seq_to_original_indices = {seq: [] for seq in deduped_seqs}

    # Initialize a mapping for each unique sequence to its occurrences in the dataset
    for idx, seq_name in enumerate(seq_names):
        if seq_name in seq_to_original_indices:
            seq_to_original_indices[seq_name].append(idx)

    # Populate site_to_seqs with deduped sequences
    for seq_name in deduped_seqs:
        sites = extract_sites_from_seqname(seq_name)
        for site in sites:
            if site not in site_to_seqs:
                site_to_seqs[site] = set()
            site_to_seqs[site].add(seq_name)

    return site_to_seqs, seq_to_original_indices

def extract_sites_from_seqname(seq_name):
    """
    Extracts sites from a sequence name.
    Example sequence name: 'C17B-F123A' -> returns [17, 123]
    """
    # Pattern to match site numbers in the sequence name
    pattern = re.compile(r'(\d+)')
    sites = pattern.findall(seq_name)
    return [int(site) for site in sites if site.isdigit()]

def compute_number_of_mutations(sequences, separator='-'):
    '''
    Compute the distribution (count, really) of the number of mutations in each sequence.
    :param sequences: List of sequences with mutations in string format, relative to a reference sequence.
    :param separator: Separator used in representing mutations.
    :return: A list with the number of mutations in each sequence. Very inefficient.
    '''
    num_mutations = [0 if seq == 'NA' else len(seq.split(separator)) for seq in sequences]

    return num_mutations

def compute_scores_and_mutations(rel_seqs, mutations, mut_scores, separator='-'):
    # Example usage
    #rel_seqs = ['A23C-B67F', 'C17B']
    #mutations = ['A23C', 'B67F', 'C17B']
    # assumes no NAs
    #mut_scores = [1.2, 0.8, 1.0]
    #scores, num_mutations = compute_scores_and_mutations(rel_seqs, mutations, mut_scores)
    # Convert mutations to a dictionary for faster lookup
    mutation_to_score = {mutation: score for mutation, score in zip(mutations, mut_scores)}

    scores = []
    num_mutations = []

    for seq in rel_seqs:
        sum_scores = 0.0
        # Split each sequence into its constituent mutations

        constituent_mutations = seq.split(separator)
        # Initialize sum of scores for this sequence
        for mutation in constituent_mutations:
            # Add score if mutation is in the dictionary
            if mutation in mutation_to_score:
                sum_scores += mutation_to_score[mutation]
        scores.append(sum_scores)
        if seq == '':  # parent sequence
            num_mutations.append(0)
        else:
            num_mutations.append(len(constituent_mutations))
    return scores, num_mutations

def compute_amino_acid_distribution(sequences, separator='-'):
    '''
    Compute the distribution of amino acids in the mutations across all sequences.
    :param sequences: List of sequences with mutations.
    :param separator: Separator used in representing mutations.
    :return: A dictionary with amino acids as keys and their counts as values.  Very inefficient.
    '''
    amino_acid_counts = {}
    for seq in sequences:
        mutations = seq.split(separator)
        for mutation in mutations:
            amino_acid = mutation[-1]  # The last character in mutation string
            amino_acid_counts[amino_acid] = amino_acid_counts.get(amino_acid, 0) + 1
    return amino_acid_counts

def create_absolute_sequences(relative_sequences, reference_sequence, separator='-'):
    """
    Create a dictionary of absolute sequences from a list of relative sequences and a reference sequence.
    The input relative sequences can be either in string format (e.g., "A1Y-N5Q") or dictionary format (e.g., {1: 'Y', 5: 'Q'}).
    Note: Outputs a dictionary, so duplicated sequences get mapped to a single entry.

    Args:
        relative_sequences (list): A list of relative sequences, each being either a string or a dictionary.
        reference_sequence (str): The reference sequence to which the mutations will be applied.
        separator (str): Separator used between mutations in the string format. Default is '-'.

    Returns:
        dict: A dictionary mapping the original relative sequence format to the absolute sequence.

    Example Usage:
        reference_seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Example reference sequence
        relative_seqs = ["A1Y-N5Q-P10Q-Y15T", {3: 'W', 20: 'R'}]  # Mixed example list of relative sequences
        absolute_seq_dict = create_absolute_sequences(relative_seqs, reference_seq)
        print(absolute_seq_dict)
    """
    absolute_sequences = {}

    for rel_seq in relative_sequences:
        if isinstance(rel_seq, str):
            if rel_seq == 'NA' or rel_seq == '':
                absolute_sequences[rel_seq] = reference_sequence
                continue
            # Process string-formatted relative sequences
            mutations = [mutation.strip() for mutation in rel_seq.split(separator)]
        elif isinstance(rel_seq, dict):
            # Process dictionary-formatted relative sequences
            mutations = [f"{reference_sequence[pos - 1]}{pos}{aa}" for pos, aa in rel_seq.items()]
            # Convert dict back to string format for uniform handling in the dictionary output
            rel_seq = separator.join(mutations)
        else:
            continue  # Skip if the format is unknown

        # Start with the reference sequence
        absolute_seq = list(reference_sequence)

        for mutation in mutations:
            original_aa, site, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
            if original_aa != reference_sequence[site - 1]:  # Check for consistency
                raise ValueError(f"Inconsistency found in {mutation}")
            absolute_seq[site - 1] = mutated_aa

        # Join the list back into a string and add to the dictionary
        absolute_sequences[rel_seq] = ''.join(absolute_seq)

    return absolute_sequences

def create_absolute_sequences_list(relative_sequences, reference_sequence, separator='-'):
    """
    Create a list of absolute sequences from a list of relative sequences and a reference sequence.
    This function supports input in both string format ("A1Y-N5Q-P10Q-Y15T") and dictionary format ({1: 'Y', 5: 'Q', 10: 'Q', 15: 'T'}).
    It maintains duplicates, unlike a version that outputs a dictionary.

    Parameters:
    relative_sequences (list): A list containing relative sequences either as strings or as dictionaries.
    reference_sequence (str): The sequence to be used as reference for applying mutations.
    separator (str): The character used to separate mutations in the string representation of relative sequences.
                     Default is '-'.

    Returns:
    list: A list of absolute sequences, preserving the order and duplicates from the input.

    Example Usage:
    reference_seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    relative_seqs = ["A1Y-N5Q-P10Q-Y15T", {"3": "W", "20": "R"}]
    absolute_seqs_list = create_absolute_sequences_list(relative_seqs, reference_seq)
    print(absolute_seqs_list)
    """
    absolute_sequences = []

    for rel_seq in relative_sequences:
        # Initialize the absolute sequence as a list for efficient mutation
        absolute_seq = list(reference_sequence)

        if rel_seq == 'NA' or rel_seq == '':  # Special case where 'NA' means use the reference sequence without changes
            pass
        elif isinstance(rel_seq, str):
            # The relative sequence is in string format: split and apply each mutation
            mutations = [mutation.strip() for mutation in rel_seq.split(separator)]
            for mutation in mutations:
                original_aa, site, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
                absolute_seq[site - 1] = mutated_aa  # Apply mutation
        elif isinstance(rel_seq, dict):
            # The relative sequence is in dictionary format: apply each mutation
            for site, mutated_aa in rel_seq.items():
                site = int(site)  # Convert site to integer
                absolute_seq[site - 1] = mutated_aa  # Apply mutation
        else:
            # Unsupported format, raise an error
            raise ValueError("Unsupported relative sequence format")

        # Join the list back into a string and add to the output list
        absolute_sequences.append(''.join(absolute_seq))

    return absolute_sequences

def apply_mutations(reference_sequence, mutations_dict):
    """
    Applies the specified mutations to the reference sequence.

    Args:
        reference_sequence (str): The original amino acid sequence.
        mutations_dict (dict): A dictionary where keys are the positions (1-based indexing) of mutations
                               and values are the mutated amino acids.

    Returns:
        str: The mutated amino acid sequence.
    """
    if mutations_dict == {}:
        return reference_sequence # no mutations
    # Convert the reference sequence to a list of characters for easy manipulation
    mutated_sequence = list(reference_sequence)
    # Iterate through the mutations and apply each one to the sequence
    for position, new_aa in mutations_dict.items():
        # Convert the 1-based indexing used in mutations_dict to 0-based indexing used in lists
        index = position - 1
        # Replace the amino acid at the specified position with the new one
        mutated_sequence[index] = new_aa
    # Convert the list back into a string and return

    return ''.join(mutated_sequence)

def calculate_amino_acid_frequencies(aminos):
    '''Return the frequencies of each amino acid in a collection of amino acids'''

    aminos = ''.join(aminos)
    p = [aminos.count(amino) for amino in AMINO_ACIDS]

    return np.asarray(p) / sum(p)

def one_hot_encode_seq(sequence, flatten=True):
    '''Return a one-hot encoding of an amino-acid sequence. If flatten is True, return a vector of
    length 20xL, else return a 2D array of shape (20, L)'''

    l = len(sequence)
    x = np.zeros((20,l))
    for i,aa in enumerate(sequence):
        x[AA_TO_IDX[aa],i] = 1
    if flatten:
        x = x.flatten()

    return x

def compute_Neff_for_probmatrix(p):
    def compute_Neff(sorted_p):
        n = len(sorted_p)
        x = np.sum(sorted_p * np.arange(1, n + 1))
        Neff = 2 * x - 1
        return Neff

    if len(p.shape) == 1:
        # p is a vector
        sorted_p = np.sort(p)[::-1]  # Sort in descending order
        Neff = compute_Neff(sorted_p)
        return {"Neff": Neff}
    elif len(p.shape) == 2:
        # p is a matrix (joint probability distribution)

        # Flatten the matrix and compute Neff
        flattened_p = p.flatten()
        sorted_flattened_p = np.sort(flattened_p)[::-1]
        Neff = compute_Neff(sorted_flattened_p)

        # Compute Neff for the marginal distribution of columns
        marginal_cols = np.sum(p, axis=0)
        sorted_marginal_cols = np.sort(marginal_cols)[::-1]
        Neff_cols = compute_Neff(sorted_marginal_cols)

        # Compute Neff for the marginal distribution of rows
        marginal_rows = np.sum(p, axis=1)
        sorted_marginal_rows = np.sort(marginal_rows)[::-1]
        Neff_rows = compute_Neff(sorted_marginal_rows)

        return {"Neff": Neff, "Neff_cols": Neff_cols, "Neff_rows": Neff_rows}
    else:
        raise ValueError("Input p must be a 1D or 2D numpy array.")

def create_score_matrix(reference_sequence, single_mutants, ml_scores, wt_scores = 0.0):
    """
    Create a score matrix from single mutant strings and their corresponding ML scores.

    Args:
        reference_sequence (str): Wild-type sequence.
        single_mutants (list[str]): List of single mutants in the format 'A456Y'.
        ml_scores (np.ndarray): Array of ML scores corresponding to the single mutants.
        wt_scores (float): the default value of wild type scores.

    Returns:
        np.ndarray: Score matrix of shape (20, L), where L is the length of the reference sequence.
    Note that because only single mutants are found, the wildtype scores are set to the default of zero.
    """
    # Initialize score matrix
    L = len(reference_sequence)
    score_matrix = np.ones((20, L))*wt_scores

    # Fill the score matrix
    for mutant, score in zip(single_mutants, ml_scores):
        wt_aa = mutant[0]
        site = int(mutant[1:-1]) - 1  # Convert to 0-based index
        mut_aa = mutant[-1]
        #wt_idx = AA_TO_IDX[wt_aa]
        mut_idx = AA_TO_IDX[mut_aa]
        score_matrix[mut_idx, site] = score

    return score_matrix
