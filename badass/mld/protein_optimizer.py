# badass/mld/protein_optimizer.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import abc
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from badass.mld.sequence_sampler import CombinatorialMutationSampler
from badass.utils.sequence_utils import AA_TO_IDX

@dataclass
class BaseOptimizerParams:
    """Parameters for the base protein sequence optimizer."""
    ref_sequence: str
    score_threshold: Optional[float] = 35
    reversal_threshold: Optional[float] = 0.0
    ref_score_value: Optional[float] = None
    ref_score_scale: Optional[float] = None
    seqs_per_iter: int = 500
    num_iter: int = 200 
    init_score_batch_size: int = 500
    T: float = 1.5
    seed: int = 7
    gamma: float = 0.5
    cooling_rate: float = 0.92
    num_mutations: int = 5
    sites_to_ignore: List[int] = field(default_factory=lambda: [1])
    boost_mutations_with_high_variance: bool = True
    sample_variety_of_mutation_numbers: bool = False
    num_sequences_proportions: List[float] = field(default_factory=lambda: [0, 1, 1, 1, 1, 1])
    n_seqs_to_keep: Optional[int] = None
    simple_simulated_annealing: bool = False
    cool_then_heat: bool = False
    normalize_scores: bool = True
    save_path: Optional[Path] = None

class BaseProteinOptimizer(abc.ABC):
    """Base class for protein sequence optimization using simulated annealing.

    Implements model agnostic logic::
        - Sequence sampling and mutation exploration
        - Temperature scheduling and annealing
        - Phase transition detection
        - Score matrix updates and normalization
        - Results tracking and preparation

    Child classes must implement:
        - setup_model(): Initialize the scoring model and set reference sequence
        - score_batch(): Score a batch of sequences using the model

    The base class handles all other optimization functionality.

    Attributes:
        params (BaseOptimizerParams): Parameters controlling the optimization process
        ref_sequence (str): Reference/wild-type sequence, set by setup_model()
        sampler (CombinatorialMutationSampler): Handles sequence mutation and sampling
        score_matrix (np.ndarray): Current score matrix for mutation sampling
        initial_score_matrix (np.ndarray): Original score matrix from single mutants
        sum_of_scores_matrix (np.ndarray): Running sum of scores per mutation
        mut_to_num_seqs_matrix (np.ndarray): Count of observations per mutation
        sum_of_scores_squared_matrix (np.ndarray): Running sum of squared scores
        scored_sequences (dict): Cache of sequence scores
        all_sequences (list): All sequences sampled
        all_scores (list): Corresponding scores
        all_variances (list): Score variance per iteration
        all_mean_ml_scores (list): Mean score per iteration
        all_Ts (list): Temperature per iteration
        all_num_sampled_seqs (list): Number of sequences per iteration
        all_num_new_seqs (list): Number of new sequences per iteration
        all_phase_transition_numbers (list): Phase transitions per iteration
        active_phase_transition (bool): Whether currently in phase transition
        first_phase_transition (int): Iteration of first phase transition
        last_phase_transition (int): Iteration of most recent phase transition
        num_phase_transitions (int): Total number of phase transitions

    Phase Transitions:
        The optimizer can detect and respond to phase transitions in the optimization landscape.
        A phase transition is detected when:
        1. The mean score crosses a threshold (params.score_threshold)
        2. Score variance shows significant change
        
        During a phase transition, the temperature schedule is modified to:
        1. Initially increase temperature to promote exploration
        2. Return to cooling once a reversal is detected
        3. Maintain phase state until stability is reached

    Temperature Scheduling:
        Temperature is updated based on several factors:
        1. Number of successful mutations per iteration
        2. Phase transition state
        3. Selected schedule (simple annealing or cool-then-heat)
        4. Current optimization stage

    Score Normalization:
        When params.normalize_scores is True:
        1. Scores are normalized using reference value and scale
        2. Reference value defaults to 80th percentile of initial scores
        3. Scale defaults to standard deviation of initial scores
        
    Matrix Updates:
        Three matrices track mutation statistics:
        1. sum_of_scores_matrix: Running sum of scores per mutation
        2. mut_to_num_seqs_matrix: Count of times each mutation is observed
        3. sum_of_scores_squared_matrix: Running sum of squared scores
        
        These are used to compute:
        1. Mean scores per mutation
        2. Score variance per mutation
        3. Variance-boosted scores when params.boost_mutations_with_high_variance is True

    Results:
        The optimize() method returns two DataFrames:
        1. Sequence results:
        - sequences: Unique sequences found
        - ml_score: Final scores
        - counts: Times each sequence was sampled
        - num_mutations: Mutations per sequence
        - iteration: First iteration each sequence was found
        
        2. Optimization statistics:
        - iteration: Optimization iteration
        - avg_ml_score: Mean score per iteration
        - var_ml_score: Score variance per iteration
        - T: Temperature per iteration
        - n_seqs: Sequences sampled per iteration
        - n_new_seqs: New sequences per iteration
        - num_phase_transitions: Cumulative phase transitions

    Example:
        class MyOptimizer(BaseProteinOptimizer):
            def setup_model(self):
                self.model = MyModel()
                
            def score_batch(self, sequences):
                return self.model.score_sequences(sequences)
                
        optimizer = MyOptimizer(params)
        results_df, stats_df = optimizer.optimize()

    See Also:
        BaseOptimizerParams: Configuration parameters
        CombinatorialMutationSampler: Sequence sampling logic
    """
    
    def __init__(self, params: BaseOptimizerParams):
        """Initialize the optimizer with parameters."""
        self.params = params
        self._setup_optimizer_params()
        self._set_seeds()
        self.ref_sequence = self.params.ref_sequence
        
        # Initialize tracking attributes
        self.all_sequences = []
        self.all_scores = []
        self.iters_for_seqs = []
        self.n_eff_joint = []
        self.n_eff_sites = []
        self.n_eff_aa = []
        self.all_Ts = []
        self.seq_iter_counts = []
        self.all_variances = []
        self.all_num_new_seqs = []
        self.all_num_sampled_seqs = []
        self.all_mean_ml_scores = []
        self.all_phase_transition_numbers = []
        self.scored_sequences = {}
        
        # Phase transition tracking
        self.active_phase_transition = False
        self.first_phase_transition = None
        self.initial_var = None
        self.initial_ml_score = None
        self.last_high_ml_score = float('-inf')
        self.last_high_var = float('-inf')
        self.last_phase_transition = None
        self.num_phase_transitions = 0
        
        # Set up model and initialize matrices
        self.setup_model()
        self.initial_score_matrix = self._compute_single_mutant_scores()
        self.score_matrix = self.initial_score_matrix.copy()
        
        if self.params.normalize_scores:
            self._normalize_score_matrix()
            
        # Initialize tracking matrices
        self.sum_of_scores_matrix = self.initial_score_matrix.copy()
        self.mut_to_num_seqs_matrix = np.ones_like(self.sum_of_scores_matrix)
        self.sum_of_scores_squared_matrix = np.square(self.sum_of_scores_matrix)
        
        # Initialize sampler
        self.sampler = CombinatorialMutationSampler(
            sequence=self.ref_sequence,
            forbidden_sites=self.params.sites_to_ignore,
            temperature=self.params.T,
            sampling="boltzmann",
            score_matrix=self.score_matrix,
            verbose=True
        )
    
    @abc.abstractmethod
    def setup_model(self):
        """Set up the scoring model and set self.ref_sequence."""
        pass
        
    @abc.abstractmethod 
    def score_batch(self, sequences: List[str]) -> List[float]:
        """Score a batch of sequences using the model."""
        pass
        
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.params.seed)
    
    def _setup_optimizer_params(self):
        """Set up derived parameters from the base parameters."""
        self.n_batch = self.params.seqs_per_iter
        self.total_iter = self.params.num_iter
        self.n_seqs = self.total_iter * self.n_batch
        
        # Phase transition parameters
        self.patience_phase_trans = 3
        self.patience = 10 if (self.params.num_mutations > 3) else 15
        self.patience += self.patience_phase_trans
        self.heating_rate = 1.4 if (self.params.num_mutations > 3) else 1.6
        self.low_temp_threshold = 0.02
        self.high_temp_threshold = 1.6

    def _normalize_score_matrix(self):
        """Normalize the score matrix using reference value and scale."""
        if self.params.ref_score_value is None:
            self.params.ref_score_value = np.quantile(self.score_matrix.flatten(), 0.8)
        if self.params.ref_score_scale is None:
            self.params.ref_score_scale = self.score_matrix.flatten().std()
            
        print(f"Reference score value: {self.params.ref_score_value:.4f}, "
              f"std dev: {self.params.ref_score_scale:.4f}. To normalize scores.")
        
        self.score_matrix = ((self.score_matrix - self.params.ref_score_value) / 
                           (self.params.ref_score_scale + 1.0))

    def _compute_single_mutant_scores(self) -> np.ndarray:
        """Compute score matrix for all single mutants."""
        L = len(self.ref_sequence)
        score_matrix = np.zeros((20, L))
        
        single_mutants = []
        for i, original_aa in enumerate(self.ref_sequence):
            site = i + 1
            if site in self.params.sites_to_ignore:
                continue
                
            for aa in AA_TO_IDX.keys():
                if aa != original_aa:
                    single_mutants.append(f"{original_aa}{site}{aa}")
        
        scores = self.score_batch(single_mutants)
        
        # Fill score matrix
        for mutant, score in zip(single_mutants, scores):
            wt_aa = mutant[0]
            site = int(mutant[1:-1]) - 1  # Convert to 0-based index
            mut_aa = mutant[-1]
            mut_idx = AA_TO_IDX[mut_aa]
            score_matrix[mut_idx, site] = score
            
        return score_matrix
    
    def score_sequences_and_update(self, sequences: List[str]) -> Tuple[List[str], List[float], int]:
        """Score sequences and update matrices."""
        # Identify new sequences
        new_indices = [i for i, seq in enumerate(sequences) if seq not in self.scored_sequences]
        n_new = len(new_indices)
        
        if n_new > 0:
            new_sequences = [sequences[i] for i in new_indices]
            new_scores = self.score_batch(new_sequences)
            
            # Update tracking
            for seq, score in zip(new_sequences, new_scores):
                self.scored_sequences[seq] = score
                
            # Update matrices
            self._update_observation_matrices(new_sequences, new_scores)
            
        # Return all sequences and scores
        final_sequences = sequences
        scores = [self.scored_sequences[seq] for seq in sequences]
        
        return final_sequences, scores, n_new
    
    def _update_observation_matrices(self, sequences: List[str], scores: List[float]):
        """Update matrices tracking mutation statistics."""
        for seq, score in zip(sequences, scores):
            mutations = seq.split('-')
            for mutation in mutations:
                wt_aa = mutation[0]
                site = int(mutation[1:-1]) - 1  # Convert to 0-based index
                mut_aa = mutation[-1]
                mut_key = (AA_TO_IDX[mut_aa], site)
                
                self.sum_of_scores_matrix[mut_key] += score
                self.mut_to_num_seqs_matrix[mut_key] += 1
                self.sum_of_scores_squared_matrix[mut_key] += score * score
    
    def _update_matrices(self):
        """Update score matrix using observation matrices."""
        self.score_matrix = self.sum_of_scores_matrix / (self.mut_to_num_seqs_matrix + 1e-8)
        
        if self.params.boost_mutations_with_high_variance and (self.params.gamma > 0.0):
            var_matrix = (self.sum_of_scores_squared_matrix / (self.mut_to_num_seqs_matrix + 1e-8) - 
                         np.square(self.score_matrix))
            var_matrix = np.clip(var_matrix, a_min=0, a_max=None)
            self.score_matrix += self.params.gamma * np.sqrt(var_matrix)
            
        if self.params.normalize_scores:
            self.score_matrix = ((self.score_matrix - self.params.ref_score_value) / 
                               (self.params.ref_score_scale + 1.0))

    def optimize(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the optimization process."""
        for i in range(self.n_seqs // self.n_batch):
            # Sample sequences
            if self.params.sample_variety_of_mutation_numbers:
                sequences = self.sampler.sample_mutant_library(
                    library_size=self.n_batch,
                    mutation_proportions=self.params.num_sequences_proportions,
                    discard_bad_sequences=False,
                    dedupe=True
                )
            else:
                sequences = self.sampler.sample_multi_mutants(
                    num_mutations=self.params.num_mutations,
                    library_size=self.n_batch,
                    discard_bad_sequences=True,
                    dedupe=True
                )
                
            sequences = list(set(sequences))
            if len(sequences) < 1:
                self._update_temperature(i, 0)
                self.sampler.update_boltzmann_distribution(
                    new_temperature=self.params.T,
                    new_score_matrix=self.score_matrix
                )
                continue
                
            self.iters_for_seqs.extend(np.repeat(i, len(sequences)))
            
            # Score sequences and update matrices
            sequences, scores, n_new = self.score_sequences_and_update(sequences)
            
            # Update tracking
            self.all_sequences.extend(sequences)
            self.all_scores.extend(scores)
            self.all_variances.append(np.var(scores))
            self.all_mean_ml_scores.append(np.mean(scores))
            self.all_Ts.append(self.params.T)
            self.all_num_sampled_seqs.append(len(sequences))
            self.all_num_new_seqs.append(n_new)
            self.all_phase_transition_numbers.append(self.num_phase_transitions)
            
            if i == 10:
                self.initial_var = np.mean(np.array(self.all_variances))
                self.initial_ml_score = np.mean(np.array(self.all_mean_ml_scores))
            
            # Handle phase transitions if appropriate
            if ((not self.active_phase_transition) and (i > 10) and 
                (not self.params.cool_then_heat) and (not self.params.simple_simulated_annealing)):
                self.detect_phase_transition(i)
                
            if ((self.active_phase_transition) and (i > self.last_phase_transition) and 
                (not self.params.simple_simulated_annealing) and (not self.params.cool_then_heat)):
                self.detect_phase_transition_reversal(i)
            
            # Update temperature and matrices
            self._update_temperature(i, len(sequences))
            if n_new > 0:
                self._update_matrices()
            
            self.sampler.update_boltzmann_distribution(
                new_temperature=self.params.T,
                new_score_matrix=self.score_matrix
            )

        self._deduplicate_results()
        return self.prepare_results()

    def detect_phase_transition(self, iteration: int):
        """Detect if a phase transition has occurred."""
        cur_mean_avg = np.mean(np.array(self.all_mean_ml_scores[-5:]))
        self.last_high_var = max(np.convolve(self.all_variances, np.ones(5)/5, mode='valid'))
        
        if cur_mean_avg > self.params.score_threshold:
            self.last_phase_transition = iteration
            self.last_high_ml_score = max(np.convolve(self.all_mean_ml_scores, np.ones(5)/5, mode='valid'))
            print(f"#### Phase transition detected. High score avg: {self.last_high_ml_score:.3g}, " 
                  f"high var: {self.last_high_var:.3g}. ####")
            self.active_phase_transition = True
            self.num_phase_transitions += 1
            if self.first_phase_transition is None:
                self.first_phase_transition = iteration
            print(f"Will start increasing temperature in {self.patience_phase_trans} iterations, "
                  "unless a phase reversal is detected first.")

    def detect_phase_transition_reversal(self, iteration: int):
        """Detect if a phase transition has reversed."""
        cur_mean_avg = np.mean(np.array(self.all_mean_ml_scores[-2:]))
        
        if cur_mean_avg < self.params.reversal_threshold:
            print("####### Phase transition reversal detected. ########")
            self.active_phase_transition = False
        
        if (iteration > self.last_phase_transition + self.patience):
            print("####### Stopping phase transition because we have not detected a phase reversal. ########")
            self.active_phase_transition = False

    def _update_temperature(self, iteration: int, num_sequences: int):
        """Update temperature based on current state and parameters."""
        if self.params.T < 0.01:
            self.params.T *= 1.4
        elif num_sequences < np.round(self.n_batch * 0.5):
            if self.params.T < 10 * self.params.T:
                self.low_temp_threshold = self.params.T * 1.10
                self.params.T *= 1.4
        elif self.params.simple_simulated_annealing:
            self.params.T *= self.params.cooling_rate
        elif self.params.cool_then_heat:
            self._update_temperature_cool_then_heat()
        elif self.active_phase_transition and (iteration > self.last_phase_transition + self.patience_phase_trans):
            self.params.T *= self.heating_rate
        elif self.last_phase_transition is None:
            self.params.T *= self.params.cooling_rate
        else:
            cooling = np.square(self.params.cooling_rate) if self.params.num_mutations < 4 else self.params.cooling_rate
            self.params.T *= np.square(cooling)

    def _update_temperature_cool_then_heat(self):
        """Update temperature for cool-then-heat schedule."""
        if not hasattr(self, 'heating_phase'):
            self.heating_phase = False
            self.cooling_iterations = 0
            self.heating_iterations = 0

        if self.heating_phase:
            if (self.heating_iterations >= 45) or (self.params.T >= self.high_temp_threshold):
                self.heating_phase = False
                self.cooling_iterations = 1
                self.params.T *= self.params.cooling_rate
            else:
                self.params.T /= self.params.cooling_rate
                self.heating_iterations += 1
        else:
            if (self.cooling_iterations >= 45) or (self.params.T <= self.low_temp_threshold):
                self.heating_phase = True
                self.heating_iterations = 1
                self.params.T /= self.params.cooling_rate
            else:
                self.params.T *= self.params.cooling_rate
                self.cooling_iterations += 1

    def _deduplicate_results(self):
        """Deduplicate the results to keep unique sequences and their mean scores."""
        unique_sequences = {}
        sequence_counts = {}
        first_seen_iteration = {}
        
        for sequence, score, iteration in zip(self.all_sequences, self.all_scores, self.iters_for_seqs):
            if sequence not in unique_sequences:
                unique_sequences[sequence] = score
                sequence_counts[sequence] = 1
                first_seen_iteration[sequence] = iteration
            else:
                unique_sequences[sequence] += score
                sequence_counts[sequence] += 1

        # Calculate mean scores
        for sequence in unique_sequences:
            unique_sequences[sequence] = unique_sequences[sequence] / sequence_counts[sequence]
            
        self.all_sequences = list(unique_sequences.keys())
        self.all_scores = list(unique_sequences.values())
        self.seq_iter_counts = list(sequence_counts.values())
        self.iter_first_seen = list(first_seen_iteration.values())

    def prepare_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare final results as DataFrames."""
        n_seqs_to_keep = self.params.n_seqs_to_keep or len(self.all_scores)
        sorted_indices = np.argsort(self.all_scores)[::-1][:n_seqs_to_keep]
        
        # Prepare sequence results
        df = pd.DataFrame({
            'sequences': [self.all_sequences[i] for i in sorted_indices],
            'ml_score': [self.all_scores[i] for i in sorted_indices],
            'counts': [self.seq_iter_counts[i] for i in sorted_indices],
            'num_mutations': [self.params.num_mutations] * len(sorted_indices),
            'iteration': [self.iter_first_seen[i] for i in sorted_indices]
        })
        
        # Prepare optimization statistics
        df_stats = pd.DataFrame({
            'iteration': range(len(self.all_mean_ml_scores)),
            'avg_ml_score': self.all_mean_ml_scores,
            'var_ml_score': self.all_variances,
            'T': self.all_Ts,
            'n_seqs': self.all_num_sampled_seqs,
            'n_new_seqs': self.all_num_new_seqs,
            'num_phase_transitions': self.all_phase_transition_numbers
        })
        
        return df, df_stats