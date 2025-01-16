# badass/mld/general_optimizer.py
from typing import Callable, List, Dict, Any, Optional
import numpy as np
from badass.mld.sequence_sampler import CombinatorialMutationSampler
from badass.utils.sequence_utils import apply_mutations, generate_single_mutants
import logging
from time import time

logger = logging.getLogger(__name__)

class ArbitraryPredictorOptimizer:
    """
    BADASS optimizer that does not assume anything about the scoring function, other than

    y=f(full_sequence)
    """
    
    def __init__(
        self,
        scoring_function: Callable[[List[str]], List[float]],
        reference_sequence: str,
        optimization_params: Dict[str, Any]
    ):
        """
        Initialize the optimizer with a scoring function and parameters.
        
        Parameters:
        -----------
        scoring_function : Callable[[List[str]], List[float]]
            Function that takes a list of sequences and returns a list of scores
        reference_sequence : str
            The reference/wild-type sequence
        optimization_params : Dict[str, Any]
            Optimization parameters dictionary
        """
        logger.info("Initializing SequenceOptimizer")
        self.scoring_function = scoring_function
        self.ref_seq = reference_sequence
        self.params = self._set_default_params(optimization_params)
        
        logger.info(f"Reference sequence length: {len(self.ref_seq)}")
        logger.info(f"Optimization parameters: {self.params}")
        
        # Initialize tracking variables
        self.scored_sequences = {}  # Full sequence -> score mapping
        self.all_sequences = []     # List of full sequences
        self.all_scores = []
        self.temperatures = []
        self.mean_scores = []
        self.score_variances = []
        self.timing_stats = {
            'sampling': [],
            'scoring': [],
            'total_iter': []
        }
        
        # Initialize sampler and score matrix
        logger.info("Computing initial score matrix...")
        start_time = time()
        self.score_matrix = self._compute_initial_score_matrix()
        logger.info(f"Score matrix computation took {time() - start_time:.2f} seconds")
        
        logger.info("Initializing sequence sampler...")
        self.sampler = self._initialize_sampler()
        
    def _set_default_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sets default parameters if not provided."""
        defaults = {
            'seqs_per_iter': 100,
            'num_iter': 100,
            'T': 1.0,
            'cooling_rate': 0.95,
            'num_mutations': 1,
            'forbidden_sites': [],
            'batch_size': 32
        }
        return {**defaults, **params}
    
    def _initialize_sampler(self) -> CombinatorialMutationSampler:
        """Initializes the sequence sampler with current parameters."""
        logger.debug(f"Initializing sampler with T={self.params['T']}")
        return CombinatorialMutationSampler(
            sequence=self.ref_seq,
            forbidden_sites=self.params['forbidden_sites'],
            temperature=self.params['T'],
            sampling="boltzmann",
            score_matrix=self.score_matrix,
            verbose=False
        )
    
    def _compute_initial_score_matrix(self) -> np.ndarray:
        """Computes initial score matrix using batch scoring of single mutants."""
        # Generate all possible single mutants
        single_mutant_mutations = generate_single_mutants(
            self.ref_seq, 
            self.params['forbidden_sites']
        )
        logger.info(f"Generated {len(single_mutant_mutations)} single mutants")
        
        # Convert mutations to full sequences
        full_sequences = []
        for mutation in single_mutant_mutations:
            mutated_sequence = self._mutations_to_sequence(mutation)
            full_sequences.append(mutated_sequence)
        
        # Score sequences in batches
        batch_size = self.params['batch_size']
        scores = []
        
        for i in range(0, len(full_sequences), batch_size):
            batch = full_sequences[i:i + batch_size]
            logger.debug(f"Scoring batch {i//batch_size + 1} of {len(full_sequences)//batch_size + 1}")
            batch_scores = self.scoring_function(batch)
            scores.extend(batch_scores)
            
            # Update cache
            for seq, score in zip(batch, batch_scores):
                self.scored_sequences[seq] = score
        
        # Score reference sequence
        wt_score = self.scoring_function([self.ref_seq])[0]
        self.scored_sequences[self.ref_seq] = wt_score
        logger.info(f"Reference sequence score: {wt_score:.3f}")
        
        return self._create_score_matrix(
            single_mutant_mutations,
            scores,
            wt_score
        )
    
    def _create_score_matrix(self, mutations: List[str], scores: List[float], wt_score: float) -> np.ndarray:
        """Creates a score matrix from mutations and their scores."""
        from badass.utils.sequence_utils import create_score_matrix
        logger.debug("Creating score matrix from mutations and scores")
        return create_score_matrix(
            self.ref_seq,
            mutations,
            scores,
            wt_scores=wt_score
        )
    
    def _mutations_to_sequence(self, mutations: str) -> str:
        """Converts a mutation string to a full sequence."""
        if not mutations:
            return self.ref_seq
            
        mutations_dict = {}
        for mutation in mutations.split('-'):
            site = int(mutation[1:-1])
            aa = mutation[-1]
            mutations_dict[site] = aa
            
        return apply_mutations(self.ref_seq, mutations_dict)
    
    def _score_sequences(self, mutation_strings: List[str]) -> List[float]:
        """
        Scores sequences in batches, using cache when possible.
        
        Parameters:
        -----------
        mutation_strings : List[str]
            List of mutation strings to score
        """
        # Convert all mutations to full sequences
        full_sequences = [self._mutations_to_sequence(m) for m in mutation_strings]
        
        # Separate cached and new sequences
        new_seqs = []
        new_seqs_indices = []
        scores = [0] * len(full_sequences)  # Initialize with zeros
        
        for i, seq in enumerate(full_sequences):
            if seq in self.scored_sequences:
                scores[i] = self.scored_sequences[seq]
            else:
                new_seqs.append(seq)
                new_seqs_indices.append(i)
        
        # Score new sequences in batches if any exist
        if new_seqs:
            logger.debug(f"Scoring {len(new_seqs)} new sequences")
            batch_size = self.params['batch_size']
            
            for i in range(0, len(new_seqs), batch_size):
                batch = new_seqs[i:i + batch_size]
                batch_scores = self.scoring_function(batch)
                
                # Update cache and scores list
                for seq, score, idx in zip(batch, batch_scores, new_seqs_indices[i:i + batch_size]):
                    self.scored_sequences[seq] = score
                    scores[idx] = score
        
        return scores
    
    def update_temperature(self, iteration: int, num_sequences: int):
        """Updates temperature based on current state."""
        old_T = self.params['T']
        if num_sequences < 0.5 * self.params['seqs_per_iter']:
            self.params['T'] *= 1.2
            logger.debug(f"Temperature increased: {old_T:.3f} -> {self.params['T']:.3f}")
        else:
            self.params['T'] *= self.params['cooling_rate']
            logger.debug(f"Temperature cooled: {old_T:.3f} -> {self.params['T']:.3f}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Runs the optimization process.
        
        Returns:
        --------
        Dict[str, Any]
            Optimization results and statistics
        """
        logger.info("Starting optimization process")
        start_time = time()
        
        for i in range(self.params['num_iter']):
            iter_start = time()
            logger.info(f"Starting iteration {i+1}/{self.params['num_iter']}")
            
            # Sample mutation strings
            sample_start = time()
            mutation_strings = self.sampler.sample_multi_mutants(
                num_mutations=self.params['num_mutations'],
                library_size=self.params['seqs_per_iter'],
                dedupe=True
            )
            self.timing_stats['sampling'].append(time() - sample_start)
            logger.debug(f"Generated {len(mutation_strings)} sequences")
            
            # Score sequences
            score_start = time()
            scores = self._score_sequences(mutation_strings)
            self.timing_stats['scoring'].append(time() - score_start)
            
            # Convert mutations to full sequences for tracking
            full_sequences = [self._mutations_to_sequence(m) for m in mutation_strings]
            
            # Update tracking
            self.all_sequences.extend(full_sequences)
            self.all_scores.extend(scores)
            self.temperatures.append(self.params['T'])
            
            mean_score = np.mean(scores)
            score_var = np.var(scores)
            self.mean_scores.append(mean_score)
            self.score_variances.append(score_var)
            
            # Update temperature
            self.update_temperature(i, len(mutation_strings))
            self.sampler.update_boltzmann_distribution(
                new_temperature=self.params['T']
            )
            
            # Log progress
            iter_time = time() - iter_start
            self.timing_stats['total_iter'].append(iter_time)
            logger.info(
                f"Iteration {i+1} complete: "
                f"mean_score={mean_score:.3f}, "
                f"score_var={score_var:.3f}, "
                f"T={self.params['T']:.3f}, "
                f"time={iter_time:.2f}s"
            )
        
        # Find best sequence
        best_idx = np.argmax(self.all_scores)
        total_time = time() - start_time
        
        logger.info(f"Optimization complete in {total_time:.2f} seconds")
        logger.info(f"Best score: {self.all_scores[best_idx]:.3f}")
        logger.info(f"Total unique sequences evaluated: {len(self.scored_sequences)}")
        
        return {
            'best_sequence': self.all_sequences[best_idx],
            'best_score': self.all_scores[best_idx],
            'all_sequences': self.all_sequences,
            'all_scores': self.all_scores,
            'statistics': {
                'temperatures': self.temperatures,
                'mean_scores': self.mean_scores,
                'score_variances': self.score_variances,
                'timing': {
                    'total_time': total_time,
                    'avg_sampling_time': np.mean(self.timing_stats['sampling']),
                    'avg_scoring_time': np.mean(self.timing_stats['scoring']),
                    'avg_iteration_time': np.mean(self.timing_stats['total_iter'])
                }
            }
        }
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot the optimization history including scores, temperature, and sequence diversity.
        
        Parameters:
        -----------
        save_path : Optional[str]
            If provided, save the plot to this path
        """
        import matplotlib.pyplot as plt
        
        logger.info("Plotting optimization history")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        iterations = range(len(self.mean_scores))
        
        # Plot 1: Scores
        ax1.plot(iterations, self.mean_scores, label='Mean Score', color='blue')
        ax1.fill_between(
            iterations,
            [m - np.sqrt(v) for m, v in zip(self.mean_scores, self.score_variances)],
            [m + np.sqrt(v) for m, v in zip(self.mean_scores, self.score_variances)],
            alpha=0.2, color='blue'
        )
        max_scores = [max(self.all_scores[:i+1]) for i in range(len(self.mean_scores))]
        ax1.plot(iterations, max_scores, label='Best Score', color='red', linestyle='--')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Score Progression')
        
        # Plot 2: Temperature
        ax2.plot(iterations, self.temperatures, color='green')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.set_title('Temperature Schedule')
        
        # Plot 3: Timing Statistics
        ax3.plot(iterations, self.timing_stats['sampling'], label='Sampling Time', color='purple')
        ax3.plot(iterations, self.timing_stats['scoring'], label='Scoring Time', color='orange')
        ax3.plot(iterations, self.timing_stats['total_iter'], label='Total Iteration Time', color='brown')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend()
        ax3.grid(True)
        ax3.set_title('Computational Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved optimization history plot to {save_path}")
        
        plt.show()

    def plot_score_distribution(self, save_path: Optional[str] = None):
        """
        Plot the distribution of scores across different stages of optimization.
        
        Parameters:
        -----------
        save_path : Optional[str]
            If provided, save the plot to this path
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        logger.info("Plotting score distribution")
        
        # Split scores into early, middle, and late stages
        n_scores = len(self.all_scores)
        early_scores = self.all_scores[:n_scores//3]
        middle_scores = self.all_scores[n_scores//3:2*n_scores//3]
        late_scores = self.all_scores[2*n_scores//3:]
        
        plt.figure(figsize=(10, 6))
        
        # Plot distributions
        sns.kdeplot(early_scores, label='Early Stage', alpha=0.5)
        sns.kdeplot(middle_scores, label='Middle Stage', alpha=0.5)
        sns.kdeplot(late_scores, label='Late Stage', alpha=0.5)
        
        # Add reference score line
        plt.axvline(x=self.scored_sequences[self.ref_seq], color='black', 
                    linestyle='--', label='Reference Score')
        
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.title('Score Distribution Across Optimization Stages')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved score distribution plot to {save_path}")
        
        plt.show()