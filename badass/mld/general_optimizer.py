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

        # Add phase transition tracking
        self.active_phase_transition = False
        self.first_phase_transition = None
        self.initial_score = None
        self.last_high_score = float('-inf')
        self.last_high_var = float('-inf')
        self.last_phase_transition = None
        self.num_phase_transitions = []  # Track per iteration
        self.patience_phase_trans = self.params.get('patience_phase_trans', 3)
        self.patience = self.params.get('patience', 10)
        if self.params['num_mutations'] > 3:
            self.patience += self.patience_phase_trans
            self.heating_rate = 1.4
        else:
            self.heating_rate = 1.6
        
    def _set_default_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sets default parameters if not provided."""
        defaults = {
            'seqs_per_iter': 100,
            'num_iter': 100,
            'T': 1.0,
            'cooling_rate': 0.95,
            'num_mutations': 1,
            'forbidden_sites': [],
            'batch_size': 32,
            'score_threshold': None,  # Will be set after initial scoring
            'reversal_threshold': None,  # Will be set after initial scoring
            'patience_phase_trans': 10,  # Increased from 3 to match ProteinOptimizer
            'patience': 10,
            'heating_rate': 1.4,
            'simple_simulated_annealing': False,
            'cool_then_heat': False,
            'boost_mutations_with_high_variance': True,
            'gamma': 0.5  # For variance boosting
        }
        params_with_defaults = {**defaults, **params}
        
        # Set temperature thresholds for different strategies
        params_with_defaults['low_temp_threshold'] = 0.02
        params_with_defaults['high_temp_threshold'] = 1.6
        
        # Initialize heating/cooling phase variables
        if params_with_defaults['cool_then_heat']:
            params_with_defaults['heating_phase'] = False
            params_with_defaults['cooling_iterations'] = 0
            params_with_defaults['heating_iterations'] = 0
        
        return params_with_defaults
    
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
    
    def detect_phase_transition(self, i: int):
        """
        Enhanced phase transition detection incorporating score variance.
        
        Parameters:
        -----------
        i : int
            Current iteration number
        """
        cur_mean_avg = np.mean(self.mean_scores[-5:])
        cur_var_avg = np.mean(self.score_variances[-5:])
        self.last_high_var = max(np.convolve(self.score_variances, np.ones(5)/5, mode='valid'))
        
        # Enhanced detection criteria using both mean and variance
        if cur_mean_avg > self.params['score_threshold']:
            self.last_phase_transition = i
            self.last_high_score = max(np.convolve(self.mean_scores, np.ones(5)/5, mode='valid'))
            logger.info(f"Phase transition detected. High score avg: {self.last_high_score:.3g}, "
                       f"high var: {self.last_high_var:.3g}")
            self.active_phase_transition = True
            self.num_phase_transitions.append(self.num_phase_transitions[-1] + 1 if self.num_phase_transitions else 1)
            
            if self.first_phase_transition is None:
                self.first_phase_transition = i
                self.initial_var = np.mean(self.score_variances)
                logger.info(f"First phase transition detected at iteration {i}")
                logger.info(f"Initial variance: {self.initial_var:.3g}")
            
            logger.info(f"Will start increasing temperature in {self.params['patience_phase_trans']} iterations "
                       "unless phase reversal is detected")
        else:
            self.num_phase_transitions.append(self.num_phase_transitions[-1] if self.num_phase_transitions else 0)

    def detect_phase_transition_reversal(self, i: int) -> bool:
        """
        Enhanced phase transition reversal detection using both score and variance.
        
        Parameters:
        -----------
        i : int
            Current iteration number
            
        Returns:
        --------
        bool
            True if phase transition has reversed
        """
        cur_var_avg = np.mean(self.score_variances[-2:])
        cur_mean_avg = np.mean(self.mean_scores[-2:])
        
        # Check both score threshold and variance conditions
        if cur_mean_avg < self.params['reversal_threshold']:
            logger.info("Phase transition reversal detected based on score threshold")
            self.active_phase_transition = False
            return True
            
        if (i > self.last_phase_transition + self.params['patience']):
            logger.info("Phase transition stopped due to patience timeout")
            self.active_phase_transition = False
            return True
            
        return False
    
    def update_temperature(self, iteration: int, num_sequences: int):
        """
        Update temperature based on current state and phase transitions.
        More sophisticated version matching ProteinOptimizer's approach.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        num_sequences : int
            Number of sequences in current iteration
        """
        old_T = self.params['T']
        
        # Prevent temperature from getting too low
        if self.params['T'] < 0.01:
            self.params['T'] *= 1.4
            logger.debug(f"Temperature increased (too low): {old_T:.3f} -> {self.params['T']:.3f}")
            return
            
        # Increase temperature if too few sequences
        if num_sequences < 0.5 * self.params['seqs_per_iter']:
            if self.params['T'] < 10 * self.params['T']:
                self.params['low_temp_threshold'] = self.params['T'] * 1.10
                self.params['T'] *= 1.4
                logger.debug(f"Temperature increased (few sequences): {old_T:.3f} -> {self.params['T']:.3f}")
            return
        
        # Handle different temperature strategies
        if self.params['simple_simulated_annealing']:
            self.params['T'] *= self.params['cooling_rate']
            logger.debug(f"Temperature cooled (simple annealing): {old_T:.3f} -> {self.params['T']:.3f}")
            
        elif self.params['cool_then_heat']:
            if self.params['heating_phase']:
                # Heating phase
                if (self.params['heating_iterations'] >= 45 or 
                    self.params['T'] >= self.params['high_temp_threshold']):
                    logger.info("Starting to cool the system")
                    self.params['heating_phase'] = False
                    self.params['cooling_iterations'] = 1
                    self.params['T'] *= self.params['cooling_rate']
                else:
                    self.params['T'] /= self.params['cooling_rate']
                    self.params['heating_iterations'] += 1
            else:
                # Cooling phase
                if (self.params['cooling_iterations'] >= 45 or 
                    self.params['T'] <= self.params['low_temp_threshold']):
                    logger.info(f"Starting to heat the system. T threshold: {self.params['low_temp_threshold']:.3f}")
                    self.params['heating_phase'] = True
                    self.params['heating_iterations'] = 1
                    self.params['T'] /= self.params['cooling_rate']
                else:
                    self.params['T'] *= self.params['cooling_rate']
                    self.params['cooling_iterations'] += 1
                    
        else:  # Adaptive temperature with phase transitions
            if self.active_phase_transition and (iteration > self.last_phase_transition + self.params['patience_phase_trans']):
                self.params['T'] *= self.params['heating_rate']
                logger.debug(f"Temperature increased (phase transition): {old_T:.3f} -> {self.params['T']:.3f}")
            elif self.last_phase_transition is None:
                self.params['T'] *= self.params['cooling_rate']
                logger.debug(f"Temperature cooled (no phase transition): {old_T:.3f} -> {self.params['T']:.3f}")
            else:
                cooling_factor = (np.square(self.params['cooling_rate']) * 
                                self.params['cooling_rate'] if self.params['num_mutations'] < 4 
                                else np.square(self.params['cooling_rate']))
                self.params['T'] *= cooling_factor
                logger.debug(f"Temperature cooled (normal): {old_T:.3f} -> {self.params['T']:.3f}")
    
    def optimize(self) -> Dict[str, Any]:
        """Runs the optimization process."""
        logger.info("Starting optimization process")
        start_time = time()
        
        # Set initial thresholds after scoring first sequences if not provided
        if self.params['score_threshold'] is None:
            initial_scores = list(self.scored_sequences.values())
            self.params['score_threshold'] = np.mean(initial_scores) + np.std(initial_scores)
            self.params['reversal_threshold'] = np.mean(initial_scores) - np.std(initial_scores)
            logger.info(f"Set score threshold to {self.params['score_threshold']:.3f}")
            logger.info(f"Set reversal threshold to {self.params['reversal_threshold']:.3f}")
        
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
            
            # Set initial score metrics after 10 iterations
            if i == 10:
                self.initial_score = np.mean(self.mean_scores)
                logger.info(f"Initial average score: {self.initial_score:.3f}")
            
            # Check for phase transitions
            if (not self.active_phase_transition and i > 10 and 
                self.initial_score is not None):
                self.detect_phase_transition(i)
                
            # Check for phase transition reversal
            if self.active_phase_transition and i > self.last_phase_transition:
                self.detect_phase_transition_reversal(i)
                
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
        logger.info(f"Total phase transitions: {self.num_phase_transitions[-1]}")
        
        return {
            'best_sequence': self.all_sequences[best_idx],
            'best_score': self.all_scores[best_idx],
            'all_sequences': self.all_sequences,
            'all_scores': self.all_scores,
            'statistics': {
                'temperatures': self.temperatures,
                'mean_scores': self.mean_scores,
                'score_variances': self.score_variances,
                'num_phase_transitions': self.num_phase_transitions,
                'timing': {
                    'total_time': total_time,
                    'avg_sampling_time': np.mean(self.timing_stats['sampling']),
                    'avg_scoring_time': np.mean(self.timing_stats['scoring']),
                    'avg_iteration_time': np.mean(self.timing_stats['total_iter'])
                }
            }
        }
    
    def _update_score_matrix(self, sequences: List[str], scores: List[float]):
        """
        Update score matrix with variance boosting option.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences that were scored
        scores : List[float]
            Corresponding scores for the sequences
        """
        # Update observation matrices
        self._update_observation_matrices(sequences, scores)
        
        # Compute new score matrix
        self.score_matrix = self.sum_of_scores_matrix / (self.mut_to_num_seqs_matrix + EPS)
        
        # Add variance boosting if enabled
        if self.params['boost_mutations_with_high_variance'] and self.params['gamma'] > 0.0:
            var_matrix = (self.sum_of_scores_squared_matrix / (self.mut_to_num_seqs_matrix + EPS) - 
                         np.square(self.score_matrix))
            var_matrix = np.clip(var_matrix, a_min=0, a_max=None)  # remove negative values
            self.score_matrix += self.params['gamma'] * np.sqrt(var_matrix)
        
        # Update sampler with new matrix
        self.sampler.update_boltzmann_distribution(
            new_temperature=self.params['T'],
            new_score_matrix=self.score_matrix
        )

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