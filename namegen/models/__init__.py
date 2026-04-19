"""Candidate-generating models.

Only the baseline character-level Markov model is wired up in M1. Phoneme
Markov, rule-based morphology composer and the neural Transformer land in
later milestones (see the project roadmap).
"""

from namegen.models.markov import CharMarkovModel

__all__ = ["CharMarkovModel"]
