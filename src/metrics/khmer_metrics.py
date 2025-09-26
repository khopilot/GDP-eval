"""
Khmer-Specific Evaluation Metrics
Specialized metrics for evaluating Khmer language models
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import Counter
import logging
from dataclasses import dataclass, field
from sacrebleu import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import editdistance

from .khmer_tokenizer import KhmerTokenizer

logger = logging.getLogger(__name__)


@dataclass
class KhmerMetricResult:
    """Container for Khmer metric results"""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.metric_name}: {self.score:.4f}"


class KhmerMetrics:
    """Khmer-specific evaluation metrics"""

    def __init__(self):
        """Initialize Khmer metrics"""
        self.tokenizer = KhmerTokenizer()
        self.smoothing = SmoothingFunction()

    def calculate_khmer_bleu(
        self,
        predictions: List[str],
        references: List[str],
        tokenization: str = "syllable",
        use_smoothing: bool = True
    ) -> KhmerMetricResult:
        """
        Calculate BLEU score with Khmer-aware tokenization

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            tokenization: Type of tokenization (char, syllable, word)
            use_smoothing: Whether to use smoothing for short texts

        Returns:
            KhmerMetricResult with BLEU score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        # Tokenize based on chosen method
        if tokenization == "char":
            pred_tokens = [self.tokenizer.tokenize_characters(p) for p in predictions]
            ref_tokens = [self.tokenizer.tokenize_characters(r) for r in references]
        elif tokenization == "syllable":
            pred_tokens = [self.tokenizer.tokenize_syllables(p) for p in predictions]
            ref_tokens = [self.tokenizer.tokenize_syllables(r) for r in references]
        else:  # word
            pred_tokens = [self.tokenizer.tokenize_words(p) for p in predictions]
            ref_tokens = [self.tokenizer.tokenize_words(r) for r in references]

        # Calculate BLEU scores
        bleu_scores = []
        for pred, ref in zip(pred_tokens, ref_tokens):
            if use_smoothing:
                score = sentence_bleu(
                    [ref],
                    pred,
                    smoothing_function=self.smoothing.method1
                )
            else:
                score = sentence_bleu([ref], pred)
            bleu_scores.append(score)

        avg_bleu = np.mean(bleu_scores)

        return KhmerMetricResult(
            metric_name=f"khmer_bleu_{tokenization}",
            score=avg_bleu,
            details={
                "tokenization": tokenization,
                "num_samples": len(predictions),
                "min_score": min(bleu_scores),
                "max_score": max(bleu_scores),
                "std_dev": np.std(bleu_scores)
            }
        )

    def calculate_character_accuracy(
        self,
        predictions: List[str],
        references: List[str],
        normalize: bool = True
    ) -> KhmerMetricResult:
        """
        Calculate character-level accuracy for Khmer text

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            normalize: Whether to normalize text before comparison

        Returns:
            KhmerMetricResult with character accuracy
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        accuracies = []

        for pred, ref in zip(predictions, references):
            # Normalize if requested
            if normalize:
                pred = self.tokenizer.normalize_khmer(pred)
                ref = self.tokenizer.normalize_khmer(ref)

            # Get character tokens
            pred_chars = self.tokenizer.tokenize_characters(pred)
            ref_chars = self.tokenizer.tokenize_characters(ref)

            # Calculate accuracy
            if len(ref_chars) == 0:
                accuracy = 1.0 if len(pred_chars) == 0 else 0.0
            else:
                # Count matching characters at same positions
                matches = sum(1 for p, r in zip(pred_chars, ref_chars) if p == r)
                accuracy = matches / max(len(pred_chars), len(ref_chars))

            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)

        return KhmerMetricResult(
            metric_name="character_accuracy",
            score=avg_accuracy,
            details={
                "normalized": normalize,
                "num_samples": len(predictions),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "std_dev": np.std(accuracies)
            }
        )

    def calculate_syllable_f1(
        self,
        predictions: List[str],
        references: List[str]
    ) -> KhmerMetricResult:
        """
        Calculate F1 score for syllable segmentation

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            KhmerMetricResult with F1 score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        f1_scores = []
        precisions = []
        recalls = []

        for pred, ref in zip(predictions, references):
            # Get syllables
            pred_syllables = set(self.tokenizer.tokenize_syllables(pred))
            ref_syllables = set(self.tokenizer.tokenize_syllables(ref))

            # Calculate precision, recall, F1
            if len(pred_syllables) == 0:
                precision = 1.0 if len(ref_syllables) == 0 else 0.0
            else:
                precision = len(pred_syllables & ref_syllables) / len(pred_syllables)

            if len(ref_syllables) == 0:
                recall = 1.0 if len(pred_syllables) == 0 else 0.0
            else:
                recall = len(pred_syllables & ref_syllables) / len(ref_syllables)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        return KhmerMetricResult(
            metric_name="syllable_f1",
            score=avg_f1,
            details={
                "precision": avg_precision,
                "recall": avg_recall,
                "num_samples": len(predictions),
                "min_f1": min(f1_scores),
                "max_f1": max(f1_scores),
                "std_dev": np.std(f1_scores)
            }
        )

    def calculate_word_segmentation_accuracy(
        self,
        predictions: List[str],
        references: List[str],
        use_zwsp: bool = True
    ) -> KhmerMetricResult:
        """
        Calculate word segmentation accuracy

        Args:
            predictions: List of predicted texts with word boundaries
            references: List of reference texts with word boundaries
            use_zwsp: Whether to use ZWSP as word boundary marker

        Returns:
            KhmerMetricResult with segmentation accuracy
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        accuracies = []
        boundary_precisions = []
        boundary_recalls = []

        for pred, ref in zip(predictions, references):
            # Get words
            pred_words = self.tokenizer.tokenize_words(pred, use_zwsp=use_zwsp)
            ref_words = self.tokenizer.tokenize_words(ref, use_zwsp=use_zwsp)

            # Calculate word-level accuracy
            if len(ref_words) == 0:
                accuracy = 1.0 if len(pred_words) == 0 else 0.0
            else:
                # Use sequence matching
                matches = 0
                for i, ref_word in enumerate(ref_words):
                    if i < len(pred_words) and pred_words[i] == ref_word:
                        matches += 1
                accuracy = matches / max(len(pred_words), len(ref_words))

            accuracies.append(accuracy)

            # Calculate boundary precision/recall
            pred_boundaries = self._get_word_boundaries(pred, use_zwsp)
            ref_boundaries = self._get_word_boundaries(ref, use_zwsp)

            if len(pred_boundaries) > 0:
                boundary_precision = len(pred_boundaries & ref_boundaries) / len(pred_boundaries)
            else:
                boundary_precision = 1.0 if len(ref_boundaries) == 0 else 0.0

            if len(ref_boundaries) > 0:
                boundary_recall = len(pred_boundaries & ref_boundaries) / len(ref_boundaries)
            else:
                boundary_recall = 1.0 if len(pred_boundaries) == 0 else 0.0

            boundary_precisions.append(boundary_precision)
            boundary_recalls.append(boundary_recall)

        avg_accuracy = np.mean(accuracies)
        avg_boundary_precision = np.mean(boundary_precisions)
        avg_boundary_recall = np.mean(boundary_recalls)

        return KhmerMetricResult(
            metric_name="word_segmentation_accuracy",
            score=avg_accuracy,
            details={
                "boundary_precision": avg_boundary_precision,
                "boundary_recall": avg_boundary_recall,
                "use_zwsp": use_zwsp,
                "num_samples": len(predictions),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "std_dev": np.std(accuracies)
            }
        )

    def calculate_edit_distance(
        self,
        predictions: List[str],
        references: List[str],
        level: str = "char"
    ) -> KhmerMetricResult:
        """
        Calculate edit distance at character or word level

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            level: Level of comparison (char, syllable, word)

        Returns:
            KhmerMetricResult with normalized edit distance
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        distances = []
        normalized_distances = []

        for pred, ref in zip(predictions, references):
            # Tokenize based on level
            if level == "char":
                pred_tokens = self.tokenizer.tokenize_characters(pred)
                ref_tokens = self.tokenizer.tokenize_characters(ref)
            elif level == "syllable":
                pred_tokens = self.tokenizer.tokenize_syllables(pred)
                ref_tokens = self.tokenizer.tokenize_syllables(ref)
            else:  # word
                pred_tokens = self.tokenizer.tokenize_words(pred)
                ref_tokens = self.tokenizer.tokenize_words(ref)

            # Calculate edit distance
            distance = editdistance.eval(pred_tokens, ref_tokens)
            distances.append(distance)

            # Normalize by maximum length
            max_len = max(len(pred_tokens), len(ref_tokens))
            if max_len > 0:
                normalized_distances.append(distance / max_len)
            else:
                normalized_distances.append(0.0)

        avg_distance = np.mean(distances)
        avg_normalized = np.mean(normalized_distances)

        # Convert to similarity score (1 - normalized distance)
        similarity_score = 1.0 - avg_normalized

        return KhmerMetricResult(
            metric_name=f"edit_distance_{level}",
            score=similarity_score,
            details={
                "avg_distance": avg_distance,
                "avg_normalized_distance": avg_normalized,
                "level": level,
                "num_samples": len(predictions),
                "min_distance": min(distances),
                "max_distance": max(distances),
                "std_dev": np.std(distances)
            }
        )

    def calculate_code_switching_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> KhmerMetricResult:
        """
        Calculate accuracy of code-switching detection

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            KhmerMetricResult with code-switching accuracy
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        accuracies = []
        precisions = []
        recalls = []

        for pred, ref in zip(predictions, references):
            # Get code-switching points
            pred_switches = set(self.tokenizer.detect_code_switching_points(pred))
            ref_switches = set(self.tokenizer.detect_code_switching_points(ref))

            # Calculate metrics
            if len(pred_switches) == 0:
                precision = 1.0 if len(ref_switches) == 0 else 0.0
            else:
                # Match switches by position (with tolerance)
                matched = 0
                for p_pos, p_from, p_to in pred_switches:
                    for r_pos, r_from, r_to in ref_switches:
                        if abs(p_pos - r_pos) <= 3 and p_from == r_from and p_to == r_to:
                            matched += 1
                            break
                precision = matched / len(pred_switches)

            if len(ref_switches) == 0:
                recall = 1.0 if len(pred_switches) == 0 else 0.0
            else:
                # Match switches by position (with tolerance)
                matched = 0
                for r_pos, r_from, r_to in ref_switches:
                    for p_pos, p_from, p_to in pred_switches:
                        if abs(p_pos - r_pos) <= 3 and p_from == r_from and p_to == r_to:
                            matched += 1
                            break
                recall = matched / len(ref_switches)

            if precision + recall == 0:
                accuracy = 0.0
            else:
                accuracy = 2 * (precision * recall) / (precision + recall)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        return KhmerMetricResult(
            metric_name="code_switching_accuracy",
            score=avg_accuracy,
            details={
                "precision": avg_precision,
                "recall": avg_recall,
                "num_samples": len(predictions),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "std_dev": np.std(accuracies)
            }
        )

    def calculate_khmer_specific_errors(
        self,
        predictions: List[str],
        references: List[str]
    ) -> KhmerMetricResult:
        """
        Analyze Khmer-specific errors (subscripts, vowels, tone marks)

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            KhmerMetricResult with error analysis
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        error_counts = {
            "subscript_errors": [],
            "vowel_errors": [],
            "sign_errors": [],
            "consonant_errors": [],
            "total_errors": []
        }

        for pred, ref in zip(predictions, references):
            # Count character types
            pred_counts = self.tokenizer.count_khmer_characters(pred)
            ref_counts = self.tokenizer.count_khmer_characters(ref)

            # Calculate errors by type
            subscript_error = abs(pred_counts['subscripts'] - ref_counts['subscripts'])
            vowel_error = abs(pred_counts['dependent_vowels'] - ref_counts['dependent_vowels'])
            sign_error = abs(pred_counts['signs'] - ref_counts['signs'])
            consonant_error = abs(pred_counts['consonants'] - ref_counts['consonants'])

            error_counts["subscript_errors"].append(subscript_error)
            error_counts["vowel_errors"].append(vowel_error)
            error_counts["sign_errors"].append(sign_error)
            error_counts["consonant_errors"].append(consonant_error)
            error_counts["total_errors"].append(
                subscript_error + vowel_error + sign_error + consonant_error
            )

        # Calculate error rates
        total_chars = sum(
            self.tokenizer.count_khmer_characters(ref)['total_khmer']
            for ref in references
        )

        if total_chars > 0:
            error_rate = sum(error_counts["total_errors"]) / total_chars
        else:
            error_rate = 0.0

        # Accuracy score (1 - error_rate)
        accuracy_score = max(0.0, 1.0 - error_rate)

        return KhmerMetricResult(
            metric_name="khmer_specific_accuracy",
            score=accuracy_score,
            details={
                "avg_subscript_errors": np.mean(error_counts["subscript_errors"]),
                "avg_vowel_errors": np.mean(error_counts["vowel_errors"]),
                "avg_sign_errors": np.mean(error_counts["sign_errors"]),
                "avg_consonant_errors": np.mean(error_counts["consonant_errors"]),
                "total_error_rate": error_rate,
                "num_samples": len(predictions)
            }
        )

    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, KhmerMetricResult]:
        """
        Calculate all available metrics

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dictionary of all metric results
        """
        results = {}

        # BLEU scores
        for tokenization in ["char", "syllable", "word"]:
            results[f"bleu_{tokenization}"] = self.calculate_khmer_bleu(
                predictions, references, tokenization
            )

        # Character accuracy
        results["character_accuracy"] = self.calculate_character_accuracy(
            predictions, references
        )

        # Syllable F1
        results["syllable_f1"] = self.calculate_syllable_f1(
            predictions, references
        )

        # Word segmentation
        results["word_segmentation"] = self.calculate_word_segmentation_accuracy(
            predictions, references
        )

        # Edit distances
        for level in ["char", "syllable", "word"]:
            results[f"edit_distance_{level}"] = self.calculate_edit_distance(
                predictions, references, level
            )

        # Code-switching
        results["code_switching"] = self.calculate_code_switching_accuracy(
            predictions, references
        )

        # Khmer-specific errors
        results["khmer_errors"] = self.calculate_khmer_specific_errors(
            predictions, references
        )

        return results

    def _get_word_boundaries(self, text: str, use_zwsp: bool) -> set:
        """Get word boundary positions in text"""
        boundaries = set()

        if use_zwsp and self.tokenizer.ZWSP in text:
            # Use ZWSP positions
            pos = 0
            for i, char in enumerate(text):
                if char == self.tokenizer.ZWSP:
                    boundaries.add(pos)
                elif char not in [' ', '\n', '\t']:
                    pos += 1
        else:
            # Use heuristic boundaries
            words = self.tokenizer.tokenize_words(text, use_zwsp=False)
            pos = 0
            for word in words:
                pos += len(word)
                boundaries.add(pos)

        return boundaries

    def format_results(
        self,
        results: Dict[str, KhmerMetricResult],
        format_type: str = "text"
    ) -> str:
        """
        Format metric results for display

        Args:
            results: Dictionary of metric results
            format_type: Output format (text, json, markdown)

        Returns:
            Formatted string
        """
        if format_type == "json":
            import json
            output = {}
            for name, result in results.items():
                output[name] = {
                    "score": result.score,
                    "details": result.details
                }
            return json.dumps(output, indent=2, ensure_ascii=False)

        elif format_type == "markdown":
            lines = ["# Khmer Evaluation Metrics\n"]
            for name, result in results.items():
                lines.append(f"## {result.metric_name}")
                lines.append(f"**Score:** {result.score:.4f}\n")
                if result.details:
                    lines.append("**Details:**")
                    for key, value in result.details.items():
                        if isinstance(value, float):
                            lines.append(f"- {key}: {value:.4f}")
                        else:
                            lines.append(f"- {key}: {value}")
                lines.append("")
            return "\n".join(lines)

        else:  # text
            lines = ["Khmer Evaluation Metrics"]
            lines.append("=" * 50)
            for name, result in results.items():
                lines.append(f"{result.metric_name}: {result.score:.4f}")
                if result.details:
                    for key, value in result.details.items():
                        if isinstance(value, float):
                            lines.append(f"  - {key}: {value:.4f}")
                        else:
                            lines.append(f"  - {key}: {value}")
            return "\n".join(lines)