"""
Evaluation Framework for Multimodal Video Retrieval System
Implements standard IR metrics: Precision, Recall, F1, MRR, nDCG
Plus Code Extraction metrics: BLEU, Edit Distance, CER, WER, Code Coverage
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
import re
from collections import Counter
import difflib


# ==============================================================================
# CODE EXTRACTION EVALUATION METRICS
# ==============================================================================

class CodeExtractionEvaluator:
    """
    Evaluates code extraction accuracy by comparing extracted code with ground truth.
    Implements metrics suitable for thesis evaluation.
    """
    
    def __init__(self):
        self.results_history = []
    
    # --------------------------------------------------------------------------
    # Character-Level Metrics
    # --------------------------------------------------------------------------
    
    def character_error_rate(self, extracted: str, ground_truth: str) -> float:
        """
        Calculate Character Error Rate (CER).
        CER = (Substitutions + Insertions + Deletions) / Total Characters in Ground Truth
        
        Lower is better. 0 = perfect match.
        """
        if not ground_truth:
            return 1.0 if extracted else 0.0
        
        # Use edit distance at character level
        distance = self._levenshtein_distance(extracted, ground_truth)
        return distance / len(ground_truth)
    
    def character_accuracy(self, extracted: str, ground_truth: str) -> float:
        """
        Calculate character-level accuracy.
        Accuracy = 1 - CER (clamped to [0, 1])
        """
        cer = self.character_error_rate(extracted, ground_truth)
        return max(0.0, 1.0 - cer)
    
    # --------------------------------------------------------------------------
    # Word/Token-Level Metrics
    # --------------------------------------------------------------------------
    
    def word_error_rate(self, extracted: str, ground_truth: str) -> float:
        """
        Calculate Word Error Rate (WER).
        WER = (Substitutions + Insertions + Deletions) / Total Words in Ground Truth
        
        Lower is better. 0 = perfect match.
        """
        extracted_words = self._tokenize(extracted)
        gt_words = self._tokenize(ground_truth)
        
        if not gt_words:
            return 1.0 if extracted_words else 0.0
        
        distance = self._levenshtein_distance(extracted_words, gt_words)
        return distance / len(gt_words)
    
    def token_accuracy(self, extracted: str, ground_truth: str) -> float:
        """
        Calculate token-level accuracy for code.
        Uses code-aware tokenization.
        """
        extracted_tokens = self._code_tokenize(extracted)
        gt_tokens = self._code_tokenize(ground_truth)
        
        if not gt_tokens:
            return 1.0 if not extracted_tokens else 0.0
        
        # Count matching tokens
        extracted_counter = Counter(extracted_tokens)
        gt_counter = Counter(gt_tokens)
        
        # Intersection
        common = extracted_counter & gt_counter
        common_count = sum(common.values())
        
        return common_count / len(gt_tokens)
    
    # --------------------------------------------------------------------------
    # BLEU Score (Standard MT Metric adapted for code)
    # --------------------------------------------------------------------------
    
    def bleu_score(self, extracted: str, ground_truth: str, max_n: int = 4) -> Dict[str, float]:
        """
        Calculate BLEU score (Bilingual Evaluation Understudy).
        Commonly used for evaluating code generation.
        
        Returns:
            Dict with BLEU-1 through BLEU-n and combined BLEU score
        """
        extracted_tokens = self._code_tokenize(extracted)
        gt_tokens = self._code_tokenize(ground_truth)
        
        if not gt_tokens or not extracted_tokens:
            return {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)} | {'bleu': 0.0}
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            extracted_ngrams = self._get_ngrams(extracted_tokens, n)
            gt_ngrams = self._get_ngrams(gt_tokens, n)
            
            if not extracted_ngrams:
                precisions.append(0.0)
                continue
            
            # Count matches
            matches = sum(min(extracted_ngrams[ng], gt_ngrams.get(ng, 0)) 
                         for ng in extracted_ngrams)
            total = sum(extracted_ngrams.values())
            
            precisions.append(matches / total if total > 0 else 0.0)
        
        # Calculate brevity penalty
        bp = min(1.0, np.exp(1 - len(gt_tokens) / len(extracted_tokens))) if extracted_tokens else 0.0
        
        # Combined BLEU (geometric mean of precisions with brevity penalty)
        if all(p > 0 for p in precisions):
            log_precisions = [np.log(p) for p in precisions]
            bleu = bp * np.exp(np.mean(log_precisions))
        else:
            bleu = 0.0
        
        results = {f'bleu_{i+1}': p for i, p in enumerate(precisions)}
        results['bleu'] = bleu
        return results
    
    # --------------------------------------------------------------------------
    # CodeBLEU (Code-specific BLEU variant)
    # --------------------------------------------------------------------------
    
    def codebleu_score(self, extracted: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate CodeBLEU - a code-specific metric that considers:
        1. N-gram match (like BLEU)
        2. Weighted n-gram match (keywords weighted higher)
        3. Syntax match (AST-based - simplified here)
        4. Dataflow match (simplified)
        
        Returns comprehensive code evaluation metrics.
        """
        # Standard BLEU component
        bleu_scores = self.bleu_score(extracted, ground_truth)
        
        # Weighted BLEU (code keywords weighted higher)
        weighted_bleu = self._weighted_ngram_match(extracted, ground_truth)
        
        # Syntax match (simplified - checks structural elements)
        syntax_match = self._syntax_match_score(extracted, ground_truth)
        
        # Dataflow match (simplified - checks variable usage patterns)
        dataflow_match = self._dataflow_match_score(extracted, ground_truth)
        
        # Combined CodeBLEU (standard weights: 0.25 each)
        codebleu = (0.25 * bleu_scores['bleu'] + 
                   0.25 * weighted_bleu + 
                   0.25 * syntax_match + 
                   0.25 * dataflow_match)
        
        return {
            'codebleu': codebleu,
            'ngram_match': bleu_scores['bleu'],
            'weighted_ngram_match': weighted_bleu,
            'syntax_match': syntax_match,
            'dataflow_match': dataflow_match,
            **bleu_scores
        }
    
    # --------------------------------------------------------------------------
    # Edit Distance Metrics
    # --------------------------------------------------------------------------
    
    def edit_distance(self, extracted: str, ground_truth: str) -> int:
        """
        Calculate Levenshtein edit distance.
        Number of single-character edits needed to transform extracted into ground_truth.
        """
        return self._levenshtein_distance(extracted, ground_truth)
    
    def normalized_edit_distance(self, extracted: str, ground_truth: str) -> float:
        """
        Normalized edit distance (0 to 1 scale).
        0 = identical, 1 = completely different.
        """
        if not extracted and not ground_truth:
            return 0.0
        
        max_len = max(len(extracted), len(ground_truth))
        distance = self._levenshtein_distance(extracted, ground_truth)
        return distance / max_len
    
    def edit_similarity(self, extracted: str, ground_truth: str) -> float:
        """
        Edit-based similarity score (1 - normalized edit distance).
        1 = identical, 0 = completely different.
        """
        return 1.0 - self.normalized_edit_distance(extracted, ground_truth)
    
    # --------------------------------------------------------------------------
    # Line-Level Metrics
    # --------------------------------------------------------------------------
    
    def line_accuracy(self, extracted: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate line-level accuracy metrics.
        
        Returns:
            Dict with exact_match_ratio, partial_match_ratio, line_precision, line_recall
        """
        extracted_lines = [l.strip() for l in extracted.split('\n') if l.strip()]
        gt_lines = [l.strip() for l in ground_truth.split('\n') if l.strip()]
        
        if not gt_lines:
            return {
                'line_exact_match': 1.0 if not extracted_lines else 0.0,
                'line_precision': 1.0 if not extracted_lines else 0.0,
                'line_recall': 1.0,
                'line_f1': 1.0 if not extracted_lines else 0.0
            }
        
        # Exact matches
        gt_set = set(gt_lines)
        extracted_set = set(extracted_lines)
        exact_matches = len(gt_set & extracted_set)
        
        # Precision and recall
        precision = exact_matches / len(extracted_set) if extracted_set else 0.0
        recall = exact_matches / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'line_exact_match': exact_matches / len(gt_lines),
            'line_precision': precision,
            'line_recall': recall,
            'line_f1': f1
        }
    
    # --------------------------------------------------------------------------
    # Code Structure Metrics
    # --------------------------------------------------------------------------
    
    def function_detection_rate(self, extracted: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate how well functions/methods are detected.
        """
        # Extract function names from both
        func_pattern = r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:\([^)]*\)|[^=])*=>|\()'
        
        extracted_funcs = set(re.findall(func_pattern, extracted))
        gt_funcs = set(re.findall(func_pattern, ground_truth))
        
        if not gt_funcs:
            return {'func_precision': 1.0, 'func_recall': 1.0, 'func_f1': 1.0}
        
        matches = len(extracted_funcs & gt_funcs)
        precision = matches / len(extracted_funcs) if extracted_funcs else 0.0
        recall = matches / len(gt_funcs)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'func_precision': precision,
            'func_recall': recall,
            'func_f1': f1,
            'funcs_detected': list(extracted_funcs),
            'funcs_expected': list(gt_funcs)
        }
    
    def import_detection_rate(self, extracted: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate how well imports are detected.
        """
        import_pattern = r"(?:import|from)\s+['\"]?([^'\";\n]+)['\"]?"
        
        extracted_imports = set(re.findall(import_pattern, extracted))
        gt_imports = set(re.findall(import_pattern, ground_truth))
        
        if not gt_imports:
            return {'import_precision': 1.0, 'import_recall': 1.0, 'import_f1': 1.0}
        
        # Normalize import paths for comparison
        extracted_normalized = {self._normalize_import(i) for i in extracted_imports}
        gt_normalized = {self._normalize_import(i) for i in gt_imports}
        
        matches = len(extracted_normalized & gt_normalized)
        precision = matches / len(extracted_normalized) if extracted_normalized else 0.0
        recall = matches / len(gt_normalized)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'import_precision': precision,
            'import_recall': recall,
            'import_f1': f1
        }
    
    def jsx_component_detection(self, extracted: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate JSX component detection (for React code).
        """
        # Match JSX components like <ComponentName ... />
        component_pattern = r'<([A-Z][a-zA-Z0-9]*)'
        
        extracted_components = set(re.findall(component_pattern, extracted))
        gt_components = set(re.findall(component_pattern, ground_truth))
        
        if not gt_components:
            return {'jsx_precision': 1.0, 'jsx_recall': 1.0, 'jsx_f1': 1.0}
        
        matches = len(extracted_components & gt_components)
        precision = matches / len(extracted_components) if extracted_components else 0.0
        recall = matches / len(gt_components)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'jsx_precision': precision,
            'jsx_recall': recall,
            'jsx_f1': f1,
            'jsx_detected': list(extracted_components),
            'jsx_expected': list(gt_components)
        }
    
    # --------------------------------------------------------------------------
    # Comprehensive Evaluation
    # --------------------------------------------------------------------------
    
    def evaluate_code_extraction(
        self, 
        extracted: str, 
        ground_truth: str,
        filename: str = "unknown"
    ) -> Dict:
        """
        Run all code extraction metrics and return comprehensive results.
        
        Args:
            extracted: The extracted code
            ground_truth: The actual ground truth code
            filename: Name of the file being evaluated
        
        Returns:
            Comprehensive evaluation results dictionary
        """
        # Clean up whitespace differences
        extracted_clean = self._normalize_code(extracted)
        gt_clean = self._normalize_code(ground_truth)
        
        results = {
            'filename': filename,
            'extracted_length': len(extracted),
            'ground_truth_length': len(ground_truth),
            
            # Character-level
            'character_error_rate': self.character_error_rate(extracted_clean, gt_clean),
            'character_accuracy': self.character_accuracy(extracted_clean, gt_clean),
            
            # Word/Token-level
            'word_error_rate': self.word_error_rate(extracted_clean, gt_clean),
            'token_accuracy': self.token_accuracy(extracted_clean, gt_clean),
            
            # BLEU scores
            **self.bleu_score(extracted_clean, gt_clean),
            
            # CodeBLEU
            **self.codebleu_score(extracted_clean, gt_clean),
            
            # Edit distance
            'edit_distance': self.edit_distance(extracted_clean, gt_clean),
            'normalized_edit_distance': self.normalized_edit_distance(extracted_clean, gt_clean),
            'edit_similarity': self.edit_similarity(extracted_clean, gt_clean),
            
            # Line-level
            **self.line_accuracy(extracted_clean, gt_clean),
            
            # Structure detection
            **self.function_detection_rate(extracted, ground_truth),
            **self.import_detection_rate(extracted, ground_truth),
            **self.jsx_component_detection(extracted, ground_truth)
        }
        
        # Store in history
        self.results_history.append(results)
        
        return results
    
    def evaluate_multiple_files(
        self, 
        extracted_files: Dict[str, str], 
        ground_truth_files: Dict[str, str]
    ) -> Dict:
        """
        Evaluate multiple files and compute aggregate metrics.
        
        Args:
            extracted_files: Dict mapping filename to extracted code
            ground_truth_files: Dict mapping filename to ground truth code
        
        Returns:
            Individual and aggregate results
        """
        individual_results = []
        
        # Evaluate each file
        for filename in ground_truth_files:
            extracted = extracted_files.get(filename, "")
            ground_truth = ground_truth_files[filename]
            
            result = self.evaluate_code_extraction(extracted, ground_truth, filename)
            individual_results.append(result)
        
        # Check for files detected but not in ground truth
        extra_files = set(extracted_files.keys()) - set(ground_truth_files.keys())
        missing_files = set(ground_truth_files.keys()) - set(extracted_files.keys())
        
        # Aggregate metrics
        numeric_keys = [k for k in individual_results[0].keys() 
                       if isinstance(individual_results[0][k], (int, float))]
        
        aggregate = {}
        for key in numeric_keys:
            values = [r[key] for r in individual_results]
            aggregate[f'mean_{key}'] = np.mean(values)
            aggregate[f'std_{key}'] = np.std(values)
            aggregate[f'min_{key}'] = np.min(values)
            aggregate[f'max_{key}'] = np.max(values)
        
        # File detection metrics
        detected_files = set(extracted_files.keys())
        expected_files = set(ground_truth_files.keys())
        file_matches = len(detected_files & expected_files)
        
        file_detection = {
            'file_precision': file_matches / len(detected_files) if detected_files else 0.0,
            'file_recall': file_matches / len(expected_files) if expected_files else 0.0,
            'files_detected': len(detected_files),
            'files_expected': len(expected_files),
            'extra_files': list(extra_files),
            'missing_files': list(missing_files)
        }
        
        file_detection['file_f1'] = (
            2 * file_detection['file_precision'] * file_detection['file_recall'] /
            (file_detection['file_precision'] + file_detection['file_recall'])
            if (file_detection['file_precision'] + file_detection['file_recall']) > 0 else 0.0
        )
        
        return {
            'individual_results': individual_results,
            'aggregate_metrics': aggregate,
            'file_detection': file_detection,
            'total_files_evaluated': len(individual_results)
        }
    
    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------
    
    def _levenshtein_distance(self, s1, s2) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return text.split()
    
    def _code_tokenize(self, code: str) -> List[str]:
        """Code-aware tokenization."""
        # Split on whitespace and punctuation, keeping meaningful tokens
        tokens = re.findall(r'\w+|[^\s\w]', code)
        return [t for t in tokens if t.strip()]
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison (remove extra whitespace)."""
        # Remove comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()
    
    def _normalize_import(self, import_path: str) -> str:
        """Normalize import path for comparison."""
        # Remove quotes and leading ./
        path = import_path.strip().strip('"\'')
        path = re.sub(r'^\./', '', path)
        return path
    
    def _weighted_ngram_match(self, extracted: str, ground_truth: str) -> float:
        """Calculate weighted n-gram match (keywords weighted higher)."""
        keywords = {'import', 'export', 'function', 'const', 'let', 'var', 'return',
                   'if', 'else', 'for', 'while', 'class', 'extends', 'async', 'await',
                   'React', 'useState', 'useEffect', 'useRef', 'props'}
        
        extracted_tokens = self._code_tokenize(extracted)
        gt_tokens = self._code_tokenize(ground_truth)
        
        if not gt_tokens:
            return 0.0
        
        # Weight keywords 2x
        def weight(token):
            return 2.0 if token in keywords else 1.0
        
        total_weight = sum(weight(t) for t in gt_tokens)
        
        extracted_counter = Counter(extracted_tokens)
        gt_counter = Counter(gt_tokens)
        
        weighted_matches = 0.0
        for token in gt_counter:
            matches = min(extracted_counter.get(token, 0), gt_counter[token])
            weighted_matches += matches * weight(token)
        
        return weighted_matches / total_weight if total_weight > 0 else 0.0
    
    def _syntax_match_score(self, extracted: str, ground_truth: str) -> float:
        """
        Simplified syntax match - checks structural elements.
        Full implementation would use AST parsing.
        """
        # Check for structural patterns
        patterns = [
            r'import\s+.*\s+from',  # ES6 imports
            r'export\s+default',     # Default export
            r'function\s+\w+\s*\(',  # Function declarations
            r'const\s+\w+\s*=',      # Const declarations
            r'return\s*\(',          # Return statements
            r'<[A-Z]\w*',            # JSX components
            r'useState\s*\(',        # React hooks
            r'useEffect\s*\(',
        ]
        
        scores = []
        for pattern in patterns:
            in_extracted = bool(re.search(pattern, extracted))
            in_gt = bool(re.search(pattern, ground_truth))
            
            if in_gt:
                scores.append(1.0 if in_extracted else 0.0)
        
        return np.mean(scores) if scores else 1.0
    
    def _dataflow_match_score(self, extracted: str, ground_truth: str) -> float:
        """
        Simplified dataflow match - checks variable usage patterns.
        """
        # Extract variable declarations and usages
        var_pattern = r'(?:const|let|var)\s+(\w+)'
        
        extracted_vars = set(re.findall(var_pattern, extracted))
        gt_vars = set(re.findall(var_pattern, ground_truth))
        
        if not gt_vars:
            return 1.0
        
        matches = len(extracted_vars & gt_vars)
        return matches / len(gt_vars)
    
    # --------------------------------------------------------------------------
    # Output Formatting for Thesis
    # --------------------------------------------------------------------------
    
    def generate_latex_table(self, results: Dict, caption: str = "Code Extraction Evaluation Results") -> str:
        """
        Generate LaTeX table for thesis.
        """
        latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Score}} \\\\
\\hline
Character Accuracy & {results.get('character_accuracy', 0):.4f} \\\\
Token Accuracy & {results.get('token_accuracy', 0):.4f} \\\\
BLEU & {results.get('bleu', 0):.4f} \\\\
CodeBLEU & {results.get('codebleu', 0):.4f} \\\\
Edit Similarity & {results.get('edit_similarity', 0):.4f} \\\\
Line F1 & {results.get('line_f1', 0):.4f} \\\\
Function F1 & {results.get('func_f1', 0):.4f} \\\\
Import F1 & {results.get('import_f1', 0):.4f} \\\\
JSX F1 & {results.get('jsx_f1', 0):.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""
        return latex
    
    def generate_summary_report(self, results: Dict) -> str:
        """
        Generate human-readable summary report.
        """
        report = []
        report.append("=" * 60)
        report.append("CODE EXTRACTION EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\nFile: {results.get('filename', 'Unknown')}")
        report.append(f"Extracted Length: {results.get('extracted_length', 0)} chars")
        report.append(f"Ground Truth Length: {results.get('ground_truth_length', 0)} chars")
        
        report.append("\n--- Character-Level Metrics ---")
        report.append(f"Character Error Rate (CER): {results.get('character_error_rate', 0):.4f}")
        report.append(f"Character Accuracy: {results.get('character_accuracy', 0):.4f}")
        
        report.append("\n--- Token-Level Metrics ---")
        report.append(f"Word Error Rate (WER): {results.get('word_error_rate', 0):.4f}")
        report.append(f"Token Accuracy: {results.get('token_accuracy', 0):.4f}")
        
        report.append("\n--- BLEU Scores ---")
        report.append(f"BLEU-1: {results.get('bleu_1', 0):.4f}")
        report.append(f"BLEU-2: {results.get('bleu_2', 0):.4f}")
        report.append(f"BLEU-3: {results.get('bleu_3', 0):.4f}")
        report.append(f"BLEU-4: {results.get('bleu_4', 0):.4f}")
        report.append(f"Combined BLEU: {results.get('bleu', 0):.4f}")
        
        report.append("\n--- CodeBLEU Metrics ---")
        report.append(f"CodeBLEU: {results.get('codebleu', 0):.4f}")
        report.append(f"  - N-gram Match: {results.get('ngram_match', 0):.4f}")
        report.append(f"  - Weighted N-gram: {results.get('weighted_ngram_match', 0):.4f}")
        report.append(f"  - Syntax Match: {results.get('syntax_match', 0):.4f}")
        report.append(f"  - Dataflow Match: {results.get('dataflow_match', 0):.4f}")
        
        report.append("\n--- Edit Distance Metrics ---")
        report.append(f"Edit Distance: {results.get('edit_distance', 0)}")
        report.append(f"Normalized Edit Distance: {results.get('normalized_edit_distance', 0):.4f}")
        report.append(f"Edit Similarity: {results.get('edit_similarity', 0):.4f}")
        
        report.append("\n--- Line-Level Metrics ---")
        report.append(f"Line Exact Match: {results.get('line_exact_match', 0):.4f}")
        report.append(f"Line Precision: {results.get('line_precision', 0):.4f}")
        report.append(f"Line Recall: {results.get('line_recall', 0):.4f}")
        report.append(f"Line F1: {results.get('line_f1', 0):.4f}")
        
        report.append("\n--- Structure Detection ---")
        report.append(f"Function F1: {results.get('func_f1', 0):.4f}")
        report.append(f"Import F1: {results.get('import_f1', 0):.4f}")
        report.append(f"JSX Component F1: {results.get('jsx_f1', 0):.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class RetrievalEvaluator:
    """
    Evaluates retrieval system performance using standard IR metrics.
    """
    
    def __init__(self, ground_truth_file: str = None):
        """
        Initialize evaluator with optional ground truth data.
        
        Args:
            ground_truth_file: Path to JSON file with ground truth relevance judgments
                Format: {query_id: {doc_id: relevance_score, ...}, ...}
        """
        self.ground_truth = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
    
    def precision_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff rank
        
        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff rank
        
        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
        return relevant_retrieved / len(relevant)
    
    def f1_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Calculate F1@K (harmonic mean of Precision@K and Recall@K)
        """
        p = self.precision_at_k(retrieved, relevant, k)
        r = self.recall_at_k(retrieved, relevant, k)
        
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: set) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Returns:
            MRR score (reciprocal of rank of first relevant result)
        """
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def dcg_at_k(self, scores: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at K
        
        Args:
            scores: List of relevance scores in retrieval order
            k: Cutoff rank
        """
        scores_at_k = scores[:k]
        gains = np.array(scores_at_k)
        discounts = np.log2(np.arange(len(gains)) + 2)  # +2 because log2(1) = 0
        return np.sum(gains / discounts)
    
    def ndcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K
        
        Args:
            retrieved: Ordered list of retrieved document IDs
            relevance_scores: Dict mapping doc_id to relevance score
            k: Cutoff rank
        
        Returns:
            nDCG@K score
        """
        # Get actual relevance scores for retrieved docs
        actual_scores = [relevance_scores.get(doc, 0) for doc in retrieved[:k]]
        
        # Calculate ideal ordering (sorted by relevance)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        
        dcg = self.dcg_at_k(actual_scores, k)
        idcg = self.dcg_at_k(ideal_scores, k)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def average_precision(self, retrieved: List[str], relevant: set) -> float:
        """
        Calculate Average Precision (AP)
        
        Returns:
            AP score
        """
        if len(relevant) == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant)
    
    def evaluate_query(
        self, 
        query_id: str,
        retrieved: List[str],
        relevant: set = None,
        relevance_scores: Dict[str, float] = None,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single query.
        
        Args:
            query_id: Query identifier
            retrieved: Ordered list of retrieved document IDs
            relevant: Set of relevant document IDs (optional if using ground_truth)
            relevance_scores: Dict of relevance scores (optional if using ground_truth)
            k_values: List of K values for @K metrics
        
        Returns:
            Dictionary of metric scores
        """
        # Use ground truth if specific relevant set not provided
        if relevant is None and query_id in self.ground_truth:
            relevance_scores = self.ground_truth[query_id]
            relevant = set(doc for doc, score in relevance_scores.items() if score > 0)
        elif relevant is None:
            relevant = set()
        
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant}
        
        results = {
            'mrr': self.mean_reciprocal_rank(retrieved, relevant),
            'ap': self.average_precision(retrieved, relevant)
        }
        
        for k in k_values:
            results[f'precision@{k}'] = self.precision_at_k(retrieved, relevant, k)
            results[f'recall@{k}'] = self.recall_at_k(retrieved, relevant, k)
            results[f'f1@{k}'] = self.f1_at_k(retrieved, relevant, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(retrieved, relevance_scores, k)
        
        return results
    
    def evaluate_batch(
        self,
        queries: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate multiple queries and compute mean metrics.
        
        Args:
            queries: List of dicts with 'query_id', 'retrieved', 'relevant' keys
            k_values: List of K values for @K metrics
        
        Returns:
            Dictionary of mean metric scores across all queries
        """
        all_results = []
        
        for q in queries:
            result = self.evaluate_query(
                query_id=q.get('query_id', ''),
                retrieved=q['retrieved'],
                relevant=q.get('relevant'),
                relevance_scores=q.get('relevance_scores'),
                k_values=k_values
            )
            all_results.append(result)
        
        # Compute means
        if not all_results:
            return {}
        
        mean_results = {}
        for key in all_results[0].keys():
            mean_results[f'mean_{key}'] = np.mean([r[key] for r in all_results])
        
        return mean_results


class VideoRetrievalBenchmark:
    """
    Benchmark suite specifically for video chunk retrieval evaluation.
    """
    
    def __init__(self, output_folder: str = 'outputs'):
        self.output_folder = output_folder
        self.evaluator = RetrievalEvaluator()
        self.benchmark_results = []
    
    def create_ground_truth_template(self, video_id: str, queries: List[str]) -> Dict:
        """
        Create a ground truth template for manual annotation.
        
        Args:
            video_id: The video ID to create template for
            queries: List of test queries
        
        Returns:
            Template dict to be filled with relevance judgments
        """
        import json
        
        # Load chunk metadata
        metadata_file = os.path.join(self.output_folder, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"No metadata for video {video_id}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        template = {
            'video_id': video_id,
            'queries': {}
        }
        
        for query in queries:
            template['queries'][query] = {
                'relevant_chunks': [],  # Fill with chunk indices
                'relevance_scores': {}  # Fill with {chunk_idx: score} for graded relevance
            }
        
        # Add chunk info for reference
        template['chunk_reference'] = [
            {
                'index': c['index'],
                'timestamp': c['timestamp'],
                'ocr_preview': c['ocr_text'][:100] if c['ocr_text'] else '',
                'audio_preview': c['audio_context'][:100] if c['audio_context'] else ''
            }
            for c in metadata['chunks']
        ]
        
        return template
    
    def run_benchmark(
        self,
        video_id: str,
        queries_with_ground_truth: List[Dict],
        search_function,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Run benchmark on a video with provided queries and ground truth.
        
        Args:
            video_id: Video to benchmark
            queries_with_ground_truth: List of dicts with 'query' and 'relevant_chunks' keys
            search_function: Function(query, video_id, top_k) -> List[chunk_indices]
            k_values: K values for metrics
        
        Returns:
            Benchmark results
        """
        query_results = []
        
        for q in queries_with_ground_truth:
            query = q['query']
            relevant = set(q['relevant_chunks'])
            relevance_scores = q.get('relevance_scores', {str(c): 1.0 for c in relevant})
            
            # Run search
            max_k = max(k_values)
            retrieved = search_function(query, video_id, max_k)
            retrieved_str = [str(r) for r in retrieved]
            
            # Evaluate
            metrics = self.evaluator.evaluate_query(
                query_id=query,
                retrieved=retrieved_str,
                relevant=set(str(c) for c in relevant),
                relevance_scores={str(k): v for k, v in relevance_scores.items()},
                k_values=k_values
            )
            
            query_results.append({
                'query': query,
                'retrieved': retrieved,
                'relevant': list(relevant),
                'metrics': metrics
            })
        
        # Compute aggregate metrics
        aggregate = {}
        if query_results:
            metric_keys = query_results[0]['metrics'].keys()
            for key in metric_keys:
                aggregate[f'mean_{key}'] = np.mean([r['metrics'][key] for r in query_results])
        
        result = {
            'video_id': video_id,
            'num_queries': len(queries_with_ground_truth),
            'individual_results': query_results,
            'aggregate_metrics': aggregate
        }
        
        self.benchmark_results.append(result)
        return result
    
    def export_results(self, filepath: str):
        """Export benchmark results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        if not self.benchmark_results:
            print("No benchmark results available.")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for result in self.benchmark_results:
            print(f"\nVideo: {result['video_id']}")
            print(f"Queries: {result['num_queries']}")
            print("\nAggregate Metrics:")
            for metric, value in result['aggregate_metrics'].items():
                print(f"  {metric}: {value:.4f}")


# Utility functions for integration with main app
def create_evaluation_endpoint_data(search_results: List[Dict], query: str) -> Dict:
    """
    Format search results for evaluation tracking.
    """
    return {
        'query': query,
        'retrieved': [r.get('chunk_index', r.get('timestamp')) for r in search_results],
        'scores': [r.get('similarity', 0) for r in search_results]
    }


# ==============================================================================
# EVALUATION RUNNER - Test the system with ground truth
# ==============================================================================

class SystemEvaluationRunner:
    """
    Run comprehensive evaluation of the video code extraction system.
    Suitable for thesis experiments.
    """
    
    def __init__(self, backend_url: str = "http://localhost:5001"):
        self.backend_url = backend_url
        self.code_evaluator = CodeExtractionEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.results = {}
    
    def evaluate_code_extraction(
        self, 
        video_id: str, 
        ground_truth_files: Dict[str, str],
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate code extraction accuracy against ground truth.
        
        Args:
            video_id: The video ID to evaluate
            ground_truth_files: Dict mapping filename to ground truth code
            save_results: Whether to save results to file
        
        Returns:
            Comprehensive evaluation results
        """
        import requests
        
        print(f"\n{'='*60}")
        print(f"EVALUATING CODE EXTRACTION FOR VIDEO: {video_id}")
        print(f"{'='*60}")
        
        # Get extracted files from the system
        extracted_files = {}
        
        for filename in ground_truth_files.keys():
            print(f"\nExtracting: {filename}")
            try:
                response = requests.post(
                    f"{self.backend_url}/api/extract-file",
                    json={"video_id": video_id, "filename": filename},
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    extracted_files[filename] = data.get('code', '')
                    print(f"  ✓ Extracted {len(data.get('code', ''))} chars from {data.get('frames_analyzed', 0)} frames")
                else:
                    print(f"  ✗ Failed: {response.json().get('error', 'Unknown error')}")
                    extracted_files[filename] = ""
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                extracted_files[filename] = ""
        
        # Run evaluation
        print("\n\nRunning evaluation metrics...")
        results = self.code_evaluator.evaluate_multiple_files(extracted_files, ground_truth_files)
        
        # Print individual file results
        print("\n--- Individual File Results ---")
        for file_result in results['individual_results']:
            print(f"\n{file_result['filename']}:")
            print(f"  Character Accuracy: {file_result['character_accuracy']:.4f}")
            print(f"  BLEU: {file_result['bleu']:.4f}")
            print(f"  CodeBLEU: {file_result['codebleu']:.4f}")
            print(f"  Edit Similarity: {file_result['edit_similarity']:.4f}")
            print(f"  Line F1: {file_result['line_f1']:.4f}")
        
        # Print aggregate results
        print("\n--- Aggregate Results ---")
        agg = results['aggregate_metrics']
        print(f"Mean Character Accuracy: {agg.get('mean_character_accuracy', 0):.4f}")
        print(f"Mean BLEU: {agg.get('mean_bleu', 0):.4f}")
        print(f"Mean CodeBLEU: {agg.get('mean_codebleu', 0):.4f}")
        print(f"Mean Edit Similarity: {agg.get('mean_edit_similarity', 0):.4f}")
        print(f"Mean Line F1: {agg.get('mean_line_f1', 0):.4f}")
        
        # Print file detection results
        print("\n--- File Detection ---")
        fd = results['file_detection']
        print(f"File Precision: {fd['file_precision']:.4f}")
        print(f"File Recall: {fd['file_recall']:.4f}")
        print(f"File F1: {fd['file_f1']:.4f}")
        
        if save_results:
            results_path = f"outputs/{video_id}_evaluation_results.json"
            with open(results_path, 'w') as f:
                # Convert non-serializable items
                serializable_results = json.loads(json.dumps(results, default=str))
                json.dump(serializable_results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
        
        self.results[video_id] = results
        return results
    
    def run_full_evaluation(
        self,
        video_id: str,
        ground_truth_files: Dict[str, str],
        test_queries: List[Dict] = None
    ) -> Dict:
        """
        Run complete evaluation including code extraction and retrieval.
        
        Args:
            video_id: Video to evaluate
            ground_truth_files: Ground truth code files
            test_queries: Optional list of test queries with relevant chunks
        
        Returns:
            Complete evaluation results
        """
        results = {
            'video_id': video_id,
            'timestamp': str(np.datetime64('now')),
            'code_extraction': None,
            'retrieval': None
        }
        
        # Code extraction evaluation
        results['code_extraction'] = self.evaluate_code_extraction(
            video_id, ground_truth_files
        )
        
        # Retrieval evaluation (if test queries provided)
        if test_queries:
            import requests
            
            print(f"\n{'='*60}")
            print("EVALUATING RETRIEVAL PERFORMANCE")
            print(f"{'='*60}")
            
            query_results = []
            for q in test_queries:
                query = q['query']
                relevant_chunks = set(q.get('relevant_chunks', []))
                
                # Run search
                response = requests.post(
                    f"{self.backend_url}/api/search",
                    json={"query": query, "video_id": video_id, "top_k": 10}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    retrieved = [str(r.get('chunk_index', i)) for i, r in enumerate(data.get('results', []))]
                    
                    metrics = self.retrieval_evaluator.evaluate_query(
                        query_id=query,
                        retrieved=retrieved,
                        relevant=set(str(c) for c in relevant_chunks)
                    )
                    
                    query_results.append({
                        'query': query,
                        'metrics': metrics
                    })
            
            if query_results:
                # Aggregate retrieval metrics
                metric_keys = query_results[0]['metrics'].keys()
                retrieval_aggregate = {}
                for key in metric_keys:
                    values = [qr['metrics'][key] for qr in query_results]
                    retrieval_aggregate[f'mean_{key}'] = np.mean(values)
                
                results['retrieval'] = {
                    'individual': query_results,
                    'aggregate': retrieval_aggregate
                }
                
                print("\n--- Retrieval Metrics ---")
                for key, value in retrieval_aggregate.items():
                    print(f"{key}: {value:.4f}")
        
        return results
    
    def generate_thesis_tables(self, results: Dict) -> str:
        """
        Generate LaTeX tables for thesis.
        """
        output = []
        
        # Code extraction table
        if results.get('code_extraction'):
            agg = results['code_extraction']['aggregate_metrics']
            
            output.append("""
\\begin{table}[h]
\\centering
\\caption{Code Extraction Evaluation Results}
\\label{tab:code_extraction}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} \\\\
\\hline""")
            
            metrics_to_show = [
                ('character_accuracy', 'Character Accuracy'),
                ('bleu', 'BLEU'),
                ('codebleu', 'CodeBLEU'),
                ('edit_similarity', 'Edit Similarity'),
                ('line_f1', 'Line F1'),
                ('func_f1', 'Function F1'),
                ('import_f1', 'Import F1'),
                ('jsx_f1', 'JSX F1')
            ]
            
            for key, label in metrics_to_show:
                mean = agg.get(f'mean_{key}', 0)
                std = agg.get(f'std_{key}', 0)
                min_val = agg.get(f'min_{key}', 0)
                max_val = agg.get(f'max_{key}', 0)
                output.append(f"{label} & {mean:.4f} & {std:.4f} & {min_val:.4f} & {max_val:.4f} \\\\")
            
            output.append("""\\hline
\\end{tabular}
\\end{table}""")
            
            # File detection table
            fd = results['code_extraction']['file_detection']
            output.append(f"""
\\begin{{table}}[h]
\\centering
\\caption{{File Detection Results}}
\\label{{tab:file_detection}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Score}} \\\\
\\hline
Precision & {fd['file_precision']:.4f} \\\\
Recall & {fd['file_recall']:.4f} \\\\
F1 Score & {fd['file_f1']:.4f} \\\\
Files Detected & {fd['files_detected']} \\\\
Files Expected & {fd['files_expected']} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}""")
        
        # Retrieval table
        if results.get('retrieval'):
            ret_agg = results['retrieval']['aggregate']
            output.append("""
\\begin{table}[h]
\\centering
\\caption{Retrieval Evaluation Results}
\\label{tab:retrieval}
\\begin{tabular}{|l|c|}
\\hline
\\textbf{Metric} & \\textbf{Score} \\\\
\\hline""")
            
            for key, value in ret_agg.items():
                label = key.replace('mean_', '').replace('_', ' ').title()
                output.append(f"{label} & {value:.4f} \\\\")
            
            output.append("""\\hline
\\end{tabular}
\\end{table}""")
        
        return "\n".join(output)


# ==============================================================================
# GROUND TRUTH DATA
# ==============================================================================

# Timeline.js ground truth (from the video OPaLnMw2i_0)
TIMELINE_JS_GROUND_TRUTH = '''import React, { useState, useRef, useEffect } from 'react';

import { CarouselButton, CarouselButtonDot, CarouselButtons, CarouselContainer, CarouselItem, CarouselItemImg, CarouselItemText, CarouselItemTitle, CarouselMobileScrollNode } from './TimeLineStyles';
import { Section, SectionDivider, SectionText, SectionTitle } from '../../styles/GlobalComponents';
import { TimeLineData } from '../../constants/constants';

const TOTAL_CAROUSEL_COUNT = TimeLineData.length;

const Timeline = () => {
  const [activeItem, setActiveItem] = useState(0);
  const carouselRef = useRef();

  const scroll = (node, left) => {
    return node.scrollTo({ left, behavior: 'smooth' });
  }

  const handleClick = (e, i) => {
    e.preventDefault();

    if (carouselRef.current) {
      const scrollLeft = Math.floor(carouselRef.current.scrollWidth * 0.7 * (i / TimeLineData.length));
      
      scroll(carouselRef.current, scrollLeft);
    }
  }

  const handleScroll = () => {
    if (carouselRef.current) {
      const index = Math.round((carouselRef.current.scrollLeft / (carouselRef.current.scrollWidth * 0.7)) * TimeLineData.length);

      setActiveItem(index);
    }
  }

  useEffect(() => {
    const handleResize = () => {
      scroll(carouselRef.current, 0);
    }

    window.addEventListener('resize', handleResize);
  }, []);

  return (
    <Section id="about">
      <SectionTitle>About Me</SectionTitle>
      <SectionText>
      The purpose of JavaScript Mastery is to help aspiring and established developers to take their development skills to the next level and build awesome apps.
      </SectionText>
      <CarouselContainer ref={carouselRef} onScroll={handleScroll}>
        <>
          {TimeLineData.map((item, index) => (
            <CarouselMobileScrollNode
              key={index}
              final={index === TOTAL_CAROUSEL_COUNT - 1}>
              <CarouselItem
                index={index}
                id={`carousel__item-${index}`}
                active={activeItem}
                onClick={(e) => handleClick(e, index)}>
                <CarouselItemTitle>
                  {`${item.year}`}
                  <CarouselItemImg
                    width="208"
                    height="6"
                    viewBox="0 0 208 6"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg">
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M2.5 5.5C3.88071 5.5 5 4.38071 5 3V3.5L208 3.50002V2.50002L5 2.5V3C5 1.61929 3.88071 0.5 2.5 0.5C1.11929 0.5 0 1.61929 0 3C0 4.38071 1.11929 5.5 2.5 5.5Z"
                      fill="url(#paint0_linear)"
                      fill-opacity="0.33"
                    />
                    <defs>
                      <linearGradient
                        id="paint0_linear"
                        x1="-4.30412e-10"
                        y1="0.5"
                        x2="208"
                        y2="0.500295"
                        gradientUnits="userSpaceOnUse">
                        <stop stop-color="white" />
                        <stop
                          offset="0.79478"
                          stop-color="white"
                          stop-opacity="0"
                        />
                      </linearGradient>
                    </defs>
                  </CarouselItemImg>
                </CarouselItemTitle>
                <CarouselItemText>{item.text}</CarouselItemText>
              </CarouselItem>
            </CarouselMobileScrollNode>
          ))}
        </>
      </CarouselContainer>
      <CarouselButtons>
        {TimeLineData.map((item, index) => {
          return (
            <CarouselButton
              key={index}
              index={index}
              active={activeItem}
              onClick={(e) => handleClick(e, index)}
              type="button">
              <CarouselButtonDot active={activeItem} />
            </CarouselButton>
          );
        })}
      </CarouselButtons>
      <SectionDivider /> 
    </Section>
  );
};

export default Timeline;'''


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Video Code Extraction System')
    parser.add_argument('--video-id', type=str, default='OPaLnMw2i_0', 
                       help='Video ID to evaluate')
    parser.add_argument('--backend-url', type=str, default='http://localhost:5001',
                       help='Backend URL')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with Timeline.js only')
    parser.add_argument('--generate-latex', action='store_true',
                       help='Generate LaTeX tables for thesis')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MULTIMODAL VIDEO CODE EXTRACTION - EVALUATION FRAMEWORK")
    print("="*70)
    
    # Initialize evaluator
    runner = SystemEvaluationRunner(backend_url=args.backend_url)
    
    if args.quick_test:
        # Quick test with just Timeline.js
        print("\nRunning quick evaluation with Timeline.js...")
        
        ground_truth = {
            'Timeline.js': TIMELINE_JS_GROUND_TRUTH
        }
        
        results = runner.evaluate_code_extraction(args.video_id, ground_truth)
        
        # Print summary
        if results['individual_results']:
            result = results['individual_results'][0]
            print("\n" + "="*60)
            print("QUICK TEST SUMMARY - Timeline.js")
            print("="*60)
            print(f"\nKey Metrics:")
            print(f"  Character Accuracy: {result['character_accuracy']:.2%}")
            print(f"  BLEU Score: {result['bleu']:.4f}")
            print(f"  CodeBLEU: {result['codebleu']:.4f}")
            print(f"  Edit Similarity: {result['edit_similarity']:.2%}")
            print(f"  Line F1: {result['line_f1']:.4f}")
            print(f"  Function Detection F1: {result['func_f1']:.4f}")
            print(f"  Import Detection F1: {result['import_f1']:.4f}")
            print(f"  JSX Component F1: {result['jsx_f1']:.4f}")
        
        if args.generate_latex:
            print("\n--- LaTeX Table ---")
            print(runner.code_evaluator.generate_latex_table(result))
    
    else:
        # Full evaluation (add more ground truth files as needed)
        print("\nRunning full evaluation...")
        
        # Add all ground truth files here
        ground_truth_files = {
            'Timeline.js': TIMELINE_JS_GROUND_TRUTH,
            # Add more files as you collect ground truth
        }
        
        results = runner.run_full_evaluation(
            args.video_id,
            ground_truth_files
        )
        
        if args.generate_latex:
            print("\n" + "="*60)
            print("LaTeX TABLES FOR THESIS")
            print("="*60)
            print(runner.generate_thesis_tables(results))
    
    print("\n✓ Evaluation complete!")

