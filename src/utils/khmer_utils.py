"""
Khmer Language Processing Utilities for GDPval Evaluation
Comprehensive toolkit for handling Khmer text in AI model evaluation
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import Counter
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class KhmerTextProcessor:
    """Main class for Khmer text processing and validation"""
    
    # Khmer Unicode ranges
    KHMER_CONSONANTS = range(0x1780, 0x17A3)  # ក to ឣ
    KHMER_DEPENDENT_VOWELS = range(0x17B6, 0x17C6)  # ា to ំ
    KHMER_SIGNS = range(0x17C6, 0x17D4)  # ំ to ៓
    KHMER_DIGITS = range(0x17E0, 0x17EA)  # ០ to ៩
    KHMER_SYMBOLS = range(0x17F0, 0x17FA)  # ៰ to ៹
    
    # Common Khmer particles and stop words
    STOP_WORDS = {
        'និង', 'ឬ', 'ប៉ុន្តែ', 'ហើយ', 'ដែល', 'នៅ', 'ពី', 'ទៅ', 'ក្នុង', 'លើ',
        'ក្រោម', 'មុន', 'ក្រោយ', 'ជាមួយ', 'ដោយ', 'សម្រាប់', 'ដើម្បី', 'នេះ', 'នោះ',
        'គឺ', 'ជា', 'មាន', 'បាន', 'នឹង', 'អាច', 'ត្រូវ', 'ទេ', 'ទាំង', 'ដូច'
    }
    
    # Technical term patterns (English within Khmer text)
    TECH_TERM_PATTERN = re.compile(r'[A-Za-z0-9_\-\.]+')
    
    # Khmer sentence delimiters
    SENTENCE_DELIMITERS = {'។', '៕', '?', '!', '៖'}
    
    # Common Khmer-English code-switching patterns
    CODE_SWITCH_MARKERS = {'/', '-', '(', ')', '[', ']'}
    
    def __init__(self, vocab_path: Optional[str] = None):
        """
        Initialize Khmer text processor
        
        Args:
            vocab_path: Optional path to custom vocabulary file
        """
        self.vocab = self._load_vocabulary(vocab_path) if vocab_path else set()
        self.technical_terms = self._load_technical_terms()
        
    def _load_vocabulary(self, vocab_path: str) -> Set[str]:
        """Load Khmer vocabulary from file"""
        vocab = set()
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        vocab.add(word)
            logger.info(f"Loaded {len(vocab)} words from vocabulary")
        except FileNotFoundError:
            logger.warning(f"Vocabulary file not found: {vocab_path}")
        return vocab
    
    def _load_technical_terms(self) -> Dict[str, str]:
        """Load technical term mappings (English to Khmer)"""
        return {
            # Finance terms
            'interest rate': 'អត្រាការប្រាក់',
            'loan': 'ឥណទាន',
            'credit': 'ឥណទាន',
            'balance sheet': 'តារាងតុល្យការ',
            'income statement': 'របាយការណ៍ចំណូល',
            'NPL': 'ឥណទានមិនដំណើរការ',
            'ROI': 'ផលចំណេញលើការវិនិយោគ',
            
            # Technology terms
            'API': 'អេភីអាយ',
            'database': 'មូលដ្ឋានទិន្នន័យ',
            'algorithm': 'ក្បួនដោះស្រាយ',
            'machine learning': 'ការរៀនដោយម៉ាស៊ីន',
            'artificial intelligence': 'បញ្ញាសិប្បនិមិត្ត',
            'neural network': 'បណ្តាញសរសៃប្រសាទ',
            
            # Agriculture terms
            'fertilizer': 'ជី',
            'irrigation': 'ប្រព័ន្ធស្រោចស្រព',
            'harvest': 'ការប្រមូលផល',
            'rice paddy': 'វាលស្រែ',
            'pesticide': 'ថ្នាំសម្លាប់សត្វល្អិត',
            
            # Healthcare terms
            'diagnosis': 'ការធ្វើរោគវិនិច្ឆ័យ',
            'treatment': 'ការព្យាបាល',
            'vaccine': 'វ៉ាក់សាំង',
            'prescription': 'វេជ្ជបញ្ជា',
            'symptom': 'រោគសញ្ញា'
        }
    
    def validate_text(self, text: str) -> bool:
        """
        Validate if text contains valid Khmer characters
        
        Args:
            text: Input text to validate
            
        Returns:
            True if text contains valid Khmer, False otherwise
        """
        if not text:
            return False
        
        khmer_chars = 0
        total_chars = 0
        
        for char in text:
            if not char.isspace():
                total_chars += 1
                if self.is_khmer_char(char):
                    khmer_chars += 1
        
        # Text should be at least 30% Khmer characters
        return (khmer_chars / total_chars) > 0.3 if total_chars > 0 else False
    
    def is_khmer_char(self, char: str) -> bool:
        """Check if character is Khmer"""
        code_point = ord(char)
        return (
            code_point in self.KHMER_CONSONANTS or
            code_point in self.KHMER_DEPENDENT_VOWELS or
            code_point in self.KHMER_SIGNS or
            code_point in self.KHMER_DIGITS or
            code_point in self.KHMER_SYMBOLS
        )
    
    def tokenize(self, text: str, preserve_english: bool = True) -> List[str]:
        """
        Tokenize Khmer text into words
        
        Args:
            text: Input text
            preserve_english: Keep English words intact
            
        Returns:
            List of tokens
        """
        tokens = []
        current_token = []
        
        for char in text:
            if self.is_khmer_char(char):
                current_token.append(char)
            elif char.isspace() or char in self.SENTENCE_DELIMITERS:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                if char in self.SENTENCE_DELIMITERS:
                    tokens.append(char)
            elif preserve_english and (char.isalnum() or char in '._-'):
                current_token.append(char)
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                if not char.isspace():
                    tokens.append(char)
        
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens
    
    def segment_words(self, text: str) -> List[str]:
        """
        Segment Khmer text into words using dictionary-based approach
        
        Args:
            text: Input text
            
        Returns:
            List of segmented words
        """
        # Remove spaces and punctuation for segmentation
        clean_text = re.sub(r'[^\u1780-\u17FF\u19E0-\u19FF\u1A00-\u1A9F]', '', text)
        
        if not clean_text:
            return []
        
        # Simple longest matching algorithm
        words = []
        i = 0
        while i < len(clean_text):
            # Try to find the longest word starting at position i
            longest_word = ''
            longest_length = 0
            
            for j in range(min(i + 20, len(clean_text)), i, -1):  # Max word length 20
                candidate = clean_text[i:j]
                if not self.vocab or candidate in self.vocab:
                    if len(candidate) > longest_length:
                        longest_word = candidate
                        longest_length = len(candidate)
            
            if longest_word:
                words.append(longest_word)
                i += longest_length
            else:
                # If no word found, take single character
                words.append(clean_text[i])
                i += 1
        
        return words
    
    def count_syllables(self, text: str) -> int:
        """
        Count syllables in Khmer text
        
        Args:
            text: Input text
            
        Returns:
            Number of syllables
        """
        # Simplified syllable counting based on consonants
        syllables = 0
        for char in text:
            if ord(char) in self.KHMER_CONSONANTS:
                syllables += 1
        return max(1, syllables)
    
    def normalize(self, text: str) -> str:
        """
        Normalize Khmer text
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove zero-width spaces and joiners
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Fix common typing errors
        text = self._fix_common_errors(text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _fix_common_errors(self, text: str) -> str:
        """Fix common Khmer typing errors"""
        replacements = {
            'ើ': 'ើ',  # Fix decomposed vowels
            'ោ': 'ោ',
            'ៅ': 'ៅ',
            'ុំ': 'ុំ',
            'ាំ': 'ាំ'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from Khmer text
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        sentences = []
        current_sentence = []
        
        for char in text:
            current_sentence.append(char)
            if char in self.SENTENCE_DELIMITERS:
                sentence = ''.join(current_sentence).strip()
                if sentence and len(sentence) > 1:  # Skip single punctuation
                    sentences.append(sentence)
                current_sentence = []
        
        # Add remaining text as last sentence if exists
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    def detect_code_switching(self, text: str) -> Dict[str, float]:
        """
        Detect code-switching between Khmer and English
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with language proportions and switch points
        """
        khmer_chars = 0
        english_chars = 0
        chinese_chars = 0
        switch_points = []
        
        prev_lang = None
        position = 0
        
        for char in text:
            current_lang = None
            
            if self.is_khmer_char(char):
                khmer_chars += 1
                current_lang = 'khmer'
            elif char.isascii() and char.isalpha():
                english_chars += 1
                current_lang = 'english'
            elif 0x4E00 <= ord(char) <= 0x9FFF:
                chinese_chars += 1
                current_lang = 'chinese'
            
            if current_lang and prev_lang and current_lang != prev_lang:
                switch_points.append(position)
            
            if current_lang:
                prev_lang = current_lang
            
            position += 1
        
        total_chars = khmer_chars + english_chars + chinese_chars
        
        return {
            'khmer_ratio': khmer_chars / total_chars if total_chars > 0 else 0,
            'english_ratio': english_chars / total_chars if total_chars > 0 else 0,
            'chinese_ratio': chinese_chars / total_chars if total_chars > 0 else 0,
            'switch_points': switch_points,
            'switch_frequency': len(switch_points) / len(text) if text else 0
        }
    
    def calculate_fluency_score(self, text: str) -> float:
        """
        Calculate fluency score for Khmer text
        
        Args:
            text: Input text
            
        Returns:
            Fluency score (0-1)
        """
        scores = []
        
        # Check character validity
        valid_chars = sum(1 for char in text if self.is_khmer_char(char) or char.isspace() or char in self.SENTENCE_DELIMITERS)
        char_validity = valid_chars / len(text) if text else 0
        scores.append(char_validity)
        
        # Check word segmentation quality
        words = self.segment_words(text)
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            # Khmer words typically 2-8 characters
            word_length_score = 1.0 if 2 <= avg_word_length <= 8 else 0.5
            scores.append(word_length_score)
        
        # Check sentence structure
        sentences = self.extract_sentences(text)
        if sentences:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            # Good Khmer sentences typically 20-150 characters
            sentence_score = 1.0 if 20 <= avg_sentence_length <= 150 else 0.7
            scores.append(sentence_score)
        
        # Check proper use of delimiters
        delimiter_count = sum(1 for char in text if char in self.SENTENCE_DELIMITERS)
        expected_delimiters = len(sentences) if sentences else 1
        delimiter_score = min(1.0, delimiter_count / expected_delimiters) if expected_delimiters > 0 else 0.5
        scores.append(delimiter_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def extract_technical_terms(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract technical terms (English) from Khmer text
        
        Args:
            text: Input text
            
        Returns:
            List of (english_term, khmer_translation) tuples
        """
        terms = []
        
        # Find English terms in text
        english_matches = self.TECH_TERM_PATTERN.findall(text)
        
        for match in english_matches:
            # Check if it's a known technical term
            lower_match = match.lower()
            for eng_term, khmer_term in self.technical_terms.items():
                if eng_term.lower() in lower_match:
                    # Check if Khmer translation is nearby in text
                    if khmer_term in text:
                        terms.append((eng_term, khmer_term))
                    else:
                        terms.append((eng_term, None))
        
        return terms
    
    def calculate_terminology_accuracy(self, text: str, domain: str) -> float:
        """
        Calculate accuracy of domain-specific terminology usage
        
        Args:
            text: Input text
            domain: Domain (finance, technology, healthcare, etc.)
            
        Returns:
            Accuracy score (0-1)
        """
        domain_terms = self._get_domain_terms(domain)
        if not domain_terms:
            return 1.0  # No domain terms to check
        
        found_terms = 0
        total_terms = 0
        
        for eng_term, khmer_term in domain_terms.items():
            # Check if concept is mentioned
            if eng_term.lower() in text.lower() or (khmer_term and khmer_term in text):
                total_terms += 1
                # Check if correct terminology is used
                if khmer_term and khmer_term in text:
                    found_terms += 1
                elif eng_term.lower() in text.lower():
                    found_terms += 0.5  # Half credit for English term
        
        return found_terms / total_terms if total_terms > 0 else 1.0
    
    def _get_domain_terms(self, domain: str) -> Dict[str, str]:
        """Get domain-specific terms"""
        domain_map = {
            'finance': {
                'interest rate': 'អត្រាការប្រាក់',
                'loan': 'ឥណទាន',
                'balance sheet': 'តារាងតុល្យការ',
                'profit': 'ប្រាក់ចំណេញ',
                'investment': 'ការវិនិយោគ'
            },
            'technology': {
                'database': 'មូលដ្ឋានទិន្នន័យ',
                'API': 'អេភីអាយ',
                'algorithm': 'ក្បួនដោះស្រាយ',
                'software': 'កម្មវិធី',
                'network': 'បណ្តាញ'
            },
            'healthcare': {
                'diagnosis': 'ការធ្វើរោគវិនិច្ឆ័យ',
                'treatment': 'ការព្យាបាល',
                'medicine': 'ថ្នាំ',
                'patient': 'អ្នកជំងឺ',
                'doctor': 'វេជ្ជបណ្ឌិត'
            },
            'agriculture': {
                'fertilizer': 'ជី',
                'harvest': 'ការប្រមូលផល',
                'irrigation': 'ប្រព័ន្ធស្រោចស្រព',
                'seed': 'គ្រាប់ពូជ',
                'crop': 'ដំណាំ'
            }
        }
        
        return domain_map.get(domain, {})
    
    def post_process(self, text: str) -> str:
        """
        Post-process generated Khmer text
        
        Args:
            text: Generated text
            
        Returns:
            Post-processed text
        """
        # Normalize text
        text = self.normalize(text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([។៕?!៖])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([។៕?!])\s*', r'\1 ', text)  # Add space after punctuation
        
        # Fix spacing around parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        # Remove duplicate punctuation
        text = re.sub(r'[។៕]{2,}', '។', text)
        
        # Ensure proper sentence ending
        if text and text[-1] not in self.SENTENCE_DELIMITERS:
            text += '។'
        
        return text.strip()
    
    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for Khmer text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of readability metrics
        """
        words = self.segment_words(text)
        sentences = self.extract_sentences(text)
        
        if not words or not sentences:
            return {
                'score': 0.0,
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0,
                'syllable_complexity': 0.0
            }
        
        # Calculate metrics
        avg_word_length = sum(len(w) for w in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Calculate syllable complexity
        total_syllables = sum(self.count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Khmer readability formula (simplified)
        # Based on average word length and sentence length
        readability_score = 100 - (avg_word_length * 5 + avg_sentence_length * 2)
        readability_score = max(0, min(100, readability_score))
        
        return {
            'score': readability_score / 100,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'syllable_complexity': avg_syllables_per_word,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }


class KhmerTranslator:
    """Helper class for Khmer-English translation in evaluation"""
    
    def __init__(self):
        self.translation_cache = {}
        
    async def translate_to_english(self, text: str) -> str:
        """
        Translate Khmer text to English for grading
        
        Args:
            text: Khmer text
            
        Returns:
            English translation
        """
        # Check cache
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        # In production, this would call a translation API
        # For now, return placeholder
        translation = f"[Translation of: {text[:50]}...]"
        
        self.translation_cache[text] = translation
        return translation
    
    async def translate_to_khmer(self, text: str) -> str:
        """
        Translate English text to Khmer
        
        Args:
            text: English text
            
        Returns:
            Khmer translation
        """
        # Check cache
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        # In production, this would call a translation API
        translation = f"[ការបកប្រែនៃ: {text[:50]}...]"
        
        self.translation_cache[text] = translation
        return translation


class BilingualMetrics:
    """Calculate metrics for bilingual (Khmer-English) text"""
    
    def __init__(self):
        self.khmer_processor = KhmerTextProcessor()
        
    def calculate_language_balance(self, text: str) -> Dict[str, float]:
        """
        Calculate balance between Khmer and English in text
        
        Args:
            text: Bilingual text
            
        Returns:
            Metrics dictionary
        """
        code_switch_info = self.khmer_processor.detect_code_switching(text)
        
        balance_score = 1.0 - abs(code_switch_info['khmer_ratio'] - code_switch_info['english_ratio'])
        
        return {
            'khmer_ratio': code_switch_info['khmer_ratio'],
            'english_ratio': code_switch_info['english_ratio'],
            'balance_score': balance_score,
            'switch_frequency': code_switch_info['switch_frequency'],
            'switch_smoothness': self._calculate_switch_smoothness(text, code_switch_info['switch_points'])
        }
    
    def _calculate_switch_smoothness(self, text: str, switch_points: List[int]) -> float:
        """
        Calculate how smooth the language switching is
        
        Args:
            text: Input text
            switch_points: Positions where language switches occur
            
        Returns:
            Smoothness score (0-1)
        """
        if not switch_points:
            return 1.0  # No switching is considered smooth
        
        # Check if switches occur at natural boundaries (spaces, punctuation)
        smooth_switches = 0
        for point in switch_points:
            if point > 0 and point < len(text):
                if text[point-1].isspace() or text[point] .isspace():
                    smooth_switches += 1
                elif text[point-1] in '.,;:()[]{}/-' or text[point] in '.,;:()[]{}/-':
                    smooth_switches += 1
        
        return smooth_switches / len(switch_points) if switch_points else 1.0


# Utility functions for evaluation
def load_khmer_evaluation_metrics(results_path: str) -> Dict[str, Any]:
    """Load and aggregate Khmer-specific evaluation metrics"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    processor = KhmerTextProcessor()
    metrics = {
        'fluency_scores': [],
        'terminology_scores': [],
        'readability_scores': [],
        'code_switching_scores': []
    }
    
    for task_result in results.get('task_results', []):
        if task_result['success'] and task_result.get('response'):
            text = task_result['response']
            
            # Calculate metrics
            fluency = processor.calculate_fluency_score(text)
            metrics['fluency_scores'].append(fluency)
            
            readability = processor.calculate_readability_score(text)
            metrics['readability_scores'].append(readability['score'])
            
            code_switch = processor.detect_code_switching(text)
            metrics['code_switching_scores'].append(code_switch['switch_frequency'])
    
    # Calculate aggregates
    return {
        'avg_fluency': np.mean(metrics['fluency_scores']) if metrics['fluency_scores'] else 0,
        'avg_readability': np.mean(metrics['readability_scores']) if metrics['readability_scores'] else 0,
        'avg_code_switching': np.mean(metrics['code_switching_scores']) if metrics['code_switching_scores'] else 0,
        'std_fluency': np.std(metrics['fluency_scores']) if metrics['fluency_scores'] else 0,
        'std_readability': np.std(metrics['readability_scores']) if metrics['readability_scores'] else 0
    }


# Example usage
if __name__ == "__main__":
    # Test Khmer text processor
    processor = KhmerTextProcessor()
    
    # Sample Khmer text
    sample_text = """
    របាយការណ៍ហិរញ្ញវត្ថុត្រីមាសទី៣ បង្ហាញពី profit margin ដែលកើនឡើង ១៥% 
    ធៀបនឹងត្រីមាសមុន។ NPL ratio បានថយចុះមកត្រឹម ៣.២% ហើយ liquidity 
    coverage ratio នៅតែរក្សាកម្រិតល្អនៅ ១៥០%។
    """
    
    print("=" * 60)
    print("Khmer Text Processing Demo")
    print("=" * 60)
    
    # Validate text
    print(f"Valid Khmer: {processor.validate_text(sample_text)}")
    
    # Normalize text
    normalized = processor.normalize(sample_text)
    print(f"\nNormalized text:\n{normalized}")
    
    # Tokenize
    tokens = processor.tokenize(normalized)
    print(f"\nTokens ({len(tokens)}):\n{tokens[:20]}...")
    
    # Extract sentences
    sentences = processor.extract_sentences(normalized)
    print(f"\nSentences ({len(sentences)}):")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    # Detect code-switching
    code_switch = processor.detect_code_switching(normalized)
    print(f"\nCode-switching analysis:")
    print(f"  Khmer: {code_switch['khmer_ratio']:.1%}")
    print(f"  English: {code_switch['english_ratio']:.1%}")
    print(f"  Switch frequency: {code_switch['switch_frequency']:.3f}")
    
    # Calculate fluency
    fluency = processor.calculate_fluency_score(normalized)
    print(f"\nFluency score: {fluency:.2f}")
    
    # Extract technical terms
    terms = processor.extract_technical_terms(normalized)
    print(f"\nTechnical terms found:")
    for eng, khm in terms:
        print(f"  {eng} -> {khm if khm else '[not translated]'}")
    
    # Calculate readability
    readability = processor.calculate_readability_score(normalized)
    print(f"\nReadability metrics:")
    print(f"  Score: {readability['score']:.2f}")
    print(f"  Avg word length: {readability['avg_word_length']:.1f}")
    print(f"  Avg sentence length: {readability['avg_sentence_length']:.1f} words")
    print(f"  Syllable complexity: {readability['syllable_complexity']:.1f}")
    
    print("\n" + "=" * 60)