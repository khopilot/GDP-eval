"""
Khmer Tokenizer for Metrics
Specialized tokenization for Khmer language evaluation
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class KhmerTokenizer:
    """Tokenizer specifically designed for Khmer text metrics"""

    # Unicode ranges for Khmer script
    KHMER_CONSONANTS = set(range(0x1780, 0x17A3))  # ក to អ
    KHMER_INDEPENDENT_VOWELS = set(range(0x17A3, 0x17B4))  # ឣ to ឳ
    KHMER_DEPENDENT_VOWELS = set(range(0x17B6, 0x17C6))  # ា to ៅ
    KHMER_SIGNS = set(range(0x17C6, 0x17D4))  # ំ to ៓
    KHMER_DIGITS = set(range(0x17E0, 0x17EA))  # ០ to ៩
    KHMER_SYMBOLS = set(range(0x17F0, 0x17FA))  # ៰ to ៹

    # Special characters
    ZWSP = '\u200B'  # Zero-width space
    ZWNJ = '\u200C'  # Zero-width non-joiner

    # Subscript consonant marker
    COENG = '\u17D2'  # ្

    # Common Khmer punctuation
    KHAN = '\u17D4'  # ។ (Khmer period)
    BARIYOOSAN = '\u17D5'  # ៕ (Khmer "end of text")

    # Syllable patterns (simplified)
    SYLLABLE_PATTERN = re.compile(
        r'[\u1780-\u17A2]'  # Initial consonant
        r'(?:\u17D2[\u1780-\u17A2])*'  # Optional subscript consonants
        r'[\u17B6-\u17C5]*'  # Optional vowels
        r'[\u17C6-\u17D3]*'  # Optional signs
        r'|[\u17A3-\u17B5]'  # Or independent vowel
        r'|[\u17E0-\u17E9]+'  # Or Khmer digits
        r'|[^\u1780-\u17F9\s]+'  # Or non-Khmer (e.g., English)
    )

    def __init__(self):
        """Initialize Khmer tokenizer"""
        self.syllable_cache = {}
        self.word_freq = Counter()

    def tokenize_characters(self, text: str) -> List[str]:
        """
        Tokenize text into individual characters (grapheme clusters)

        Args:
            text: Input text

        Returns:
            List of characters
        """
        # Remove ZWSP for character tokenization
        text = text.replace(self.ZWSP, '')

        characters = []
        i = 0
        while i < len(text):
            char = text[i]
            char_code = ord(char)

            # Check if it's a consonant that might have subscripts/vowels
            if char_code in self.KHMER_CONSONANTS:
                cluster = char
                j = i + 1

                # Collect subscript consonants
                while j < len(text) and text[j] == self.COENG:
                    if j + 1 < len(text) and ord(text[j + 1]) in self.KHMER_CONSONANTS:
                        cluster += text[j:j + 2]
                        j += 2
                    else:
                        break

                # Collect vowels and signs
                while j < len(text):
                    next_char = ord(text[j])
                    if (next_char in self.KHMER_DEPENDENT_VOWELS or
                            next_char in self.KHMER_SIGNS):
                        cluster += text[j]
                        j += 1
                    else:
                        break

                characters.append(cluster)
                i = j
            else:
                characters.append(char)
                i += 1

        return characters

    def tokenize_syllables(self, text: str) -> List[str]:
        """
        Tokenize text into Khmer syllables

        Args:
            text: Input text

        Returns:
            List of syllables
        """
        # Cache lookup
        if text in self.syllable_cache:
            return self.syllable_cache[text]

        # Remove ZWSP for syllable detection
        clean_text = text.replace(self.ZWSP, '')

        syllables = []
        matches = self.SYLLABLE_PATTERN.finditer(clean_text)

        for match in matches:
            syllable = match.group()
            if syllable and not syllable.isspace():
                syllables.append(syllable)

        # Cache result
        if len(text) < 1000:  # Only cache short texts
            self.syllable_cache[text] = syllables

        return syllables

    def tokenize_words(self, text: str, use_zwsp: bool = True) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text
            use_zwsp: Whether to use ZWSP as word boundary

        Returns:
            List of words
        """
        words = []

        if use_zwsp and self.ZWSP in text:
            # Use ZWSP as word delimiter
            segments = text.split(self.ZWSP)
            for segment in segments:
                # Further split by spaces and punctuation
                sub_words = re.split(r'[\s\u17D4\u17D5]+', segment)
                words.extend([w for w in sub_words if w])
        else:
            # Use heuristic word segmentation
            words = self._segment_words_heuristic(text)

        # Update word frequency
        self.word_freq.update(words)

        return words

    def _segment_words_heuristic(self, text: str) -> List[str]:
        """
        Segment words using heuristics (when ZWSP not available)

        Args:
            text: Input text

        Returns:
            List of words
        """
        # This is a simplified heuristic approach
        # In production, you might use a dictionary-based or ML approach

        words = []
        current_word = []
        syllables = self.tokenize_syllables(text)

        for syllable in syllables:
            # Check if syllable is non-Khmer (e.g., English word)
            if not any(ord(c) in range(0x1780, 0x17FA) for c in syllable):
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
                words.append(syllable)
            else:
                current_word.append(syllable)

                # Simple heuristic: max 4 syllables per word
                if len(current_word) >= 4:
                    words.append(''.join(current_word))
                    current_word = []

        if current_word:
            words.append(''.join(current_word))

        return words

    def tokenize_mixed(self, text: str) -> Dict[str, List[str]]:
        """
        Tokenize mixed Khmer-English text

        Args:
            text: Input text

        Returns:
            Dictionary with 'khmer' and 'english' token lists
        """
        khmer_tokens = []
        english_tokens = []

        # Split into potential tokens
        tokens = re.findall(r'[\u1780-\u17F9]+|[A-Za-z]+|\d+|[^\u1780-\u17F9A-Za-z\d\s]+|\s+', text)

        for token in tokens:
            if re.match(r'[\u1780-\u17F9]+', token):
                # Khmer token
                khmer_tokens.append(token)
            elif re.match(r'[A-Za-z]+', token):
                # English token
                english_tokens.append(token.lower())
            # Skip whitespace and punctuation for this analysis

        return {
            'khmer': khmer_tokens,
            'english': english_tokens,
            'all': tokens
        }

    def count_khmer_characters(self, text: str) -> Dict[str, int]:
        """
        Count different types of Khmer characters

        Args:
            text: Input text

        Returns:
            Dictionary with character type counts
        """
        counts = {
            'consonants': 0,
            'independent_vowels': 0,
            'dependent_vowels': 0,
            'signs': 0,
            'digits': 0,
            'symbols': 0,
            'subscripts': 0,
            'zwsp': 0,
            'total_khmer': 0
        }

        for char in text:
            char_code = ord(char)

            if char_code in self.KHMER_CONSONANTS:
                counts['consonants'] += 1
            elif char_code in self.KHMER_INDEPENDENT_VOWELS:
                counts['independent_vowels'] += 1
            elif char_code in self.KHMER_DEPENDENT_VOWELS:
                counts['dependent_vowels'] += 1
            elif char_code in self.KHMER_SIGNS:
                counts['signs'] += 1
            elif char_code in self.KHMER_DIGITS:
                counts['digits'] += 1
            elif char_code in self.KHMER_SYMBOLS:
                counts['symbols'] += 1
            elif char == self.COENG:
                counts['subscripts'] += 1
            elif char == self.ZWSP:
                counts['zwsp'] += 1

            if 0x1780 <= char_code <= 0x17F9:
                counts['total_khmer'] += 1

        return counts

    def normalize_khmer(self, text: str) -> str:
        """
        Normalize Khmer text for comparison

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # NFD normalization
        text = unicodedata.normalize('NFD', text)

        # Remove ZWSP and ZWNJ
        text = text.replace(self.ZWSP, '').replace(self.ZWNJ, '')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove duplicate signs
        text = re.sub(r'(\u17C6)+', '\u17C6', text)  # Remove duplicate និគ្គហិត

        return text.strip()

    def detect_code_switching_points(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Detect points where code-switching occurs

        Args:
            text: Input text

        Returns:
            List of (position, from_script, to_script) tuples
        """
        switches = []
        prev_script = None

        tokens = re.findall(r'[\u1780-\u17F9]+|[A-Za-z]+|[0-9]+|.', text)

        position = 0
        for token in tokens:
            if re.match(r'[\u1780-\u17F9]+', token):
                current_script = 'khmer'
            elif re.match(r'[A-Za-z]+', token):
                current_script = 'english'
            elif re.match(r'[0-9]+', token):
                current_script = 'number'
            else:
                current_script = 'other'

            if prev_script and prev_script != current_script and current_script != 'other':
                switches.append((position, prev_script, current_script))

            if current_script != 'other':
                prev_script = current_script

            position += len(token)

        return switches

    def extract_numbers(self, text: str) -> List[Dict[str, any]]:
        """
        Extract Khmer and Arabic numbers from text

        Args:
            text: Input text

        Returns:
            List of number dictionaries
        """
        numbers = []

        # Khmer number mapping
        khmer_digit_map = {
            '០': '0', '១': '1', '២': '2', '៣': '3', '៤': '4',
            '៥': '5', '៦': '6', '៧': '7', '៨': '8', '៩': '9'
        }

        # Find Khmer numbers
        khmer_numbers = re.findall(r'[\u17E0-\u17E9]+', text)
        for kh_num in khmer_numbers:
            arabic = ''.join(khmer_digit_map.get(d, d) for d in kh_num)
            numbers.append({
                'original': kh_num,
                'value': int(arabic) if arabic.isdigit() else 0,
                'type': 'khmer'
            })

        # Find Arabic numbers
        arabic_numbers = re.findall(r'\d+', text)
        for ar_num in arabic_numbers:
            numbers.append({
                'original': ar_num,
                'value': int(ar_num),
                'type': 'arabic'
            })

        return numbers

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Khmer sentence delimiters
        delimiters = [
            self.KHAN,  # ។
            self.BARIYOOSAN,  # ៕
            '.',  # English period
            '!',  # Exclamation
            '?',  # Question mark
        ]

        # Create regex pattern
        delimiter_pattern = '|'.join(re.escape(d) for d in delimiters)
        sentences = re.split(f'({delimiter_pattern})', text)

        # Reconstruct sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)

        # Add last sentence if no delimiter at end
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return result

    def get_vocabulary_stats(self, texts: List[str]) -> Dict[str, any]:
        """
        Get vocabulary statistics from multiple texts

        Args:
            texts: List of texts

        Returns:
            Vocabulary statistics
        """
        all_words = []
        all_syllables = []
        all_characters = []

        for text in texts:
            all_words.extend(self.tokenize_words(text))
            all_syllables.extend(self.tokenize_syllables(text))
            all_characters.extend(self.tokenize_characters(text))

        return {
            'unique_words': len(set(all_words)),
            'total_words': len(all_words),
            'unique_syllables': len(set(all_syllables)),
            'total_syllables': len(all_syllables),
            'unique_characters': len(set(all_characters)),
            'total_characters': len(all_characters),
            'avg_word_length': sum(len(w) for w in all_words) / len(all_words) if all_words else 0,
            'avg_syllables_per_word': len(all_syllables) / len(all_words) if all_words else 0,
            'most_common_words': Counter(all_words).most_common(10),
            'most_common_syllables': Counter(all_syllables).most_common(10)
        }