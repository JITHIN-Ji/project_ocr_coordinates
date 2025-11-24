import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger("ocr_matcher")


class OCRMatchHandler:
    """Class to handle OCR matching and coordinate extraction with context-aware matching."""

    def __init__(self, fuzzy_threshold=0.80):
        """
        Initialize the OCRMatchHandler class.

        Args:
            fuzzy_threshold (float): The minimum similarity score for a match to be considered valid.
        """
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def normalize_text_for_matching(text):
        """Converts to lowercase and removes non-alphanumeric chars, normalizes whitespace."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)  
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def calculate_fuzzy_score(s1, s2):
        """Calculates similarity score between two normalized strings."""
        return SequenceMatcher(None, s1, s2).ratio()

    def find_best_match_for_value(self, target_value_str, structured_data, context_words=None):
        """
        Finds the best match for a target string within the structured word data.
        Uses context words to prefer matches that appear near other parts of the same name.

        Args:
            target_value_str (str): The target string to match.
            structured_data (list): The structured word data.
            context_words (list): List of other name parts that should be nearby (e.g., ["Gibbs"] when searching for "John")

        Returns:
            tuple: (best_match_info, modified_structured_data) - Information about the best match and the modified structured data
        """
        if not target_value_str or not structured_data:
            return None, structured_data

        
        best_match_info, modified_data = self._search_with_case_sensitivity(
            target_value_str, structured_data, case_sensitive=True, context_words=context_words
        )
        
        
        if best_match_info is None:
            best_match_info, modified_data = self._search_with_case_sensitivity(
                target_value_str, structured_data, case_sensitive=False, context_words=context_words
            )
        
        return best_match_info, modified_data

    def _calculate_context_bonus(self, page_words, match_position, match_length, context_words, case_sensitive):
        """
        Calculate a bonus score if context words appear nearby.
        
        Args:
            page_words (list): List of word objects on the page
            match_position (int): Starting position of the current match
            match_length (int): Length of the matched phrase in words
            context_words (list): Words to look for nearby
            case_sensitive (bool): Whether matching is case-sensitive
            
        Returns:
            float: Bonus score (0.0 to 1.0)
        """
        if not context_words:
            return 0.0
        
        
        window_size = 5
        start_idx = max(0, match_position - window_size)
        end_idx = min(len(page_words), match_position + match_length + window_size)
        
        nearby_words = []
        for i in range(start_idx, end_idx):
            
            if i < match_position or i >= match_position + match_length:
                word_text = page_words[i]["text"]
                if case_sensitive:
                    normalized = re.sub(r'[^\w\s]', '', word_text)
                    normalized = re.sub(r'\s+', ' ', normalized).strip()
                else:
                    normalized = self.normalize_text_for_matching(word_text)
                nearby_words.append(normalized)
        
        
        matches_found = 0
        for context_word in context_words:
            if case_sensitive:
                normalized_context = re.sub(r'[^\w\s]', '', context_word)
                normalized_context = re.sub(r'\s+', ' ', normalized_context).strip()
            else:
                normalized_context = self.normalize_text_for_matching(context_word)
            
            for nearby_word in nearby_words:
                if self.calculate_fuzzy_score(normalized_context, nearby_word) >= 0.85:
                    matches_found += 1
                    break
        
        
        if context_words:
            return matches_found / len(context_words)
        return 0.0

    def _search_with_case_sensitivity(self, target_value_str, structured_data, case_sensitive=True, context_words=None):
        """
        Internal method to search for matches with specified case sensitivity.
        
        Args:
            target_value_str (str): The target string to match.
            structured_data (list): The structured word data.
            case_sensitive (bool): Whether to perform case-sensitive matching.
            context_words (list): Other name parts that should be nearby.
            
        Returns:
            tuple: (best_match_info, modified_structured_data)
        """
        if case_sensitive:
            
            normalized_target = re.sub(r'[^\w\s]', '', target_value_str)
            normalized_target = re.sub(r'\s+', ' ', normalized_target).strip()
        else:
            
            normalized_target = self.normalize_text_for_matching(target_value_str)
        
        target_words_list = normalized_target.split()
        if not target_words_list:
            return None, structured_data

        best_match_info = None
        highest_score = -1.0
        best_match_location = None

        
        for page_idx, page_content in enumerate(structured_data):
            page_words = page_content.get("words", [])
            if not page_words:
                continue

            for i in range(len(page_words) - len(target_words_list) + 1):
                
                current_pdf_phrase_objects = page_words[i: i + len(target_words_list)]
                current_pdf_phrase_texts = [word_obj["text"] for word_obj in current_pdf_phrase_objects]

                # Normalize the constructed PDF phrase based on case sensitivity
                pdf_phrase_text = " ".join(current_pdf_phrase_texts)
                if case_sensitive:
                    # For case-sensitive, only remove punctuation and normalize whitespace
                    normalized_pdf_phrase = re.sub(r'[^\w\s]', '', pdf_phrase_text)
                    normalized_pdf_phrase = re.sub(r'\s+', ' ', normalized_pdf_phrase).strip()
                else:
                    # For case-insensitive, use the existing normalization method
                    normalized_pdf_phrase = self.normalize_text_for_matching(pdf_phrase_text)

                # Calculate base fuzzy match score
                base_score = self.calculate_fuzzy_score(normalized_target, normalized_pdf_phrase)

                # Only continue if base score meets threshold
                if base_score < self.fuzzy_threshold:
                    continue

                
                context_bonus = 0.0
                if context_words:
                    context_bonus = self._calculate_context_bonus(
                        page_words, i, len(target_words_list), context_words, case_sensitive
                    )
                
                
                final_score = base_score * 0.7 + context_bonus * 0.3

                if final_score > highest_score:
                    highest_score = final_score

                    
                    min_x0 = min(word["x0"] for word in current_pdf_phrase_objects)
                    min_top = min(word["top"] for word in current_pdf_phrase_objects)
                    max_x1 = max(word["x1"] for word in current_pdf_phrase_objects)
                    max_bottom = max(word["bottom"] for word in current_pdf_phrase_objects)

                    best_match_info = {
                        "text_from_pdf": " ".join(current_pdf_phrase_texts),
                        "x0": min_x0,
                        "top": min_top,
                        "x1": max_x1,
                        "bottom": max_bottom,
                        "page_number": page_content["page_number"],
                        "page_width": page_content["page_width"],
                        "page_height": page_content["page_height"],
                        "score": final_score,
                        "base_score": base_score,
                        "context_bonus": context_bonus
                    }
                    
                    best_match_location = (page_idx, i, len(target_words_list))
                    
                    
                    if final_score >= 0.95 and (not context_words or context_bonus > 0.5):
                        break

            
            if best_match_info and highest_score >= 0.95:
                if not context_words or best_match_info.get("context_bonus", 0) > 0.5:
                    break

        
        return best_match_info, structured_data
