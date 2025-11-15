import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger("ocr_matcher")


class OCRMatchHandler:
    """Class to handle OCR matching and coordinate extraction."""

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
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Keep only alphanumeric and spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def calculate_fuzzy_score(s1, s2):
        """Calculates similarity score between two normalized strings."""
        return SequenceMatcher(None, s1, s2).ratio()

    def find_best_match_for_value(self, target_value_str, structured_data):
        """
        Finds the best match for a target string within the structured word data and removes it from the data.
        First tries case-sensitive matching, then falls back to case-insensitive if no match is found.

        Args:
            target_value_str (str): The target string to match.
            structured_data (list): The structured word data.

        Returns:
            tuple: (best_match_info, modified_structured_data) - Information about the best match and the modified structured data
        """
        if not target_value_str or not structured_data:
            return None, structured_data

        # First try case-sensitive matching
        best_match_info, modified_data = self._search_with_case_sensitivity(
            target_value_str, structured_data, case_sensitive=True
        )
        
        # If no match found with case-sensitive, try case-insensitive
        if best_match_info is None:
            best_match_info, modified_data = self._search_with_case_sensitivity(
                target_value_str, structured_data, case_sensitive=False
            )
        
        return best_match_info, modified_data

    def _search_with_case_sensitivity(self, target_value_str, structured_data, case_sensitive=True):
        """
        Internal method to search for matches with specified case sensitivity.
        
        Args:
            target_value_str (str): The target string to match.
            structured_data (list): The structured word data.
            case_sensitive (bool): Whether to perform case-sensitive matching.
            
        Returns:
            tuple: (best_match_info, modified_structured_data)
        """
        if case_sensitive:
            # For case-sensitive, only remove punctuation and normalize whitespace
            normalized_target = re.sub(r'[^\w\s]', '', target_value_str)
            normalized_target = re.sub(r'\s+', ' ', normalized_target).strip()
        else:
            # For case-insensitive, use the existing normalization method
            normalized_target = self.normalize_text_for_matching(target_value_str)
        
        target_words_list = normalized_target.split()
        if not target_words_list:
            return None, structured_data

        best_match_info = None
        highest_score = -1.0
        best_match_location = None

        # Search through the original data directly
        for page_idx, page_content in enumerate(structured_data):
            page_words = page_content.get("words", [])
            if not page_words:
                continue

            for i in range(len(page_words) - len(target_words_list) + 1):
                # Construct phrase from PDF words
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

                score = self.calculate_fuzzy_score(normalized_target, normalized_pdf_phrase)

                if score > highest_score and score >= self.fuzzy_threshold:
                    highest_score = score

                    # Calculate combined bounding box
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
                        "score": score
                    }
                    # Save the location of this best match
                    best_match_location = (page_idx, i, len(target_words_list))
                    
                    if score == 1.0:  # Perfect match
                        break

            if best_match_info and highest_score == 1.0:
                break

        # If a match was found, remove it from the original structured data
        if best_match_location:
            page_idx, start_idx, length = best_match_location
            if page_idx < len(structured_data):
                del structured_data[page_idx]["words"][start_idx:start_idx + length]
                
        return best_match_info, structured_data