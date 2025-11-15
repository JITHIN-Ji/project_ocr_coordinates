import os
import json
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)


def extract_names_from_image(image: Image.Image):
    """
    Extract names from columns and split each name into individual words.
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: JSON response with extracted names split into individual words
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Prompt for extracting names from columns
        prompt = """
        You are a document analysis expert. This image contains names organized in columns.
        
        YOUR TASK:
        1. Identify ALL names in the document, reading from left to right, top to bottom
        2. Each person's name may be in format like "Bayly Francis Wilson" or "Amal Krishna Rajesh"
        3. For EACH complete name you find, split it into individual words
        
        IMPORTANT RULES:
        - A complete name consists of all words belonging to one person
        - Split each complete name into separate individual words
        - Preserve the exact spelling of each word
        - Maintain the order of words as they appear
        
        OUTPUT FORMAT - Return ONLY this JSON:
        {
            "names": [
                {
                    "full_name": "Bayly Francis Wilson",
                    "name_parts": ["Bayly", "Francis", "Wilson"]
                },
                {
                    "full_name": "Amal Krishna Rajesh",
                    "name_parts": ["Amal", "Krishna", "Rajesh"]
                },
                {
                    "full_name": "Goodman Timothy",
                    "name_parts": ["Goodman", "Timothy"]
                }
            ]
        }
        
        CRITICAL:
        - full_name: The complete name as it appears (all words together)
        - name_parts: Array of individual words from that name
        - Include EVERY name you can see
        - Each word in name_parts should be exactly as written in the document
        - Return ONLY the JSON, no markdown, no explanation
        """
        
        # Generate response
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()
        
        print(f"Raw Gemini response: {response_text[:300]}...")
        
        # Clean markdown formatting
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Validate and log
        if "names" in result:
            print(f"\n{'='*60}")
            print(f"GEMINI EXTRACTED {len(result['names'])} NAMES:")
            print(f"{'='*60}")
            for idx, name_obj in enumerate(result["names"], 1):
                full_name = name_obj.get("full_name", "")
                name_parts = name_obj.get("name_parts", [])
                print(f"{idx}. Full Name: {full_name}")
                print(f"   Parts: {name_parts}")
                print()
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Response text: {response_text}")
        return {"names": [], "error": f"JSON parsing failed: {str(e)}"}
    except Exception as e:
        print(f"Gemini extraction error: {str(e)}")
        return {"names": [], "error": str(e)}


def prepare_gemini_output_for_matching(gemini_result):
    """
    Convert Gemini result to format ready for coordinate matching.
    Returns list with full_name and individual name_parts.
    
    Args:
        gemini_result: Dict with 'names' list from Gemini
        
    Returns:
        list: Names with parts ready for coordinate matching
    """
    formatted_output = []
    
    if "names" in gemini_result:
        for idx, name_obj in enumerate(gemini_result["names"], 1):
            full_name = name_obj.get("full_name", "").strip()
            name_parts = name_obj.get("name_parts", [])
            
            # Clean name parts
            name_parts = [part.strip() for part in name_parts if part.strip()]
            
            if full_name and name_parts:
                formatted_item = {
                    "full_name": full_name,
                    "name_parts": name_parts,
                    "person_id": idx
                }
                formatted_output.append(formatted_item)
                print(f"Prepared Person {idx}: {full_name} -> {name_parts}")
    
    print(f"\nTotal names prepared for matching: {len(formatted_output)}")
    return formatted_output


# Test function
def test_extraction(image_path: str):
    """Test the extraction with an image file."""
    try:
        img = Image.open(image_path)
        result = extract_names_from_image(img)
        print("\n" + "="*60)
        print("EXTRACTION RESULT:")
        print("="*60)
        print(json.dumps(result, indent=2))
        print("\n" + "="*60)
        print("PREPARED FOR MATCHING:")
        print("="*60)
        formatted = prepare_gemini_output_for_matching(result)
        print(json.dumps(formatted, indent=2))
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    print("Gemini field extraction module loaded successfully")
    print("Extracts names and splits them into individual words")
    print("Use extract_names_from_image(image) to extract names")
    print("Use prepare_gemini_output_for_matching(result) to prepare for OCR matching")