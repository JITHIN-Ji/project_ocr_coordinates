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
1. Identify **all column headings** in the document. For each column heading, determine whether the column is a **name column**.
   A column is a NAME COLUMN ONLY IF its heading contains one of the following keywords:
       - "Christian Name"
       - "Surname"
       - "Names of Each Voter"
       - "Full Name"
       - "Name of Voter"
       - "Name" (ONLY when used for persons, NOT properties)

   EXPLICITLY **IGNORE** ANY COLUMN with headings such as:
       - "Name of the Property"
       - "Name of the Tenant"
       - "Street"
       - "Lane"
       - "Place"
       - "Property"
       - "Farm"
       - "House"
       - "Occupier"
       - "Where Property Is Situated"

   **If the heading is not clearly a person-name column, DO NOT extract anything from that column.**

2. **CRITICAL SURNAME RULE:**
   - If a column heading indicates surname-first ordering — specifically when the heading contains the phrase:
       "Name of Each Voter at Full Length the Surname Being First"
     then:
       * Return the **surname as given_name**
       * Return the **given name as surname**
   - In all other headings, use **normal order**:
       * First word = given name
       * Remaining words = surname

3. **SPACE-BASED NAME SPLITTING RULES:**
   
   **NORMAL ORDER (given name first):**
   - Find the **first space** in the name text
   - Everything **BEFORE the first space** = given_name
   - Everything **AFTER the first space** = surname
   
   Example: "WalshJohnHenry Arnold"
   - First space appears before "Arnold"
   - given_name: "WalshJohnHenry"
   - surname: "Arnold"
   
   **SURNAME-FIRST ORDER (when column heading indicates "surname being first"):**
   - Find the **first space** in the name text
   - Everything **BEFORE the first space** = surname (but return as given_name due to swap)
   - Everything **AFTER the first space** = given_name (but return as surname due to swap)
   
   Example: "WalshJohnHenry Arnold" (in surname-first column)
   - First space appears before "Arnold"
   - Actual: WalshJohnHenry (surname), Arnold (given name)
   - **SWAPPED OUTPUT:**
     - given_name: "Arnold"
     - surname: "WalshJohnHenry"

4. Identify names that are **split across multiple lines** (e.g., "jit" on one line and "hin" on the next).

5. For each complete name:
    - Determine the **given name** and **surname** based on the space-finding rule above.
    - Split each part into individual words where spaces exist.
    - **If a word is split across lines, keep the fragments as separate strings in the parts array.**
    - Preserve original spelling and order.

NAME EXTRACTION RULES:
- **Normal case (no surname-first indicator in heading):** 
  Everything before first space = given name; everything after first space = surname.
  Example:
    "WalshJohnHenry Arnold"
        given_name: "WalshJohnHenry"
        surname: "Arnold"
        
- **Surname-first case (column heading indicates "surname being first" or similar):**
  **SWAP THE VALUES:** Return surname as given_name, and given name as surname.
  
  Example 1:
    Actual name in document: "Portal Robert" (where Portal is surname, Robert is given name)
    Return as:
        given_name: "Robert"  
        surname: "Portal"  
  
  Example 2:
    Actual name in document: "WalshJohnHenry Arnold" (where WalshJohnHenry is surname, Arnold is given name)
    Return as:
        given_name: "Arnold"  
        surname: "WalshJohnHenry"  
        
- **If a word is broken across lines, keep fragments separate:**
    "jit" on line 1 + "hin" on line 2 → ["jit", "hin"]

OUTPUT FORMAT — Return ONLY this JSON (NO markdown, NO explanation):
{
    "names": [
        {
            "full_name": "Amal Krishna Rajesh",
            "given_name": "Amal",
            "given_name_parts": ["Amal"],
            "surname": "Krishna Rajesh",
            "surname_parts": ["Krishna", "Rajesh"]
        },
        {
            "full_name": "Jithin Rajesh",
            "given_name": "Jithin",
            "given_name_parts": ["jithin"],
            "surname": "Rajesh",
            "surname_parts": ["Rajesh"]
        },
        {
            "full_name": "WalshJohnHenry Arnold",
            "given_name": "WalshJohnHenry",
            "given_name_parts": ["WalshJohnHenry"],
            "surname": "Arnold",
            "surname_parts": ["Arnold"]
        }
    ]
}

CRITICAL REQUIREMENTS:
- Extract ONLY from name-related columns (e.g., "Christian Names", "Names of Each Voter", "Full Name", "Name") — ignore all other columns.
- **Use the first space to determine where given_name ends and surname begins.**
- **If a word/name is split across lines, keep the fragments as separate strings in the parts array** (e.g., ["jit", "hin"]).
- **CRITICAL:** If heading indicates "surname being first", SWAP the values - return surname as given_name and given name as surname.
- Do NOT invent or assume names.
- Preserve exact spelling from the document.
- Maintain left-to-right, top-to-bottom extraction order"""
        
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
    Convert Gemini result (new format) to a structure ready for coordinate matching.

    Expected Gemini format:
    {
        "names": [
            {
                "full_name": "Amal Krishna Rajesh",
                "given_name": "Amal",
                "given_name_parts": ["Amal"],
                "surname": "Krishna Rajesh",
                "surname_parts": ["Krishna", "Rajesh"]
            }
        ]
    }

    Returns:
        list: Cleaned and standardized output list for matching.
    """

    formatted_output = []

    if "names" in gemini_result:
        for idx, name_obj in enumerate(gemini_result["names"], 1):

            full_name = name_obj.get("full_name", "").strip()

            given_name = name_obj.get("given_name", "").strip()
            given_name_parts = [p.strip() for p in name_obj.get("given_name_parts", []) if p.strip()]

            surname = name_obj.get("surname", "").strip()
            surname_parts = [p.strip() for p in name_obj.get("surname_parts", []) if p.strip()]

            # Validate that the record contains at least a full name
            if not full_name:
                print(f"Skipping Person {idx}: Missing full_name")
                continue

            formatted_item = {
                "person_id": idx,
                "full_name": full_name,
                "given_name": given_name,
                "given_name_parts": given_name_parts,
                "surname": surname,
                "surname_parts": surname_parts,
            }

            formatted_output.append(formatted_item)

            print(
                f"Prepared Person {idx}: Full='{full_name}', "
                f"Given='{given_name_parts}', Surname='{surname_parts}'"
            )

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
