import os
from typing import ClassVar, Literal, Optional, List, Type, Dict, Any
from pydantic import BaseModel, Field
import base64
import requests
from dotenv import load_dotenv
import json
import logging
from pathlib import Path
from google import genai
import os


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables and setup constants
load_dotenv()

API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not found")

client = genai.Client(api_key=API_KEY)

MODEL_ID = "gemini-2.0-flash"
INPUT_DATA_PATH = "data/biography_list/biography_list.json"
OUTPUT_DATA_PATH = Path("data/structured_biographies")

# Ensure output directory exists
OUTPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Model definitions
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Occupation(BaseModel):
    occupation: str = Field(..., description="Occupation in Swedish")
    occupation_english: Optional[str] = Field(None, description="Occupation translated into English")


class Parent(BaseModel):
    first_name: str = Field(..., description="Parent's first name or initials")
    last_name: str = Field(..., description="Parent's last name")
    middle_name: Optional[str] = Field(None, description="Parent's middle name")
    gender: Literal["Male", "Female"] = Field(..., description="Parent's gender, inferred from the name")
    occupation: Optional[Occupation] = Field(None, description="Parent's occupation title")


class Person(BaseModel):
    first_name: str = Field(..., description="Person's first name or initials")
    last_name: str = Field(..., description="Person's last name")
    middle_name: Optional[str] = Field(None, description="Person's middle name")
    gender: Literal["Male", "Female"] = Field(..., description="Person's gender, inferred from the name")
    birth_date: Optional[str] = Field(None, description="Person's birth date, in DD-MM-YYYY format")
    birth_place: Optional[str] = Field(None, description="Person's place of birth")
    occupation: Optional[Occupation] = Field(None, description="Person's occupation title")
    parents: Optional[List[Parent]] = Field(None, description="List of parents with their details")


class EducationItem(BaseModel):
    degree: str = Field(..., description="Name of the education")
    degree_level: Literal["Schooling", "Bachelor's", "Master's", "Doctorate"] = Field(
        ..., description="Level of the education"
    )
    year: str = Field(..., description="Year the education was obtained")
    institution: str = Field(..., description="Name of the institution where the degree was obtained")


class CareerItem(BaseModel):
    position: str = Field(..., description="Position held in the career")
    start_year: int = Field(..., description="Year when the career started")
    end_year: Optional[int] = Field(None, description="Year when the career ended (optional)")
    organization: Optional[str] = Field(None, description="Name of the organization")
    location: Optional[str] = Field(None, description="Location of the organization")


class Child(BaseModel):
    first_name: str = Field(..., description="Child's name")
    last_name: str = Field(..., description="Child's name")
    middle_name: Optional[str] = Field(None, description="Child's middle name")
    gender: Literal["Male", "Female"] = Field(..., description="Child's gender, inferred from the name")
    birth_year: Optional[str] = Field(None, description="Child's birth year")


class Family(BaseModel):
    spouse: Optional[Person] = Field(None, description="Details about the spouse")
    marriage_year: Optional[str] = Field(None, description="Year of marriage")
    divorce_year: Optional[str] = Field(None, description="Year of divorce if applicable")
    children: Optional[List[Child]] = Field(None, description="List of children")


# Expanded Publication Schema
class Edition(BaseModel):
    edition_number: str = Field(..., description="Edition number (e.g., '2:a')")
    year: str = Field(..., description="Year of this edition")
    publisher: Optional[str] = Field(None, description="Publisher of this edition")
    notes: Optional[str] = Field(None, description="Additional notes about this edition")


class Publication(BaseModel):
    title: str = Field(..., description="Title of the publication")
    year: str = Field(..., description="Year of publication")
    type: str = Field(..., description="Type of publication (e.g., book, article, etc.)")
    editions: Optional[List[Edition]] = Field(None, description="List of editions/reprints")
    publisher: Optional[str] = Field(None, description="Publisher name")
    reception: Optional[str] = Field(None, description="Notes on critical reception or impact")
    signature: Optional[str] = Field(None, description="Pseudonym or signature used for this work")
    volumes: Optional[str] = Field(None, description="Volume information (e.g., I-II)")


# Committees and Organizations
class CommitteeMembership(BaseModel):
    committee_name: str = Field(..., description="Name of the committee or organization")
    role: str = Field(..., description="Role in the committee (e.g., 'ordf' for chairperson)")
    start_year: int = Field(..., description="Year when the membership started")
    end_year: Optional[int] = Field(None, description="Year when the membership ended, or 'sed' (since) for ongoing")
    organization: Optional[str] = Field(None, description="Parent organization if applicable")
    significance: Optional[str] = Field(None, description="Significance of this membership")


# Academic Positions
class AcademicPosition(BaseModel):
    position: str = Field(..., description="Academic position title in Swedish (e.g., 'doc', 'prof')")
    position_english: Optional[str] = Field(None, description="Academic position translated (e.g., 'docent')")
    institution: str = Field(..., description="Institution name (possibly abbreviated)")
    department: Optional[str] = Field(None, description="Department or faculty")
    start_year: int = Field(..., description="Year when the position started")
    end_year: Optional[int] = Field(None, description="Year when the position ended, or 'sed' for ongoing")
    field: Optional[str] = Field(None, description="Academic field")


# Special Achievements
class Achievement(BaseModel):
    title: str = Field(..., description="Title or name of the achievement")
    field: str = Field(..., description="Field in which achievement was made")
    year: Optional[str] = Field(None, description="Year of the achievement")
    significance: Optional[str] = Field(None, description="Description of significance")
    recognized_by: Optional[str] = Field(None, description="Organization or entity recognizing achievement")


# Political/Advocacy Positions
class PoliticalPosition(BaseModel):
    organization: str = Field(..., description="Organization name")
    position: str = Field(..., description="Position held")
    start_year: int = Field(..., description="Start year")
    end_year: Optional[int] = Field(None, description="End year or 'sed' for ongoing")
    focus_area: Optional[str] = Field(None, description="Focus area (e.g., environment, culture)")
    significance: Optional[str] = Field(None, description="Significance of this position")


class CommunityInvolvement(BaseModel):
    role: str = Field(..., description="Role in the community organization")
    organization: str = Field(..., description="Name of the community organization")


class BoardMembership(BaseModel):
    position: str = Field(..., description="Position held on the board")
    organization: str = Field(..., description="Name of the organization")


class HonoraryTitle(BaseModel):
    title: str = Field(..., description="Honorary title received")
    institution: str = Field(..., description="Institution that awarded the title")
    year: str = Field(..., description="Year the title was awarded")


class Travel(BaseModel):
    country: str = Field(..., description="Country visited")
    year: Optional[str] = Field(None, description="Year of travel")


class BioSchema(BaseModel):
    # Using Person as the base for biographical information
    person: Person = Field(..., description="Core biographical information about the person")
    current_location: Optional[str] = Field(None, description="Current location associated with the person")
    
    # Education, career and academic positions
    education: Optional[List[EducationItem]] = Field(None, description="List of educational qualifications")
    academic_positions: Optional[List[AcademicPosition]] = Field(None, description="Academic roles and positions")
    career: List[CareerItem] = Field(..., description="Career history of the person")
    
    # Family information
    family: Optional[Family] = Field(None, description="Family details including spouse and children")
    
    # Publications and creative works
    publications: Optional[List[Publication]] = Field(None, description="List of publications with detailed information")
    
    # Organizations, committees, and positions
    committee_memberships: Optional[List[CommitteeMembership]] = Field(
        None, description="Committee memberships with date ranges"
    )
    political_positions: Optional[List[PoliticalPosition]] = Field(
        None, description="Political and advocacy positions"
    )
    community_involvement: Optional[List[CommunityInvolvement]] = Field(
        None, description="Community roles and involvement"
    )
    board_memberships: Optional[List[BoardMembership]] = Field(
        None, description="Board memberships held by the person"
    )
    
    # Achievements and recognition
    achievements: Optional[List[Achievement]] = Field(None, description="Special achievements and contributions")
    honorary_titles: Optional[List[HonoraryTitle]] = Field(None, description="List of honorary titles received")
    awards: Optional[List[str]] = Field(None, description="List of awards received")
    honors: Optional[List[str]] = Field(None, description="Honors received by the person")
    
    # Personal details
    hobbies: Optional[List[str]] = Field(None, description="List of hobbies")
    travels: Optional[List[Travel]] = Field(None, description="Travel details")
    leadership_roles: Optional[List[str]] = Field(None, description="List of leadership roles held")
    languages_spoken: Optional[List[str]] = Field(None, description="Languages spoken by the person")
    military_service: Optional[str] = Field(None, description="Military service details")
    


# Create Gemini API client
def create_client():
    """Create and return a configured Gemini API client."""
    return genai.Client(api_key=API_KEY)


# Define the prompt template
PROMPT_TEMPLATE = """
You are an expert on Swedish biographies and will structure the biographies of individuals 
from the 20th century biographical dictionary 'Vem är Vem' that is provided below. 

### Task:
1. Use the schema to organize the information in the biography.
2. Keep the biographic descriptions in Swedish and remove any abbreviations based on your knowledge, 
   e.g. 'fil. kand.' is 'filosofie kandidat', and 'Skarab. l.' is 'Skaraborgs Län' etc. 
3. For missing data in a required field, include the field with a `None` value.
4. Ensure fields are correctly labeled and structured as per the schema.
5. Put years in full based on context. Put dates in DD-MM-YYYY format where possible. 
"""


def structure_biography(filename: str, file_path: str) -> Optional[Dict[str, Any]]:
    """
    Process a biography file and structure its content according to the BioSchema.
    
    Args:
        filename: The name of the biography file without extension
        file_path: The path to the biography file
        
    Returns:
        Structured biography data as a dictionary, or None if processing failed
    """
    # Create output path
    output_path = OUTPUT_DATA_PATH / f"{filename}.json"
    
    # Skip if already processed
    if output_path.exists():
        logger.info(f"Output for {filename} already exists, skipping.")
        return None
    
    try:
        # Read biography content
        with open(file_path, "r", encoding="utf-8") as f:
            biography_text = f.read()
        
        # Generate structured content
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[PROMPT_TEMPLATE, biography_text],
            config={
                "response_mime_type": "application/json",
                "response_schema": BioSchema.model_json_schema()
            }
        )
        
        # Process the response
        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("Empty response from Gemini API")
            
        # Extract JSON from response
        json_text = response.candidates[0].content.parts[0].text
        result = json.loads(json_text)
        
        # Save structured data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Successfully processed and saved {filename}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error for {filename}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {filename}: {e}")
    except IOError as e:
        logger.error(f"IO error for {filename}: {e}")
    except ValueError as e:
        logger.error(f"Value error for {filename}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing {filename}: {e}", exc_info=True)
    
    return None


def process_biographies(limit: int = float('inf')) -> None:
    """
    Process multiple biographies up to the specified limit.
    
    Args:
        limit: Maximum number of biographies to process
    """
    try:
        # Load the list of biographies to process
        with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
            biographies = json.load(f)
        
        logger.info(f"Found {len(biographies)} biographies to process")
        
        # Process biographies up to the limit
        processed_count = 0
        for entry in biographies:
            if processed_count >= limit:
                logger.info(f"Reached limit of {limit} biographies, stopping")
                break
                
            filename = entry.get('filename')
            file_path = entry.get('relative_path')
            
            if not filename or not file_path:
                logger.warning(f"Missing filename or path in entry: {entry}")
                continue
                
            # Process the biography
            result = structure_biography(filename, file_path)
            
            # Count only if we actually processed it (not skipped)
            if result is not None:
                processed_count += 1
                
    except IOError as e:
        logger.error(f"Error reading biography list: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing biography list JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    # Process all biographies (or set a specific limit)
    process_biographies(limit=10)