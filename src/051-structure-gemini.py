from typing import ClassVar, Literal, Optional
from pydantic import BaseModel, Field
import base64
import requests
from dotenv import load_dotenv
import os
import json
import logging
from typing import List, Optional
from google import genai


# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

# Create a client
client = genai.Client(api_key=api_key)

MODEL_ID =  "gemini-2.0-flash" 

PROCESSED_PATH = "data/structured_biographies"


class Occupation(BaseModel):
    occupation: str = Field(..., description="Occupation in Swedish")
    occupation_english: Optional[str] = Field(None, description="Occupation translated into English")


class Parent(BaseModel):
    name: str = Field(..., description="Parent's name")
    occupation: Optional[Occupation] = Field(None, description="Parent's occupation title")


class Person(BaseModel):
    first_name: str = Field(..., description="Person's first name or initials")
    last_name: str = Field(..., description="Person's last name")
    middle_name: Optional[str] = Field(None, description="Person's middle name")
    gender: Literal["Male", "Female"] = Field(..., description="Person's gender, inferred from the name")
    birth_date: Optional[str] = Field(None, description="Person's birth date, in DD-MM-YYYY format")
    occupation: Optional[Occupation] = Field(None, description="Person's occupation title")


class BirthDetails(BaseModel):
    date: str = Field(..., description="Birth date")
    place: str = Field(..., description="Place of birth")
    parents: Optional[List[Parent]] = Field(None, description="List of parents with their details")


class EducationItem(BaseModel):
    degree: str = Field(..., description="Name of the education")
    degree_level: Literal["Schooling", "Bachelor's", "Master's", "Doctorate"] = Field(..., description="Level of the education")
    year: str = Field(..., description="Year the education was obtained")
    institution: str = Field(..., description="Name of the institution where the degree was obtained")


class CareerItem(BaseModel):
    position: str = Field(..., description="Position held in the career")
    start_year: int = Field(..., description="Year when the career started")
    end_year: Optional[int] = Field(None, description="Year when the career ended (optional)")
    organization: Optional[str] = Field(None, description="Name of the organization")
    location: Optional[str] = Field(None, description="Location of the organization")


class Child(BaseModel):
    name: str = Field(..., description="Child's name")
    birth_year: Optional[str] = Field(None, description="Child's birth year")


class Family(BaseModel):
    spouse: Optional[Person] = Field(None, description="Details about the spouse")
    children: Optional[List[Child]] = Field(None, description="List of children")


class Publication(BaseModel):
    title: str = Field(..., description="Title of the publication")
    year: str = Field(..., description="Year of publication")
    type: str = Field(..., description="Type of publication (e.g., book, article, etc.)")


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
    full_name: str = Field(..., description="Full name of the person")
    location: Optional[str] = Field(None, description="Location associated with the person")
    occupation: Occupation = Field(..., description="Occupation details of the person")
    birth_details: BirthDetails = Field(..., description="Details about the person's birth")
    education: Optional[List[EducationItem]] = Field(None, description="List of educational qualifications")
    career: List[CareerItem] = Field(..., description="Career history of the person")
    family: Optional[Family] = Field(None, description="Family details including spouse and children")
    publications: Optional[List[Publication]] = Field(None, description="List of publications")
    community_involvement: Optional[List[CommunityInvolvement]] = Field(None, description="Community roles and involvement")
    board_memberships: Optional[List[BoardMembership]] = Field(None, description="Board memberships held by the person")
    honorary_titles: Optional[List[HonoraryTitle]] = Field(None, description="List of honorary titles received")
    hobbies: Optional[List[str]] = Field(None, description="List of hobbies")
    travels: Optional[List[Travel]] = Field(None, description="Travel details")
    awards: Optional[List[str]] = Field(None, description="List of awards received")
    leadership_roles: Optional[List[str]] = Field(None, description="List of leadership roles held")
    languages_spoken: Optional[List[str]] = Field(None, description="Languages spoken by the person")
    military_service: Optional[str] = Field(None, description="Military service details")
    honors: Optional[str] = Field(None, description="Honors received by the person")
    death_date: Optional[str] = Field(None, description="Date of death")



# Craft the prompt for zero-shot structuring of biographies
prompt = f"""
You are an expert on Swedish biographies and will structure the biographies of individuals from the 20th century biographical dictionary 'Vem är Vem' that is provided below. 

### Task:
1. Use the schema to organize the information in the biography.
2. Keep the biographic descriptions in Swedish and remove any abbreviations based on your knowledge, e.g. 'fil. kand.' is 'filosofie kandidat', and 'Skarab. l.' is 'Skaraborgs Län' etc. 
3. For missing data in a required field, include the field with a `None` value.
4. Ensure fields are correctly labeled and structured as per the schema.
5. Put years in full based on context. Put dates in DD-MM-YYYY format where possible. 

"""

def structure_biography(filename, relative_path, model: BaseModel):

    # Read the content of the biography from the file
    text_in = open(relative_path, "r", encoding="utf-8").read()

    try:
        response = client.models.generate_content(
            model = MODEL_ID,
            contents = [prompt, text_in],
            config = {
                "response_mime_type": "application/json",
                "response_schema": model
            }

        json_object = response.model_dump()

        # Save the structured biography to a JSON file
        output_path = f"data/structured_biographies/{filename}.json"
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(json_object, output_file, indent=4, ensure_ascii=False)
        logging.info(f"Structured biography saved to {output_path}")
        return json_object
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Error occurred while structuring biography for {filename}: {e}")
        return None


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def structure_biographies(limit=10):
    # Path to the file containing the list of documents to classify
    bios_list_path = "data/biography_list/biography_list.json"

    # Load the document list
    with open(bios_list_path, 'r') as bios_file:
        bios_to_structure = json.load(bios_file)

    # Initialize counter
    counter = 0

    # Loop over each document in the list
    for entry in bios_to_structure:
        # Stop after processing the first 10 documents
        if counter >= limit:
            logging.info(f"Processed {limit} documents, stopping.")
            break

        filename = entry.get('filename')

        relative_path = entry.get('relative_path')

        if not filename:
            logging.warning(
                f"Document does not contain 'filename', skipping: {entry}")
            continue

        # Define the path where the output should be stored
        output_path = f"data/structured_biographies/{filename}.json"

        # Check if the output file already exists
        if os.path.exists(output_path):
            # logging.info(f"Output for filename {filename} already exists, skipping.")
            continue

        # If the output doesn't exist, classify the document
        logging.info(f"Classifying document with filename {filename}")

        # structure the biography
        json_object = structure_biography(filename, relative_path)

        logging.info(f"Saved output for filename {filename} to {output_path} as {json_object}.")

        # Increment counter after processing each document
        counter += 1


# Test the structure function
structure_biographies(limit=1e10)
