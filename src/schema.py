from typing import List, Optional
from pydantic import BaseModel


class Occupation(BaseModel):
    occupation: str  # Occupation in Swedish
    occupation_english: Optional[str] = None  # Occupation translated into English


class Parent(BaseModel):
    name: str
    occupation: Optional[Occupation] = None  # Parent's occupation details


class Person(BaseModel):
    name: str
    birth_date: Optional[str] = None
    occupation: Optional[Occupation] = None


class BirthDetails(BaseModel):
    date: str
    place: str
    parents: Optional[List[Parent]] = None  # List of parents with their details


class EducationItem(BaseModel):
    degree: str
    year: str
    institution: str


class CareerYears(BaseModel):
    start_year: int
    end_year: Optional[int] = None


class CareerItem(BaseModel):
    position: str
    years: CareerYears
    organization: Optional[str] = None


class Child(BaseModel):
    name: str
    birth_year: Optional[str] = None


class Family(BaseModel):
    spouse: Optional[Person] = None  # Spouse details as a `Person` model
    children: Optional[List[Child]] = None  # List of children


class Publication(BaseModel):
    title: str
    year: str
    type: str


class CommunityInvolvement(BaseModel):
    role: str
    organization: str
    years: str


class BoardMembership(BaseModel):
    position: str
    organization: str
    years: str


class HonoraryTitle(BaseModel):
    title: str
    institution: str
    year: str


class Travel(BaseModel):
    country: str
    year: Optional[str] = None


class Schema(BaseModel):
    full_name: str
    location: Optional[str] = None
    occupation: Occupation
    birth_details: BirthDetails
    education: Optional[List[EducationItem]] = None
    career: List[CareerItem]
    family: Optional[Family] = None
    publications: Optional[List[Publication]] = None
    community_involvement: Optional[List[CommunityInvolvement]] = None
    board_memberships: Optional[List[BoardMembership]] = None
    honorary_titles: Optional[List[HonoraryTitle]] = None
    hobbies: Optional[List[str]] = None
    travels: Optional[List[Travel]] = None
    awards: Optional[List[str]] = None
    leadership_roles: Optional[List[str]] = None
    languages_spoken: Optional[List[str]] = None
    military_service: Optional[str] = None
    honors: Optional[str] = None
    death_date: Optional[str] = None
