const fs = require('fs');
const path = require('path');

// Function to extract engineer information from a biography
function extractEngineerInfo(biography) {
  const person = biography.person;
  
  // Check if this person is an engineer based on HISCO codes
  let isEngineer = false;
  let occupation = null;
  
  if (person?.occupation) {
    const hiscoSwedish = person.occupation.hisco_code_swedish;
    const hiscoEnglish = person.occupation.hisco_code_english;
    
    // Convert HISCO codes to numbers and check if they're in the engineer range
    if (hiscoSwedish) {
      const hiscoCode = parseFloat(hiscoSwedish);
      if (hiscoCode >= 2000 && hiscoCode < 2900) {
        isEngineer = true;
        occupation = person.occupation;
      }
    }
    
    if (!isEngineer && hiscoEnglish) {
      const hiscoCode = parseFloat(hiscoEnglish);
      if (hiscoCode >= 2000 && hiscoCode < 2900) {
        isEngineer = true;
        occupation = person.occupation;
      }
    }
  }
  
  // If not an engineer, return null
  if (!isEngineer) {
    return null;
  }
  
  // Extract birth date and determine decade of birth
  let birthDecade = null;
  if (person.birth_date) {
    // Parse different date formats (DD-MM-YYYY, YYYY-MM-DD, etc.)
    const parts = person.birth_date.split('-');
    
    // Try to identify which part is the year (4-digit number)
    let yearPart = null;
    for (const part of parts) {
      if (part && part.length === 4 && !isNaN(parseInt(part))) {
        yearPart = part;
        break;
      }
    }
    
    // If a 4-digit year was found
    if (yearPart) {
      const birthYear = parseInt(yearPart);
      birthDecade = Math.floor(birthYear / 10) * 10;
    }
  }
  
  // Extract name information
  const fullName = {
    first_name: person.first_name || "",
    middle_name: person.middle_name || "",
    last_name: person.last_name || ""
  };
  
  // Extract education information and determine highest level
  let education = biography.education || [];
  let highestDegreeLevel = null;
  const degreeLevelRanking = {
    "Schooling": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "Doctorate": 4
  };
  
  // Find the highest degree level
  education.forEach(edu => {
    const currentLevel = edu.degree_level;
    if (currentLevel && degreeLevelRanking[currentLevel]) {
      if (!highestDegreeLevel || degreeLevelRanking[currentLevel] > degreeLevelRanking[highestDegreeLevel]) {
        highestDegreeLevel = currentLevel;
      }
    }
  });
  
  return {
    id: path.basename(biography.id || "unknown", ".json"),
    name: fullName,
    birth_date: person.birth_date,
    birth_decade: birthDecade,
    occupation: occupation,
    education: education,
    highest_degree_level: highestDegreeLevel
  };
}

// Function to aggregate data by decade
function aggregateByDecade(engineerData) {
  const aggregated = {};
  
  engineerData.forEach(engineer => {
    if (engineer.birth_decade && engineer.highest_degree_level) {
      const decade = `${engineer.birth_decade}s`;
      
      if (!aggregated[decade]) {
        aggregated[decade] = {
          "Schooling": 0,
          "Bachelor's": 0,
          "Master's": 0,
          "Doctorate": 0
        };
      }
      
      aggregated[decade][engineer.highest_degree_level]++;
    }
  });
  
  return aggregated;
}

// For educational institutions classification
function extractUniqueInstitutions(engineerData) {
  const institutions = new Map();
  
  engineerData.forEach(engineer => {
    if (engineer.education) {
      engineer.education.forEach(edu => {
        if (edu.institution && edu.institution !== "None") {
          const count = institutions.get(edu.institution) || 0;
          institutions.set(edu.institution, count + 1);
        }
      });
    }
  });
  
  // Convert to array of objects with name and count
  return Array.from(institutions.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count); // Sort by frequency
}

// Main function to process all files in a directory
function processEngineerFiles(directoryPath) {
  const allEngineers = [];
  
  // Read all files in the directory
  const files = fs.readdirSync(directoryPath);
  
  files.forEach(file => {
    if (file.endsWith('.json')) {
      const filePath = path.join(directoryPath, file);
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      
      // Add the file ID to the data
      data.id = file;
      
      const engineerInfo = extractEngineerInfo(data);
      if (engineerInfo) {
        allEngineers.push(engineerInfo);
      }
    }
  });
  
  // Aggregate by decade
  const byDecade = aggregateByDecade(allEngineers);
  
  // Extract unique institutions
  const institutions = extractUniqueInstitutions(allEngineers);
  
  // Create the result data
  const resultData = {
    total_engineers: allEngineers.length,
    engineers_by_decade: byDecade,
    unique_institutions: institutions,
    engineer_details: allEngineers
  };
  
  // Write to output file
  fs.writeFileSync('analysis/engineer_education_analysis.json', JSON.stringify(resultData, null, 2));
  
  console.log(`Analysis complete. Found ${allEngineers.length} engineers.`);
  console.log(`Data saved to engineer_education_analysis.json`);
  
  return resultData;
}

// Usage:
processEngineerFiles('data/enriched_biographies_with_hisco_codes');