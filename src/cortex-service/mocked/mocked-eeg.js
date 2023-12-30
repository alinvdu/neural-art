const EEG_DATA_HARDCODED = [];
const EEG_METADATA_HARDCODED = [];

const fs = require("fs");
const path = require("path");
const csv = require("csv-parse"); // Ensure you have the 'csv-parse' package installed

const recordingsPath = "recordings";
const sampleRate = 1; // Assuming 1 sample per second, adjust as necessary

async function readCsv(filePath) {
  const content = fs.readFileSync(filePath);
  return csv.parse(content, {
    columns: true,
    skip_empty_lines: true,
  });
}

async function processTrial(trialPath) {
  const metadataPath = path.join(trialPath, "eegMetadata.csv");
  const dataPath = path.join(trialPath, "eegData");

  const metadata = await readCsv(metadataPath);
  const eegData = await readCsv(dataPath);

  // Skip the first row of eegData as it contains irrelevant data
  eegData.shift();

  const result = {
    eegMetadata: [],
    eegData: [],
  };

  metadata.forEach((marker) => {
    const markerTimestamp = parseInt(marker.timestamp);
    const startIndex = eegData.findIndex(
      (row) => parseInt(row.timestamp) === markerTimestamp
    );
    const endIndex = startIndex + 8 * sampleRate; // Adjust for 8 seconds

    if (startIndex !== -1) {
      result.eegMetadata.push({
        type: marker.type,
        timestamp: markerTimestamp,
      });

      const eegSegment = eegData.slice(startIndex, endIndex).map((row) => {
        return [row.timestamp, ...row.slice(5, 22)]; // Extracting relevant EEG channels
      });
      result.eegData.push(eegSegment);
    }
  });

  return result;
}

async function main() {
  const trials = fs
    .readdirSync(recordingsPath)
    .filter((folder) =>
      fs.statSync(path.join(recordingsPath, folder)).isDirectory()
    );

  for (const trial of trials) {
    const trialPath = path.join(recordingsPath, trial);
    const trialData = await processTrial(trialPath);
    console.log(`Processed ${trial}:`, trialData);
    EEG_DATA_HARDCODED.push(trialData.eegData);
    EEG_METADATA_HARDCODED.push(trialData.eegMetadata);
  }
}

main().catch(console.error);

function getRandomArrayIndex(array) {
  return Math.floor(Math.random() * array.length);
}

const extractRandomEegFromHardcoded = () => {
  const randomId = getRandomArrayIndex(EEG_DATA_HARDCODED);
  return {
    eegData: EEG_DATA_HARDCODED[randomId],
    eegMetadata: EEG_METADATA_HARDCODED[randomId],
  };
};

module.exports = extractRandomEegFromHardcoded;
