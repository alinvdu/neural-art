const EEG_IMAGINE_HARDCODED = [];
const EEG_BASELINE_HARDCODED = [];
const EEG_METADATA_HARDCODED = [];

const eegChannels = [
  "EEG.AF3",
  "EEG.F7",
  "EEG.F3",
  "EEG.FC5",
  "EEG.T7",
  "EEG.P7",
  "EEG.O1",
  "EEG.O2",
  "EEG.P8",
  "EEG.T8",
  "EEG.FC6",
  "EEG.F4",
  "EEG.F8",
  "EEG.AF4",
];

const fs = require("fs");
const path = require("path");
const { parse } = require("csv-parse/sync"); // Ensure you have the 'csv-parse' package installed

const recordingsPath = "/usr/src/app/mocked/recordings";
const sampleRate = 128; // Assuming 1 sample per second, adjust as necessary

async function readCsv(filePath, fromLine = 1) {
  const content = fs.readFileSync(filePath);
  return parse(content, {
    from_line: fromLine,
    columns: true,
    skip_empty_lines: true,
  });
}

async function processTrial(trialPath) {
  const metadataPath = path.join(trialPath, "eegMetadata.csv");
  const dataPath = path.join(trialPath, "eegData.csv");

  const metadata = await readCsv(metadataPath);
  const eegData = await readCsv(dataPath, (fromLine = 2));

  const result = {
    eegMetadata: [],
    eegData: [],
    baselineData: [],
  };

  metadata.forEach((entry) => {
    const fields =
      entry["latency;duration;type;marker_value;key;timestamp;marker_id"].split(
        ";"
      );

    const type = fields[2];
    const markerTimestamp = parseFloat(fields[5]);

    const startIndex = eegData.findIndex(
      (row) => parseFloat(row["Timestamp"]) === markerTimestamp
    );
    const endIndex = startIndex + 8 * sampleRate; // Adjust for 8 seconds

    if (startIndex !== -1) {
      result.eegMetadata.push({ type: type, timestamp: markerTimestamp });

      const eegSegment = eegData.slice(startIndex, endIndex).map((row) => {
        let eegRow = { timestamp: parseFloat(row["Timestamp"]) };
        eegChannels.forEach((channel) => {
          eegRow[channel] = parseFloat(row[channel]);
        });
        return eegRow;
      });

      if (type.includes("eyesopen")) {
        result.baselineData.push(...eegSegment);
      } else {
        result.eegData.push(...eegSegment);
      }
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
    EEG_IMAGINE_HARDCODED.push(trialData.eegData);
    EEG_BASELINE_HARDCODED.push(trialData.baselineData);
    EEG_METADATA_HARDCODED.push(trialData.eegMetadata);
  }
}

main().catch(console.error);

function getRandomArrayIndex(array) {
  return Math.floor(Math.random() * array.length);
}

const extractRandomEegFromHardcoded = () => {
  const randomId = getRandomArrayIndex(EEG_IMAGINE_HARDCODED);
  return {
    eegData: EEG_IMAGINE_HARDCODED[randomId],
    eegMetadata: EEG_METADATA_HARDCODED[randomId],
    baseline: EEG_BASELINE_HARDCODED[randomId],
  };
};

module.exports = extractRandomEegFromHardcoded;
