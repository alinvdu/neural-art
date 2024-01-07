const {
  extractRandomEegFromHardcoded,
  processEEGTrial,
} = require("./mocked/mocked-eeg");

let loadedEegData = null;
let loadedEegMetadata = null;

function parseCsv(csvString, skipFirstLine) {
  const rows = skipFirstLine
    ? csvString.split("\n").slice(1)
    : csvString.split("\n");
  const headers = rows[0].split(",");
  return rows
    .slice(1)
    .filter((row) => !!row)
    .map((row) => {
      const obj = {};
      row.split(",").forEach((value, i) => {
        obj[headers[i]] = value;
      });

      return obj;
    });
}

const handleClientLogic = (
  ws,
  cortex,
  parsedMessage,
  generationService,
  clientConnection,
  eegData
) => {
  const id = parsedMessage.id;
  switch (parsedMessage.action) {
    case "START_RECORD": {
      try {
        const runStartRecordLogic = () => {
          const startRecord = () => {
            const recordName = `${id}_record`;
            cortex
              .startRecord(cortex.authToken, cortex.sessionId, recordName)
              .then((recordId) => {
                console.log("record started");
                clientConnection.ongoingRecords[recordId] = true;

                ws.send(
                  JSON.stringify({
                    action: "RECORD_STARTED",
                  })
                );
              })
              .catch((err) => {
                ws.send(
                  JSON.stringify({
                    action: "ERROR",
                    err: {
                      message: err.message,
                    },
                  })
                );
              });
          };

          // if (!clientConnection.hasStreamSubscription) {
          //   cortex
          //     .sub(["eeg"], (cortexMessage) => {
          //       // get the recording id
          //       // for now we have to mock it
          //       const recordId = Object.keys(
          //         clientConnection.ongoingRecords
          //       )[0];
          //       // here goes the update of record id with data and metadata: TO-DO -> waiting for EEG API access
          //     })
          //     .then(() => {
          //       startRecord();
          //     })
          //     .catch((err) => {
          //       console.log(err);
          //       ws.send(
          //         JSON.stringify({
          //           action: "ERROR",
          //           err,
          //         })
          //       );
          //     });
          // } else {
          startRecord();
          //}
        };

        if (!cortex.sessionId || !cortex.authToken) {
          cortex
            .checkGrantAccessAndQuerySessionInfo()
            .then(() => {
              runStartRecordLogic();
            })
            .catch((err) => {
              ws.send(
                JSON.stringify({
                  action: "ERROR",
                  error: err,
                })
              );
            });
        } else {
          runStartRecordLogic();
        }
      } catch (err) {
        ws.send(
          JSON.stringify({
            action: "ERROR",
            error: err,
          })
        );
      }
      break;
    }
    case "STOP_RECORD": {
      const recordName = `${id}_record`;
      // not needed for now - we hardcode to timestamp + 8 seconds
      // cortex
      //   .updateMarkerRequest(
      //     cortex.authToken,
      //     cortex.sessionId,
      //     parsedMessage.markerId,
      //     parsedMessage.time
      //   )
      //   .then(() => {
      //     console.log("marked imagine end");
      //   })
      //   .catch((err) => {
      //     JSON.stringify({
      //       action: "ERROR",
      //       err: {
      //         message: err.message,
      //       },
      //     });
      //   });
      cortex
        .stopRecord(cortex.authToken, cortex.sessionId, recordName)
        .then(({ result }) => {
          // generationService.send(
          //   JSON.stringify({
          //     action: "GENERATE_IMAGES",
          //     clientId: id,
          //     eeg: eegData[id],
          //   })
          // );
          ws.send(
            JSON.stringify({
              action: "RECORD_STOPPED",
            })
          );

          // cortex
          //   .exportRecord(
          //     cortex.authToken,
          //     cortex.sessionId,
          //     result.record.uuid
          //   )
          //   .then((res) => {
          //     console.log(res.result.failure[0]);
          //     console.log("exported");
          //   })
          //   .catch((err) => {
          //     console.log(err);
          //   });
        })
        .catch((err) => {
          console.log(err);
          console.log("problem stopping record");
        });
      break;
    }
    case "MARK_BASELINE": {
      cortex
        .injectMarkerRequest(
          cortex.authToken,
          cortex.sessionId,
          "baseline",
          "baseline",
          parsedMessage.time
        )
        .then((markerId) => {
          console.log(
            "injected marker for baseline at time" + parsedMessage.time
          );
          // console.log("marker id is", markerId);
          // ws.send(
          //   JSON.stringify({
          //     action: "SET_BASELINE_MARKER",
          //     markerId,
          //   })
          // );
        })
        .catch((err) => {
          console.log(err);
          ws.send(
            JSON.stringify({
              action: "ERROR",
              err: {
                message: err.message,
              },
            })
          );
        });
      break;
    }
    // not needed for now - we hardcode at timestamp + 8s
    // case "SET_MARKER_END": {
    //   cortex
    //     .updateMarkerRequest(
    //       cortex.authToken,
    //       cortex.sessionId,
    //       parsedMessage.markerId,
    //       parsedMessage.time
    //     )
    //     .then(() => {
    //       console.log("marked baseline end");
    //     })
    //     .catch((err) => {
    //       JSON.stringify({
    //         action: "ERROR",
    //         err: {
    //           message: err.message,
    //         },
    //       });
    //     });
    //   break;
    // }
    case "MARK_IMAGINE_START": {
      cortex
        .injectMarkerRequest(
          cortex.authToken,
          cortex.sessionId,
          "imagine",
          "imagine",
          parsedMessage.time
        )
        .then((markerId) => {
          console.log(
            "added marker for imagine at timestamp: ",
            parsedMessage.time
          );
          // ws.send(
          //   JSON.stringify({
          //     action: "SET_IMAGINE_MARKER_ID",
          //     markerId,
          //   })
          // );
        })
        .catch((err) => {
          ws.send(
            JSON.stringify({
              action: "ERROR",
              err: {
                message: err.message,
              },
            })
          );
        });
      break;
    }
    case "UPLOAD_FILE": {
      const isEEGData = parsedMessage.fileData.isEEGData;
      console.log("is eeg data", isEEGData);
      const buffer = Buffer.from(parsedMessage.fileData.content);
      let csvString = buffer.toString("utf8"); // Convert buffer to string

      if (isEEGData) {
        loadedEegData = parseCsv(csvString, (skipFirstLine = true));
      } else {
        loadedEegMetadata = parseCsv(csvString);
      }

      if (loadedEegData && loadedEegMetadata) {
        console.log("FILE uploadded!");
        ws.send(
          JSON.stringify({
            action: "FILE_UPLOADED",
          })
        );

        const eegData = processEEGTrial(loadedEegMetadata, loadedEegData);

        // process files
        generationService.send(
          JSON.stringify({
            eeg: eegData,
            clientId: id,
            action: "GENERATE_IMAGES",
          })
        );
      }
      break;
    }
  }
};

module.exports = handleClientLogic;
