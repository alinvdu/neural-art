const extractRandomEegFromHardcoded = require("./mocked/mocked-eeg");

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
        cortex
          .checkGrantAccessAndQuerySessionInfo()
          .then(() => {
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
            // first subscribe to the stream if not yet subscribed
            console.log(clientConnection.hasStreamSubscription);
            if (!clientConnection.hasStreamSubscription) {
              cortex
                .sub(["eeg"], (cortexMessage) => {
                  // get the recording id
                  // for now we have to mock it
                  const recordId = Object.keys(
                    clientConnection.ongoingRecords
                  )[0];
                  // here goes the update of record id with data and metadata: TO-DO -> waiting for EEG API access
                })
                .then(() => {
                  startRecord();
                })
                .catch((err) => {
                  console.log(err);
                  ws.send(
                    JSON.stringify({
                      action: "ERROR",
                      err,
                    })
                  );
                });
            } else {
              startRecord();
            }
          })
          .catch((err) => {
            console.log("err");
            ws.send(
              JSON.stringify({
                action: "ERROR",
                err: {
                  message: err.message,
                },
              })
            );
          });
      } catch (err) {
        ws.send(
          JSON.stringify({
            error: err,
          })
        );
      }
      break;
    }
    case "STOP_RECORD": {
      const recordName = `${id}_record`;
      cortex
        .stopRecord(cortex.authToken, cortex.sessionId, recordName)
        .then((recordId) => {
          console.log("record stopped");
          clientConnection.ongoingRecords[recordId] = false;
          // send data to the generation service for further processing and image generation
          // mock it for now
          eegData[recordId] = extractRandomEegFromHardcoded();

          generationService.send(
            JSON.stringify({
              action: "GENERATE_IMAGES",
              clientId: id,
              eegData: eegData[id],
            })
          );
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
          clientConnection.authToken,
          clientConnection.sessionId,
          "baseline"
        )
        .then(() => {
          console.log(
            "injected marker for baseline at time" + parsedMessage.time
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
      break;
    }
    case "MARK_IMAGINE_START": {
      cortex
        .injectMarkerRequest(
          clientConnection.authToken,
          clientConnection.sessionId,
          "imagine"
        )
        .then(() => {
          console.log("injected marker for art at time" + parsedMessage.time);
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
  }
};

module.exports = handleClientLogic;
