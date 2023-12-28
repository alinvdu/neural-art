// server.js
const express = require("express");
const cors = require("cors");
const Cortex = require("./cortex");
const http = require("http");

const { v4: uuidv4 } = require("uuid");

const app = express();
const PORT = process.env.PORT || 9000;

// Middlewares
app.use(cors());
app.use(express.json());

// Sample route
app.get("/", (req, res) => {
  res.send("Hello from Cortex-Service!");
});

const clients = {};

const WebSocket = require("ws");
const profileName = "alindumitru";

class ClientConnection {
  constructor(cortex) {
    const server = http.createServer();
    const wss = new WebSocket.Server({ server });
    server.listen(9001);
    this.hasStreamSubscription = false;
    this.ongoingRecords = {};

    wss.on("connection", (ws) => {
      this.readyState = ws.readyState;
      this.ws = ws;

      const randomUuid = uuidv4();
      clients[randomUuid] = randomUuid;

      this.ws.send(
        JSON.stringify({
          id: randomUuid,
        })
      );

      this.ws.on("message", (message) => {
        const id = message.id;
        const parsedMessage = JSON.parse(message);
        switch (parsedMessage.action) {
          case "START_RECORD": {
            try {
              cortex
                .checkGrantAccessAndQuerySessionInfo()
                .then(() => {
                  const startRecord = () => {
                    const recordName = `${id}_record`;
                    cortex
                      .startRecord(
                        cortex.authToken,
                        cortex.sessionId,
                        recordName
                      )
                      .then((recordId) => {
                        console.log("record started");
                        this.ongoingRecords[recordId] = true;
                        this.ws.send(
                          JSON.stringify({
                            action: "RECORD_STARTED",
                          })
                        );
                      })
                      .catch((err) => {
                        this.ws.send(
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
                  if (!this.hasStreamSubscription) {
                    cortex
                      .sub(["eeg"])
                      .then(() => {
                        startRecord();
                      })
                      .catch((err) => {
                        this.ws.send(
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
                  this.ws.send(
                    JSON.stringify({
                      action: "ERROR",
                      err: {
                        message: err.message,
                      },
                    })
                  );
                });
            } catch (err) {
              this.ws.send(
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
                this.ongoingRecords[recordId] = false;
                cortex.unsub;
              })
              .catch(() => {
                console.log("problem stopping record");
              });
            break;
          }
          case "MARK_BASELINE": {
            cortex
              .injectMarkerRequest()
              .then(() => {
                console.log(
                  "injected marker for baseline at time" + parsedMessage.time
                );
              })
              .catch((err) => {
                this.ws.send(
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
              .injectMarkerRequest()
              .then(() => {
                console.log(
                  "injected marker for art at time" + parsedMessage.time
                );
              })
              .catch((err) => {
                this.ws.send(
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
      });
    });
  }
}
let user = {
  clientId: process.env.CLIENT_ID,
  clientSecret: process.env.CLIENT_SECRET,
  debit: 1,
};

let socketUrl = "wss://host.docker.internal:6868";

let c = new Cortex(user, socketUrl, profileName);
new ClientConnection(c);

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
