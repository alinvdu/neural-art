// server.js
const Cortex = require("./cortex");
const http = require("http");

const { v4: uuidv4 } = require("uuid");
const clients = {};

const WebSocket = require("ws");
const handleClientLogic = require("./client-logic");
const profileName = "alindumitru";

class ClientConnection {
  constructor(cortex) {
    const server = http.createServer();
    const wss = new WebSocket.Server({ server });
    server.listen(9001);
    this.hasStreamSubscription = true;
    this.ongoingRecords = {};
    this.clients = {};
    this.generationService = null;
    this.eegData = {};

    wss.on("connection", (ws) => {
      this.readyState = ws.readyState;
      this.ws = ws;

      this.ws.on("message", (message) => {
        const parsedMessage = JSON.parse(message);
        if (parsedMessage.client === "generation-service") {
          if (!this.generationService) {
            this.generationService = ws;
            console.log("Generation service connected");
          } else {
            handleGenerationServiceLogic(this.ws, parsedMessage, this.clients);
          }
        } else {
          if (!this.clients[parsedMessage.id]) {
            const randomUuid = uuidv4();
            this.clients[randomUuid] = ws;
            ws.send(JSON.stringify({ id: randomUuid }));
            console.log("client connected", randomUuid);
          }
          handleClientLogic(
            this.ws,
            cortex,
            parsedMessage,
            this.generationService,
            this,
            this.eegData
          );
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
