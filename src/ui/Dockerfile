# Use the official lightweight Node.js 14 image to build the React app.
# https://hub.docker.com/_/node
FROM node:latest

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
