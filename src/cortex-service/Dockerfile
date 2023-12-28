# Use the official Node.js 14 image.
# https://hub.docker.com/_/node
FROM node:14

# Create and change to the app directory.
WORKDIR /usr/src/app

# Copy package.json and package-lock.json to work directory
COPY package*.json ./

# Install production dependencies.
RUN npm install

# Copy local code to the container image.
COPY . .

# Expose port 9000 for the application.
EXPOSE 9000

# Run the web service on container startup.
CMD [ "node", "server.js" ]
