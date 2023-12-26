# Use the official lightweight Node.js 14 image to build the React app.
# https://hub.docker.com/_/node
FROM node:14 as build

# Create and change to the app directory.
WORKDIR /usr/src/app

# Copy package.json and package-lock.json to work directory
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy local code to the container image.
COPY . .

# Build the React application.
RUN npm run build

# Use the official nginx image for a production build.
# https://hub.docker.com/_/nginx
FROM nginx:stable

# Copy the build from the first stage.
COPY --from=build /usr/src/app/build /usr/share/nginx/html

# Expose port 80 for the application.
EXPOSE 80

# No need to specify a CMD command as the base image handles that.
