FROM node:22-alpine

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Install serve package
RUN npm install -g serve

# Copy source code
COPY . .

ARG REACT_APP_API_URL=http://localhost:5000/api/game
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

RUN npm run build

EXPOSE 3000

# Run the app with serve
CMD ["serve", "-s", "dist", "-l", "3000"]