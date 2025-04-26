# Currently This is just being used for linux testing
FROM ubuntu:latest

# Install dependencies
RUN apt-get update && \
    apt-get install -y cmake g++ pkg-config libsdl2-dev libespeak-ng-dev

# Create a working directory
WORKDIR /app

COPY . .

RUN rm -rf ./build

RUN cmake -B build && \
    cmake --build build --config Release

