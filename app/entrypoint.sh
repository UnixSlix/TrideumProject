#!/bin/sh 

# Reference for entrypoint solution to wait for ollama serve to spin
#    up before pulling llama3.2:
# https://stackoverflow.com/questions/78232178/ollama-in-docker-pulls-models-via-interactive-shell-but-not-via-run-command-in-t

echo "Starting Ollama server..."
ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" = "" ]; do
    sleep 1
done

echo "Pulling llama3.2 ..."
ollama pull llama3.2

echo "Pulling mxbai-embed-large ..."
ollama pull mxbai-embed-large

echo "Ollama ready, waiting"
wait $SERVER_PID