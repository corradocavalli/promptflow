# Code Generator Spike

This repository contains the code and documentation for the Code generation project.  
The spike uses `Python` in order to better evaluate the results.

## Azure

1. Setup an OpenAI service with two deployments `GPT-4` and `Text-Embeddings-ADA002`
2. Setup a AI Search service with ranker enabled.

## Installation

1. Clone the repository.
2. Open the project inside included devcontainer.
3. Copy the `.env copy` file to `.env` and add required info

## Upload documents into Azure AI Search

1. run `sh scripts/upload_docs.sh` (test docs are, at the moment, represented by the file `data\json\library_data.json`)

## Other

Inside `src` there are different tests of AI Search data uploading.
