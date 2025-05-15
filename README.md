# Depot-pour-le-goat-des-llm-



![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)


#NOTE 
Dataset from StackOverflow
llm-mistral
PyPI Changelog Tests License

LLM plugin providing access to Mistral models using the Mistral API

Installation
Install this plugin in the same environment as LLM:

llm install llm-mistral
Usage
First, obtain an API key for the Mistral API.

Configure the key using the llm keys set mistral command:

llm keys set mistral
<paste key here>
You can now access the Mistral hosted models. Run llm models for a list.

To run a prompt through mistral-tiny:

llm -m mistral-tiny 'A sassy name for a pet sasquatch'
To start an interactive chat session with mistral-small:

llm chat -m mistral-small
Chatting with mistral-small
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
> three proud names for a pet walrus
1. "Nanuq," the Inuit word for walrus, which symbolizes strength and resilience.
2. "Sir Tuskalot," a playful and regal name that highlights the walrus' distinctive tusks.
3. "Glacier," a name that reflects the walrus' icy Arctic habitat and majestic presence.
To use a system prompt with mistral-medium to explain some code:

cat example.py | llm -m mistral-medium -s 'explain this code'
Vision
The Pixtral models are capable of interpreting images. You can use those like this:

llm -m pixtral-large 'describe this image' \
  -a https://static.simonwillison.net/static/2025/two-pelicans.jpg
Output:

This image features two pelicans in flight against a clear blue sky. Pelicans are large water birds known for their long beaks and distinctive throat pouches, which they use for catching fish. In this photo, the birds are flying close to each other, showcasing their expansive wings and characteristic beaks. The clear sky provides a stark contrast, highlighting the details of their feathers and the graceful curves of their wings. The image captures a moment of synchronicity and elegance in nature.

You can pass filenames instead of URLs.

Schemas
Mistral models (with the exception of codestral-mamba) also support schemas:

llm -m mistral-small --schema 'name,bio:one sentence' 'invent a cool dog'
Output:

{
  "name": "CyberHound",
  "bio": "A futuristic dog with glowing cybernetic enhancements and the ability to hack into any system."
}
Model options
All three models accept the following options, using -o name value syntax:

-o temperature 0.7: The sampling temperature, between 0 and 1. Higher increases randomness, lower values are more focused and deterministic.
-o top_p 0.1: 0.1 means consider only tokens in the top 10% probability mass. Use this or temperature but not both.
-o max_tokens 20: Maximum number of tokens to generate in the completion.
-o safe_mode 1: Turns on safe mode, which adds a system prompt to add guardrails to the model output.
-o random_seed 123: Set an integer random seed to generate deterministic results.
-o prefix 'Prefix here: Set a prefix that will be used for the start of the response. Try { to encourage JSON or GlaDOS:  to encourage a roleplay from a specific character.
Available models
Run llm models for a full list of Mistral models. This plugin configures the following alias shortcuts:

mistral-tiny for mistral/mistral-tiny
mistral-nemo for mistral/open-mistral-nemo
mistral-small-2312 for mistral/mistral-small-2312
mistral-small-2402 for mistral/mistral-small-2402
mistral-small-2409 for mistral/mistral-small-2409
mistral-small-2501 for mistral/mistral-small-2501
mistral-small for mistral/mistral-small-latest
mistral-medium-2312 for mistral/mistral-medium-2312
mistral-medium-2505 for mistral/mistral-medium-2505
mistral-medium for mistral/mistral-medium-latest
mistral-large for mistral/mistral-large-latest
codestral-mamba for mistral/codestral-mamba-latest
codestral for mistral/codestral-latest
ministral-3b for mistral/ministral-3b-latest
ministral-8b for mistral/ministral-8b-latest
pixtral-12b for mistral/pixtral-12b-latest
pixtral-large for mistral/pixtral-large-latest
Refreshing the model list
Mistral sometimes release new models.

To make those models available to an existing installation of llm-mistral run this command:

llm mistral refresh
This will fetch and cache the latest list of available models. They should then become available in the output of the llm models command.

Embeddings
The Mistral Embeddings API can be used to generate 1,024 dimensional embeddings for any text.

To embed a single string:

llm embed -m mistral-embed -c 'this is text'
This will return a JSON array of 1,024 floating point numbers.

The LLM documentation has more, including how to embed in bulk and store the results in a SQLite database.

See LLM now provides tools for working with embeddings and Embeddings: What they are and why they matter for more about embeddings.

Development
To set up this plugin locally, first checkout the code. Then create a new virtual environment:

cd llm-mistral
python3 -m venv venv
source venv/bin/activate
Now install the dependencies and test dependencies:

llm install -e '.[test]'
To run the tests:

pytest