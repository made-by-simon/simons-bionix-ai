# bionix-chatbot

This repository contains the code for Simon's Bionix Chatbot, a Discord bot deployed on Replit, with LLM inference provided by Groq, augmented by a custom-code TF-IDF (term frequency inverse document frequency) RAG (retrieval-augmented generation) system. 

This chatbot uses a parellel short and long term recall strategy: 
- Short term recall handled by directly passing recent messages inside the current message prompt.
- Long term recall handled by the TF-IDF system which carries out semantic search before passing results to the current message prompt.

The Bionix Chatbot is accessible in the "🤖┃bionix-chatbot" channel of the Bionix Discord server. It currently operates using only the free tiers of Replit and Groq, with no billing information provided. 

[System status page.](https://simons-bionix-ai--made-by-simon.replit.app/)

