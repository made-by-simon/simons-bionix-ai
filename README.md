# Bionix Discord Chatbot

A Discord chatbot with custom Retrieval-Augmented Generation (RAG) for the Alberta Bionix organization.

## Overview

Within many organizations, software such as Discord is used for communication and collaboration. Alberta Bionix uses its Discord server extensively, with some people being pinged for questions that could be answered by simply reading old messages. However, with over 200 members in the server, it is often impractical to go looking for a specific past message.

The Bionix Chatbot addresses this challenge by harnessing the power of large language models (LLMs), augmented with both short-term and long-term contextual awareness:

- **Short-term context**: Handles recent messages from across the Discord server's 40+ channels
- **Long-term context**: Handles 30,000+ past messages using a custom TF-IDF retrieval system

## Technical Implementation

| Component | Details |
|-----------|---------|
| **Deployment** | Replit |
| **LLM Inference** | Groq (Llama 3.3 70B Versatile) |
| **Short-term Context** | Recent messages passed directly to LLM prompts |
| **Long-term Context** | TF-IDF lexical similarity search |
| **TF-IDF Vectorizer** | 5,000-feature limit, 1-to-3-word n-grams |
| **Similarity Metric** | Cosine similarity |
| **Index Rebuild** | Automatic hourly rebuild |

## Context Token Limits

- Bot channel: 1,000 tokens
- Other channels: 1,500 tokens
- Semantic search: 1,500 tokens

## Commands

| Command | Description |
|---------|-------------|
| `!status` | Show comprehensive system status |
| `!search <query>` | Test lexical similarity search without generating a response |
| `!recent [limit]` | Show recent stored messages (default: 10, max: 50) |
| `!help` | Display help message |

## Results

Preliminary evaluation shows that the Bionix Chatbot accurately retrieves and answers questions from past messages, without requiring multiple attempts or careful prompt engineering. These results suggest that members can rely on the chatbot to retrieve relevant information effectively even for queries it has never seen before.

By automating knowledge retrieval, the Bionix Chatbot reduces unnecessary interruptions, minimizes repeated questions, and improves overall productivity by allowing members to focus on higher-value work.

## Environment Variables

- `DISCORD_TOKEN` - Discord bot token
- `GROQ_API_KEY` - Groq API key
- `BOT_CHANNEL_ID` - Channel ID where the bot responds to messages