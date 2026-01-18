import os
import asyncio
from collections import Counter, deque
from datetime import datetime, timedelta

import discord
import numpy as np
import psutil
from discord.ext import commands
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import keep_alive
import web_status

# Configuration constants.
BOT_CHANNEL_ID = int(os.getenv('BOT_CHANNEL_ID'))
MESSAGES_PER_CHANNEL = 1000  # Limit for TF-IDF indexing per channel.
CHANNEL_DELAY = 2
TFIDF_REBUILD_INTERVAL = 3600  # One hour.
MAX_COMMAND_LIMIT = 50

# Context limits for token management.
BOT_CHANNEL_LIMIT = 50
OTHER_CHANNEL_LIMIT = 10
MAX_OTHER_CHANNELS = 15
SEMANTIC_TOP_K = 5
BOT_MSG_MAX_LEN = 200
OTHER_MSG_MAX_LEN = 150
SEMANTIC_MSG_MAX_LEN = 200

# Initialize Discord bot.
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)
bot.remove_command('help')

# Initialize Groq client.
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
print(f"[{datetime.now()}] Groq client initialized")

# Initialize TF-IDF vectorizer.
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95
)
print(f"[{datetime.now()}] TF-IDF vectorizer initialized")

# Global state.
message_history = deque()  # No upper bound.
tfidf_matrix = None
bot_start_time = datetime.now()
last_tfidf_rebuild = None
next_tfidf_rebuild = None
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens_used = 0
total_api_calls = 0

print(f"[{datetime.now()}] Bot channel ID: {BOT_CHANNEL_ID}")


async def tfidf_rebuild_loop():
    """Background task to rebuild TF-IDF matrix every hour."""
    global next_tfidf_rebuild
    await bot.wait_until_ready()
    print(f"[{datetime.now()}] TF-IDF rebuild loop started")

    while not bot.is_closed():
        next_tfidf_rebuild = datetime.now() + timedelta(seconds=TFIDF_REBUILD_INTERVAL)
        await asyncio.sleep(TFIDF_REBUILD_INTERVAL)
        print(f"[{datetime.now()}] Scheduled TF-IDF rebuild triggered")
        rebuild_tfidf_matrix()


@bot.event
async def on_ready():
    """Handle bot ready event and initialize historical message loading."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] Bot logged in as {bot.user}")
    print(f"[{datetime.now()}] Bot ID: {bot.user.id}")
    print(f"{'='*60}\n")

    await load_historical_messages()
    print(f"[{datetime.now()}] Total messages loaded: {len(message_history)}")
    print(f"[{datetime.now()}] Bot is ready")
    bot.loop.create_task(tfidf_rebuild_loop())


async def load_historical_messages():
    """Load past messages from all text channels on startup."""
    total_loaded = 0

    print(f"[{datetime.now()}] Beginning historical message scan")

    for guild in bot.guilds:
        print(f"[{datetime.now()}] Scanning server: {guild.name}")

        for channel in guild.text_channels:
            if not channel.permissions_for(guild.me).read_message_history:
                continue

            try:
                message_count = 0
                async for message in channel.history(limit=MESSAGES_PER_CHANNEL):
                    if not message.content:
                        continue

                    message_history.append({
                        "content": message.content,
                        "author": message.author.name,
                        "channel": str(channel.name),
                        "timestamp": message.created_at.isoformat(),
                        "channel_id": channel.id,
                        "is_bot": message.author == bot.user
                    })
                    message_count += 1

                    if message_count % 100 == 0:
                        await asyncio.sleep(0.5)

                total_loaded += message_count
                print(f"[{datetime.now()}] #{channel.name}: {message_count} messages")
                await asyncio.sleep(CHANNEL_DELAY)

            except (discord.Forbidden, discord.HTTPException) as e:
                print(f"[{datetime.now()}] Error on #{channel.name}: {e}")
                continue

    print(f"[{datetime.now()}] Total messages loaded: {total_loaded}")

    if message_history:
        print(f"[{datetime.now()}] Building initial TF-IDF matrix...")
        rebuild_tfidf_matrix()


@bot.event
async def on_message(message):
    """Handle incoming messages from all channels."""
    # Store all messages.
    if message.content:
        store_message(message)

    # Skip bot's own messages for response generation.
    if message.author == bot.user:
        await bot.process_commands(message)
        return

    # Respond to non-command messages in bot channel.
    if message.channel.id == BOT_CHANNEL_ID and not message.content.startswith('!'):
        await handle_query(message)

    await bot.process_commands(message)


def store_message(message):
    """Store message in in-memory storage."""
    global tfidf_matrix

    message_history.append({
        "content": message.content,
        "author": message.author.name,
        "channel": str(message.channel.name),
        "timestamp": message.created_at.isoformat(),
        "channel_id": message.channel.id,
        "is_bot": message.author == bot.user
    })

    if tfidf_matrix is None and len(message_history) >= 10:
        print(f"[{datetime.now()}] Initial TF-IDF build triggered ({len(message_history)} messages)")
        rebuild_tfidf_matrix()


def rebuild_tfidf_matrix():
    """Rebuild TF-IDF matrix from all stored messages."""
    global tfidf_matrix, last_tfidf_rebuild

    if not message_history:
        return

    tfidf_matrix = tfidf_vectorizer.fit_transform(msg['content'] for msg in message_history)
    last_tfidf_rebuild = datetime.now()
    print(f"[{datetime.now()}] TF-IDF matrix rebuilt. Shape: {tfidf_matrix.shape}")


def get_recent_messages_by_channel(channel_id, limit):
    """Get recent messages from a specific channel."""
    messages = [msg for msg in message_history if msg['channel_id'] == channel_id]
    return messages[-limit:]


def format_message(msg, max_length):
    """Format a message with author and truncated content."""
    author = f"{msg['author']}(b)" if msg.get('is_bot') else msg['author']
    content = msg['content'][:max_length]
    if len(msg['content']) > max_length:
        content += "..."
    return f"{author}: {content}"


def get_recent_bot_channel_context():
    """Get recent messages from the bot channel."""
    messages = get_recent_messages_by_channel(BOT_CHANNEL_ID, BOT_CHANNEL_LIMIT)
    if not messages:
        return ""
    return "\n".join(format_message(msg, BOT_MSG_MAX_LEN) for msg in messages)


def get_recent_other_channels_context():
    """Get recent messages from other channels."""
    # Group messages by channel.
    channel_messages = {}
    for msg in reversed(message_history):
        if msg['channel_id'] != BOT_CHANNEL_ID:
            channel_id = msg['channel_id']
            if channel_id not in channel_messages:
                channel_messages[channel_id] = []
            if len(channel_messages[channel_id]) < OTHER_CHANNEL_LIMIT:
                channel_messages[channel_id].append(msg)

    if not channel_messages:
        return ""

    # Limit to top channels by message count.
    sorted_channels = sorted(channel_messages.items(), key=lambda x: len(x[1]), reverse=True)[:MAX_OTHER_CHANNELS]

    context_parts = []
    for channel_id, messages in sorted_channels:
        messages.reverse()
        channel_name = messages[0]['channel']
        context_parts.append(f"#{channel_name}:")
        context_parts.extend(format_message(msg, OTHER_MSG_MAX_LEN) for msg in messages)

    return "\n".join(context_parts)


def search_messages_semantic(query):
    """Semantic search using TF-IDF and cosine similarity."""
    if tfidf_matrix is None or not message_history:
        return ""

    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-SEMANTIC_TOP_K:][::-1]
    relevant_indices = [idx for idx in top_indices if similarities[idx] > 0.1]

    if not relevant_indices:
        return ""

    messages_list = list(message_history)
    context_parts = [
        f"[{messages_list[idx]['channel']}] {format_message(messages_list[idx], SEMANTIC_MSG_MAX_LEN)}"
        for idx in relevant_indices
    ]
    return "\n".join(context_parts)


async def send_chunked(message, response):
    """Send a response, splitting into chunks if needed."""
    chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
    await message.reply(chunks[0])
    for chunk in chunks[1:]:
        await message.channel.send(chunk)


async def handle_query(message):
    """Handle user query using RAG with semantic search."""
    print(f"[{datetime.now()}] Query from {message.author.name}: {message.content[:50]}")

    try:
        async with message.channel.typing():
            response = generate_response(
                message.content,
                get_recent_bot_channel_context(),
                get_recent_other_channels_context(),
                search_messages_semantic(message.content)
            )
            await send_chunked(message, response)
        print(f"[{datetime.now()}] Response sent")
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: {e}")
        await message.reply(f"Error: {e}")


def generate_response(query, recent_bot_context, recent_other_context, semantic_context):
    """Generate response using Groq LLM with context."""
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, total_api_calls

    # Build compact prompt with only non-empty contexts.
    context_parts = []

    if recent_bot_context:
        context_parts.append(f"RECENT BOT CHANNEL:\n{recent_bot_context}")

    if recent_other_context:
        context_parts.append(f"OTHER CHANNELS:\n{recent_other_context}")

    if semantic_context:
        context_parts.append(f"RELEVANT:\n{semantic_context}")

    context_str = "\n\n".join(context_parts) if context_parts else "No context available."

    prompt = f"""You are a helpful Discord chatbot with access to conversation history. Provide concise responses based on context. If unsure, say as little as possible. Never use "@" in responses. Messages marked "(b)" are your own.

Prioritize recent bot channel context for conversation flow, other channels for server activity, and semantic search for broader knowledge. For general knowledge, use what you know from pretraining. 

If asked about yourself: "Hello! I am a Discord chatbot created by Simon. I am deployed on Replit, with LLM inference provided by Groq, augmented by a custom-coded TF-IDF semantic search system."

{context_str}

User question: {query}

Respond based on context above. Prioritize bot channel for conversation flow, other channels for server activity, semantic search for broader knowledge. Never use "@" in your responses."""

    print(f"[{datetime.now()}] Prompt length: {len(prompt)} chars")
    print(f"[{datetime.now()}] === PROMPT START ===")
    print(prompt)
    print(f"[{datetime.now()}] === PROMPT END ===")

    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=False
    )

    # Track token usage.
    if hasattr(response, 'usage') and response.usage:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens_used += total_tokens
        total_api_calls += 1

        print(f"[{datetime.now()}] Tokens: {prompt_tokens}p + {completion_tokens}c = {total_tokens}t")

    return response.choices[0].message.content


@bot.command()
async def help(ctx):
    """Display all available bot commands."""
    help_msg = "**Discord RAG Bot - Available Commands**\n\n"
    help_msg += "`!status` - Show comprehensive system status\n"
    help_msg += "`!search <query>` - Test semantic search without generating a response\n"
    help_msg += "`!recent [limit]` - Show recent stored messages (default: 10, max: 50)\n"
    help_msg += "`!help` - Display this help message\n"

    await ctx.send(help_msg)


def _section(title):
    """Format a status section header."""
    return f"\n{'-' * 50}\n{title}\n{'-' * 50}\n"


@bot.command()
async def status(ctx):
    """Show comprehensive system status."""
    uptime = str(datetime.now() - bot_start_time).split('.')[0]
    bot_msg_count = sum(1 for msg in message_history if msg.get('is_bot'))
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    avg_tokens = total_tokens_used / total_api_calls if total_api_calls else 0

    lines = ["**System Status**\n```"]
    lines.append(_section("BOT INFO"))
    lines.append(f"Status:          Online\nBot User:        {bot.user.name}#{bot.user.discriminator}")
    lines.append(f"Bot ID:          {bot.user.id}\nUptime:          {uptime}\nLatency:         {bot.latency*1000:.1f}ms")

    lines.append(_section("MEMORY & STORAGE"))
    lines.append(f"RAM Usage:       {memory_mb:.1f} MB\nMessages Stored: {len(message_history):,}")
    lines.append(f"  - User msgs:   {len(message_history) - bot_msg_count:,}\n  - Bot msgs:    {bot_msg_count:,}")

    lines.append(_section("TF-IDF CONFIGURATION"))
    lines.append(f"Status:          {'Built' if tfidf_matrix is not None else 'Not Built'}")
    if tfidf_matrix is not None:
        lines.append(f"Matrix Shape:    {tfidf_matrix.shape}\nVocabulary Size: {len(tfidf_vectorizer.vocabulary_):,}")
    lines.append("Max Features:    5,000\nN-gram Range:    (1, 3)\nMax Doc Freq:    0.95\nRebuild Interval: 1 hour")
    if last_tfidf_rebuild:
        lines.append(f"Last Rebuild:    {str(datetime.now() - last_tfidf_rebuild).split('.')[0]} ago")
    if next_tfidf_rebuild and (secs := (next_tfidf_rebuild - datetime.now()).total_seconds()) > 0:
        lines.append(f"Next Rebuild:    {int(secs // 60)} minute{'s' if int(secs // 60) != 1 else ''}")

    lines.append(_section("LLM CONFIGURATION"))
    lines.append("Provider:        Groq\nModel:           openai/gpt-oss-20b\nTemperature:     1.0")
    lines.append("Max Tokens:      8192\nReasoning Effort: medium")

    lines.append(_section("TOKEN USAGE"))
    lines.append(f"Total API Calls:     {total_api_calls:,}\nPrompt Tokens:       {total_prompt_tokens:,}")
    lines.append(f"Completion Tokens:   {total_completion_tokens:,}\nTotal Tokens Used:   {total_tokens_used:,}")
    lines.append(f"Avg Tokens/Call:     {avg_tokens:.1f}")

    lines.append(_section("CONTEXT LIMITS"))
    lines.append(f"Bot Channel Msgs:    {BOT_CHANNEL_LIMIT}\nOther Channel Msgs:  {OTHER_CHANNEL_LIMIT}")
    lines.append(f"Max Other Channels:  {MAX_OTHER_CHANNELS}\nSemantic Top K:      {SEMANTIC_TOP_K}")
    lines.append(f"Bot Msg Max Len:     {BOT_MSG_MAX_LEN}\nOther Msg Max Len:   {OTHER_MSG_MAX_LEN}")
    lines.append(f"Semantic Msg Max Len: {SEMANTIC_MSG_MAX_LEN}")

    lines.append(_section("SERVER INFO"))
    lines.append(f"Connected Servers: {len(bot.guilds)}")
    lines.extend(f"  - {g.name}: {g.member_count} members" for g in bot.guilds)
    lines.append("```\n")

    await ctx.send("\n".join(lines))

    channel_counts = Counter(msg['channel'] for msg in message_history)
    if channel_counts:
        max_count = max(channel_counts.values())
        sorted_channels = channel_counts.most_common(15)
        channel_lines = ["**Messages by Channel**\n```"]
        for channel, count in sorted_channels:
            bar_len = int(count / max_count * 20)
            channel_lines.append(f"#{channel[:15]:<15} [{'#' * bar_len}{'.' * (20 - bar_len)}] {count:,}")
        if len(channel_counts) > 15:
            channel_lines.append(f"... and {len(channel_counts) - 15} more channels")
        channel_lines.append("```")
        await ctx.send("\n".join(channel_lines))


@bot.command()
async def recent(ctx, limit: int = 10):
    """Show recent messages stored."""
    if not message_history:
        await ctx.send("No messages stored yet.")
        return

    recent_msgs = list(message_history)[-min(limit, MAX_COMMAND_LIMIT):]
    lines = [f"**Last {len(recent_msgs)} messages stored:**\n```"]
    for m in recent_msgs:
        preview = m['content'][:50] + "..." if len(m['content']) > 50 else m['content']
        bot_tag = " (bot)" if m.get('is_bot') else ""
        lines.append(f"[{m['channel']}] {m['author']}{bot_tag}: {preview}")
    lines.append("```")
    await ctx.send("\n".join(lines))


@bot.command()
async def search(ctx, *, query: str):
    """Test semantic search without generating a response."""
    if tfidf_matrix is None or not message_history:
        await ctx.send("No messages available for search.")
        return

    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]

    messages_list = list(message_history)
    results = [
        f"[{messages_list[idx]['channel']}] {format_message(messages_list[idx], 100)} (score: {similarities[idx]:.3f})"
        for idx in top_indices if similarities[idx] > 0.05
    ]

    if not results:
        await ctx.send(f"**No relevant results for:** {query}")
        return

    context = "\n\n".join(results)[:1900]
    await ctx.send(f"**Search results for:** {query}\n```\n{context}\n```")


def get_tfidf_matrix():
    """Getter function for TF-IDF matrix."""
    return tfidf_matrix


def get_token_stats():
    """Getter function for token statistics."""
    return {
        'total_api_calls': total_api_calls,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_tokens_used': total_tokens_used
    }


def get_last_rebuild():
    """Getter function for last TF-IDF rebuild time."""
    return last_tfidf_rebuild


def get_next_rebuild():
    """Getter function for next TF-IDF rebuild time."""
    return next_tfidf_rebuild


# Configure and start the web status server.
print(f"[{datetime.now()}] Configuring web status server...")
web_status.configure(
    bot=bot,
    message_history=message_history,
    tfidf_vectorizer=tfidf_vectorizer,
    tfidf_matrix_getter=get_tfidf_matrix,
    start_time=bot_start_time,
    last_rebuild_getter=get_last_rebuild,
    next_rebuild_getter=get_next_rebuild,
    token_stats_getter=get_token_stats,
    context_limits={
        'bot_channel_limit': BOT_CHANNEL_LIMIT,
        'other_channel_limit': OTHER_CHANNEL_LIMIT,
        'max_other_channels': MAX_OTHER_CHANNELS,
        'semantic_top_k': SEMANTIC_TOP_K,
        'bot_msg_max_len': BOT_MSG_MAX_LEN,
        'other_msg_max_len': OTHER_MSG_MAX_LEN,
        'semantic_msg_max_len': SEMANTIC_MSG_MAX_LEN,
    },
)

print(f"[{datetime.now()}] Starting web status server...")
web_status.start()


@bot.event
async def on_connect():
    """Start keep-alive task when bot connects."""
    keep_alive.start(bot.loop)


# Run the bot.
print(f"[{datetime.now()}] Starting Discord bot...")
bot.run(os.getenv('DISCORD_TOKEN'))