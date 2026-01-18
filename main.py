import os
import discord
from discord.ext import commands
from groq import Groq
import web_status
import keep_alive
from collections import deque
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from datetime import datetime
import psutil

# Configuration constants.
MAX_MESSAGES = 1000000
BOT_CHANNEL_ID = int(os.getenv('BOT_CHANNEL_ID'))
MESSAGES_PER_CHANNEL = 10000
CHANNEL_DELAY = 2
TFIDF_REBUILD_INTERVAL = 3600  # One hour.
MAX_COMMAND_LIMIT = 50

# Context limits for token management.
BOT_CHANNEL_LIMIT = 25
OTHER_CHANNEL_LIMIT = 3
MAX_OTHER_CHANNELS = 10
SEMANTIC_TOP_K = 2
BOT_MSG_MAX_LEN = 120
OTHER_MSG_MAX_LEN = 80
SEMANTIC_MSG_MAX_LEN = 100

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
message_history = deque(maxlen=MAX_MESSAGES)
tfidf_matrix = None
bot_start_time = datetime.now()
last_tfidf_rebuild = None
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens_used = 0
total_api_calls = 0

print(f"[{datetime.now()}] Bot channel ID: {BOT_CHANNEL_ID}")
print(f"[{datetime.now()}] Message capacity: {MAX_MESSAGES}")


async def tfidf_rebuild_loop():
    """Background task to rebuild TF-IDF matrix every hour."""
    await bot.wait_until_ready()
    print(f"[{datetime.now()}] TF-IDF rebuild loop started")

    while not bot.is_closed():
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

    if len(message_history) > 0:
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
    message_history.append({
        "content": message.content,
        "author": message.author.name,
        "channel": str(message.channel.name),
        "timestamp": message.created_at.isoformat(),
        "channel_id": message.channel.id,
        "is_bot": message.author == bot.user
    })


def rebuild_tfidf_matrix():
    """Rebuild TF-IDF matrix from all stored messages."""
    global tfidf_matrix, last_tfidf_rebuild

    if len(message_history) == 0:
        return

    contents = [msg['content'] for msg in message_history]
    tfidf_matrix = tfidf_vectorizer.fit_transform(contents)
    last_tfidf_rebuild = datetime.now()
    print(f"[{datetime.now()}] TF-IDF matrix rebuilt. Shape: {tfidf_matrix.shape}")


def get_recent_messages_by_channel(channel_id, limit):
    """Get recent messages from a specific channel."""
    messages = []
    for msg in reversed(message_history):
        if msg['channel_id'] == channel_id:
            messages.append(msg)
            if len(messages) >= limit:
                break
    messages.reverse()
    return messages


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
    if tfidf_matrix is None or len(message_history) == 0:
        return ""

    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-SEMANTIC_TOP_K:][::-1]
    relevant_indices = [idx for idx in top_indices if similarities[idx] > 0.1]

    if not relevant_indices:
        return ""

    messages_list = list(message_history)
    context_parts = []
    for idx in relevant_indices:
        msg = messages_list[idx]
        author = f"{msg['author']}(b)" if msg.get('is_bot') else msg['author']
        content = msg['content'][:SEMANTIC_MSG_MAX_LEN]
        if len(msg['content']) > SEMANTIC_MSG_MAX_LEN:
            content += "..."
        context_parts.append(f"[{msg['channel']}] {author}: {content}")

    return "\n".join(context_parts)


async def handle_query(message):
    """Handle user query using RAG with semantic search."""
    print(f"[{datetime.now()}] Query from {message.author.name}: {message.content[:50]}")

    try:
        async with message.channel.typing():
            # Gather context.
            recent_bot = get_recent_bot_channel_context()
            recent_other = get_recent_other_channels_context()
            semantic = search_messages_semantic(message.content)

            # Generate and send response.
            response = generate_response(message.content, recent_bot, recent_other, semantic)

            # Send response in chunks if needed.
            if len(response) > 2000:
                chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
                await message.reply(chunks[0])
                for chunk in chunks[1:]:
                    await message.channel.send(chunk)
            else:
                await message.reply(response)

        print(f"[{datetime.now()}] Response sent")

    except Exception as e:
        print(f"[{datetime.now()}] ERROR: {str(e)}")
        await message.reply(f"Error: {str(e)}")


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

Respond based on context above. Prioritize bot channel for conversation flow, other channels for server activity, semantic search for broader knowledge."""

    print(f"[{datetime.now()}] Prompt length: {len(prompt)} chars")

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


@bot.command()
async def status(ctx):
    """Show comprehensive system status."""
    # Calculate metrics.
    uptime = str(datetime.now() - bot_start_time).split('.')[0]
    bot_message_count = sum(1 for msg in message_history if msg.get('is_bot', False))
    user_message_count = len(message_history) - bot_message_count
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    avg_tokens_per_call = total_tokens_used / total_api_calls if total_api_calls > 0 else 0

    # Build status message.
    status_msg = f"**System Status**\n```\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"BOT INFO\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"Bot User:        {bot.user.name}#{bot.user.discriminator}\n"
    status_msg += f"Bot ID:          {bot.user.id}\n"
    status_msg += f"Uptime:          {uptime}\n"
    status_msg += f"Latency:         {bot.latency*1000:.1f}ms\n"
    status_msg += f"\n{'='*40}\n"
    status_msg += f"MEMORY & STORAGE\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"RAM Usage:       {memory_mb:.1f} MB\n"
    status_msg += f"Messages Stored: {len(message_history):,} / {MAX_MESSAGES:,}\n"
    status_msg += f"  - User msgs:   {user_message_count:,}\n"
    status_msg += f"  - Bot msgs:    {bot_message_count:,}\n"
    status_msg += f"Storage Used:    {len(message_history)/MAX_MESSAGES*100:.2f}%\n"
    status_msg += f"\n{'='*40}\n"
    status_msg += f"TF-IDF CONFIGURATION\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"Status:          {'Active' if tfidf_matrix is not None else 'Not Built'}\n"

    if tfidf_matrix is not None:
        status_msg += f"Matrix Shape:    {tfidf_matrix.shape}\n"
        status_msg += f"Vocabulary Size: {len(tfidf_vectorizer.vocabulary_):,}\n"
        if last_tfidf_rebuild:
            time_since = str(datetime.now() - last_tfidf_rebuild).split('.')[0]
            status_msg += f"Last Rebuild:    {time_since} ago\n"

    status_msg += f"Max Features:    5,000\n"
    status_msg += f"N-gram Range:    (1, 3)\n"
    status_msg += f"Max Doc Freq:    0.95\n"
    status_msg += f"Rebuild Interval: 1 hour\n"
    status_msg += f"\n{'='*40}\n"
    status_msg += f"LLM CONFIGURATION\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"Provider:        Groq\n"
    status_msg += f"Model:           openai/gpt-oss-20b\n"
    status_msg += f"Temperature:     1.0\n"
    status_msg += f"Max Tokens:      8192\n"
    status_msg += f"Reasoning Effort: medium\n"
    status_msg += f"\n{'='*40}\n"
    status_msg += f"TOKEN USAGE\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"Total API Calls:     {total_api_calls:,}\n"
    status_msg += f"Prompt Tokens:       {total_prompt_tokens:,}\n"
    status_msg += f"Completion Tokens:   {total_completion_tokens:,}\n"
    status_msg += f"Total Tokens Used:   {total_tokens_used:,}\n"
    status_msg += f"Avg Tokens/Call:     {avg_tokens_per_call:.1f}\n"
    status_msg += f"\n{'='*40}\n"
    status_msg += f"CONTEXT LIMITS\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"Bot Channel Msgs:    {BOT_CHANNEL_LIMIT}\n"
    status_msg += f"Other Channel Msgs:  {OTHER_CHANNEL_LIMIT}\n"
    status_msg += f"Max Other Channels:  {MAX_OTHER_CHANNELS}\n"
    status_msg += f"Semantic Top K:      {SEMANTIC_TOP_K}\n"
    status_msg += f"\n{'='*40}\n"
    status_msg += f"SERVER INFO\n"
    status_msg += f"{'='*40}\n"
    status_msg += f"Connected Servers: {len(bot.guilds)}\n"

    for guild in bot.guilds:
        status_msg += f"  - {guild.name}: {guild.member_count} members\n"

    status_msg += "```\n"

    await ctx.send(status_msg)

    # Send channel breakdown.
    channel_counts = {}
    for msg in message_history:
        channel_name = msg['channel']
        channel_counts[channel_name] = channel_counts.get(channel_name, 0) + 1

    if channel_counts:
        channel_msg = "**Messages by Channel**\n```\n"
        sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)

        for channel, count in sorted_channels[:15]:
            bar_length = int(count / max(channel_counts.values()) * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            channel_msg += f"#{channel[:15]:<15} {bar} {count:,}\n"

        if len(sorted_channels) > 15:
            channel_msg += f"\n... and {len(sorted_channels) - 15} more channels\n"

        channel_msg += "```"
        await ctx.send(channel_msg)


@bot.command()
async def recent(ctx, limit: int = 10):
    """Show recent messages stored."""
    limit = min(limit, MAX_COMMAND_LIMIT)

    if len(message_history) == 0:
        await ctx.send("No messages stored yet.")
        return

    recent_msgs = list(message_history)[-limit:]

    msg = f"**Last {len(recent_msgs)} messages stored:**\n```\n"
    for m in recent_msgs:
        content_preview = m['content'][:50] + "..." if len(m['content']) > 50 else m['content']
        bot_indicator = " (bot)" if m.get('is_bot', False) else ""
        msg += f"[{m['channel']}] {m['author']}{bot_indicator}: {content_preview}\n"
    msg += "```"

    await ctx.send(msg)


@bot.command()
async def search(ctx, *, query: str):
    """Test semantic search without generating a response."""
    if tfidf_matrix is None or len(message_history) == 0:
        await ctx.send("No messages available for search.")
        return

    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]

    messages_list = list(message_history)
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            msg = messages_list[idx]
            author = f"{msg['author']}(b)" if msg.get('is_bot') else msg['author']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            results.append(f"[{msg['channel']}] {author}: {content} (score: {similarities[idx]:.3f})")

    if not results:
        await ctx.send(f"**No relevant results for:** {query}")
        return

    context = "\n\n".join(results)
    context = context[:1900] + "..." if len(context) > 1900 else context

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


# Configure and start the web status server.
print(f"[{datetime.now()}] Configuring web status server...")
web_status.configure(
    bot=bot,
    message_history=message_history,
    tfidf_vectorizer=tfidf_vectorizer,
    tfidf_matrix_getter=get_tfidf_matrix,
    start_time=bot_start_time,
    max_messages=MAX_MESSAGES,
    last_rebuild_getter=get_last_rebuild,
    token_stats_getter=get_token_stats,
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