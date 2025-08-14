import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import numpy as np
import plotly.express as px
import io
from scipy import stats
import math
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import asyncio
import aiohttp


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Batangas&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Batangas', serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Get the folder where chatbot.py is located
base_path = os.path.dirname(__file__)

logo = [
    os.path.join(base_path, "Scholarra (1).png"),
]


# Track current page
if "page" not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def last_page():
    st.session_state.page -= 1

# ---------------- PAGE 1 ----------------
if st.session_state.page == 1:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(logo[0], use_container_width=True)  # Works with GIFs too

    # Inject Batangas font CSS once
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Batangas&display=swap');
        .batangas-font {
            font-family: 'Batangas', serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use the CSS class on the heading
    st.markdown(
        "<h1 class='batangas-font' style='text-align: center;'>Smarter study starts here.</h1>",
        unsafe_allow_html=True,
    )
    st.button("Next", on_click=next_page)


# ---------------- PAGE 2 ----------------
elif st.session_state.page == 2:
    st.markdown(
        "<h3 style='text-align: center;'>Scholarra is an online, machine learning study supplement for students, by students..</h3>",
        unsafe_allow_html=True
    )
    st.header("")
    st.markdown(
        "<h5 style='text-align: center;'>How do we help students?</h5>",
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown(
        "<h6 style='text-align: center;'>Students can interface with course supplements and various study tools to gain better understanding of class material using adaptive machine learning without the intrusive AI-produced work, stunting their growth.</h6>",
        unsafe_allow_html=True
    )
    st.header("")
    st.markdown(
        "<h5 style='text-align: center;'>Teachers, dont feel too left out!</h5>",
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown(
        "<h6 style='text-align: center;'>By giving students an outlet for independent study, they may produce higher quality work with less teacher guidance, helping them to become more independent, alleviating teacher work load.</h6>",
        unsafe_allow_html=True
    )
    st.header("")
    st.markdown(
        "<h5 style='text-align: center;'>What is Scholarra?</h5>",
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown(
        "<h6 style='text-align: center;'>Scholarra is a student made study tool, meaning it's built with student interests and needs in mind while enforcing academic integrity through its safeguards. Scholarra, our machine learning tutor, powered by openai, is programmed to disallow essay rewriting, and cheating. Nexstats, our graphing and statistics calculator can graph and calculate neccisary statistics for courses such as, AP Biology, AP Psychology, and math courses up to Pre-Calculus!</h6>",
        unsafe_allow_html=True
    )
    st.write("")
    # Scatterplot image
    scatter_path = os.path.join(base_path, "scatter_plot.png")
    st.image(scatter_path, caption="Example scatter plot generated with the Scholistics function")
    st.header("")
    key = st.text_input("Enter use key")
    access_keys = ["pibble67"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("Back", on_click=last_page)
    with col2:
        if key in access_keys or key == "Scholar-EG-01":
            st.button("Next", on_click=next_page)


# ---------------- PAGE 3 (Student Chat) ----------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if st.session_state.page == 3 or st.session_state.page == 4 or st.session_state.page == 5 or st.session_state.page == 6:
    col1, col2, col3, col4 = st.columns(4)
    if col2.button("Grapher"):
        st.session_state.page = 4
    if col3.button("Messager"):
        st.session_state.page = 3
    if col1.button("Login"):
        st.session_state.page = 1

if st.session_state.page == 3:
    selection = st.selectbox("AI Mode", ["Standard", "Scholarly"])
    if selection == "Standard":
        st.title("Scholarra interface")
        st.markdown("""Powered by Open AI APIs""")
    
        # Initialize chat history with system prompt if not exists
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and ethical tutor. Explain concepts clearly and factually, using only scholarly and widely accepted academic sources and guide the user to learn by themselves. "
                        "Do NOT write essays, complete homework, think for the user, or provide opinion/analysis of material or do the user's work. Instead, priorotize encouraging critical thinking and provide hints or explanations, with intext citations and a full sources link set at the bottom.\n\n"
                        "If the user asks you to write an essay or do their homework, politely refuse by saying something like: "
                        "\"I'm here to help you understand the topic better, but I can't do your assignments for you.\"\n\n"
                        "Use a friendly, patient, high-school friendly, and encouraging tone. And also remember to always cite every source used with intext citations and links at the end of each message."
                    )
                }
            ]
    
        # If chat history only contains the system prompt, send initial greeting
        if len(st.session_state.chat_history) == 1:
            with st.spinner("Loading AI tutor..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.chat_history + [{"role": "user", "content": "start"}]
                    )
                    ai_message = response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_message})
                except Exception as e:
                    st.error(f"Error contacting AI: {e}")
    
        user_input = st.chat_input("Ask me something about your coursework...")
    
        # Buttons below the chat input
        if user_input:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
    
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.chat_history
                    )
                    ai_message = response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_message})
                except Exception as e:
                    st.error(f"Error contacting AI: {e}")
    
            # Display chat messages except the system prompt
            for msg in st.session_state.chat_history:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
    if selection == "Scholarly":
        
        # -----------------------------
        # Base URLs by source (with multiple "modes")
        # -----------------------------
        BASE_URLS = {
            "britannica": {
                "general": "https://www.britannica.com",
                "event": "https://www.britannica.com/event",
                "person": "https://www.britannica.com/biography",
                "place": "https://www.britannica.com/place"
            },
            "history_com": {
                "general": "https://www.history.com/topics",
                "event": "https://www.history.com/topics",
                "person": "https://www.history.com/topics/people",
                "place": "https://www.history.com/topics/places"
            },
            "stanford_phil": {
                "general": "https://plato.stanford.edu",
                "event": "https://plato.stanford.edu/search/search.html?query=[TOPIC]",
                "person": "https://plato.stanford.edu/search/search.html?query=[TOPIC]",
                "place": "https://plato.stanford.edu/search/search.html?query=[TOPIC]"
            },
            "iep": {
                "general": "https://iep.utm.edu",
                "event": "https://iep.utm.edu/search/?q=[TOPIC]",
                "person": "https://iep.utm.edu/search/?q=[TOPIC]",
                "place": "https://iep.utm.edu/search/?q=[TOPIC]"
            },
            "nature": {
                "general": "https://www.nature.com",
                "event": "https://www.nature.com/search?q=[TOPIC]",
                "person": "https://www.nature.com/search?q=[TOPIC]",
                "place": "https://www.nature.com/search?q=[TOPIC]"
            },
            "sciencedirect": {
                "general": "https://www.sciencedirect.com",
                "event": "https://www.sciencedirect.com/search?qs=[TOPIC]",
                "person": "https://www.sciencedirect.com/search?qs=[TOPIC]",
                "place": "https://www.sciencedirect.com/search?qs=[TOPIC]"
            },
            "jstor": {
                "general": "https://www.jstor.org/action/doBasicSearch?Query=[TOPIC]&so=rel",
                "event": "https://www.jstor.org/action/doBasicSearch?Query=[TOPIC]&so=rel",
                "person": "https://www.jstor.org/action/doBasicSearch?Query=[TOPIC]&so=rel",
                "place": "https://www.jstor.org/action/doBasicSearch?Query=[TOPIC]&so=rel"
            },
            "mathworld": {
                "general": "https://mathworld.wolfram.com",
                "event": "https://mathworld.wolfram.com/search/?query=[TOPIC]",
                "person": "https://mathworld.wolfram.com/search/?query=[TOPIC]",
                "place": "https://mathworld.wolfram.com/search/?query=[TOPIC]"
            },
            "poetryfoundation": {
                "general": "https://www.poetryfoundation.org",
                "event": "https://www.poetryfoundation.org/search?query=[TOPIC]",
                "person": "https://www.poetryfoundation.org/search?query=[TOPIC]",
                "place": "https://www.poetryfoundation.org/search?query=[TOPIC]"
            }
        }
        
        # -------------------------------
        # Topic normalization mapping
        # -------------------------------
        TOPIC_NORMALIZATION = {
            "ww2": "World War 2",
            "world war ii": "World War 2",
            "wwii": "World War 2",
            "ww1": "World War 1",
            "wwi": "World War 1",
            "holocost": "Holocaust",
            "holocaust": "Holocaust",
            "french rev": "French Revolution",
            "french revolution": "French Revolution"
        }
        
        # Build known topics list
        KNOWN_TOPICS = list(TOPIC_NORMALIZATION.values())
        
        # -------------------------------
        # Comprehensive extraction & normalization
        # -------------------------------
        def extract_and_normalize_topic(query, fuzzy_threshold=70, typo_threshold=80):
            """
            Extracts and normalizes a main topic from a user query.
            """
            # Step 1: Preprocess query
            query_clean = query.lower()
            query_clean = re.sub(r'\b(what|who|when|where|is|are|was|were|the|a|an|of)\b', '', query_clean)
            query_clean = re.sub(r'[^\w\s]', '', query_clean)
            query_clean = re.sub(r'\s+', ' ', query_clean).strip()
            
            # Step 2: Normalize using mapping
            normalized_topic = TOPIC_NORMALIZATION.get(query_clean, query_clean.title())
            
            # Step 3: Fuzzy match for typo correction
            best_match = process.extractOne(normalized_topic, KNOWN_TOPICS)
            if best_match:
                match_topic, score, _ = best_match
                if score >= fuzzy_threshold:
                    return match_topic
            
            return normalized_topic
        
        # -------------------------------
        # AI-generated related terms
        # -------------------------------
        def generate_related_terms(user_query, max_terms=5):
            """
            Extracts and normalizes the topic, then generates AI-suggested related terms.
            """
            # Normalize the topic first
            topic = extract_and_normalize_topic(user_query)
            
            prompt = (
                "You are an academic assistant.\n"
                f'Given the topic: "{topic}",\n'
                f"provide up to {max_terms} short, relevant keywords or related topics\n"
                "that could help expand a search for scholarly sources.\n"
                "Reply with a comma-separated list only."
            )
            
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            
            related_text = resp.choices[0].message.content.strip()
            related_terms = [t.strip() for t in related_text.split(",") if t.strip()]
            return related_terms
        
        # -------------------------------
        # Determine query type
        # -------------------------------
        def determine_query_type(term, use_ai=True):
            """
            Determine the query type (Person, Event, Place, General) for a given term.
            
            Args:
                term (str): The topic or query to classify.
                use_ai (bool): If True, attempt AI classification first, fallback to heuristic if needed.
                
            Returns:
                str: One of "Person", "Event", "Place", "General".
            """
            category = None
        
            if use_ai:
                try:
                    # AI-driven classification
                    prompt = (
                        "You are an academic assistant.\n"
                        "Classify the following query into one of these categories: "
                        "Person, Event, Place, General.\n"
                        f"Query: \"{term}\"\n"
                        "Reply with only one word: Person, Event, Place, or General."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "system", "content": prompt}]
                    )
                    category = resp.choices[0].message.content.strip().capitalize()
                except Exception:
                    category = None
        
            # Validate AI response; fallback to heuristic if invalid
            if category not in ["Person", "Event", "Place", "General"]:
                category = get_mode_priority(term)[0].capitalize()  # fallback heuristic
        
            return category

        
        # -------------------------------
        # Construct dynamic URLs
        # -------------------------------

        def sanitize_topic_for_url(topic, is_event=False):
            """
            Prepare topic string for URL:
            - Convert to lowercase
            - Replace spaces with hyphens
            - Remove punctuation
            - Optionally prepend 'The-' for events
            """
            topic = topic.strip()
            topic = topic.lower()
            topic = re.sub(r'\s+', '-', topic)       # spaces -> hyphens
            topic = re.sub(r'[^\w\-]', '', topic)   # remove punctuation
            if is_event:
                topic = "The-" + topic
            return topic
        
        
        def construct_source_url(source_name, topic, mode="general"):
            """
            Construct a URL for any source in BASE_URLS.
            """
            if source_name not in BASE_URLS:
                raise ValueError(f"Source '{source_name}' not found in BASE_URLS.")
            
            # Pick the base URL for the given mode; fallback to general
            base_url = BASE_URLS[source_name].get(mode, BASE_URLS[source_name]["general"])
            
            sanitized_topic = sanitize_topic_for_url(topic, is_event=(mode=="event"))
            
            # Replace [TOPIC] placeholder if present
            if "[TOPIC]" in base_url:
                return base_url.replace("[TOPIC]", sanitized_topic)
            
            # Otherwise, just append the topic
            return f"{base_url}/{sanitized_topic}"

        # -------------------------------
        # Retrieve text chunks from URL
        # -------------------------------
        def retrieve_source_chunks(url, chunk_size=500):
            """
            Fetch text content from a URL and split it into chunks of specified size.
        
            Args:
                url (str): URL to fetch.
                chunk_size (int): Number of words per chunk.
        
            Returns:
                list[str]: List of text chunks.
            """
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()  # Raise HTTPError for bad responses
                
                soup = BeautifulSoup(resp.text, "html.parser")
                paragraphs = soup.find_all("p")
                
                # Extract and clean text
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
                words = text.split()
                
                # Split into chunks
                chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
                return chunks
            
            except Exception as e:
                st.warning(f"Could not fetch content from {url}: {e}")
                return []
                
        # -------------------------------
        # Generate embeddings
        # -------------------------------
        def get_embeddings(texts, model="text-embedding-3-small"):
            """
            Generate embeddings for a list of texts.
            """
            embeddings = []
            for t in texts:
                resp = client.embeddings.create(model=model, input=t)
                embeddings.append(resp.data[0].embedding)
            return np.array(embeddings)
        
        # -------------------------------
        # Semantic search
        # -------------------------------
        def semantic_search(chunks, user_query, top_k=3):
            """
            Return top_k most relevant chunks to the user query using embeddings.
            """
            if not chunks:
                return []
            
            chunk_embeddings = get_embeddings(chunks)
            query_embedding = get_embeddings([user_query])[0].reshape(1, -1)
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            top_indices = similarities.argsort()[::-1][:top_k]
            return [chunks[i] for i in top_indices]
        
        # -------------------------------
        # AI-generated related terms
        # -------------------------------
        def generate_related_terms(query, max_terms=5):
            """
            Generate AI-suggested related terms for a normalized topic.
            """
            topic = extract_and_normalize_topic(query)  # ensure topic is normalized
            
            prompt = (
                "You are an academic assistant.\n"
                f'Given the topic: "{topic}",\n'
                f"provide up to {max_terms} short, relevant keywords or related topics\n"
                "that could help expand a search for scholarly sources.\n"
                "Reply with a comma-separated list only."
            )
            
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            
            related_text = resp.choices[0].message.content.strip()
            return [t.strip() for t in related_text.split(",") if t.strip()]

        # -------------------------------
        # Determine mode priority for a term
        # -------------------------------
        def get_mode_priority(term):
            """
            Heuristic fallback to determine query type.
            Returns a list of modes to try in order of priority.
            """
            term_lower = term.lower()
            words = term.split()
        
            # Event heuristic
            event_keywords = ["war", "revolution", "holocaust", "battle", "conflict"]
            if any(word in term_lower for word in event_keywords):
                return ["event", "general"]
        
            # Person heuristic: at least 2 words, all capitalized
            if len(words) >= 2 and all(w[0].isupper() for w in words if w):
                return ["person", "general"]
        
            # Place heuristic
            place_keywords = ["city", "country", "mountain", "river", "state"]
            if any(word in term_lower for word in place_keywords):
                return ["place", "general"]
        
            # Default
            return ["general"]

        
        # -------------------------------
        # Async fetch function for chunks
        # -------------------------------

        async def fetch_chunks(session, url, source_name):
            """
            Fetch chunks from a URL asynchronously.
            
            Returns a list of dicts with keys: 'text', 'name', 'url'.
            """
            try:
                chunks = await retrieve_source_chunks_async(session, url)
                return [{"text": c, "name": source_name, "url": url} for c in chunks]
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
                return []
        
        
        async def gather_all_chunks(terms):
            """
            Gather all content chunks from all sources concurrently.
            
            Args:
                terms (list[str]): List of normalized topics.
            
            Returns:
                list[dict]: Each dict contains 'text', 'name', and 'url'.
            """
            gathered_chunks = []
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for term in terms:
                    term_normalized = extract_and_normalize_topic(term)  # normalize input
                    modes = get_mode_priority(term_normalized)            # get mode priority
                    
                    for source_name, source_modes in BASE_URLS.items():
                        for mode in modes:
                            if mode not in source_modes:
                                continue
                            url = construct_source_url(source_name, term_normalized, mode=mode)
                            tasks.append(fetch_chunks(session, url, source_name))
                
                results = await asyncio.gather(*tasks)
                for res in results:
                    gathered_chunks.extend(res)
            
            return gathered_chunks
        
        # -------------------------------
        # Scholarly answer generation
        # -------------------------------
        def generate_scholarly_answer(query, max_related_terms=3):
            """
            Generate a scholarly answer to a query using AI, semantic search, and multiple sources.
            
            Steps:
            1. Normalize main topic.
            2. Generate AI-suggested related terms.
            3. Prepare term variants (title case, optional 'The ' prefix for events).
            4. Fetch all source chunks asynchronously.
            5. Perform semantic search to select top relevant chunks.
            6. Fallback if no chunks found.
            7. Combine chunks for GPT context and generate scholarly response.
            """
            # --- Step 1: Normalize main topic ---
            main_topic = extract_and_normalize_topic(query)
        
            # --- Step 2: Generate related terms ---
            related_terms = generate_related_terms(main_topic)[:max_related_terms]
        
            # --- Step 3: Prepare term variants ---
            all_terms = set()
            for term in [main_topic] + related_terms:
                all_terms.add(term.title())
                if any(word in term.lower() for word in ["war", "revolution", "holocaust", "battle", "conflict"]):
                    all_terms.add("The " + term.title())
            all_terms = list(all_terms)
        
            # --- Step 4: Fetch all chunks asynchronously ---
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            gathered_chunks = loop.run_until_complete(gather_all_chunks(all_terms))
        
            # --- Step 5: Semantic search ---
            if gathered_chunks:
                combined_texts = [c["text"] for c in gathered_chunks]
                top_chunks_texts = semantic_search(combined_texts, query)
                gathered_chunks = [c for c in gathered_chunks if c["text"] in top_chunks_texts]
        
            # --- Step 6: Fallback if no content ---
            if not gathered_chunks:
                fallback_urls = [construct_source_url(src, main_topic) for src in BASE_URLS.keys()]
                return (f"No scholarly content found for '{query}'. "
                        f"You can check the sources manually: {', '.join(fallback_urls)}")
        
            # --- Step 7: Combine chunks for GPT context ---
            context_text = "\n\n".join(
                [f"{c['text']} (Source: {c['name']}, {c['url']})" for c in gathered_chunks]
            )
        
            # --- Step 8: GPT prompt ---
            prompt = f"""
        You are an academic assistant. Based on the following source texts, provide a scholarly, factual response to the user's query.
        Include in-text citations and a list of sources at the end.
        
        User Query: {query}
        
        Source Texts:
        {context_text}
        """
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            
            return resp.choices[0].message.content.strip()

        # -------------------------------
        # Streamlit UI
        # -------------------------------
         st.title("Scholarly Assistant")
        
        query = st.text_input("Enter your academic question:")
        
        if query:
            status_placeholder = st.empty()  # Placeholder for dynamic status messages
            with st.spinner("Starting scholarly search..."):
                try:
                    # Step 1: Normalize topic
                    status_placeholder.text("Normalizing your query...")
                    main_topic = extract_and_normalize_topic(query)
        
                    # Step 2: Generate related terms
                    status_placeholder.text("Generating related terms...")
                    related_terms = generate_related_terms(main_topic)
        
                    # Step 3: Prepare all term variants
                    status_placeholder.text("Preparing search terms...")
                    all_terms = set([main_topic.title()])
                    for term in related_terms:
                        all_terms.add(term.title())
                        if any(word in term.lower() for word in ["war", "revolution", "holocaust", "battle", "conflict"]):
                            all_terms.add("The " + term.title())
                    all_terms = list(all_terms)
        
                    # Step 4: Fetch chunks asynchronously
                    status_placeholder.text("Fetching sources...")
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    gathered_chunks = loop.run_until_complete(gather_all_chunks(all_terms))
        
                    # Step 5: Semantic search
                    status_placeholder.text("Performing semantic search...")
                    if gathered_chunks:
                        combined_texts = [c["text"] for c in gathered_chunks]
                        top_chunks_texts = semantic_search(combined_texts, query)
                        gathered_chunks = [c for c in gathered_chunks if c["text"] in top_chunks_texts]
        
                    # Step 6: Fallback check
                    if not gathered_chunks:
                        status_placeholder.text("No relevant sources found, using fallback URLs...")
                        fallback_urls = [construct_source_url(src, main_topic) for src in BASE_URLS.keys()]
                        st.warning(f"No scholarly content found. Check manually: {', '.join(fallback_urls)}")
                    else:
                        # Step 7 & 8: Combine context and generate answer
                        status_placeholder.text("Generating scholarly answer with AI...")
                        context_text = "\n\n".join(
                            [f"{c['text']} (Source: {c['name']}, {c['url']})" for c in gathered_chunks]
                        )
                        prompt = f"""
        You are an academic assistant. Based on the following source texts, provide a scholarly, factual response to the user's query.
        Include in-text citations and a list of sources at the end.
        
        User Query: {query}
        
        Source Texts:
        {context_text}
        """
                        resp = client.chat.completions.create(
                            model="gpt-4-turbo",
                            messages=[{"role": "system", "content": prompt}]
                        )
                        answer = resp.choices[0].message.content.strip()
        
                        status_placeholder.empty()  # Clear status once complete
                        st.markdown("### Scholarly Answer")
                        st.write(answer)
        
                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"An error occurred: {e}")
# ---------------- PAGE 4 (Grapher) ----------------

def parse_xy_input(text):
    points = []
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid format (missing ':'): {pair}")
        x_str, y_str = pair.split(":", 1)
        x = float(x_str.strip())
        y = float(y_str.strip())
        points.append((x, y))
    return points

def calculate_stats(df):
    """Return dict of stats for dataframe with 'x' and 'y' columns"""
    result = {}

    for axis in ["x", "y"]:
        series = df[axis]
        result[f"{axis}_mean"] = series.mean()
        result[f"{axis}_median"] = series.median()
        mode_vals = series.mode()
        if mode_vals.empty:
            result[f"{axis}_mode"] = None
        else:
            result[f"{axis}_mode"] = mode_vals.tolist()
        result[f"{axis}_std"] = series.std()
        result[f"{axis}_sem"] = stats.sem(series) if len(series) > 1 else np.nan

    slope, intercept, r_value, p_value, std_err = stats.linregress(df["x"], df["y"])
    result["regression_slope"] = slope
    result["regression_intercept"] = intercept
    result["regression_r_value"] = r_value
    result["regression_p_value"] = p_value
    result["regression_std_err"] = std_err

    if slope != 0:
        result["x_intercept"] = -intercept / slope
    else:
        result["x_intercept"] = np.nan

    # Correlation coefficient (Pearson's r)
    if len(df) > 1:
        corr_coef, _ = stats.pearsonr(df["x"], df["y"])
        result["correlation_coefficient"] = corr_coef
    else:
        result["correlation_coefficient"] = np.nan

    return result

if st.session_state.page == 4:
    st.title("Scholistics")
    st.header("Next level graphing and statistics calculator.")

    st.write("""
    Enter your data points as comma-separated pairs `x:y`.  
    Example: `1:2, 2:3, 3:5, 4:8`
    """)
    graph_label = st.text_input("Graph label:")
    data_input_1 = st.text_input("Data for Dataset 1 (x:y pairs):")
    data_name_1 = st.text_input("Dataset 1 label:")
    data_input_2 = st.text_input("Data for Dataset 2 (optional, x:y pairs):")
    data_name_2 = st.text_input("Dataset 2 label:")


    x_label = st.text_input("X-axis label:", value="x")
    y_label = st.text_input("Y-axis label:", value="y")

    func_input = st.text_input("Enter a function of x to plot (optional):", value="")

    def safe_eval_func(expr, x_vals):
        # Use numexpr if available for safer evaluation, else fallback
        try:
            y_vals = numexpr.evaluate(expr, local_dict={"x": x_vals, "np": np})
            return y_vals
        except Exception as e:
            # Fallback to eval but warn user
            try:
                # VERY basic safety - only allow 'x', 'np', and math functions from numpy
                allowed_names = {"x": x_vals, "np": np}
                y_vals = eval(expr, {"__builtins__": {}}, allowed_names)
                return y_vals
            except Exception as e2:
                st.error(f"Error evaluating function: {e2}")
                return None

    graph_types = st.multiselect(
        "Select one or more graph types to display:",
        ["Line chart", "Bar chart", "Area chart", "Scatter plot"],
        default=["Line chart"]
    )

    stat_functions = st.multiselect(
        "Select additional statistical calculations:",
        [
            "Mean",
            "Median",
            "Mode",
            "Standard Deviation",
            "Standard Error of the Mean",
            "Linear Regression (slope & intercept)",
            "Statistical Significance (p-value for slope)",
            "X Intercept",
            "Correlation Coefficient",
            "T-test: Compare means of Dataset 1 and Dataset 2"
        ],
        default=[]
    )
    
    # New visualization options for error bars
    visualization_options = st.multiselect(
        "Select additional visual elements on graphs:",
        [
            "Show Standard Deviation as error bars",
            "Show Standard Error of the Mean (SEM) as error bars"
        ],
        default=[]
    )

    calc_on_option = st.selectbox(
        "Calculate statistics on:",
        options=["Dataset 1", "Dataset 2", "Both Combined"],
        index=0
    )

    num_points = st.selectbox(
        "If no data entered in Dataset 1, select number of points to generate:",
        options=list(range(1, 101)),
        index=9
    )

    # Parse Dataset 1 or generate fallback
    if data_input_1.strip():
        try:
            points_1 = parse_xy_input(data_input_1)
            df1 = pd.DataFrame(points_1, columns=["x", "y"])
        except ValueError as e:
            st.error(f"Error parsing Dataset 1 input: {e}")
            st.stop()
    else:
        np.random.seed(42)
        x_vals = list(range(1, num_points + 1))
        y_vals = np.random.randn(num_points).cumsum()
        df1 = pd.DataFrame({"x": x_vals, "y": y_vals})

    # Parse Dataset 2 or None
    if data_input_2.strip():
        try:
            points_2 = parse_xy_input(data_input_2)
            df2 = pd.DataFrame(points_2, columns=["x", "y"])
        except ValueError as e:
            st.error(f"Error parsing Dataset 2 input: {e}")
            st.stop()
    else:
        df2 = None

    if graph_types:
        st.write(f"Displaying {len(graph_types)} graph(s):")

        # Prepare labeled combined dataframe for plotting
        if df2 is not None:
            df1_labeled = df1.copy()
            df1_labeled["dataset"] = data_name_1
            df2_labeled = df2.copy()
            df2_labeled["dataset"] = data_name_2
            df_all = pd.concat([df1_labeled, df2_labeled], ignore_index=True)
        else:
            df_all = df1.copy()
            df_all["dataset"] = data_name_1

        # Calculate stats for each dataset for error bar values
        stats_df1 = calculate_stats(df1)
        stats_df2 = calculate_stats(df2) if df2 is not None else None

        for graph_type in graph_types:
            st.subheader(graph_type)

            # Base figure
            if graph_type == "Line chart":
                fig = px.line(df_all, x="x", y="y", color="dataset", title=graph_label)
            elif graph_type == "Bar chart":
                fig = px.bar(df_all, x="x", y="y", color="dataset", title=graph_label)
            elif graph_type == "Area chart":
                fig = px.area(df_all, x="x", y="y", color="dataset", title=graph_label)
            elif graph_type == "Scatter plot":
                fig = px.scatter(df_all, x="x", y="y", color="dataset", title=graph_label)
            else:
                fig = None
            
                st.markdown(
                    """
                    <style>
                        body {
                            background-color: white;
                        }
                        .stApp {
                            background-color: white;
                        }
                        [data-testid="stSidebar"] {
                            background-color: white;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                       # Define font_size before using it
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    margin=dict(t=80, r=150),
                    title=dict(
                        text=graph_label,
                        x=0.5,
                        xanchor='center',
                        yanchor='top',
                        textfont=dict(
                            size=font_size,
                            color='black'
                        )
                    ),
                    xaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        zerolinecolor='rgba(0,0,0,0.2)',
                        tickangle=45,
                        automargin=True
                    ),
                    yaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        zerolinecolor='rgba(0,0,0,0.2)'
                    ),
                    legend=dict(
                        textfont=dict(color='black')
                    )
                )
            # BOTTOM OF THE CODE FOR CENTERING

            
            if func_input.strip():
                # Use a fine grid of x for smooth plotting of function
                x_min = df_all["x"].min()
                x_max = df_all["x"].max()
                x_func = np.linspace(x_min, x_max, 300)
                y_func = safe_eval_func(func_input, x_func)

                if y_func is not None:
                    # Add function plot trace
                    fig.add_scatter(x=x_func, y=y_func, mode="lines", name="User Function", line=dict(dash="dash", color="black"))

            # Add error bars if selected
            # We'll add constant error bars per dataset, applied on y
            if fig and visualization_options:
                for dataset_name in df_all['dataset'].unique():
                    # Select corresponding stats and data
                    if dataset_name == "Dataset 1":
                        df_sub = df1
                        stats_sub = stats_df1
                    else:
                        df_sub = df2
                        stats_sub = stats_df2

                    if df_sub is None or stats_sub is None:
                        continue

                    # Determine error value (std or SEM)
                    if "Show Standard Deviation as error bars" in visualization_options:
                        err_val = stats_sub["y_std"]
                    elif "Show Standard Error of the Mean (SEM) as error bars" in visualization_options:
                        err_val = stats_sub["y_sem"]
                    else:
                        err_val = None

                    if err_val is None or math.isnan(err_val):
                        continue

                    # Add error bars trace (constant error)
                    fig.add_scatter(
                        x=df_sub["x"],
                        y=df_sub["y"],
                        mode='markers' if graph_type == "Scatter plot" else 'lines+markers',
                        name=f"{dataset_name} Error Bars",
                        error_y=dict(
                            type='constant',
                            value=err_val,
                            visible=True
                        ),
                        marker=dict(symbol='line-ns-open', size=0),  # invisible markers to only show error bars
                        showlegend=False
                    )

            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
                title_x=0.5
            )

            st.plotly_chart(fig, use_container_width=True)

            # Defunct image download
            #img_bytes = fig.to_image(format="png")
            #btn_label = f"Download {graph_type} as PNG"
            #st.download_button(
            #    label=btn_label,
            #    data=img_bytes,
            #    file_name=f"{graph_type.replace(' ', '_').lower()}.png",
            #    mime="image/png"
            #)


        # Statistical calculations display
        if stat_functions:
            st.subheader("Additional Statistical Calculations")

            if calc_on_option == "Dataset 1":
                df_stat = df1
                st.write("Calculating stats on **Dataset 1**")
            elif calc_on_option == "Dataset 2":
                if df2 is None:
                    st.warning("Dataset 2 not provided. Showing Dataset 1 stats instead.")
                    df_stat = df1
                else:
                    df_stat = df2
                    st.write("Calculating stats on **Dataset 2**")
            else:
                if df2 is None:
                    st.warning("Dataset 2 not provided. Showing Dataset 1 stats instead.")
                    df_stat = df1
                else:
                    df_stat = pd.concat([df1, df2], ignore_index=True)
                    st.write("Calculating stats on **Both Datasets Combined**")

            stats_dict = calculate_stats(df_stat)

            # Basic stats table
            rows = []
            if "Mean" in stat_functions:
                rows.append(["Mean", f"{stats_dict['x_mean']:.4f}", f"{stats_dict['y_mean']:.4f}"])
            if "Median" in stat_functions:
                rows.append(["Median", f"{stats_dict['x_median']:.4f}", f"{stats_dict['y_median']:.4f}"])
            if "Mode" in stat_functions:
                x_mode = stats_dict['x_mode']
                y_mode = stats_dict['y_mode']
                x_mode_str = ", ".join(map(str, x_mode)) if x_mode else "No mode"
                y_mode_str = ", ".join(map(str, y_mode)) if y_mode else "No mode"
                rows.append(["Mode", x_mode_str, y_mode_str])
            if "Standard Deviation" in stat_functions:
                rows.append(["Standard Deviation", f"{stats_dict['x_std']:.4f}", f"{stats_dict['y_std']:.4f}"])
            if "Standard Error of the Mean" in stat_functions:
                sem_x = stats_dict['x_sem']
                sem_y = stats_dict['y_sem']
                sem_x_str = f"{sem_x:.4f}" if not math.isnan(sem_x) else "N/A"
                sem_y_str = f"{sem_y:.4f}" if not math.isnan(sem_y) else "N/A"
                rows.append(["Standard Error of the Mean", sem_x_str, sem_y_str])
            if "Correlation Coefficient" in stat_functions:
                corr = stats_dict.get("correlation_coefficient", np.nan)
                corr_str = f"{corr:.4f}" if not math.isnan(corr) else "N/A"
                rows.append(["Correlation Coefficient (Pearson's r)", corr_str, ""])

            if rows:
                stats_df = pd.DataFrame(rows, columns=["Statistic", "X", "Y"])
                st.table(stats_df)
            else:
                st.write("No basic stats selected.")

            # Regression stats
            reg_rows = []
            if "Linear Regression (slope & intercept)" in stat_functions:
                reg_rows.append(f"Slope: {stats_dict['regression_slope']:.4f}")
                reg_rows.append(f"Intercept (Y-intercept): {stats_dict['regression_intercept']:.4f}")
            if "Statistical Significance (p-value for slope)" in stat_functions:
                reg_rows.append(f"P-value for slope: {stats_dict['regression_p_value']:.6f}")
            if "X Intercept" in stat_functions:
                xi = stats_dict["x_intercept"]
                xi_str = f"{xi:.4f}" if not math.isnan(xi) else "Undefined (slope=0)"
                reg_rows.append(f"X-intercept: {xi_str}")

            if reg_rows:
                st.markdown("**Regression Statistics:**")
                for line in reg_rows:
                    st.write(f"- {line}")
            if "Linear Regression (slope & intercept)" in stat_functions:
                reg_rows.append(f"Slope: {stats_dict['regression_slope']:.4f}")
                reg_rows.append(f"Y Intercept: {stats_dict['regression_intercept']:.4f}")

            # T-test comparing means of y-values in Dataset 1 and Dataset 2
            if "T-test: Compare means of Dataset 1 and Dataset 2" in stat_functions:
                if df2 is None:
                    st.warning("Dataset 2 not provided. Cannot perform T-test.")
                else:
                    t_stat, p_val = stats.ttest_ind(df1['y'], df2['y'], equal_var=False)
                    st.write("### T-test: Compare means of Dataset 1 and Dataset 2 (y values)")
                    st.write(f"T-statistic: {t_stat:.4f}")
                    st.write(f"P-value: {p_val:.6f}")

                    alpha = 0.05
                    if p_val < alpha:
                        st.success(f"Reject the null hypothesis at α={alpha}: The means are statistically different.")
                    else:
                        st.info(f"Fail to reject the null hypothesis at α={alpha}: No statistically significant difference between means.")

    else:
        st.info("Select at least one graph type to display.")

# ---------------- PAGE 3-5 (Notes) ----------------

if st.session_state.page >= 3:
    with st.sidebar:
        st.header("Side Notes")

        if "sidebar_note" not in st.session_state:
            st.session_state.sidebar_note = ""

        st.session_state.sidebar_note = st.text_area(
            "Enter your note here",
            value=st.session_state.sidebar_note,
            key="sidebar_note_area",
            height=800
        )

# ---------------- PAGE 5 (User Info) ----------------


































































