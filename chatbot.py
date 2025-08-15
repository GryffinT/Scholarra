import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import numpy as np
import httpx
import plotly.express as px
from urllib.parse import quote
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

if st.session_state.page >= 3:
    main_switch = st.selectbox("Function selection", ["Messager", "Grapher", "Login", "Account Info", "Analytics", "Courses"])
    if main_switch == "Grapher":
        st.session_state.page = 4
    if main_switch == "Messager":
        st.session_state.page = 3
    if main_switch == "Login":
        st.session_state.page = 1
    if main_switch == "Account Info":
        st.session_state.page = 5
    if main_switch == "Analytics":
        st.session_state.page = 6
    if main_switch == "Courses":
        st.session_state.page = 7

if st.session_state.page == 3:
    selection = st.selectbox("AI Mode", ["Standard", "Research (Beta)"])
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
                        model="gpt-5",
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
                        model="gpt-5",
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
    if selection == "Research (Beta)":
        
        # -----------------------------
        # Hidden character injection
        # -----------------------------
        def obfuscate_text(text):
            # inject zero-width spaces randomly between characters
            zwsp = "\u200b"
            return zwsp.join(list(text))
        
        # -----------------------------
        # Topic + type classification
        # -----------------------------
        def classify_topic(user_input):
            
            prompt = (
                f"Classify the topic of this input: '{user_input}'. "
                "Return ONLY main_topic and sub_type separated by a comma. "
                "Example output: History, event"
            )
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}]
            )
            classification = response.choices[0].message.content.strip()
            try:
                main_topic, sub_type = classification.split(",")
                return main_topic.strip(), sub_type.strip()
            except ValueError:
                print("NO TOPIC ERROR")
                return "History", "event"  # fallback
        
        # -----------------------------
        # Build URLs for sources
        # -----------------------------
        def build_urls(user_input, main_topic, sub_type):
            encoded_query = quote(user_input)
            urls = []
            for source_name, variants in SOURCES.get(main_topic, {}).items():
                if sub_type in variants:
                    url = variants[sub_type].replace("[TOPIC]", encoded_query)
                    urls.append(url)
            return urls
        
        # -----------------------------
        # Generate GPT answer
        # -----------------------------
        def answer_user(user_input):
            main_topic, sub_type = classify_topic(user_input)
            urls = build_urls(user_input, main_topic, sub_type)
         # -----------------------------
         # Sources dictionary
         # -----------------------------
        SOURCES = { "MATH": [ ("NIST Digital Library of Mathematical Functions", "https://dlmf.nist.gov/"), ("Encyclopedia of Mathematics (Springer)", "https://encyclopediaofmath.org/"), ("Notices of the American Mathematical Society", "https://www.ams.org/journals/notices/") ],
                    "HISTORY": [ ("Library of Congress Digital Collections", "https://www.loc.gov/collections/"), ("Encyclopaedia Britannica", "https://www.britannica.com/"), ("JSTOR", "https://www.jstor.org/") ],
                    "CHEMISTRY": [ ("IUPAC Gold Book", "https://goldbook.iupac.org/"), ("NIST Chemistry WebBook", "https://webbook.nist.gov/chemistry/"), ("PubChem (NCBI)", "https://pubchem.ncbi.nlm.nih.gov/") ],
                    "BIOLOGY": [ ("NCBI Bookshelf", "https://www.ncbi.nlm.nih.gov/books/"), ("Encyclopedia of Life", "https://eol.org/"), ("PubMed (NLM)", "https://pubmed.ncbi.nlm.nih.gov/") ],
                    "EARTH SCIENCES": [ ("U.S. Geological Survey (USGS)", "https://www.usgs.gov/"), ("National Oceanic and Atmospheric Administration (NOAA)", "https://www.noaa.gov/"), ("NASA Earth Observatory", "https://earthobservatory.nasa.gov/") ], 
                    "COMPUTER SCIENCE": [ ("ACM Digital Library", "https://dl.acm.org/"), ("IEEE Xplore", "https://ieeexplore.ieee.org/"), ("MIT OpenCourseWare (EECS)", "https://ocw.mit.edu/collections/electrical-engineering-computer-science/") ], 
                    "LANGUAGE": [ ("World Atlas of Language Structures (WALS)", "https://wals.info/"), ("Glottolog", "https://glottolog.org/"), ("Linguistic Society of America (LSA)", "https://www.linguisticsociety.org/") ], 
                    "RELIGION": [ ("Oxford Research Encyclopedia of Religion", "https://oxfordre.com/religion"), ("Pew Research Center: Religion & Public Life", "https://www.pewresearch.org/religion/"), ("Stanford Encyclopedia of Philosophy (Philosophy of Religion)", "https://plato.stanford.edu/") ],
                    "GOVERNANCE": [ ("World Bank Worldwide Governance Indicators", "https://info.worldbank.org/governance/wgi/"), ("OECD Public Governance", "https://www.oecd.org/governance/"), ("International IDEA", "https://www.idea.int/") ], 
                    "HEALTH": [ ("World Health Organization (WHO)", "https://www.who.int/"), ("Centers for Disease Control and Prevention (CDC)", "https://www.cdc.gov/"), ("Cochrane Library", "https://www.cochranelibrary.com/") ],
                    "BUSINESS": [ ("Academy of Management Journal", "https://journals.aom.org/journal/amj"), ("Harvard Business Review", "https://hbr.org/"), ("U.S. SEC EDGAR", "https://www.sec.gov/edgar") ], 
                    "ECONOMICS": [ ("National Bureau of Economic Research (NBER)", "https://www.nber.org/"), ("International Monetary Fund — Publications", "https://www.imf.org/en/Publications"), ("Journal of Economic Perspectives (AEA)", "https://www.aeaweb.org/journals/jep") ] }
        
        def answer_user(user_input):
            main_topic, sub_type = classify_topic(user_input)
            urls = build_urls(user_input, main_topic, sub_type)
            
            topic_sources = SOURCES.get(main_topic.upper(), [])
            
            search_instruction = (
                f"Fetch factual information about '{user_input}' from the top 5 most relevant of these sources: {topic_sources}. "
                "If there are no sources, search from only verified academic/scholarly sources. "
                "Synthesize a concise, verbatim, and academic answer, using quotation marks when applicable. "
                "Each answer should have at least 1 quote (<500 words), cite the sources with in-text citation, "
                "and insert hidden characters (zero-width) between letters to prevent direct copy-paste while maintaining text wrap. "
                "It is important that every statement is politically neutral, 100% factually based, cited correctly, "
                "and that each response contains at least 5 quotes from the aforementioned sources, with quotation marks and citation."
            )
            
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": search_instruction}]
            )
            
            return response.choices[0].message.content
            
        def extract_sources(Intake_message):
            generation_instructions = " DO THIS: Scan through this text and locate all sources within it, moreover note down each source used and write a little description on each source" + Intake_message.content
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role":"user", "content":generation_instructions}]
            )
            return response
        
        # -----------------------------
        # Streamlit UI
        # -----------------------------
        st.title("Research Assistant")
        user_input = st.text_input("Ask me about anything:")
        
        if st.button("Get Answer") and user_input.strip():
            with st.spinner("Fetching answer..."):
                try:
                    answer = answer_user(user_input)
                    st.markdown(answer)
                    source_expander = st.expander(label="Sources")
                    with source_expander:
                        source_text = extract_sources(answer)
                        st.write(source_text)
                except Exception as e:
                    st.error(f"Error fetching answer: {e}")
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

if st.session_state.page == 5:
    key_expandable = st.expander(label="Personal Key")
    with key_expandable:
        st.write(st.session_state.key)































