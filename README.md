# Smart Idea Validator Pro 💡🚀



An AI-powered startup idea validation chatbot that gives entrepreneurs real-time market intelligence, ML-based scoring, live competitor discovery via web search, and downloadable PDF reports — all through a conversational Streamlit interface. Reduces traditional validation from **weeks to under 30 seconds**.

---

## 🎯 Goal

Help aspiring founders validate their startup ideas by:

- Identifying real market competitors through live web search (SerpAPI + DuckDuckGo)
- Scoring the idea across 4 key dimensions using ML models
- Analyzing market trends via Google Trends (PyTrends)
- Generating expert-level narrative analysis using a local LLM (TinyLlama via Ollama)
- Producing a downloadable professional PDF validation report

> "Democratizing access to expert-level validation capabilities that apply state-of-the-art artificial intelligence to de-risk ventures from day one." — Project Abstract

---

## 🧩 Key Technologies

| Component | Tool Used | Justification |
|---|---|---|
| UI / Frontend | Streamlit (Python) | Rapid development of data-intensive web apps; native Python integration |
| LLM Analysis | TinyLlama via Ollama (local) | Local, private, cost-free inference; no data leaves the host machine |
| Competitor Search | SerpAPI + DuckDuckGo fallback | Structured JSON interface for Google results; bypasses anti-bot issues |
| Market Trends | PyTrends (Google Trends API) | Unique temporal market interest data not available in static databases |
| Semantic Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) | 384-dim vectors; optimal balance of accuracy and computational efficiency |
| ML Scoring | Scikit-learn (RandomForestRegressor + rule-based) | Robust standard algorithms with clear migration path to trained models |
| Vector Similarity Search | FAISS | Efficient high-dimensional similarity search for novelty scoring |
| PDF Generation | ReportLab | Full control over professional report layout with dynamic pagination |
| Text-to-Speech | pyttsx3 (offline) | Accessible audio readout of analysis summary |
| NLP Preprocessing | NLTK + TF-IDF + POS Tagging | Key phrase extraction for targeted API queries |

---

## 🔁 How It Works

### 📌 Architecture Overview

<!-- 💡 SUGGESTED IMAGE: Add your system architecture/data flow diagram here -->
<!-- Example: ![System Architecture](assets/screenshots/architecture_diagram.png) -->

```
User submits idea (Streamlit UI)
         │
         ▼
   NLP Module (NLTK)
   ├── Text preprocessing & normalization
   ├── Semantic embedding (all-MiniLM-L6-v2 → 384-dim vector)
   └── Key phrase extraction (TF-IDF + POS tagging)
         │
   ┌─────┴──────────────┐
   ▼                    ▼
Competitor Module    Market Trends Module
(SerpAPI / DDG)      (PyTrends)
Real-time Google     Search volume
search results       time-series data
   └─────┬──────────────┘
         ▼
   ML Scoring Engine
   ├── Novelty Score      = 1 − max(cosine_similarity(idea, competitors))
   ├── Market Opportunity = 0.6 × trend_momentum + 0.4 × market_volume
   ├── Competitive Score  = inverse_normalization(competitor_density)
   └── Overall Score      = 0.3×Novelty + 0.4×Market + 0.3×Competitive
         │
         ▼
   TinyLlama (Ollama) — Local LLM Synthesis
   Generates: Executive Summary, Strengths & Opportunities,
              Risks & Challenges, Actionable Recommendations
         │
    ┌────┴──────┐
    ▼           ▼
Chat Response  PDF Report
(Streamlit)    (ReportLab)
                + TTS Readout (pyttsx3)
```

**The full pipeline in sequence:**

1. 🧠 User submits a startup idea through the fixed-position Streamlit chat input
2. 🔤 **NLP Module** preprocesses text, generates a 384-dim semantic embedding, and extracts key phrases
3. 🌐 **Competitor Search** — SerpAPI queries Google for real competing companies; DuckDuckGo as fallback; semantic relevance filtering removes false positives
4. 📈 **Market Trends** — PyTrends retrieves 12-month relative interest data; exponential backoff handles API instability; 24hr file-cache reduces redundant calls
5. 🤖 **ML Scoring** — rule-based engine (compatible with RandomForestRegressor) produces 4 dimensional scores and an Overall Validation Score (0–10)
6. 💬 **LLM Synthesis** — TinyLlama (via Ollama, running locally) generates a structured expert narrative from quantitative scores + raw data
7. 🗣️ **TTS Readout** — pyttsx3 speaks a summary score aloud
8. 📄 **PDF Export** — ReportLab compiles the full multi-section report for download

---

## 📁 Project Structure

```
smart-idea-validator/
│
├── app/
│   ├── chatbot.py                    # Main Streamlit app, UI orchestration
│   ├── utils.py                      # Embeddings, trends, PDF gen, LLM prompt
│   ├── ml_scorer.py                  # ML scoring class (RandomForest + rule-based)
│   ├── real_competitor_extractor.py  # SerpAPI + DuckDuckGo competitor search
│   └── assets/                       # Logo images
│       ├── smart_idea_logo.png
│       ├── magnify.jpg
│       ├── external.png
│       ├── ml_score.png
│       ├── dashboard_logo.png
│       ├── valid.jpg
│       ├── detail_expert.png
│       └── summary.png
│
├── models/
│   └── startup_scorer.joblib         # Pre-trained ML model (auto-generated on first train)
│
├── .env                              # API keys (SERPAPI_KEY)
└── requirements.txt
```

---

## 🛠️ Code Highlights

```python
# Step 1 – NLP: extract keywords and generate semantic embedding
keywords = extract_keywords(user_input)          # TF-IDF + POS tagging
embedding = model_st.encode([user_input])        # 384-dim MiniLM vector

# Step 2 – Real competitor discovery via SerpAPI
real_competitors = search_competitors_serpapi(user_input)

# Step 3 – Market trend data with caching and retry logic
trend_df = get_trend_data(keywords)              # PyTrends, 12-week window

# Step 4 – ML scoring across 4 dimensions
ml_results = get_ml_scores(user_input, real_competitors, trend_summary)
# Returns: novelty, market_opportunity, competitive_landscape, feasibility + overall

# Step 5 – LLM expert narrative (local TinyLlama via Ollama)
raw_analysis = get_ollama_feedback(user_input, real_competitors, trend_summary, ml_results)

# Step 6 – PDF report generation
pdf_file = generate_pdf_report(idea, analysis, competitors, trend_summary, ml_scores)
```

---

## 📐 Scoring Formula

```
Novelty Score            = 1 − max(cosine_similarity(idea_vector, competitor_vectors))
Market Opportunity Score = 0.6 × trend_momentum + 0.4 × market_volume
Competitive Landscape    = inverse_normalization(competitor_density)

Overall Validation Score = (0.3 × Novelty) + (0.4 × Market Opportunity) + (0.3 × Competitive Landscape)
```

---

## 📊 Sample Output

```
Idea Submitted:     "AI-Powered Personal Tutor for Programming"

Competitors Found:  Codecademy · Educative · Coursera · freeCodeCamp (7 total)
Market Trend:       🔥 High Demand — 45% YoY growth in "programming tutor" searches

ML Scores:
  Novelty:              8.4 / 10
  Market Opportunity:   8.1 / 10
  Competitive Edge:     7.2 / 10
  Feasibility:          8.9 / 10
  ──────────────────────────────
  Overall Score:        8.2 / 10   ✅ Viable Opportunity

Processing time:    ~28.7 seconds
TTS Output:         "Analysis complete. Overall score: 8.2 out of 10."
PDF Report:         startup_validation_1718293847.pdf  ✅
```

---

## 🖼️ Screenshots & Dashboard

> **To contributors:** Capture the following screenshots from your running app and add them to `assets/screenshots/`. Then uncomment the lines below.

```markdown
<!-- 1. Homepage / initial chat screen with logo and bot greeting -->
<!-- ![Chat Interface](assets/screenshots/chat_interface.png) -->

<!-- 2. Analysis in progress (spinner steps visible) -->
<!-- ![Processing](assets/screenshots/processing_steps.png) -->

<!-- 3. Full chat response with score breakdown -->
<!-- ![Validation Report](assets/screenshots/validation_report.png) -->

<!-- 4. Score summary section (overall score card + progress bars) -->
<!-- ![Score Cards](assets/screenshots/score_cards.png) -->

<!-- 5. Competitor results with clickable company URLs -->
<!-- ![Competitor Results](assets/screenshots/competitor_results.png) -->

<!-- 6. Downloaded PDF — executive summary page -->
<!-- ![PDF Report](assets/screenshots/pdf_report.png) -->
```

**6 screenshots to capture for a complete README:**

| # | What to capture | Where in the app |
|---|---|---|
| 1 | Homepage with logo + bot greeting + validation component cards | On fresh load before any input |
| 2 | Processing spinners in action | Right after submitting an idea |
| 3 | Full chat response with scores rendered in chat bubbles | After analysis completes |
| 4 | Score summary section — 3-column layout with score card and PDF button | Scroll down after analysis |
| 5 | Competitor results section — company names + URLs | Bottom of page after analysis |
| 6 | One page of the generated PDF report | Open the downloaded PDF |

---

## 🧪 Test Results

Tested across **25 startup archetypes** in three categories:

| Archetype | Example Idea | Score | Outcome |
|---|---|---|---|
| High-Feasibility | AI-Powered Personal Tutor (programming) | 8.2/10 | Viable, growing market; 7 competitors identified |
| Blue Ocean | Blockchain supply chain for small farmers | 7.8/10 | High novelty (9.2), limited direct competitors |
| Concept Validation | Subscription meals for pet rocks | 2.1/10 | Zero search volume — correctly flagged as high-risk |

### Key Performance Metrics

| Metric | Value |
|---|---|
| Avg. end-to-end processing time | 28.7s (σ = 4.2s) |
| 95th percentile completion | < 35 seconds |
| NLP concept accuracy | 94% (vs. 67% basic keyword extraction) |
| SerpAPI competitor relevance | 92% success; 89% false-positive filtering |
| PyTrends data retrieval (with cache) | 92% effective success rate |
| ML scoring test-retest reliability | r = 0.91 |
| Peak memory usage | ~2.3 GB (mostly LLM inference) |
| Users who found new blind spots | 87% of 15 early-stage founders tested |
| Users preferring unified interface | 94% over multiple separate tools |

---

## 🚧 Project Status

This is a **working end-to-end prototype** running locally. Core functionality is fully operational:

- ✅ Live competitor search via SerpAPI + DuckDuckGo fallback
- ✅ Real-time market trend analysis with exponential backoff + 24hr caching
- ✅ ML-powered 4-dimension scoring (rule-based, scikit-learn compatible)
- ✅ Local LLM expert analysis (TinyLlama via Ollama — fully private, no cloud)
- ✅ PDF report download (ReportLab)
- ✅ Voice readout (TTS via pyttsx3)
- ✅ Session state management (Streamlit)
- 🔄 Full Gradio interface — in progress
- 🔄 Trained RandomForestRegressor — rule-based fallback currently active

---

## 🔮 Future Work (5-Phase Roadmap)

**Phase 1 — ML Model Training**
Train the RandomForestRegressor on real startup outcome data to replace the rule-based scorer and enable adaptive scoring based on current market conditions.

**Phase 2 — Investor Matching System**
Identify relevant VCs, angel investors, and accelerators based on startup domain and maturity level; automate pitch deck optimization suggestions per investor preference.

**Phase 3 — Advanced Visualization**
Replace static report sections with interactive dashboards: competitive density heat maps, predictive trend projections, geographic market opportunity mapping, and scenario sensitivity analysis tools.

**Phase 4 — Expanded Data Integration**
Add multi-platform social media sentiment (Twitter, LinkedIn, Reddit), patent analysis for technological novelty, global funding landscape data, and multilingual NLP for non-English markets.

**Phase 5 — Ecosystem Integration**
Build API interfaces for CRM/project management tools, team collaboration features, milestone tracking, and a community for co-founder and advisor discovery — evolving the tool into a full startup intelligence ecosystem.

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally
- TinyLlama model pulled: `ollama pull tinyllama`
- SerpAPI key (optional — falls back to DuckDuckGo without it)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/smart-idea-validator.git
cd smart-idea-validator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your SerpAPI key in .env
echo "SERPAPI_KEY=your_key_here" > .env

# 4. Start Ollama with TinyLlama
ollama serve
ollama run tinyllama

# 5. Launch the app
streamlit run app/chatbot.py
```

> **No SerpAPI key?** The app works fully — it falls back to DuckDuckGo, then to rule-based competitor suggestions if both fail.

### Key Dependencies

```
streamlit==1.28.0
scikit-learn==1.3.0
sentence-transformers==2.2.2
pytrends==4.9.2
reportlab==4.0.4
faiss-cpu==1.7.4
ollama==0.1.7
pyttsx3
requests
python-dotenv
nltk
numpy
pandas
```

---

## 📚 Research Background

Key references that informed the architecture and methodology:

- Adamson et al. (2024) — LLMs for idea generation in innovation
- Al-Ani & Al-Ani (2023) — Role of AI in entrepreneurship development
- Jha (2024) — AI in business development and startup growth
- Al-Saidi et al. (2025) — AI for startup idea validation: systematic review
- Full references available in the project thesis (BCE497J)

---

## 👨‍💻 Contributors

| Name | Student ID |
|---|---|
| Garima Mishra | 22BAI1153 |
| Amritha K | 22BAI1318 |
| Shifana Mehar | 22BAI1455 |

**Project Guide:** Dr. Thanikachalam V, School of Computer Science and Engineering, VIT Chennai

---

## 📌 Tags

`#StartupValidation` `#TinyLlama` `#Ollama` `#Streamlit` `#SerpAPI` `#MachineLearning` `#NLP` `#FAISS` `#ReportLab` `#SentenceTransformers` `#EntrepreneurshipAI` `#VIT` `#BTech2025`
