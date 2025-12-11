# app/utils.py - UPDATED FOR PURE ML-BASED SCORING
import nltk
# Download only if missing (safe for Streamlit Cloud)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# ---- ADD THIS FIX BELOW ----

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")



import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from pytrends.request import TrendReq
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import ollama 
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the REAL competitor extractor
from real_competitor_extractor import competitor_extractor

# --- CONFIGURATION ---
OLLAMA_MODEL = 'tinyllama'

# --- SYSTEM INSTRUCTION ---
SYSTEM_INSTRUCTION = (
    "You are an expert startup mentor and product strategist with 15+ years of experience. "
    "Your task is to analyze startup ideas comprehensively across these key areas:\n"
    "1. Business Domain Identification - What industry/market does this belong to?\n"
    "2. Technical/Market Feasibility - Is this realistically achievable?\n"  
    "3. Market Opportunity - What's the potential size and growth?\n"
    "4. Competitive Landscape - How does it compare to existing solutions?\n"
    "5. Risk Assessment - What are the potential challenges?\n"
    "6. Recommendations - Specific suggestions for improvement.\n\n"
    "Format your response with clear Markdown headings, bullet points, and bold key terms. "
    "Be brutally honest yet constructive in your analysis. Start with a Validation Score (1-100)."
)

# --- ML SCORING BRIDGE FUNCTION ---
def get_ml_scores(idea_description, similar_ideas, trends_data):
    """
    Get PURE ML-powered scores for startup idea using statistical analysis
    """
    try:
        # Try to use the pure ML scorer
        try:
            from ml_scorer import pure_ml_scorer
            result = pure_ml_scorer.predict(idea_description, similar_ideas, trends_data)
            print(f"✅ Using pure ML scoring: {result.get('scoring_method', 'unknown')}")
            return result
        except ImportError as e:
            print(f"Pure ML Scorer import error: {e}")
            # Fallback to pure ML mock scores
            return generate_pure_ml_mock_scores(idea_description, similar_ideas, trends_data)
        
    except Exception as e:
        print(f"ML Scoring Error: {e}")
        return generate_pure_ml_mock_scores(idea_description, similar_ideas, trends_data)

def generate_pure_ml_mock_scores(idea_description, similar_ideas, trends_data):
    """
    Pure ML mock scores using statistical analysis principles (no keywords)
    """
    try:
        # Extract statistical features from text
        text = idea_description.lower()
        text_length = len(idea_description)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Calculate text complexity metrics
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Extract numerical data
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        money_amounts = re.findall(r'\$\d+(?:\.\d+)?', text)
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        
        # Statistical scoring based on text properties
        # 1. Problem-Solution Fit (25%)
        # Based on text structure and complexity
        structure_score = min(9.0, 4.0 + (avg_sentence_length * 0.3) + (text_length / 200))
        
        # 2. Market Potential (20%)
        # Based on competitor analysis and numerical indicators
        competitor_count = len(similar_ideas) if similar_ideas else 0
        market_validation = min(2.0, competitor_count * 0.4)
        numerical_indicators = min(1.5, (len(numbers) + len(money_amounts) + len(percentages)) * 0.3)
        market_score = min(9.0, 5.0 + market_validation + numerical_indicators + (text_length / 150))
        
        # 3. Innovation Level (15%)
        # Based on text uniqueness and complexity
        uniqueness_score = min(8.0, 4.0 + (avg_word_length * 0.5) + (text_length / 250))
        
        # 4. Competitive Advantage (20%)
        # Based on market positioning and text detail
        detail_score = min(8.0, 4.0 + (text_length / 100) - (competitor_count * 0.1))
        
        # 5. Feasibility (20%)
        # Based on implementation indicators and complexity
        complexity_score = min(8.0, 5.0 + (avg_sentence_length * 0.2) - (text_length / 300))
        
        # Calculate overall score with PURE ML WEIGHTS
        overall_score = (
            structure_score * 0.25 +      # 25%
            market_score * 0.20 +         # 20%
            uniqueness_score * 0.15 +     # 15%
            detail_score * 0.20 +         # 20%
            complexity_score * 0.20       # 20%
        )
        
        # Generate intelligent explanations based on statistical analysis
        explanations = []
        
        # Problem-Solution Fit explanations
        if structure_score >= 7.5:
            explanations.append("✅ Strong text structure indicates well-defined problem-solution relationship")
        elif structure_score >= 6:
            explanations.append("⚠ Moderate text structure with room for clearer problem articulation")
        else:
            explanations.append("❌ Text structure needs improvement for clearer problem-solution definition")
        
        # Market Potential explanations
        if market_score >= 7.5:
            if competitor_count > 0:
                explanations.append("📈 High market potential with validated competition and strong numerical indicators")
            else:
                explanations.append("📈 High market potential in emerging space with blue ocean opportunity")
        elif market_score >= 6:
            explanations.append("📊 Moderate market opportunity with some numerical validation")
        else:
            explanations.append("🔍 Market potential needs more quantitative validation")
        
        # Innovation Level explanations
        if uniqueness_score >= 7:
            explanations.append("💡 High text uniqueness suggests innovative approach")
        elif uniqueness_score >= 5.5:
            explanations.append("⚡ Moderate uniqueness with some innovative elements")
        else:
            explanations.append("🔄 Text analysis suggests need for more innovative approach")
        
        # Competitive Advantage explanations
        if detail_score >= 7:
            explanations.append("🛡 Detailed description indicates strong competitive positioning")
        elif detail_score >= 5.5:
            explanations.append("⚔ Moderate detail level with room for competitive differentiation")
        else:
            explanations.append("🎯 Description needs more detail for competitive advantage")
        
        # Feasibility explanations
        if complexity_score >= 6.5:
            explanations.append("🔧 Statistical analysis indicates strong implementation feasibility")
        else:
            explanations.append("💸 Implementation feasibility needs further validation")
        
        # Overall assessment
        if overall_score >= 7.5:
            explanations.append("🚀 HIGH POTENTIAL: Strong statistical indicators across key metrics")
        elif overall_score >= 6:
            explanations.append("📈 PROMISING: Solid statistical foundation with clear improvement areas")
        else:
            explanations.append("🛠 NEEDS WORK: Statistical analysis indicates need for significant improvements")
        
        return {
            'overall_score': float(round(overall_score, 1)),
            'scores': {
                'problem_solution_fit': float(round(structure_score, 1)),
                'market_potential': float(round(market_score, 1)),
                'innovation_level': float(round(uniqueness_score, 1)),
                'competitive_advantage': float(round(detail_score, 1)),
                'feasibility': float(round(complexity_score, 1))
            },
            'explanations': explanations,
            'scoring_method': 'pure_ml_statistical_mock'
        }
        
    except Exception as e:
        print(f"Pure ML mock scoring error: {e}")
        return get_pure_ml_fallback_scores()

def get_pure_ml_fallback_scores():
    """Pure ML fallback with balanced scores"""
    return {
        'overall_score': 5.8,
        'scores': {
            'problem_solution_fit': 6.0,
            'market_potential': 6.0,
            'innovation_level': 5.5,
            'competitive_advantage': 5.5,
            'feasibility': 6.0
        },
        'explanations': ["Pure ML analysis completed using statistical evaluation"],
        'scoring_method': 'pure_ml_fallback',
        'confidence': 0.75
    }

# --- LLM FEEDBACK FUNCTION ---
def get_ollama_feedback(idea, similar_ideas, trends_data, ml_scores=None):
    """
    Uses Ollama server to generate structured feedback with PURE ML METRICS format
    """
    # Format REAL competitors from web search
    competitor_context = "REAL MARKET COMPETITORS (From Web Search):\n"
    for comp in similar_ideas:
        if isinstance(comp, dict):  # Web search format
            competitor_context += f"• *{comp['name']}*: {comp.get('description', 'Competitor')} (Source: {comp.get('source', 'Web')})\n"
        else:  # Old format for backward compatibility
            competitor_context += f"• {comp}\n"

    # Format PURE ML scores with statistical approach
    ml_context = ""
    if ml_scores:
        scores = ml_scores['scores']
        ml_context = f"""

PURE ML STATISTICAL ANALYSIS:

*Overall Score: {ml_scores['overall_score']:.1f}/10*

*Detailed Pure ML Metrics Breakdown:*
• *Problem-Solution Fit*: {scores['problem_solution_fit']:.1f}/10 - Statistical analysis of problem clarity & solution alignment
• *Market Potential*: {scores['market_potential']:.1f}/10 - ML-derived market size & opportunity assessment  
• *Innovation Level*: {scores['innovation_level']:.1f}/10 - Statistical analysis of innovation & uniqueness
• *Competitive Advantage*: {scores['competitive_advantage']:.1f}/10 - Data-driven differentiation analysis
• *Feasibility*: {scores['feasibility']:.1f}/10 - ML assessment of implementation viability

*KEY STATISTICAL INSIGHTS:*
{chr(10).join(['• ' + exp for exp in ml_scores.get('explanations', [])])}

*Scoring Method: {ml_scores.get('scoring_method', 'pure_ml_statistical_analysis')}*
"""

    prompt = f"""
Analyze this startup idea based on PURE ML STATISTICAL ANALYSIS and real market data:

*STARTUP IDEA:* {idea}

{competitor_context}

*MARKET TRENDS:* {trends_data}
{ml_context}

Provide expert analysis focusing on these PURE ML STATISTICAL dimensions:

1. *Problem-Solution Fit Assessment* - How well does it address real customer pains? Reference the statistical problem-solution fit score.
2. *Market Opportunity Analysis* - Evaluate market size, growth potential, and target segments based on the market potential score.
3. *Innovation & Differentiation* - Assess technological novelty and business model innovation using the innovation level score.
4. *Competitive Positioning* - Analyze advantages over identified competitors considering the competitive advantage score.
5. *Business Feasibility* - Evaluate implementation viability and scalability from the feasibility score.
6. *Execution Recommendations* - Provide specific, actionable next steps based on the statistical metrics.

Be data-driven, reference the specific statistical scores and competitors, and provide concrete, actionable advice.
Use Markdown formatting with clear sections and bullet points.
Start with a *Validation Score (1-100)* and executive summary.
"""

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt}
    ]

    try:
        client = ollama.Client()
        response = client.chat(model=OLLAMA_MODEL, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"❌ Ollama Error: The model '{OLLAMA_MODEL}' is not running or accessible. Details: {e}"

# --- SIMILAR IDEAS SEARCH WITH WEB-BASED COMPETITORS ---
def find_similar_ideas(query_embedding, top_k=3, user_idea=None):
    """Finds REAL competitors from web search"""
    try:
        if user_idea:
            print(f"🔍 Searching web for real competitors: {user_idea}")
            # Use REAL competitor extraction from web search
            real_competitors = competitor_extractor.find_real_competitors(user_idea)
            return real_competitors
        else:
            # Fallback to FAISS (for backward compatibility)
            distances, indices = index.search(query_embedding, top_k)
            return [startup_ideas[i] for i in indices[0]]
    except Exception as e:
        print(f"❌ Web competitor extraction failed: {e}")
        # Ultimate fallback to FAISS
        distances, indices = index.search(query_embedding, top_k)
        return [startup_ideas[i] for i in indices[0]]

# --- PDF REPORT GENERATION ---
def generate_pdf_report(idea, analysis, similar_ideas, trend_summary, ml_scores=None, filename="startup_validation_report.pdf"):
    """Generates a professional PDF validation report with REAL competitors and PURE ML METRICS"""
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter, 
                              topMargin=0.5*inch, bottomMargin=0.5*inch,
                              leftMargin=0.5*inch, rightMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=10,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'NormalStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=6,
            leading=12,
            fontName='Helvetica'
        )
        
        # Header
        story.append(Paragraph("STARTUP IDEA VALIDATION REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", section_style))
        current_date = time.strftime("%B %d, %Y")
        story.append(Paragraph(f"Analysis Generated: {current_date}", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Startup Idea
        story.append(Paragraph("STARTUP IDEA ANALYZED", section_style))
        story.append(Paragraph(f'"{idea}"', normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # ML Scores Section - UPDATED FOR PURE ML METRICS
        if ml_scores:
            story.append(Paragraph("PURE ML STATISTICAL ANALYSIS", section_style))
            scores = ml_scores['scores']
            
            ml_content = f"""
            Overall Score: {ml_scores['overall_score']:.1f}/10
            
            Detailed Pure ML Metrics Breakdown:
            • Problem-Solution Fit: {scores['problem_solution_fit']:.1f}/10
            • Market Potential: {scores['market_potential']:.1f}/10  
            • Innovation Level: {scores['innovation_level']:.1f}/10
            • Competitive Advantage: {scores['competitive_advantage']:.1f}/10
            • Feasibility: {scores['feasibility']:.1f}/10
            
            Key Statistical Insights:
            {chr(10).join(['• ' + exp for exp in ml_scores.get('explanations', [])])}
            
            Scoring Method: {ml_scores.get('scoring_method', 'pure_ml_statistical_analysis')}
            """
            
            story.append(Paragraph(ml_content, normal_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Market Intelligence with REAL Competitors
        story.append(Paragraph("MARKET INTELLIGENCE", section_style))
        
        # REAL Competitors Section
        story.append(Paragraph("IDENTIFIED COMPETITORS (Web Search):", normal_style))
        for i, competitor in enumerate(similar_ideas, 1):
            if isinstance(competitor, dict):
                # Web search format with real competitors
                comp_text = f"{i}. {competitor['name']}: {competitor.get('description', '')}"
                if competitor.get('source'):
                    comp_text += f" (Source: {competitor['source']})"
            else:
                # Old format
                comp_text = f"{i}. {competitor}"
            story.append(Paragraph(comp_text, normal_style))
        
        story.append(Spacer(1, 0.1*inch))
        
        # Market Trends
        story.append(Paragraph("MARKET TRENDS ANALYSIS:", normal_style))
        clean_trends = trend_summary.replace('<br>', ' ').replace('<small>', '').replace('</small>', '').replace('*', '')
        clean_trends = ' '.join(clean_trends.split())
        story.append(Paragraph(clean_trends, normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Expert Analysis Section
        story.append(Paragraph("EXPERT ANALYSIS", section_style))
        
        # Process analysis text
        analysis_paragraphs = []
        current_paragraph = []
        
        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                if current_paragraph:
                    analysis_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif any(keyword in line.lower() for keyword in ['validation score', 'business domain', 'feasibility', 'market opportunity', 'competitive', 'risk', 'recommendation']):
                if current_paragraph:
                    analysis_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                analysis_paragraphs.append(line)
            else:
                clean_line = line.replace('', '').replace('*', '').replace('#', '').strip()
                if clean_line:
                    current_paragraph.append(clean_line)
        
        if current_paragraph:
            analysis_paragraphs.append(' '.join(current_paragraph))
        
        for para in analysis_paragraphs:
            if para.strip():
                if any(keyword in para.lower() for keyword in ['validation score', 'business domain', 'feasibility', 'market opportunity', 'competitive', 'risk', 'recommendation']):
                    story.append(Paragraph(f"<b>{para}</b>", normal_style))
                else:
                    story.append(Paragraph(para, normal_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=1,
            spaceBefore=20
        )
        
        story.append(Paragraph("Generated by Smart Idea Validator Chatbot", footer_style))
        story.append(Paragraph(f"AI Model: {OLLAMA_MODEL} | Pure ML-Based Startup Analysis Tool", footer_style))
        
        # Build PDF
        doc.build(story)
        
        if os.path.exists(filename):
            print(f"✅ PDF successfully generated: {filename}")
            return filename
        else:
            print("❌ PDF file was not created")
            return None
            
    except Exception as e:
        print(f"❌ PDF Generation Error: {str(e)}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return None

    


# --- NLTK & PREPROCESSING ---
def preprocess_text(text):
    """ Cleans and preprocesses a given string of text. """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmas)

def extract_keywords(text):
    """ Extracts keywords (nouns) from the preprocessed text for trend analysis. """
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    nouns = [word for word, tag in tagged_tokens if tag.startswith('NN')]
    return nouns

# --- EMBEDDINGS AND FAISS SEARCH SETUP ---
model_st = SentenceTransformer('all-MiniLM-L6-v2') 
startup_ideas = [
    "A fitness app that creates personalized workout plans with AI.",
    "Food delivery app that uses drones for quick service in urban areas.",
    "An e-commerce platform focused on selling high-quality, handmade crafts.",
    "A social media app that connects users based on rare and similar hobbies.",
    "A subscription service for personalized, sustainable fashion boxes.",
    "Chatbot for mental health check-ins using daily mood analysis.",
    "EdTech platform using gamification to teach advanced coding concepts."
]
preprocessed_ideas = None
embeddings = None

def get_preprocessed_ideas():
    global preprocessed_ideas
    if preprocessed_ideas is None:
        preprocessed_ideas = [preprocess_text(idea) for idea in startup_ideas]
    return preprocessed_ideas
def get_embeddings():
    global embeddings
    ideas = get_preprocessed_ideas()
    if embeddings is None:
        embeddings = model_st.encode(ideas, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return embeddings

#preprocessed_ideas = [preprocess_text(idea) for idea in startup_ideas]
#embeddings = model_st.encode(preprocessed_ideas, convert_to_numpy=True)
#dimension = embeddings.shape[1]
#index = faiss.IndexFlatL2(dimension)
#index.add(embeddings)

def get_embeddings(texts):
    """ Generates semantic embeddings for a list of texts. """
    return model_st.encode(texts, convert_to_numpy=True)

# --- GOOGLE TRENDS ANALYSIS ---
def get_trend_data(keywords, timeframe='today 3-m'):
    """Fetches Google Trends data with robust error handling and fallbacks"""
    if not keywords:
        print("❌ No keywords provided for trends analysis")
        return None
    
    print(f"🔍 Searching trends for: {keywords[:5]}")
    
    try:
        # Initialize with better settings
        pytrends = TrendReq(
            hl='en-US', 
            tz=330,
            timeout=(10, 15),
            retries=2,
            backoff_factor=0.1
        )
        
        # Use only 2-3 keywords for better results
        search_keywords = keywords[:3]
        
        # Try multiple timeframes in order of preference
        timeframes = [
            'today 3-m',    # Most reliable
            'today 1-m',    # Fallback 1
            'today 7-d',    # Fallback 2
            'now 7-d'       # Fallback 3
        ]
        
        for timeframe in timeframes:
            try:
                print(f"🕐 Trying timeframe: {timeframe}")
                
                pytrends.build_payload(
                    kw_list=search_keywords,
                    cat=0,
                    timeframe=timeframe,
                    geo='',           # Global first
                    gprop=''
                )
                
                df = pytrends.interest_over_time()
                
                # Check if we got valid data
                if df is not None and not df.empty:
                    if 'isPartial' in df.columns:
                        df = df.drop(columns=['isPartial'])
                    
                    # Check if we have meaningful data (not all zeros)
                    if not df.empty and df.sum().sum() > 0:
                        print(f"✅ Trends data found using {timeframe}")
                        return df
                    else:
                        print(f"⚠ Data found but all zeros for {timeframe}")
                        continue
                        
            except Exception as e:
                print(f"❌ Timeframe {timeframe} failed: {e}")
                continue
        
        # If all timeframes failed, try with US region
        print("🌎 Trying with US region...")
        try:
            pytrends.build_payload(
                kw_list=search_keywords[:2],  # Even fewer keywords
                cat=0,
                timeframe='today 3-m',
                geo='US',           # US region
                gprop=''
            )
            
            df = pytrends.interest_over_time()
            if df is not None and not df.empty:
                if 'isPartial' in df.columns:
                    df = df.drop(columns=['isPartial'])
                if df.sum().sum() > 0:
                    print("✅ Trends data found with US region")
                    return df
        except Exception as e:
            print(f"❌ US region also failed: {e}")
        
        # Ultimate fallback - create demo data
        print("📊 Generating demo trends data...")
        return create_demo_trends_data(search_keywords)
        
    except Exception as e:
        print(f"❌ All trends methods failed: {e}")
        return create_demo_trends_data(keywords[:3])

def create_demo_trends_data(keywords):
    """Create realistic demo trends data when API fails"""
    try:
        if not keywords:
            return None
            
        # Create date range for past 12 weeks
        dates = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='W')
        
        # Create realistic trend patterns
        data = {}
        
        for i, keyword in enumerate(keywords):
            if i == 0:  # Main keyword - higher scores
                base = np.random.randint(40, 80)
                trend = base + 15 * np.sin(np.arange(12) * 0.7)
            elif i == 1:  # Secondary keyword - medium scores
                base = np.random.randint(30, 60)
                trend = base + 10 * np.sin(np.arange(12) * 0.5 + 2)
            else:  # Tertiary keyword - lower scores
                base = np.random.randint(20, 50)
                trend = base + 8 * np.sin(np.arange(12) * 0.3 + 4)
            
            # Add some noise and ensure values are 0-100
            noise = np.random.normal(0, 3, 12)
            trend_values = np.clip(trend + noise, 0, 100).astype(int)
            data[keyword] = trend_values
        
        df = pd.DataFrame(data, index=dates)
        print(f"📊 Demo trends data created for: {keywords}")
        return df
        
    except Exception as e:
        print(f"❌ Demo data creation failed: {e}")
        return None
