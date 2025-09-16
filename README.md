# AI CV Analyzer (LangChain + LangGraph + Streamlit)

A production-leaning multi-agent app to analyze a candidate CV against a target role, extract skills, compare with market demand, and produce a tidy Markdown report (English or Indonesian).

## Features
- Real LLM parsing (Gemini/Mistral) for structured CV extraction (no mocks)
- Market intelligence from real web search (Tavily) + LLM synthesis
- Fixed bilingual Markdown sections with strengths and gaps tables
- Orchestration via LangGraph state machine
- Inputs: CV `.pdf` or `.txt`, target role, language
- Outputs: clean Markdown on screen + downloadable `.md`

## Prerequisites
- Python 3.10+
- Required API keys (no fallbacks):
  - Gemini: GEMINI_API_KEY (or GOOGLE_API_KEY), optional GEMINI_MODEL (default `gemini-2.0-flash`)
  - Mistral: MISTRAL_API_KEY, optional MISTRAL_MODEL (default `mistral-large-latest`)
  - Tavily: TAVILY_API_KEY

You can provide keys in a local `.env` file or Streamlit secrets.

Example `.env`:

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-large-latest
TAVILY_API_KEY=tvly-...
```

## Setup
```bash
python -m venv .venv  # Windows: .venv\Scripts\activate
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Streamlit App
Run locally:

```bash
streamlit run app.py
```

Sidebar inputs:
- Target role (text)
- Language (english | indonesia)
- LLM Provider: Auto | Gemini | Mistral
- Upload CV (.pdf or .txt)
- Optional: Use sample CV (samples/cv.pdf)

Output:
- Tidy Markdown report with fixed sections
- Download button to save as `.md` (filename includes role and language)

### CLI usage
Run the analyzer from command line:

```bash
python main.py --cv samples/cv.pdf --role "Senior AI Engineer" --provider auto --out report.md
```

Provider choices: `auto | gemini | mistral`.

### Streamlit Cloud
Add the following secrets:

```
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-2.0-flash"
MISTRAL_API_KEY = "..."
MISTRAL_MODEL = "mistral-large-latest"
TAVILY_API_KEY = "..."
```

Notes:
- The app requires real API keys. It will error early if keys are missing.
- Market skills are synthesized from Tavily results; no static fallback remains.
- Auto provider tries: Gemini → Mistral, failing over if a provider can't initialize or call.

Note: OpenAI/Azure support has been removed per project scope.
