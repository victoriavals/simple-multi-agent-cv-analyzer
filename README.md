# AI CV Analyzer

A production-ready multi-agent application that analyzes candidate CVs against target roles using LangChain, LangGraph, and Streamlit. The system extracts skills from resumes, compares them with live market requirements, and generates structured reports in English or Indonesian.

## Overview

This application solves the challenge of matching candidate skills with current market demands through:

- **Intelligent CV Parsing**: Uses LLMs (Gemini/Mistral) to extract structured data from PDF/text resumes, including skills, experience, projects, and education
- **Real-time Market Intelligence**: Fetches live job market data via Tavily search and synthesizes current skill requirements for target roles
- **Skill Gap Analysis**: Compares candidate skills (explicit and implicit) against market demands to identify strengths and development areas
- **Multilingual Reports**: Generates clean, structured Markdown reports in English or Indonesian with consistent formatting

The app is designed for HR professionals, recruiters, and career counselors who need accurate, data-driven insights for candidate assessment and development planning.

## Architecture

The system uses a multi-agent architecture orchestrated by LangGraph:

### 1. CV Parser Agent (`cv_parser.py`)
- **Purpose**: Extracts structured data from raw CV text
- **Input**: PDF/text file content
- **Output**: Structured JSON with name, summary, skills, experience, projects, education
- **Method**: LLM-based parsing with robust fallback to rule-based extraction

### 2. Skill Analyst Agent (`skill_analyst.py`)
- **Purpose**: Analyzes and enriches skill information
- **Input**: Structured CV data
- **Output**: Explicit skills (found in CV) + implicit skills (inferred capabilities)
- **Method**: LLM extraction + heuristic mapping for frameworks/tools

### 3. Market Intelligence Agent (`market_intel.py`)
- **Purpose**: Gathers current market requirements for target roles
- **Input**: Target role name
- **Output**: List of in-demand skills and technologies
- **Method**: Tavily web search + LLM synthesis of job market data

### 4. Report Generator Agent (`report_agent.py`)
- **Purpose**: Creates structured, multilingual analysis reports
- **Input**: CV data, skills analysis, market requirements
- **Output**: Clean Markdown with tables, upskilling plans, and recommendations
- **Method**: Pydantic models for structured data + deterministic Markdown rendering

### LangGraph Workflow (`workflow.py`)
The agents are orchestrated in a sequential pipeline:
```
Load CV → Parse CV → Analyze Skills → Fetch Market Data → Generate Report
```

Each step validates inputs and handles errors gracefully, with state preserved in `PipelineState`.

## Setup

### Prerequisites
- Python 3.10+
- API keys for:
  - **Gemini**: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
  - **Mistral**: `MISTRAL_API_KEY` 
  - **Tavily**: `TAVILY_API_KEY`

### Installation

1. **Create virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API keys:**

Create a `.env` file in the project root:
```env
# Gemini (Google AI)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-pro

# Mistral
MISTRAL_API_KEY=your_mistral_key_here
MISTRAL_MODEL=mistral-large-latest

# Tavily (Market Search)
TAVILY_API_KEY=tvly-your_tavily_key_here
```

**Alternative**: For Streamlit Cloud deployment, add these as secrets in your Streamlit app settings.

## Usage

### CLI Examples

**Basic usage with defaults (Indonesian, auto provider):**
```bash
python main.py --cv samples/sample_cv.txt --role "Senior AI Engineer"
```

**Specify language and provider:**
```bash
python main.py --cv samples/sample_cv.txt --role "Data Scientist" --language english --provider gemini --out data_scientist_report.md
```

**PDF input with custom output:**
```bash
python main.py --cv resume.pdf --role "DevOps Engineer" --language indonesia --provider mistral --out devops_analysis.md
```

**Available options:**
- `--cv`: Path to CV file (.txt or .pdf)
- `--role`: Target job role (e.g., "Senior AI Engineer")
- `--language`: Report language (`english` | `indonesia`, default: `indonesia`)
- `--provider`: LLM provider (`auto` | `gemini` | `mistral`, default: `auto`)
- `--out`: Output file path (default: `report.md`)

### Streamlit Web App

**Start the app:**
```bash
streamlit run app.py
```

**Interface:**
- **Target Role**: Enter the job position to analyze against
- **Language**: Choose between English or Indonesian reports
- **LLM Provider**: Select AI model (Auto recommended)
- **CV Upload**: Upload PDF or text files
- **Demo Mode**: Use included sample CV for testing

The app displays the report on-screen and provides a download button for the exact same content.

## Output Format

### Indonesian Report Structure
```markdown
## Ikhtisar
Brief candidate summary and role fit assessment

## Kekuatan
| Skill | Catatan |
|-------|---------|
| Python | Sesuai dengan kebutuhan pasar |
| Docker | Pengalaman praktis dalam deployment |

## Kesenjangan Keahlian
| Skill | Catatan |
|-------|---------|
| Kubernetes | Diperlukan untuk role senior |
| GraphQL | Teknologi yang berkembang |

## Rencana Upskilling
**Minggu 1 — Dasar & Instalasi**
- Pelajari dasar Kubernetes
- Setup environment development
- Ikuti tutorial resmi

**Minggu 2 — Latihan Inti**
- Deploy aplikasi sederhana
- Praktikkan networking concepts
- Konfigurasi persistent storage

## Catatan Akhir
Summary and next steps for development
```

### English Report Structure
Uses equivalent sections: Overview, Strengths, Skill Gaps, Actionable Upskilling Plan, Final Notes.

### Key Features
- **Consistent Structure**: Same sections every time, no missing parts
- **Tabular Format**: Clean tables for skills comparison
- **Actionable Plans**: Week-by-week development roadmap
- **Screen-Download Parity**: Downloaded file matches displayed content exactly

## Sample Output

See [samples/sample_output.md](samples/sample_output.md) for a complete example report generated from `samples/sample_cv.txt` for the role "Senior AI Engineer" in Indonesian.

### Generate Sample Output

To regenerate the sample output file (useful for testing or after making changes):

```bash
python scripts/generate_sample_output.py
```

This script:
- Reads `samples/sample_cv.txt`
- Analyzes for role "Senior AI Engineer" in Indonesian
- Runs the complete pipeline with auto provider selection
- Writes output to `samples/sample_output.md`
- Exits with code 0 on success, 1 on failure

**Requirements**: All API keys must be configured before running.

## Testing

Run basic smoke test: `bash scripts/smoke_test.sh`

## Troubleshooting

### Common Issues

**Missing API Keys**
```
Error: Missing TAVILY_API_KEY
```
- **Solution**: Add API keys to `.env` file or Streamlit secrets
- **Check**: Ensure key names match exactly (case-sensitive)

**Tavily Empty Results**
```
Error: No Tavily results. Try a different role or check your API key limits.
```
- **Solution**: Try a more specific or common job role
- **Check**: Verify Tavily API key has remaining quota
- **Alternative**: Use different search terms like "Software Engineer" instead of very niche roles

**LLM Provider Errors**
```
Error: LLM init error (gemini): Invalid API key
```
- **Solution**: Verify API key is valid and has sufficient quota
- **Workaround**: Switch to different provider using `--provider mistral`

**CV Parsing Failures**
```
Warning: CV parse error: Failed to extract structured data
```
- **Solution**: Ensure CV file is readable text or properly formatted PDF
- **Check**: Try converting PDF to text first if parsing fails

**Memory Issues with Large Files**
- **Solution**: Keep CV files under 5MB
- **Workaround**: Convert to text format for better processing

### Debug Tips
1. **Check logs**: Enable verbose output with `streamlit run app.py --logger.level=debug`
2. **Test with sample**: Use demo mode to verify setup works
3. **Validate keys**: Test each API independently before running full pipeline
4. **File formats**: Prefer `.txt` files for most reliable parsing

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**Need help?** Check the troubleshooting section above or review the sample files in the `samples/` directory for working examples.
