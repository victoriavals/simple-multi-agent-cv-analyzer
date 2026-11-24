# CV Analyzer project handoff

Use the [project README](../README.md) for setup, CLI examples, and deployment
steps. The notes below describe how the entry points fit together and capture
the evidence from the release checks on 2026-11-24.

## Architecture

~~~mermaid
flowchart LR
    streamlit["Streamlit app.py"] --> workflow["LangGraph workflow"]
    cli["CLI main.py"] --> workflow
    browser["React and Next.js"] --> proxy["Next.js /api/analyze"]
    proxy --> api["FastAPI /analyze"]
    api --> workflow
    workflow --> provider["SumoPod chat completions"]
    workflow --> sources["DDGS or supplied job description"]
~~~

| Area | Source |
| --- | --- |
| Streamlit form, session state, and reports | [app.py](../app.py) and [Streamlit configuration](../.streamlit/config.toml) |
| Next.js form, validation, and loading states | [workspace.tsx](../web/components/workspace.tsx) and [globals.css](../web/app/globals.css) |
| Next.js report tabs and Markdown actions | [report-panel.tsx](../web/components/report-panel.tsx) and [report-body.tsx](../web/components/report-body.tsx) |
| Upload and streaming adapters | [Next.js route](../web/app/api/analyze/route.ts), [FastAPI adapter](../src/http_api.py), and [client contract](../web/lib/analysis.ts) |
| Pipeline state and orchestration | [state.py](../src/state.py) and [workflow.py](../src/graph/workflow.py) |
| CV text and agent steps | [utils.py](../src/utils.py), [CV parser](../src/agents/cv_parser.py), [skill analysis](../src/agents/skill_analyst.py), [market agent](../src/agents/market_intel.py), and [report generation](../src/agents/report_agent.py) |
| Provider and requirements sources | [SumoPod adapter](../src/llm_provider.py), [settings](../src/config.py), and [market search](../src/tools/market_search.py) |
| Local CLI and fixtures | [main.py](../main.py), [sample CV](../samples/sample_cv.txt), [sample job description](../samples/sample_job_description.txt), and [sample report](../samples/sample_output.md) |

The shared workflow extracts readable PDF or UTF-8 text, structures the CV,
identifies explicit and inferred skills, gathers role requirements, and renders
an Indonesian or English Markdown report. The four user-facing progress events
are `reading`, `skills`, `requirements`, and `report`. The workflow creates
one SumoPod client after reading the CV and closes it when the run ends.
Provider failures stop the workflow before the remaining steps.

## Configuration and data handling

| Setting | Behavior |
| --- | --- |
| `API_KEY` | SumoPod credential required for live analysis |
| `MODEL` | SumoPod model ID. The default is `glm-5.3-flash` |
| `BASE_URL` | The default is `https://ai.sumopod.com/v1` |
| `CV_ANALYZER_API_URL` | Server-side Next.js setting. The default is `http://127.0.0.1:8000` |

Python reads nonempty Streamlit secrets first, then environment variables,
then defaults. The Python entry points load the root `.env` with existing
environment variables taking precedence. [Local template](../.env.example) ·
[Cloud Secrets template](../.streamlit/secrets.toml.example) ·
[Next.js proxy template](../web/.env.local.example)

- SumoPod makes hosted chat-completion calls with a 45-second HTTP timeout
  and up to one retry per request. An analysis can make multiple calls.
- CV text and the selected requirements text go to SumoPod. The DDGS query
  uses the target role and year. Git ignores local credential files, and the
  Python process supplies credentials for provider requests.
- `ddgs` searches DuckDuckGo and Yahoo. It takes up to five snippets of 600
  characters each, with an eight-second HTTP timeout, two worker threads per
  search, and two concurrent searches per Python process.
- Public web requirements are cached for six hours with up to 128 entries.
  The key uses the normalized role, current year, model, and endpoint. Each
  entry contains extracted requirements and their public source links.
- `job-description` extracts requirements from the supplied description.
  DDGS failures use a supplied description as fallback and mark the result
  with that source. When neither source is available, the analysis shows
  an actionable error.
- Both web entry points limit uploads to 5 MB and support PDF with embedded
  text or UTF-8 TXT. The CLI also accepts Markdown input. Scanned PDFs need
  OCR to supply readable text.
- Streamlit and the API each limit analysis to two simultaneous runs per
  process. A temporary analysis copy is removed when processing ends, including
  on failure. Streamlit retains the uploader selection and report in the
  user's session. Next.js displays the report in the browser. The CLI persists
  only the requested Markdown output.

## HTTP contract

`GET /health` returns process health from FastAPI. The Next.js frontend sends
multipart form requests to `POST /api/analyze`, which proxies them to Python
`POST /analyze`. Run both services for a Next.js browser smoke test.

| Multipart field | Contract |
| --- | --- |
| `role` | Required target role, up to 150 characters |
| `language` | `indonesia` by default, or `english` |
| `source` | `ddgs` by default, or `job-description` |
| `description` | Required in direct description mode, optional fallback in web mode, up to 15,000 characters |
| `cv` and `demo` | Supply a PDF or TXT file, or set `demo=true` for the bundled sample CV |

Validation errors return JSON with a stable error code. An accepted request
streams newline-delimited JSON with `progress` events and a terminal
`complete` or `error` event. Complete reports include Markdown, role, language,
source (`ddgs` or `job_description`), and `usedFallback`. The Node.js proxy
pipes the response without caching, limits the multipart body to the 5 MiB file
allowance plus 128 KiB of overhead, and uses a 300-second upstream deadline.

## Interface and artwork

The Next.js redesign moved the form into the main page and placed the results
alongside it on desktop. Mobile uses a single column. The form provides field
validation and recovery guidance, live progress, sample results, and report
tabs for the summary, skills, and learning plan. The interface is in Indonesian
and offers both report languages. The Streamlit release uses its native widget
layout with equivalent analysis inputs, progress, sample report, and Markdown
download and copy.

| Asset | Usage |
| --- | --- |
| [cv-analyzer-logo.png](../web/public/cv-analyzer-logo.png) | 384 × 384 transparent ImageGen artwork used in both headers |
| [icon.png](../web/app/icon.png) | 64 × 64 transparent variant of the same artwork. Next.js serves it as App Router icon metadata and Streamlit loads it through `st.set_page_config` |
| [cv-inspection.webp](../web/public/cv-inspection.webp) | Transparent ImageGen illustration for the Next.js hero |

The folded CV and white checkmark communicate document review at favicon
size. The purple tile uses the interface accent `#593BD4`. Next.js uses
Manrope and follows the system reduced-motion preference for report transitions.
When replacing the brand mark, update the header asset and the icon together.

<details>
<summary>ImageGen prompt for the logo</summary>

~~~text
Create one original flat app logo for CV Analyzer, a resume and skill analysis web app. On a square transparent canvas, center a single solid violet (#593BD4) rounded-square tile with precise corner radii, occupying about 90% of the canvas width. Inside, draw a crisp bold white folded-corner document symbol, with one short horizontal resume rule and a checkmark integrated into the lower right edge of the document. Make the mark optically balanced and instantly recognizable as both a 37 px website header logo and a 16 px browser favicon. Use two solid colors, generous whitespace, and clean vector-like geometry. No words, initials, gradient, texture, shadows, lighting, border, frame, or extra graphics. Transparent outside the tile.
~~~

</details>

## Release verification

The [production QA report](QA-PRODUCTION.md) records the later test matrix,
browser coverage, and defects to address before release.

| Gate | Evidence recorded on 2026-11-24 |
| --- | --- |
| Community Cloud dependency parity | Installed the root requirements in a clean Debian 11 environment with Python 3.11.14. All 20 offline tests passed |
| Next.js and Python API | Production build, lint, and TypeScript checks passed on Windows. Three FastAPI tests passed. A synthetic PDF completed through the Next.js proxy with live SumoPod, progress, Markdown download, and copy |
| Streamlit live analysis | Browser test on Debian read SumoPod credentials from Streamlit Secrets. PDF with a direct job description produced English Markdown. TXT in web mode produced Indonesian Markdown through the fallback. Clipboard contents matched the download |
| Source availability | Earlier Windows runs returned DDGS sources. During Debian verification DuckDuckGo was unreachable and Yahoo returned no results. The fallback report showed its source and completed in about 33 seconds |
| Streamlit browser checks | Sample report without a key, required field messages, corrupt PDF handling, 5 MB upload limit, stale report clearing, mobile width, and PNG favicon passed. No temporary analysis directories remained after the runs |
| Next.js UX checks | Desktop and 320 px and 390 px mobile widths, keyboard tab controls, first-error focus, and simulated search recovery were checked. Axe found no violations in the initial and sample-report states checked |
| Resource observation | The Streamlit process peaked at about 177 MiB resident memory in the Debian smoke test |
| Hosted deployment | The smoke test on the actual `.streamlit.app` URL remains pending |

The [test commands](../README.md#testing) run offline unless explicitly using
a live smoke path. Use synthetic inputs for browser checks. Local captures and
reports live under the Git-ignored `outputs/` directory, including
`outputs/ux-redesign/` and `outputs/streamlit-release/`.

For the Cloud release, push a branch containing `app.py`, `src/`, the
requirements, the Streamlit configuration, the samples, and both logo files.
Select Python 3.11 and set the three provider values in Cloud Secrets. Verify
the deployed URL with the sample CV and a direct job description, then repeat
with web search and the description as fallback. The DDGS paths depend on
network access from the deployed instance. [Deployment and smoke checklist](../README.md#deploy-on-streamlit-community-cloud)

## Optional development tools

The project build uses [Python requirements](../requirements.txt) and
[the npm lockfile](../web/package-lock.json). The workstation had the
following global UI tools available during verification.

| Purpose | Tools |
| --- | --- |
| Animation and assets | `motion`, `motion-ai`, `gsap`, `@gsap/react`, `@lottiefiles/dotlottie-react`, `@rive-app/react-webgl2` |
| UI components | `shadcn` and `jsrepo` for registry workflows such as React Bits |
| Browser testing | `@playwright/mcp` and the Playwright MCP tools |
| Design and agent guidance | `skills`, `ui-ux-pro-max-cli`, Motion and shadcn skills, UI UX Pro Max, frontend-design, and React best practices |

Motion is used in the Next.js report transition. The Motion MCP provides
animation references, shadcn MCP provides component guidance, and Playwright
MCP supports browser regression checks.
