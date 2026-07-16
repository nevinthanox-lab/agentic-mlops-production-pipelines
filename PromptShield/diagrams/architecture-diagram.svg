<svg viewBox="0 0 1300 900" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#5f6368"/>
    </marker>
    <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#c0392b"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1300" height="900" fill="#ffffff"/>

  <!-- Title -->
  <text x="650" y="34" text-anchor="middle" font-size="22" font-weight="700" fill="#1a1a1a">PromptShield — System Architecture</text>
  <text x="650" y="54" text-anchor="middle" font-size="12.5" fill="#6b6b6b">Ingress → Vector Pre-Filter → Classifier → Fail-Closed CrewAI Quarantine → Persistence &amp; Alerting</text>

  <!-- Band labels -->
  <text x="30" y="98" font-size="11" font-weight="700" fill="#9aa0a6" letter-spacing="1">INGRESS</text>
  <text x="30" y="228" font-size="11" font-weight="700" fill="#9aa0a6" letter-spacing="1">PRE-FILTER + CLASSIFY</text>
  <text x="30" y="418" font-size="11" font-weight="700" fill="#9aa0a6" letter-spacing="1">CREWAI QUARANTINE CREW</text>
  <text x="30" y="778" font-size="11" font-weight="700" fill="#9aa0a6" letter-spacing="1">PERSISTENCE + ALERTING</text>

  <!-- band separators -->
  <line x1="20" y1="112" x2="1280" y2="112" stroke="#eeeeee" stroke-width="1"/>
  <line x1="20" y1="292" x2="1280" y2="292" stroke="#eeeeee" stroke-width="1"/>
  <line x1="20" y1="738" x2="1280" y2="738" stroke="#eeeeee" stroke-width="1"/>

  <!-- 1. Client -->
  <rect x="580" y="80" width="240" height="52" rx="10" fill="#E8F0FE" stroke="#4285F4" stroke-width="1.5"/>
  <text x="700" y="102" text-anchor="middle" font-size="13.5" font-weight="600" fill="#1a1a1a">Client / LLM Application</text>
  <text x="700" y="119" text-anchor="middle" font-size="11" fill="#555">Incoming prompt via n8n Webhook</text>

  <line x1="700" y1="132" x2="700" y2="160" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 2. Chroma pre-filter -->
  <rect x="540" y="160" width="320" height="56" rx="10" fill="#E6F4EA" stroke="#34A853" stroke-width="1.5"/>
  <text x="700" y="183" text-anchor="middle" font-size="13.5" font-weight="600" fill="#1a1a1a">Chroma Vector Pre-Filter</text>
  <text x="700" y="200" text-anchor="middle" font-size="11" fill="#555">601 known-attack embeddings · similarity short-circuit</text>

  <line x1="700" y1="216" x2="700" y2="244" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 3. FastAPI classify -->
  <rect x="520" y="244" width="360" height="56" rx="10" fill="#E6F4EA" stroke="#34A853" stroke-width="1.5"/>
  <text x="700" y="267" text-anchor="middle" font-size="13.5" font-weight="600" fill="#1a1a1a">FastAPI · POST /classify</text>
  <text x="700" y="284" text-anchor="middle" font-size="11" fill="#555">Bake-off winner: DistilBERT (94.67% acc.) · GBM &amp; LogReg baselines</text>

  <line x1="700" y1="300" x2="700" y2="326" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 4. Risk diamond -->
  <polygon points="700,326 790,378 700,430 610,378" fill="#FFF3E0" stroke="#FB8C00" stroke-width="1.5"/>
  <text x="700" y="374" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1a1a1a">Risk Tier?</text>
  <text x="700" y="390" text-anchor="middle" font-size="10.5" fill="#555">low / medium / high</text>

  <!-- low -> allow -->
  <line x1="610" y1="378" x2="330" y2="378" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="470" y="368" text-anchor="middle" font-size="10.5" fill="#5f6368">low</text>
  <rect x="180" y="352" width="300" height="52" rx="10" fill="#F1F3F4" stroke="#9aa0a6" stroke-width="1.5"/>
  <text x="330" y="374" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1a1a">Allow — pass through</text>
  <text x="330" y="390" text-anchor="middle" font-size="10.5" fill="#555">forwarded to LLM unmodified</text>
  <line x1="330" y1="404" x2="330" y2="742" stroke="#9aa0a6" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <text x="215" y="580" font-size="10.5" fill="#9aa0a6" transform="rotate(-90 215 580)">still fully traced</text>

  <!-- medium/high -> triage -->
  <line x1="700" y1="430" x2="700" y2="456" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="740" y="446" text-anchor="middle" font-size="10.5" fill="#5f6368">medium / high</text>

  <!-- Agent 1 Triage -->
  <rect x="560" y="456" width="280" height="58" rx="10" fill="#F3E8FD" stroke="#8E24AA" stroke-width="1.5"/>
  <text x="700" y="480" text-anchor="middle" font-size="13.5" font-weight="600" fill="#1a1a1a">Agent 1 — Triage Analyst</text>
  <text x="700" y="497" text-anchor="middle" font-size="10.5" fill="#555">allow / sanitize / block / escalate · Pydantic-validated</text>

  <line x1="700" y1="514" x2="700" y2="536" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- sanitize branch -->
  <polygon points="1010,470 1080,500 1010,530 940,500" fill="#FFF3E0" stroke="#FB8C00" stroke-width="1.5"/>
  <text x="1010" y="496" text-anchor="middle" font-size="10.5" font-weight="600" fill="#1a1a1a">Sanitize</text>
  <text x="1010" y="510" text-anchor="middle" font-size="9.5" fill="#555">needed?</text>
  <line x1="840" y1="485" x2="940" y2="497" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>
  <rect x="960" y="556" width="220" height="52" rx="10" fill="#F3E8FD" stroke="#8E24AA" stroke-width="1.5"/>
  <text x="1070" y="577" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1a1a1a">Agent 2 — Sanitizer</text>
  <text x="1070" y="593" text-anchor="middle" font-size="10" fill="#555">medium-risk prompt rewrite</text>
  <line x1="1010" y1="530" x2="1010" y2="556" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="1030" y="546" font-size="10" fill="#5f6368">yes</text>
  <path d="M960 582 C 860 582, 840 570, 840 555" fill="none" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Agent 3 Judge -->
  <rect x="560" y="536" width="280" height="58" rx="10" fill="#F3E8FD" stroke="#8E24AA" stroke-width="1.5"/>
  <text x="700" y="560" text-anchor="middle" font-size="13.5" font-weight="600" fill="#1a1a1a">Agent 3 — LLM-Judge</text>
  <text x="700" y="577" text-anchor="middle" font-size="10.5" fill="#555">independent red-team re-review · local Ollama llama3.2</text>

  <line x1="700" y1="594" x2="700" y2="616" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- disposition diamond -->
  <polygon points="700,616 800,668 700,720 600,668" fill="#FFF3E0" stroke="#FB8C00" stroke-width="1.5"/>
  <text x="700" y="662" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1a1a1a">High risk OR</text>
  <text x="700" y="678" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1a1a1a">Judge disagrees?</text>

  <!-- escalate branch -->
  <line x1="800" y1="668" x2="1070" y2="668" stroke="#c0392b" stroke-width="1.5" marker-end="url(#arrowRed)"/>
  <text x="940" y="658" text-anchor="middle" font-size="10.5" fill="#c0392b">yes</text>
  <rect x="960" y="642" width="220" height="52" rx="10" fill="#FCE4E4" stroke="#c0392b" stroke-width="1.5"/>
  <text x="1070" y="663" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1a1a1a">Agent 4 — Escalation</text>
  <text x="1070" y="679" text-anchor="middle" font-size="10" fill="#555">deterministic fallback if LLM fails</text>

  <!-- no -> straight down -->
  <line x1="700" y1="720" x2="700" y2="742" stroke="#5f6368" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="668" y="710" text-anchor="middle" font-size="10.5" fill="#5f6368">no</text>

  <line x1="1070" y1="694" x2="1070" y2="742" stroke="#c0392b" stroke-width="1.5" marker-end="url(#arrowRed)"/>

  <!-- Storage row -->
  <rect x="200" y="742" width="260" height="56" rx="10" fill="#E0F7FA" stroke="#00838F" stroke-width="1.5"/>
  <text x="330" y="765" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1a1a">MongoDB</text>
  <text x="330" y="782" text-anchor="middle" font-size="10.5" fill="#555">full pipeline_traces document</text>

  <rect x="570" y="742" width="260" height="56" rx="10" fill="#E0F7FA" stroke="#00838F" stroke-width="1.5"/>
  <text x="700" y="765" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1a1a">PostgreSQL</text>
  <text x="700" y="782" text-anchor="middle" font-size="10.5" fill="#555">incident_events + daily_metrics_rollup</text>

  <rect x="940" y="742" width="260" height="56" rx="10" fill="#F3E5F5" stroke="#4A154B" stroke-width="1.5"/>
  <text x="1070" y="765" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1a1a">Slack (via n8n)</text>
  <text x="1070" y="782" text-anchor="middle" font-size="10.5" fill="#555">real-time incident alert</text>

  <line x1="460" y1="770" x2="570" y2="770" stroke="#5f6368" stroke-width="1.5" stroke-dasharray="3 3"/>
  <line x1="830" y1="770" x2="940" y2="770" stroke="#5f6368" stroke-width="1.5" stroke-dasharray="3 3"/>

  <!-- Legend -->
  <rect x="20" y="826" width="14" height="14" rx="3" fill="#E6F4EA" stroke="#34A853"/>
  <text x="40" y="837" font-size="10.5" fill="#555">Classification</text>
  <rect x="150" y="826" width="14" height="14" rx="3" fill="#F3E8FD" stroke="#8E24AA"/>
  <text x="170" y="837" font-size="10.5" fill="#555">CrewAI Agent</text>
  <rect x="280" y="826" width="14" height="14" rx="3" fill="#FFF3E0" stroke="#FB8C00"/>
  <text x="300" y="837" font-size="10.5" fill="#555">Decision</text>
  <rect x="390" y="826" width="14" height="14" rx="3" fill="#E0F7FA" stroke="#00838F"/>
  <text x="410" y="837" font-size="10.5" fill="#555">Storage</text>
  <rect x="480" y="826" width="14" height="14" rx="3" fill="#FCE4E4" stroke="#c0392b"/>
  <text x="500" y="837" font-size="10.5" fill="#555">Escalation path</text>

  <text x="1280" y="890" text-anchor="end" font-size="10" fill="#b0b0b0">PromptShield · fail-closed by design — every path terminates in Persistence, never in silence</text>
</svg>
