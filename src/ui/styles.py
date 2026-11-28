from functools import lru_cache


@lru_cache(maxsize=1)
def get_app_css() -> str:
    # apple hig inspired light-mode design system tokens and responsive layout rules
    return """
    <style>
    :root {
        --bg-canvas: #f5f5f7;
        --bg-surface: #ffffff;
        --bg-note: #fbfbfd;
        --text-primary: #1d1d1f;
        --text-secondary: #515154;
        --text-muted: #6e6e73;
        --accent-blue: #0066cc;
        --accent-blue-hover: #0055b3;
        --border-hairline: #d2d2d7;
        --border-subtle: #e5e5ea;
        --status-ok-fg: #15803d;
        --status-warn-fg: #92400e;
        --status-warn-bg: #fffbeb;
        --status-warn-border: #fcd34d;
        --status-err-fg: #b91c1c;
        --radius-card: 16px;
        --radius-notice: 12px;
        --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 14px rgba(0, 0, 0, 0.03);
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stDecoration"] {
        display: none !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
        color: var(--text-primary);
        background-color: var(--bg-canvas) !important;
        -webkit-font-smoothing: antialiased;
        overflow-x: hidden !important;
    }

    /* keep streamlit top utility icons clickable without blocking header nav below */
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] * {
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        border-bottom: none !important;
        height: 2.5rem !important;
        min-height: 0 !important;
        z-index: 110 !important;
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"] *,
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] * {
        pointer-events: auto !important;
    }
    [data-testid="stAppDeployButton"],
    .stAppDeployButton {
        display: none !important;
    }

    .block-container {
        max-width: 1380px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-top: 4.65rem !important;
        padding-bottom: 4.25rem !important;
        padding-left: clamp(1rem, 3vw, 2.75rem) !important;
        padding-right: clamp(1rem, 3vw, 2.75rem) !important;
    }

    /* sticky full-bleed global header shell with brand left and top-level navigation right on desktop */
    div[data-testid="stVerticalBlock"].st-key-global_header_shell {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 95 !important;
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
        background: rgba(245, 245, 247, 0.92) !important;
        backdrop-filter: saturate(180%) blur(12px) !important;
        border-bottom: 1px solid var(--border-hairline) !important;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid="stVerticalBlock"].st-key-global_header_shell > [data-testid="stLayoutWrapper"] {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div.st-key-global_header_inner {
        max-width: 1380px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 8px clamp(1rem, 3vw, 2.75rem) !important;
        box-sizing: border-box !important;
        align-items: center !important;
    }
    .st-key-global_header_inner > [data-testid="stElementContainer"]:first-child {
        width: auto !important;
        flex: 0 1 auto !important;
    }
    /* neutralize streamlit's default -1rem markdown bottom margin so flex row height never collapses */
    .st-key-global_header_inner [data-testid="stMarkdown"],
    .st-key-global_header_inner [data-testid="stMarkdownContainer"],
    .st-key-global_header_inner .stMarkdown,
    .st-key-global_header_inner [data-testid="stHeadingWithActionElements"] {
        margin: 0 !important;
        padding: 0 !important;
        height: auto !important;
    }
    /* hide streamlit heading hover anchor links globally for clean editorial headers */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    h1.global-brand-title {
        font-size: clamp(20px, 1.85vw, 24px) !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.024em !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
    }

    /* workspace intro & region selector strip */
    .scope-lead {
        font-size: 14.5px;
        line-height: 1.5;
        color: var(--text-secondary);
        margin: 0 !important;
        text-align: left;
    }
    div[data-testid="stElementContainer"]:has(.scope-lead) {
        margin-top: 2px !important;
        margin-bottom: -2px !important;
    }
    .region-caption {
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin: 0 !important;
        text-align: left;
    }
    div[data-testid="stElementContainer"]:has(.region-caption) {
        margin-top: 0 !important;
        margin-bottom: -10px !important;
    }

    /* normalize streamlit markdown and heading boxes so every section obeys Gestalt proximity (~32px above headings, ~8px below) */
    .block-container [data-testid="stMarkdownContainer"],
    .block-container [data-testid="stHeadingWithActionElements"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    h2 {
        font-size: clamp(19px, 1.65vw, 22px) !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    h3 {
        font-size: 17.5px !important;
        font-weight: 650 !important;
        line-height: 1.3 !important;
        letter-spacing: -0.016em !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    h4 {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stElementContainer"]:has(h2),
    div[data-testid="stHorizontalBlock"]:has(h2) {
        margin-top: 16px !important;
    }
    div[data-testid="stElementContainer"]:has(h2) {
        margin-bottom: -10px !important;
    }
    div[data-testid="stElementContainer"]:has(h3) {
        margin-top: 18px !important;
        margin-bottom: -8px !important;
    }
    div[data-testid="stElementContainer"]:has(.section-lead) {
        margin-top: 0 !important;
        margin-bottom: -6px !important;
    }

    .section-lead {
        font-size: 14px;
        line-height: 1.5;
        color: var(--text-secondary);
        margin: 0 !important;
    }

    /* quiet two-column context notes strip */
    .notice-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin: 0 !important;
    }
    .notice-stack {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin: 0 !important;
    }
    .notice-bar {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 16px;
        border-radius: var(--radius-notice);
        font-size: 13px;
        line-height: 1.5;
        border: 1px solid var(--border-hairline);
        background-color: var(--bg-note);
        color: var(--text-secondary);
    }
    .notice-bar strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    .notice-bar.notice-advisory {
        background-color: var(--status-warn-bg);
        border-color: var(--status-warn-border);
        color: var(--status-warn-fg);
    }
    .notice-bar.notice-advisory strong {
        color: #78350f;
    }
    .notice-meta {
        color: var(--text-muted);
    }

    /* primary hero kpi cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 0 !important;
    }
    .kpi-card, .metric-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-hairline);
        border-radius: var(--radius-card);
        padding: 16px 18px;
        box-shadow: var(--shadow-card);
        overflow: visible !important;
    }
    .kpi-label, .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.015em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }
    .kpi-value, .metric-value {
        font-size: clamp(22px, 1.9vw, 27px);
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.15;
        color: var(--text-primary);
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
        margin-bottom: 6px;
        word-break: normal !important;
    }
    .kpi-meta, .metric-caption {
        font-size: 12.5px;
        line-height: 1.42;
        color: var(--text-secondary);
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
    }

    /* semantic status text colors */
    .status-chip {
        display: inline-flex;
        align-items: center;
        font-weight: 650;
        font-size: clamp(18px, 1.5vw, 22px);
        line-height: 1.2;
        letter-spacing: -0.015em;
    }
    .status-ok { color: var(--status-ok-fg); }
    .status-warn { color: var(--status-warn-fg); }
    .status-err { color: var(--status-err-fg); }

    /* left-align top region selector and right-align forecast horizon control */
    .st-key-region_control:not([data-retained="true"]),
    .st-key-region_control:not([data-retained="true"]) > div,
    .st-key-region_control:not([data-retained="true"]) [data-testid="stButtonGroup"] {
        display: flex !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }
    .st-key-region_control:not([data-retained="true"]) [role="radiogroup"] {
        margin-left: 0 !important;
        justify-content: flex-start !important;
    }
    .st-key-horizon_ctrl:not([data-retained="true"]),
    .st-key-horizon_ctrl:not([data-retained="true"]) > div,
    .st-key-horizon_ctrl:not([data-retained="true"]) [data-testid="stButtonGroup"] {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
    }
    .st-key-horizon_ctrl:not([data-retained="true"]) [role="radiogroup"] {
        margin-left: auto !important;
        justify-content: flex-end !important;
    }

    [data-testid="stButtonGroup"] [role="radiogroup"] {
        gap: 6px !important;
    }
    [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"],
    [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] {
        margin: 0 !important;
        border-radius: 10px !important;
        border-top-left-radius: 10px !important;
        border-top-right-radius: 10px !important;
        border-bottom-left-radius: 10px !important;
        border-bottom-right-radius: 10px !important;
        font-size: 13px !important;
        border: 1px solid var(--border-hairline) !important;
        border-width: 1px !important;
        border-style: solid !important;
        border-color: var(--border-hairline) !important;
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"]:hover,
    [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"]:hover {
        background-color: #f2f2f7 !important;
        border-color: #b8b8bf !important;
    }
    [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"][aria-checked="true"],
    [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"][data-selected="true"],
    [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"][aria-checked="true"] {
        background-color: #eaf2ff !important;
        border: 1px solid var(--accent-blue) !important;
        border-width: 1px !important;
        border-style: solid !important;
        border-color: var(--accent-blue) !important;
        color: var(--accent-blue) !important;
        box-shadow: inset 0 0 0 0.5px var(--accent-blue) !important;
        font-weight: 600 !important;
    }
    [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"]:focus,
    [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"]:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"][aria-checked="true"]:focus,
    [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"][aria-checked="true"]:focus {
        box-shadow: inset 0 0 0 0.5px var(--accent-blue) !important;
    }
    [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"]:focus-visible,
    [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"]:focus-visible {
        outline: 2px solid var(--accent-blue) !important;
        outline-offset: -2px !important;
    }

    /* top header navigation bar styled cleanly without filter-pill borders */
    div.st-key-nav_control:not([data-retained="true"]),
    div.st-key-nav_control:not([data-retained="true"]) > div,
    div.st-key-nav_control:not([data-retained="true"]) [data-testid="stButtonGroup"] {
        width: auto !important;
        display: flex !important;
        justify-content: flex-end !important;
    }
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] {
        gap: 4px !important;
        background-color: rgba(0, 0, 0, 0.045) !important;
        border-radius: 10px !important;
        padding: 3px !important;
    }
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"],
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] {
        border: none !important;
        border-width: 0 !important;
        border-radius: 8px !important;
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        padding: 6px 13px !important;
        min-height: 30px !important;
        box-shadow: none !important;
        margin: 0 !important;
    }
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"]:hover,
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"]:hover {
        background-color: rgba(255, 255, 255, 0.65) !important;
        border: none !important;
        border-width: 0 !important;
        color: var(--text-primary) !important;
    }
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"][aria-checked="true"],
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"][data-selected="true"],
    div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"][aria-checked="true"] {
        background-color: #ffffff !important;
        border: none !important;
        border-width: 0 !important;
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    /* single clean chart and table surfaces with matching 16px curves and zero height overflow */
    div[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]),
    div[data-testid="stLayoutWrapper"]:has([data-testid="stPlotlyChart"]) {
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        width: 100% !important;
        margin: 0 !important;
        overflow: visible !important;
    }
    div.stPlotlyChart[data-testid="stPlotlyChart"],
    [data-testid="stPlotlyChart"] {
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-hairline) !important;
        border-radius: var(--radius-card) !important;
        padding: 0 !important;
        box-shadow: var(--shadow-card) !important;
        height: auto !important;
        max-height: none !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stPlotlyChart"] > div {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        height: auto !important;
        overflow: visible !important;
    }

    [data-testid="stDataFrame"] {
        margin: 0 !important;
        border-radius: var(--radius-card) !important;
        box-shadow: var(--shadow-card) !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"] > div {
        border-radius: var(--radius-card) !important;
        overflow: hidden !important;
    }

    .details-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-hairline);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-card);
        overflow: hidden;
        margin: 0 !important;
    }
    .detail-row {
        display: grid;
        grid-template-columns: minmax(220px, 0.9fr) 1.5fr;
        gap: 10px 20px;
        padding: 14px 18px;
        border-bottom: 1px solid var(--border-subtle);
        align-items: start;
    }
    .detail-row:last-child {
        border-bottom: none;
    }
    .detail-key {
        font-size: 13.5px;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.48;
    }
    .detail-val {
        font-size: 13.5px;
        color: var(--text-secondary);
        line-height: 1.55;
    }
    .details-card code {
        font-size: 12px;
        background-color: #f2f2f7;
        color: var(--text-primary);
        padding: 2px 6px;
        border-radius: 5px;
    }
    .formula-box {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        line-height: 1.55;
        color: var(--text-primary);
        background-color: #f5f5f7;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 10px 12px;
        margin: 8px 0 10px 0;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .readme-cta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 14px 18px;
        border-radius: var(--radius-card);
        font-size: 13.5px;
        line-height: 1.5;
        border: 1px solid var(--border-hairline);
        background-color: var(--bg-surface);
        color: var(--text-secondary);
        box-shadow: var(--shadow-card);
    }
    .readme-cta strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    .readme-cta a {
        color: var(--accent-blue);
        font-weight: 600;
        text-decoration: none;
    }
    .readme-cta a:hover {
        text-decoration: underline;
    }

    .grid-3 {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 0 !important;
    }
    .grid-2x3 {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 0 !important;
    }
    .spec-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-hairline);
        border-radius: var(--radius-card);
        padding: 16px 18px;
        box-shadow: var(--shadow-card);
    }
    .spec-step {
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent-blue);
        margin-bottom: 4px;
    }
    .spec-title {
        font-size: 15.5px;
        font-weight: 650;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    .spec-body {
        font-size: 13.5px;
        line-height: 1.55;
        color: var(--text-secondary);
    }

    .info-tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background-color: #f2f2f7;
        border: 1px solid var(--border-hairline);
        color: var(--text-secondary);
        font-size: 10.5px;
        font-weight: 600;
        cursor: pointer;
        outline: none;
        position: relative;
        vertical-align: middle;
        line-height: 1;
        flex-shrink: 0;
        transition: all 0.15s ease;
    }
    .info-tip:hover,
    .info-tip:focus,
    .info-tip:focus-visible,
    .info-tip:active {
        background-color: var(--accent-blue);
        border-color: var(--accent-blue);
        color: #ffffff;
    }
    .info-tip:focus-visible {
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.28);
    }
    .info-tip:hover::after,
    .info-tip:focus::after,
    .info-tip:focus-visible::after,
    .info-tip:active::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 145%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #1d1d1f;
        color: #f5f5f7;
        font-size: 12px;
        font-weight: 400;
        line-height: 1.48;
        padding: 9px 12px;
        border-radius: 10px;
        white-space: normal;
        width: 260px;
        z-index: 9999999;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
        pointer-events: none;
        text-transform: none;
        letter-spacing: normal;
    }
    .info-tip.tip-right:hover::after,
    .info-tip.tip-right:focus::after,
    .info-tip.tip-right:focus-visible::after,
    .info-tip.tip-right:active::after {
        left: auto;
        right: -6px;
        transform: none;
    }

    div[data-testid="stDownloadButton"] button {
        background-color: var(--accent-blue) !important;
        border: 1px solid var(--accent-blue) !important;
        border-radius: 10px !important;
        padding: 8px 22px !important;
        margin-top: 6px;
        transition: background-color 0.15s ease !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: var(--accent-blue-hover) !important;
        border-color: var(--accent-blue-hover) !important;
    }
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button * {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
    }

    /* suppress whole streamlit container for mobile-only hints on desktop so zero empty gap remains */
    div[data-testid="stElementContainer"]:has(.chart-mobile-hint),
    .chart-mobile-hint {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
    }

    .app-footer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 95 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: rgba(245, 245, 247, 0.92) !important;
        backdrop-filter: saturate(180%) blur(12px) !important;
        border-top: 1px solid var(--border-hairline) !important;
    }
    .app-footer-inner {
        max-width: 1380px;
        width: 100%;
        margin: 0 auto;
        padding: 9px clamp(1rem, 3vw, 2.75rem);
        box-sizing: border-box;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12.5px;
        line-height: 1.4;
        color: var(--text-secondary);
        gap: 10px 16px;
    }
    .footer-author,
    .footer-links {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
    }
    .app-footer a {
        color: var(--accent-blue);
        text-decoration: none;
        font-weight: 500;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }

    /* calm apple-style loading windows, skeleton shimmers, and rerun indicators */
    .loading-window {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-hairline);
        border-radius: var(--radius-notice);
        padding: 12px 16px;
        margin: 0.45rem 0 0.85rem 0;
        box-shadow: var(--shadow-card);
    }
    .loading-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 9px;
    }
    .loading-title {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
    }
    .loading-meta {
        font-size: 12px;
        color: var(--text-secondary);
    }
    .loading-bar-track {
        width: 100%;
        height: 4px;
        border-radius: 999px;
        background-color: #e5e5ea;
        overflow: hidden;
    }
    .loading-bar-fill {
        width: 36%;
        height: 100%;
        border-radius: 999px;
        background-color: var(--accent-blue);
        animation: gte-progress-slide 1.25s ease-in-out infinite;
    }
    .skeleton-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-hairline);
        border-radius: var(--radius-card);
        padding: 16px 18px;
        box-shadow: var(--shadow-card);
    }
    .skeleton-notice {
        background-color: var(--bg-note);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-notice);
        padding: 14px 16px;
    }
    .skeleton-chart {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-hairline);
        border-radius: var(--radius-card);
        padding: 16px 18px;
        box-shadow: var(--shadow-card);
    }
    .skeleton-line {
        border-radius: 6px;
        background: linear-gradient(90deg, #ececf1 25%, #f5f5f7 50%, #ececf1 75%);
        background-size: 200% 100%;
        animation: gte-shimmer 1.35s linear infinite;
    }
    [data-testid="stSpinner"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-hairline) !important;
        border-radius: var(--radius-notice) !important;
        padding: 10px 14px !important;
        box-shadow: var(--shadow-card) !important;
        margin: 0.4rem 0 !important;
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    /* hide retained outgoing nodes immediately so replaced widgets never ghost during reruns */
    div[data-testid="stElementContainer"][data-retained="true"],
    .stApp [data-retained="true"] {
        display: none !important;
    }
    .stApp [data-stale="true"] {
        transition: opacity 0.14s ease !important;
    }
    @keyframes gte-progress-slide {
        0% { transform: translateX(-110%); }
        100% { transform: translateX(310%); }
    }
    @keyframes gte-shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    @media (prefers-reduced-motion: reduce) {
        .loading-bar-fill,
        .skeleton-line {
            animation: none !important;
        }
    }

    @media (max-width: 1480px) {
        div.st-key-global_header_inner {
            padding-right: 3.1rem !important;
        }
    }

    @media (max-width: 960px) {
        div.st-key-global_header_inner {
            padding: 8px 3rem 8px 1rem !important;
            gap: 8px 14px !important;
        }
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"],
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] {
            padding: 5px 9px !important;
            font-size: 12.5px !important;
        }
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"] p,
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"] span,
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] p,
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] span {
            font-size: 12.5px !important;
            margin: 0 !important;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .notice-grid, .grid-3, .grid-2x3 {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .scope-lead {
            font-size: 14px;
        }
        div[data-testid="stElementContainer"].st-key-region_control,
        div[data-testid="stElementContainer"].st-key-region_control > div,
        .st-key-region_control [data-testid="stButtonGroup"],
        .st-key-region_control [data-testid="stButtonGroup"] > div {
            justify-content: flex-start !important;
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .st-key-region_control [role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            gap: 8px !important;
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
            margin-left: 0 !important;
        }
        .st-key-region_control [role="radiogroup"] button {
            width: 100% !important;
            min-height: 36px !important;
            margin: 0 !important;
            border-radius: 10px !important;
            justify-content: center !important;
        }
    }

    @media (max-width: 768px) {
        header[data-testid="stHeader"] {
            height: 2.85rem !important;
        }
        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 4.25rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .block-container:has(.st-key-nav_control) {
            padding-top: 5.35rem !important;
        }
        div[data-testid="stVerticalBlock"].st-key-global_header_shell {
            background: transparent !important;
            backdrop-filter: none !important;
            border-bottom: none !important;
            box-shadow: none !important;
        }
        div.st-key-global_header_inner {
            display: flex !important;
            flex-direction: column !important;
            flex-wrap: nowrap !important;
            align-items: stretch !important;
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
            gap: 0 !important;
        }
        div.st-key-global_header_inner > [data-testid="stElementContainer"]:first-child {
            width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            flex: 0 0 auto !important;
            margin: 0 !important;
            padding: 11px 2.75rem 11px 0.75rem !important;
            box-sizing: border-box !important;
            background: var(--bg-surface) !important;
            border-bottom: 1px solid var(--border-subtle) !important;
        }
        h1.global-brand-title {
            display: block !important;
            font-size: 18px !important;
            line-height: 1.25 !important;
            white-space: normal !important;
            margin: 0 !important;
        }
        div[data-testid="stElementContainer"].st-key-nav_control:not([data-retained="true"]) {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            flex: 0 0 auto !important;
            margin: 0 !important;
            padding: 6px 0.75rem !important;
            box-sizing: border-box !important;
            background: rgba(245, 245, 247, 0.96) !important;
            backdrop-filter: saturate(180%) blur(12px) !important;
            border-bottom: 1px solid var(--border-hairline) !important;
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.04) !important;
        }
        div[data-testid="stElementContainer"].st-key-nav_control > div,
        div.st-key-nav_control [data-testid="stButtonGroup"],
        div.st-key-nav_control [data-testid="stButtonGroup"] > div,
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 100% !important;
            justify-content: space-between !important;
            box-sizing: border-box !important;
        }
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            gap: 2px !important;
            padding: 3px !important;
        }
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"],
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            justify-content: center !important;
            font-size: clamp(11.8px, 3.45vw, 13.5px) !important;
            line-height: 1.2 !important;
            padding: 6px 3px !important;
            min-height: 32px !important;
            white-space: nowrap !important;
            letter-spacing: -0.015em !important;
            overflow: visible !important;
        }
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"] p,
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[data-variant="segmented_control"] span,
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] p,
        div.st-key-nav_control [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] span {
            font-size: clamp(11.8px, 3.45vw, 13.5px) !important;
            line-height: 1.2 !important;
            letter-spacing: -0.015em !important;
            margin: 0 !important;
            white-space: nowrap !important;
        }
        .app-footer-inner {
            padding: 9px 0.75rem;
        }
        .notice-grid {
            grid-template-columns: 1fr;
        }
        .st-key-region_control [role="radiogroup"] {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            gap: 8px !important;
        }
        .st-key-region_control [role="radiogroup"] button,
        .st-key-region_control [role="radiogroup"] button p,
        .st-key-region_control [role="radiogroup"] button span {
            font-size: 14px !important;
        }
        .st-key-region_control [role="radiogroup"] button {
            min-height: 40px !important;
            padding: 6px 10px !important;
            white-space: nowrap !important;
        }
        div[data-testid="stElementContainer"].st-key-horizon_ctrl,
        div[data-testid="stElementContainer"].st-key-horizon_ctrl > div,
        .st-key-horizon_ctrl [data-testid="stButtonGroup"],
        .st-key-horizon_ctrl [data-testid="stButtonGroup"] > div,
        div[data-testid="stElementContainer"].st-key-scenario_ctrl,
        div[data-testid="stElementContainer"].st-key-scenario_ctrl > div,
        .st-key-scenario_ctrl [data-testid="stButtonGroup"],
        .st-key-scenario_ctrl [data-testid="stButtonGroup"] > div {
            justify-content: flex-start !important;
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .st-key-horizon_ctrl [role="radiogroup"],
        .st-key-scenario_ctrl [role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            gap: 6px !important;
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
            margin-left: 0 !important;
        }
        .st-key-horizon_ctrl [role="radiogroup"] button,
        .st-key-horizon_ctrl [role="radiogroup"] button p,
        .st-key-horizon_ctrl [role="radiogroup"] button span,
        .st-key-scenario_ctrl [role="radiogroup"] button,
        .st-key-scenario_ctrl [role="radiogroup"] button p,
        .st-key-scenario_ctrl [role="radiogroup"] button span {
            font-size: clamp(12.5px, 3.6vw, 13.5px) !important;
        }
        .st-key-horizon_ctrl [role="radiogroup"] button,
        .st-key-scenario_ctrl [role="radiogroup"] button {
            width: 100% !important;
            min-height: 38px !important;
            margin: 0 !important;
            border-radius: 10px !important;
            justify-content: center !important;
            padding: 6px 6px !important;
            white-space: nowrap !important;
        }
        .detail-row {
            grid-template-columns: 1fr;
            gap: 4px;
            padding: 12px 14px;
        }
        .notice-bar {
            flex-direction: column;
            align-items: flex-start;
        }
        div[data-testid="stElementContainer"]:not([data-retained="true"]):has(.chart-mobile-hint) {
            display: block !important;
            height: auto !important;
        }
        .chart-mobile-hint {
            display: block !important;
            font-size: 12px !important;
            color: var(--text-secondary);
            background-color: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: 10px;
            padding: 8px 12px !important;
            margin-bottom: 10px !important;
            height: auto !important;
        }
        .info-tip {
            width: 20px;
            height: 20px;
            font-size: 11px;
        }
        div[data-testid="stElementContainer"]:has(> div[data-testid="stDownloadButton"]),
        div[data-testid="stDownloadButton"],
        div[data-testid="stDownloadButton"] > div,
        div[data-testid="stDownloadButton"] button {
            width: 100% !important;
            max-width: 100% !important;
        }
        div[data-testid="stDownloadButton"] button {
            display: flex !important;
            justify-content: center !important;
        }
    }

    @media (max-width: 540px) {
        .kpi-grid, .grid-3, .grid-2x3 {
            grid-template-columns: 1fr;
        }
        .info-tip:hover::after,
        .info-tip:focus::after {
            left: auto;
            right: -6px;
            transform: none;
            width: min(220px, 76vw);
        }
    }
    </style>
    """