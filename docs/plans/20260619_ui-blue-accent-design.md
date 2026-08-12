# UI Blue Accent Design — 2026-03-22

## Goal
Add cool blue accent details to the existing frosted glass UI to make it feel like a polished SaaS tool while preserving the mountain background aesthetic.

## Approach
Option 1: Accent borders + blue-tinted chat area. Keep white glass panels, add a blue gradient top border to each card, and tint the chat messages area blue instead of plain white.

## Design Tokens
- Primary blue: `#3b82f6`
- Secondary blue: `#0ea5e9`
- Chat area tint: `rgba(59, 130, 246, 0.12)`
- Hero card top border: `3px solid #3b82f6`
- Section panel top border: `2px solid rgba(59, 130, 246, 0.65)`

## Changes Per Page

### Main Page (HOME_PAGE_HTML)
- `.hero-card` — add `border-top: 3px solid #3b82f6`
- `.section` — add `border-top: 2px solid rgba(59, 130, 246, 0.65)`
- `h2` — add `border-left: 3px solid #3b82f6; padding-left: 10px`
- `#chat-messages` — change background to `rgba(59, 130, 246, 0.12)`
- `.tab-btn.active` — change to `background: rgba(59,130,246,0.22); color: #e0f0ff`
- `.settings-toggle` — change border to `rgba(59,130,246,0.45)`

### Health Page (HEALTH_CHECK_HTML)
- `.header`, `.component-card`, `.metrics-section` — add `border-top: 2px solid rgba(59, 130, 246, 0.65)`
- `.status-main` — keep dynamic `border-left` colored by health status

### Stats Page (STATS_PAGE_HTML)
- `.header`, `.stat-card`, `.config-section` — add `border-top: 2px solid rgba(59, 130, 246, 0.65)`
- `.config-value` — update color to `#0ea5e9`
