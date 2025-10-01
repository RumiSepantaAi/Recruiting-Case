#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, pathlib, json, math, textwrap
from typing import List, Dict, Tuple
from dataclasses import dataclass

import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# --- OpenAI (optionaler Einsatz) ---
from openai import OpenAI

# ---------- Hilfsfunktionen ----------
YOUTUBE_OEMBED = "https://www.youtube.com/oembed"

def get_video_id(url: str) -> str:
    # direkte ID?
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url or ""):
        return url
    # youtu.be/<id>
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    # watch?v=<id>
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    # shorts/<id>
    m = re.search(r"shorts/([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    raise ValueError("Konnte keine YouTube-Video-ID extrahieren.")

def fetch_title(url: str) -> str:
    try:
        r = requests.get(YOUTUBE_OEMBED, params={"url": url, "format": "json"}, timeout=10)
        if r.status_code == 200:
            return r.json().get("title") or ""
    except requests.RequestException:
        pass
    return ""

def mmss(seconds: float) -> str:
    total = int(round(seconds))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"

def slugify(text: str, fallback: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text or fallback

@dataclass
class Segment:
    start: float
    text: str

def fetch_transcript(video_id: str, lang_priority: List[str]) -> List[Segment]:
    try_order = lang_priority + list({"de","en"} - set(lang_priority))
    # Direkter Call mit Sprachen
    for langs in [try_order, lang_priority, ["de"], ["en"]]:
        try:
            raw = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            return [Segment(start=i["start"], text=i["text"].strip()) for i in raw if i.get("text")]
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            continue
        except Exception:
            continue
    # Fallback Listing (manuell/auto/translate)
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        # manuell bevorzugen
        for code in try_order:
            try:
                tr = listing.find_transcript([code])
                raw = tr.fetch()
                return [Segment(start=i["start"], text=i["text"].strip()) for i in raw if i.get("text")]
            except Exception:
                pass
        # übersetzt
        for code in try_order:
            try:
                tr = listing.find_transcript([code]).translate(code)
                raw = tr.fetch()
                return [Segment(start=i["start"], text=i["text"].strip()) for i in raw if i.get("text")]
            except Exception:
                pass
    except Exception:
        pass
    return []

def format_transcript_for_llm(segments: List[Segment]) -> str:
    # Jede Zeile: [MM:SS] Text  — so kann das LLM Zitate mit Timestamp extrahieren
    lines = []
    for seg in segments:
        t = seg.text.replace("\n", " ").strip()
        if t:
            lines.append(f"[{mmss(seg.start)}] {t}")
    return "\n".join(lines)

# ---------- LLM Aufrufe ----------
def openai_client_or_none():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def llm_single_pass_markdown(client, model: str, title: str, url: str, transcript_block: str,
                             language_hint: str, max_tldr: int, max_keypoints: int,
                             include: List[str]) -> str:
    """
    Ein einziger Prompt, das LLM soll direkt die fertige Markdown-Notiz ausgeben.
    """
    include_set = set(include or [])
    wants_quotes = "quotes" in include_set
    wants_questions = "questions" in include_set
    wants_glossary = "glossary" in include_set

    system = f"""Du bist ein präziser Notizen-Agent. Erzeuge aus einem YouTube-Transcript eine klare, sachliche Markdown-Notiz in {language_hint.upper()}.
Halte dich strikt an die geforderte Struktur und füge nichts hinzu, was nicht durch das Transcript gedeckt ist."""
    user = f"""
Metadaten:
- Titel: {title}
- Quelle: {url}

Zielstruktur (GENAU einhalten):
# {{Titel}}
- Quelle: {{URL}}

## TL;DR (3–5 Bullet Points)
(bis zu {max_tldr} kurze, prägnante Bullet Points)

## Kernaussagen
(bis zu {max_keypoints} Punkte)

## Struktur / Outline
(3–6 nummerierte Punkte; starke, inhaltliche Zwischenüberschriften)

{"## Zitate mit Zeitstempel\n(stich aus dem Transcript 2–4 markante, wörtliche Zitate, jedes mit [MM:SS])" if wants_quotes else ""}
{"## Offene Fragen\n(1–3 Fragen, die im Video offenbleiben)" if wants_questions else ""}
{"## Glossar (Begriffe)\n(2–5 zentrale Begriffe, jeweils sehr kurz erklärt)" if wants_glossary else ""}

WICHTIG:
- Schreibe NUR die Markdown-Notiz, keine Erklärtexte.
- Halluziniere NICHT.
- Nutze Zitate nur wörtlich aus dem Transcript, mit vorhandenem [MM:SS].

Hier ist das Transcript (Zeile je Segment):


"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )
    return resp.choices[0].message.content.strip()

def chunk(text: str, max_chars: int = 12000) -> List[str]:
    # Grobe, robuste Char-basierte Chunking-Strategie mit Zeilen-Schnitt
    if len(text) <= max_chars:
        return [text]
    parts, buf = [], []
    size = 0
    for line in text.splitlines(True):
        if size + len(line) > max_chars and buf:
            parts.append("".join(buf))
            buf, size = [line], len(line)
        else:
            buf.append(line); size += len(line)
    if buf: parts.append("".join(buf))
    return parts

def llm_map_reduce(client, model: str, title: str, url: str, transcript_block: str,
                   language_hint: str, max_tldr: int, max_keypoints: int, include: List[str]) -> str:
    """
    2-stufig: 1) Chunk-Zusammenfassungen, 2) Finaler Merge → Markdown.
    """
    # 1) Map: Chunk-Zusammenfassungen
    summaries = []
    for i, part in enumerate(chunk(transcript_block), 1):
        sys = f"Fasse folgendes Transkript-Teil in {language_hint.upper()} extrem prägnant zusammen. Gib 5–8 Bulletpoints aus. Keine Halluzinationen."
        usr = f"## Transcript-Teil {i}\n```\n{part}\n```"
        r = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}]
        )
        summaries.append(r.choices[0].message.content.strip())

    # 2) Reduce: Finales Markdown bauen
    merged = "\n".join(f"- {s}" for s in summaries)
    sys_final = f"Baue aus Stichpunkten eine klare Markdown-Notiz in {language_hint.upper()} nach Vorgabe. Strikt, prägnant, keine Halluzination."
    user_final = f"""
Metadaten:
- Titel: {title}
- Quelle: {url}

Stichpunkte aus allen Teilen:
{merged}

Gib die Notiz exakt in folgendem Format aus (wie oben im Ein-Pass-Prompt beschrieben), ohne Extrakommentare.
"""
    r2 = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role":"system","content":sys_final},{"role":"user","content":user_final}]
    )
    return r2.choices[0].message.content.strip()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="YouTube → Markdown (LLM-only)")
    ap.add_argument("url", type=str, help="YouTube-URL oder Video-ID")
    ap.add_argument("--lang", nargs="+", default=["de","en"], help="Sprach-Priorität für Transcript (z.B. de en)")
    ap.add_argument("--outdir", type=str, default="output", help="Output-Verzeichnis")
    ap.add_argument("--max-tldr", type=int, default=4, help="TL;DR Bullet-Anzahl")
    ap.add_argument("--max-keypoints", type=int, default=6, help="Kernaussagen max.")
    ap.add_argument("--include", nargs="*", default=["quotes","questions","glossary"],
                    help="Optionale Sektionen: quotes questions glossary")
    ap.add_argument("--max-prompt-chars", type=int, default=12000, help="Fallback-Chunking, wenn Transcript größer ist")
    args = ap.parse_args()

    url = args.url
    try:
        vid = get_video_id(url)
    except ValueError as e:
        print(f"[FEHLER] {e}")
        return

    title = fetch_title(url) or f"YouTube-Video {vid}"
    slug = slugify(title, fallback=vid)

    print(f"[INFO] Video-ID: {vid}")
    print(f"[INFO] Titel: {title}")

    segments = fetch_transcript(vid, args.lang)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{slug}.md"

    if not segments:
        print("[WARNUNG] Kein Transcript gefunden. Erzeuge Platzhalter-Notiz.")
        md = f"""# {title}
- Quelle: {url}

## TL;DR (3–5 Bullet Points)
- (Kein Transcript verfügbar.) Bitte anderes Video wählen.

## Kernaussagen
- —

## Struktur / Outline
1. —
"""
        outfile.write_text(md, encoding="utf-8"); print(f"[OK] Geschrieben: {outfile}"); return

    transcript_block = format_transcript_for_llm(segments)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = openai_client_or_none()

    if not client:
        # Kein LLM? Minimaler Fallback: Roh-Transcript in einfache Notiz.
        print("[HINWEIS] Kein OPENAI_API_KEY gesetzt. Schreibe einfache, nicht-verdichtete Notiz.")
        raw_md = f"""# {title}
- Quelle: {url}

## TL;DR (3–5 Bullet Points)
- (Kein LLM aktiv) Bitte OPENAI_API_KEY setzen für automatische Verdichtung.

## Kernaussagen
- Transcript-Export unten.

## Struktur / Outline
1. (Nicht generiert ohne LLM)

## Zitate mit Zeitstempel
(Nur mit LLM.)

## Transcript (Roh)


"""
        outfile.write_text(raw_md, encoding="utf-8"); print(f"[OK] Geschrieben: {outfile}"); return

    # Ein-Pass wenn kurz genug, sonst Map-Reduce
    if len(transcript_block) <= args.max_prompt_chars:
        md = llm_single_pass_markdown(
            client, model, title, url, transcript_block,
            language_hint=args.lang[0], max_tldr=args.max_tldr,
            max_keypoints=args.max_keypoints, include=args.include
        )
    else:
        print("[INFO] Transcript ist groß – Map-Reduce Verdichtung aktiv.")
        md = llm_map_reduce(
            client, model, title, url, transcript_block,
            language_hint=args.lang[0], max_tldr=args.max_tldr,
            max_keypoints=args.max_keypoints, include=args.include
        )

    # Sicherstellen, dass # Titel + Quelle vorhanden sind (falls Modell weglässt)
    if "# " not in md:
        md = f"# {title}\n- Quelle: {url}\n\n" + md.strip() + "\n"

    outfile.write_text(md, encoding="utf-8")
    print(f"[OK] Geschrieben: {outfile}")

if __name__ == "__main__":
    main()


