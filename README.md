# Recruiting-Case
YouTube → strukturierte Markdown-Notiz


Ein sehr schlanker CLI-Agent:
1) Transcript per `youtube-transcript-api`
2) Ein einzelner LLM-Call erzeugt die Notiz im geforderten Format
3) Output nach `output/<slug>.md`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
