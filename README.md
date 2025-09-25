# Tamil WSD API

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import stanza; stanza.download('ta')"
uvicorn app:app --reload
