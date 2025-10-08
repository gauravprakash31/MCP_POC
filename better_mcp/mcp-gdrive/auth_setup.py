# auth_setup.py

import json, pathlib
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

CREDS_PATH = pathlib.Path("credentials.json")
TOKEN_PATH = pathlib.Path("token.json")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def main():
    if not CREDS_PATH.exists():
        raise SystemExit("Missing credentials.json in project folder.")
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
    # this opens a browser for you to approve once
    """
    if token expires, delete token.json and rerun auth_setup.py to generate a new token"""
    creds = flow.run_local_server(port=0)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"Wrote {TOKEN_PATH} — you're authenticated")

if __name__ == "__main__":
    main()