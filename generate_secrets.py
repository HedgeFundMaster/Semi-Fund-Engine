import json
import os

# Load your existing service account JSON key
json_path = "gcp_key.json"  # adjust if your file is elsewhere
with open(json_path, "r") as j:
    data = json.load(j)

# Prepare the TOML lines
lines = []
lines.append('type = "{}"'.format(data["type"]))
lines.append('project_id = "{}"'.format(data["project_id"]))
lines.append('private_key_id = "{}"'.format(data["private_key_id"]))

# Convert the PEM into a single-line string with \n escapes
pem_lines = data["private_key"].strip().splitlines()
escaped_pem = "\\n".join(pem_lines)
lines.append('private_key = "{}\\n"'.format(escaped_pem))

lines.append('client_email = "{}"'.format(data["client_email"]))
lines.append('client_id = "{}"'.format(data["client_id"]))
lines.append('auth_uri = "{}"'.format(data["auth_uri"]))
lines.append('token_uri = "{}"'.format(data["token_uri"]))
lines.append('auth_provider_x509_cert_url = "{}"'.format(data["auth_provider_x509_cert_url"]))
lines.append('client_x509_cert_url = "{}"'.format(data["client_x509_cert_url"]))

# Write to .streamlit/secrets.toml
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/secrets.toml", "w") as f:
    f.write("\n".join(lines))

print("✅ Generated .streamlit/secrets.toml successfully.") 