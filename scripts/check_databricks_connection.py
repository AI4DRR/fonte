import os
import logging
import certifi
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("databricks.sql").setLevel(logging.DEBUG)

host = os.environ["DATABRICKS_SERVER_HOSTNAME"]
path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]

print("HOST:", host, flush=True)
print("PATH:", path, flush=True)
print("TOKEN FOUND:", token is not None, flush=True)

with sql.connect(
    server_hostname=host,
    http_path=path,
    access_token=token,
    _tls_trusted_ca_file=certifi.where(),
) as conn:
    print("CONNECTED", flush=True)
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        print(cur.fetchall(), flush=True)