import os
from pathlib import Path
from dotenv import load_dotenv
import json
import duckdb
from openai import OpenAI
import argparse
import logging
import time
import uuid
import pandas as pd

from distribution import (
    pick_observed_data,
    compute_observed_distribution_fare,
    OBSERVED_DATA,
    TITANIC_HISTORICAL_DATA
)


# CONFIGURATION -------------------------------------

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n", type=int, required=False, default=5,
    help="Nombre de passagers à générer"
)
args = parser.parse_args()
N = args.n

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

con = duckdb.connect()

# LLM template
template = Path("src/synthetic_data/prompt_generation.md").read_text(encoding="utf-8")
with open("src/synthetic_data/schema_generation.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

# Ensure schema consistency with N
schema["schema"]["minItems"] = N
schema["schema"]["maxItems"] = N


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://llm.lab.sspcloud.fr/api"
)


# OBSERVED DATA PICKED FROM FRENCH CENSUS DATA -------------------------

logger.info(f"Sélection de {N} passagers observés")
passengers = pick_observed_data(con, OBSERVED_DATA, n=N)

logger.debug(passengers)


fare_stats = compute_observed_distribution_fare(
    con,
    TITANIC_HISTORICAL_DATA
)

logger.info(
    "Fare données historiques utilisées pour générer des valeurs vraisemblables"
)
logger.info(fare_stats)


# INJECTION DANS LE PROMPT --------------------------

prompt = template.format(
    N=N,
    fare_min=fare_stats["fare_min"],
    fare_q1=fare_stats["fare_q1"],
    fare_q2=fare_stats["fare_q2"],
    fare_q3=fare_stats["fare_q3"],
    fare_max=fare_stats["fare_max"],
    input=json.dumps(passengers, ensure_ascii=False)
)

logger.debug(prompt)


# REQUETE GENERATIVE ------------------------------

logger.info("Envoi de la requête au LLM")


t0 = time.perf_counter()

resp = client.chat.completions.create(
    model="qwen3-5-9b",
    messages=[{"role": "user", "content": prompt}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            **schema,
            "strict": True
        }
    }
)

elapsed = time.perf_counter() - t0
logger.info("Temps de réponse du LLM: %.3f secondes", elapsed)

out = resp.choices[0].message.content
result = json.loads(out)

logger.info("Résultat reçu: %s lignes", len(result))
logger.info("Résultat brut: %s", result)


# SAVE OBSERVED DATA -----------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

random_filename = f"titanic_{uuid.uuid4().hex}.parquet"
output_path = output_dir / random_filename

df = pd.DataFrame(result)
df.to_parquet(output_path, index=False)

logger.info("Parquet enregistré dans %s", output_path)

logger.debug(f"Saved to: {output_path}")