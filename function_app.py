import os
import json
import logging
import azure.functions as func
from openai import AzureOpenAI

# =========================
# Azure Functions v2 model
# =========================
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# =========================
# Azure OpenAI client
# =========================
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT):
    raise RuntimeError("Faltan variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,  # solo base; sin /openai/deployments
    api_key=AZURE_OPENAI_API_KEY,
)

# =========================
# Prompting
# =========================
SYSTEM_PROMPT = (
    "You are a culinary menu generator and diet-aware scorer.\n"
    "Goal: When given ONLY a restaurant name and location (no raw menu), infer a plausible, representative English menu "
    "and evaluate each dish for digestive safety based on the user's profile.\n"
    "\n"
    "INFERENCE RULES:\n"
    "- Infer the likely cuisine/style from the restaurant name and location (city/country). If ambiguous, pick a broadly popular style for that region (e.g., Spanish tapas in Madrid, taquería in Mexico City, American diner in NYC).\n"
    "- Generate a concise, representative menu of canonical, well-known dishes for that cuisine; reflect common local adaptations when relevant.\n"
    "- Do NOT invent brand-specific, proprietary, or overly niche items; avoid prices and emojis.\n"
    "- Prefer 6–10 dishes unless the user specifies otherwise; cover a mix of starters/mains/sides as appropriate.\n"
    "- English ONLY. Each 'contents' is 1–2 sentences with key ingredients, sauces, sides, and cooking method.\n"
    "- Be conservative: if an allergen or ingredient is typical for a dish (e.g., peanuts in Pad Thai), treat it as present unless contradicted by the context.\n"
    "\n"
    "DIET SCORING RULES:\n"
    "- Consider dietary restrictions: Gluten-free, Dairy-free, Nut-free, Soy-free, Egg-free, Shellfish-free, Vegan, Vegetarian, Low-FODMAP.\n"
    "- Consider health conditions: IBS, Celiac Disease, Crohn's Disease, Ulcerative Colitis, GERD, Lactose intolerance.\n"
    "- Start each dish at 100 and subtract for conflicts; major conflicts (e.g., gluten for Gluten-free or clear high-FODMAP for IBS) should push score <50 and set recommended=false.\n"
    "- 'good_for_you_tags' are short positives (e.g., 'Lean protein', 'Easy to digest carbs').\n"
    "- 'potential_triggers' are brief and concrete (e.g., 'Spicy chili', 'Cream/dairy').\n"
    "- 'reason_short' is a one-sentence justification.\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "- Strictly match the provided JSON schema (restaurant_name, items[dish_name, contents, evaluation[score_percent, good_for_you_tags, potential_triggers, recommended, reason_short]]).\n"
)


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "personalized_menu_schema",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["restaurant_name", "items"],
            "properties": {
                "restaurant_name": {"type": ["string", "null"]},
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["dish_name", "contents", "evaluation"],
                        "properties": {
                            "dish_name": {"type": "string", "minLength": 1},
                            "contents": {"type": "string", "minLength": 10},
                            "evaluation": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "score_percent",
                                    "good_for_you_tags",
                                    "potential_triggers",
                                    "recommended",
                                    "reason_short"
                                ],
                                "properties": {
                                    "score_percent": {"type": "integer", "minimum": 0, "maximum": 100},
                                    "good_for_you_tags": {"type": "array", "items": {"type": "string"}},
                                    "potential_triggers": {"type": "array", "items": {"type": "string"}},
                                    "recommended": {"type": "boolean"},
                                    "reason_short": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "strict": True
    }
}

def build_user_prompt(payload: dict) -> str:
    """
    Minimal prompt: only restaurant_name, location, plus optional hints:
      - cuisine_hint (string) para forzar/afinar el tipo de cocina
      - max_items (int) para controlar cantidad de platos (default 8)
      - health_conditions (list[str]) y dietary_restrictions (list[str]) para el perfil
    """
    name = payload.get("restaurant_name", "").strip()
    location = payload.get("location", "").strip()

    cuisine_hint = payload.get("cuisine_hint")  # opcional
    max_items = payload.get("max_items") or 8   # opcional

    hc = payload.get("health_conditions") or []
    dr = payload.get("dietary_restrictions") or []

    lines = [
        "TASK INPUT:",
        f"- Restaurant name: {name}",
        f"- Location: {location}",
        f"- Max items to generate: {max_items}",
    ]
    if cuisine_hint:
        lines.append(f"- Cuisine hint (optional): {cuisine_hint}")
    if hc:
        lines.append(f"- Health conditions: {', '.join(hc)}")
    if dr:
        lines.append(f"- Dietary restrictions: {', '.join(dr)}")

    lines.append(
        "\nINSTRUCTIONS FOR THIS INPUT:\n"
        f"- Infer a plausible {(' ' + cuisine_hint) if cuisine_hint else ''} menu typical for the given name/location.\n"
        "- Return between 4 and the requested max items (inclusive), aiming for the requested max.\n"
        "- Keep items canonical to the inferred cuisine; avoid proprietary or highly obscure dishes.\n"
    )
    return "\n".join(lines)



# =========================
# HTTP route (POST /generate-menu)
# =========================
@app.route(route="generate-menu", methods=[func.HttpMethod.POST])
def generate_menu(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON body", status_code=400)

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.1,
            response_format=RESPONSE_FORMAT,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(body)}
            ],
        )

        parsed = getattr(resp.choices[0].message, "parsed", None)
        if parsed is None:
            parsed = json.loads(resp.choices[0].message.content)

        if not parsed.get("restaurant_name") and body.get("restaurant_name"):
            parsed["restaurant_name"] = body["restaurant_name"]

        return func.HttpResponse(
            body=json.dumps(parsed, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("OpenAI error")
        return func.HttpResponse(
            body=json.dumps({"error": f"{e}"}),
            status_code=500,
            mimetype="application/json"
        )

