import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. THE DATA STRUCTURE
class DailyRoutine(BaseModel):
    morning: list[str] = Field(description="Morning skincare steps")
    evening: list[str] = Field(description="Evening skincare steps")

# --- CHANGED: Split products into two price categories ---
class ProductRecommendations(BaseModel):
    affordable: list[str] = Field(description="Affordable/drugstore real-world products (e.g., CeraVe, The Ordinary, Cetaphil)")
    high_end: list[str] = Field(description="High-end/luxury real-world products (e.g., SkinCeuticals, Drunk Elephant, Tatcha)")
    avoid: list[str] = Field(description="Types of products to avoid")

class IngredientGuide(BaseModel):
    look_for: list[str] = Field(description="Beneficial ingredients")
    avoid: list[str] = Field(description="Ingredients to avoid")

class SkinRecommendation(BaseModel):
    summary: str = Field(description="Short summary of analysis")
    daily_routine: DailyRoutine = Field(description="Routines")
    products: ProductRecommendations = Field(description="Products")
    ingredients: IngredientGuide = Field(description="Ingredients")
    lifestyle_tips: list[str] = Field(description="3 lifestyle tips")

# 2. SETUP
API_KEY = "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"
llm = ChatGroq(api_key=API_KEY, model="llama-3.3-70b-versatile", temperature=0.5)
parser = PydanticOutputParser(pydantic_object=SkinRecommendation)

conversation_history = []

# 3. ANALYSIS FUNCTION
def get_recommendation(skin_type: str, condition: str) -> dict:
    print(f"🔄 Analyzing {skin_type}...")
    
    prompt = ChatPromptTemplate.from_template(
        "You are SkinE, a top dermatologist. Analyze this:\n"
        "Type: {skin_type}\nCondition: {condition}\n\n"
        "CRITICAL RULE: For recommended products, you MUST provide REAL brand names and specific products. "
        "Divide them strictly into 'affordable' (budget/drugstore) and 'high_end' (luxury/expensive) options.\n\n"
        "Return ONLY JSON based on these instructions: {format_instructions}"
    )
    
    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "skin_type": skin_type, 
            "condition": condition,
            "format_instructions": parser.get_format_instructions()
        })
        return result.model_dump() 
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

# 4. CHAT FUNCTION
def chat(user_message: str) -> str:
    try:
        response = llm.invoke(user_message)
        return response.content
    except Exception as e:
        return f"Chat Error: {e}"

def reset_session():
    global conversation_history
    conversation_history = []

if __name__ == "__main__":
    print("--- TESTING ENGINE ---")
    res = get_recommendation("Oily", "Acne")
    print(json.dumps(res, indent=2))