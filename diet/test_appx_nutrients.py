from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

meal_schema = {
    "{meal_name}": {
        "ingredients": "{ingredients}",
        "total_calories": 0,
        "total_protein": 0,
        "protein_calories": 0,
        "carb_calories": 0,
        "vegetable_calories": 0,
        "fat_calories": 0,
        "fiber": 0,
        "sodium": 0,
        "saturated_fat": 0,
        "sugars": 0,
        "vitamin_d": 0,
        "calcium": 0,
        "iron": 0,
        "potassium": 0,
        "cholesterol": 0,
        "omega_3": 0,
        "omega_6": 0,
    }
}


def approximate_nutritional_values(meal_name, additional_info, meal_schema):
    llm = OpenAI(temperature=0.4)
    prompt = PromptTemplate.from_template(
        "You must return a JSON object with the following schema: {schema} "
        "What are the nutritional values for {meal_name} given the following information: {additional_info}? "
        "If the additional_info does not provide enough information then infer as much as you can to make approximations. "
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run(
        meal_name=meal_name, additional_info=additional_info, schema=meal_schema
    )

    return resp


meal_name = "Breakfast Burrito"
additional_info = "It says it has 700 calories and 20g of protein, its a tortilla with eggs and sausage"

# Get the nutritional values as a dict
nutritional_values = approximate_nutritional_values(
    meal_name, additional_info, meal_schema
)

# Print them to the console
print(json.dumps(nutritional_values, indent=4))
