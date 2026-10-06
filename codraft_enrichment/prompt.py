PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are an expert Intellectual Property Analyst specializing in the NICE Classification system. "
    "Use 'Chain of Draft' reasoning to disambiguate product names."
)

USER_PROMPT = """Analyze the following LIST of products. 
    For each item, strictly follow the CoDraft steps:
    1. Brainstorm: List potential meanings.
    2. Critique: Compare with the provided NICE Class ID & Description to select the correct context.
    3. Define: Specify the Nature, Purpose, and create an Expanded Name.

    ---
    EXAMPLE 1 (Good Reasoning):
    Input: Term: "Apple", Class: 31, Description: "Agricultural, aquacultural, horticultural and forestry products; Raw and unprocessed grains..."
    Step 1 (Brainstorm): Could be a fruit, a tech company, or a record label.
    Step 2 (Critique): Class 31 covers horticultural products. Tech companies (Class 9) and labels (Class 41) are incorrect here. Correct meaning: The edible fruit.
    Nature: Fresh fruit
    Purpose: Human consumption
    Expanded Name: Fresh apples

    EXAMPLE 2 (Good Reasoning):
    Input: Term: "Mercury", Class: 5, Description: "Pharmaceuticals, medical and veterinary preparations; Dietetic food and substances..."
    Step 1 (Brainstorm): Could be a planet, a chemical element (metal), or a pharmaceutical ingredient.
    Step 2 (Critique): Class 5 covers medical preparations. Element mercury (Class 1) or planets (N/A) are incorrect. Correct meaning: A substance used in medicine.
    Nature: Pharmaceutical preparation
    Purpose: Medical treatment
    Expanded Name: Mercury-based pharmaceutical preparations
    ---

    INPUT DATA TO PROCESS:
    {batch_json}
    """
