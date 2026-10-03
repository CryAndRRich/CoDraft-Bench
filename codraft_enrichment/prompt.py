
ENRICHMENT_PROMPTS = {
    "v1": {
        "system": "You are an expert Intellectual Property Analyst specializing in the NICE Classification system. Use 'Chain of Draft' reasoning to disambiguate product names.",
        "user": """Analyze the following LIST of products. 
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
    },
    "v2": {
        "system": "You are an expert Intellectual Property Analyst specializing in the NICE Classification system. Use a 3-tier 'Chain of Draft' reasoning (Class -> Sub-group -> Term) to disambiguate product names.",
        "user": """Analyze the following LIST of products. 
        For each item, strictly follow the CoDraft steps using the 3-tier context:
        1. Brainstorm: List potential meanings of the term in a vacuum.
        2. Critique: Filter the meanings first by the broad 'Class Heading', then STRICTLY align it with the specific 'Sub-group'. Reject meanings that don't fit the Sub-group.
        3. Define: Specify the Nature, Purpose, and create an Expanded Name that fits perfectly within the Sub-group.

        ---
        EXAMPLE 1 (Good 3-Tier Reasoning):
        Input: 
        - Term: "Apple"
        - Class: 31
        - Class Heading: "Agricultural, aquacultural, horticultural and forestry products..."
        - Sub-group: "Fresh fruits and vegetables"
        
        Step 1 (Brainstorm): Could be a fruit, a tech company, or a record label.
        Step 2 (Critique): Class 31 covers agriculture. More specifically, the Sub-group is "Fresh fruits and vegetables". Tech companies (Class 9) and labels (Class 41) are eliminated. Correct meaning: The edible raw fruit.
        Nature: Fresh fruit
        Purpose: Human consumption
        Expanded Name: Fresh apples

        EXAMPLE 2 (Good 3-Tier Reasoning):
        Input: 
        - Term: "Mercury"
        - Class: 5
        - Class Heading: "Pharmaceuticals, medical and veterinary preparations; Dietetic food and substances..."
        - Sub-group: "Pharmaceuticals, medical and veterinary preparations"
        
        Step 1 (Brainstorm): Could be a planet, a chemical element (metal), or a pharmaceutical ingredient.
        Step 2 (Critique): While Class 5 covers many things (like dietetic food), the specific Sub-group is "Pharmaceuticals". Element mercury (Class 1) or planets are incorrect. Correct meaning: A substance used in medical preparations.
        Nature: Pharmaceutical preparation
        Purpose: Medical treatment
        Expanded Name: Mercury-based pharmaceutical preparations
        ---

        INPUT DATA TO PROCESS:
        {batch_json}
        """
    }
}