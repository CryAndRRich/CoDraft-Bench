from pydantic import BaseModel, Field

class NiceCoDraft(BaseModel):
    original_term: str = Field(..., description="The exact product name from the input list.")
    step1_brainstorming: list[str] = Field(
        ...,
        description="List 3 potential meanings of the Product Name in general contexts (e.g., 'Apple' -> Fruit, Tech Company, Record Label)."
    )
    step2_context_verification: str = Field(
        ...,
        description="Compare the drafted meanings with the provided NICE Class ID & Description. Explicitly reject mismatched meanings and select the correct one."
    )

    nature: str = Field(
        ...,
        description="The specific physical nature of the product (Noun). DO NOT use generic words like 'Product', 'Item'. Examples: 'Downloadable software application', 'Chemical cleaning solution', 'Printed magazine'."
    )

    purpose: str = Field(
        ...,
        description="The primary specific function or usage (Verb-ing phrase). Examples: 'Managing customer data', 'Removing grease from surfaces', 'Protecting skin from UV rays'."
    )

    expanded_name: str = Field(
        ...,
        description="The fully expanded, professional product name based on its nature and context."
    )

class NiceCoDraft2(BaseModel):
    original_term: str = Field(..., description="The exact product name from the input list.")
    
    step1_brainstorming: list[str] = Field(
        ...,
        description="List 3 potential meanings of the Product Name in general contexts (e.g., 'Apple' -> Fruit, Tech Company, Record Label)."
    )
    
    step2_context_verification: str = Field(
        ...,
        description="Compare the drafted meanings with BOTH the general 'Class Heading' AND the specific 'Sub-group'. Explicitly explain why mismatched meanings are rejected based on the Sub-group, and select the correct one."
    )

    nature: str = Field(
        ...,
        description="The specific physical nature of the product (Noun). DO NOT use generic words like 'Product', 'Item'. Examples: 'Downloadable software application', 'Chemical cleaning solution', 'Printed magazine'."
    )

    purpose: str = Field(
        ...,
        description="The primary specific function or usage (Verb-ing phrase). Examples: 'Managing customer data', 'Removing grease from surfaces', 'Protecting skin from UV rays'."
    )

    expanded_name: str = Field(
        ...,
        description="The fully expanded, professional product name strictly aligned with the Sub-group context."
    )
class BatchAnalysis(BaseModel):
    items: list[NiceCoDraft] = Field(..., description="List of analyzed products.")

class BatchAnalysis2(BaseModel):
    items: list[NiceCoDraft2] = Field(..., description="List of analyzed products.")

SCHEMA_MAP = {
    "v1": BatchAnalysis,
    "v2": BatchAnalysis2
}