from pydantic import BaseModel, ConfigDict, Field


class CategoryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=3,
        max_length=1000,
    )


class CategoryResponse(BaseModel):
    category: str