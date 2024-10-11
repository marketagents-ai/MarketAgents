from pydantic import BaseModel

class InputSchema(BaseModel):
    tool_name: str
    tool_input_type: str
    tool_input_value: dict
