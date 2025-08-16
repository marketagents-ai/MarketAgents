from miniference.models import CallableTool
from pydantic import BaseModel
from typing import Dict, Any, List


class WorkflowStep(BaseModel):
    id: str
    tool: CallableTool
    inputs: Dict[str, Any]
    result: Any = None

class WorkflowDefinition(BaseModel):
    steps: List[WorkflowStep]


class WorkflowExecutor:
    def __init__(self, workflow: WorkflowDefinition):
        self.workflow = workflow

    def _resolve_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve any string input references of the form 'stepX.result'."""
        resolved = {}
        for key, value in inputs.items():
            if isinstance(value, str) and ".result" in value:
                # For simplicity, assume the format is "step_id.result" (or with attribute, e.g. "step1.result.numbers")
                parts = value.split(".")
                step_id = parts[0]
                # Find the corresponding step:
                step = next((s for s in self.workflow.steps if s.id == step_id), None)
                if not step or step.result is None:
                    raise ValueError(f"Step reference '{value}' cannot be resolved; step '{step_id}' has no result yet.")
                # If additional attribute access is specified, walk through them.
                attr_value = step.result
                for attr in parts[2:]:
                    attr_value = getattr(attr_value, attr)
                resolved[key] = attr_value
            else:
                resolved[key] = value
        return resolved

    async def execute(self) -> WorkflowDefinition:
        for step in self.workflow.steps:
            processed_inputs = self._resolve_inputs(step.inputs)
            print(f"Executing step {step.id} with inputs: {processed_inputs}")

            step.result = step.tool.call(processed_inputs)
            print(f"Step {step.id} executed; result: {step.result}")
        return self.workflow
