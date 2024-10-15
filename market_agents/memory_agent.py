from market_agents.simple_agent import SimpleAgent, BidTool, AskTool, SimpleAgentState, AuctionInput
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.econ_models import Basket, Good, Trade, Endowment, Bid, Ask, SellerPreferenceSchedule, BuyerPreferenceSchedule
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import LLMPromptContext, StructuredTool, LLMConfig, LLMOutput
from market_agents.environments.mechanisms.auction import AuctionLocalObservation, AuctionGlobalObservation
from typing import Optional, Union, Dict, Any, List
from pydantic import Field, field_validator, model_validator, computed_field, BaseModel
import json

class MemoryAnalysis(BaseModel):
    """ Class defining the structured output of the MemoryProcessor it is a comprehensive analyis of the history of 
    Inputs and Outputs from the SequentialAgent """
    analysis: str = Field(description="A detailed analysis of the history of inputs and outputs and the causal consequences that can be derived")
    action_suggestion: str = Field(description="Qualitative analyisis of the bid or ask that the agent should make based on the analysis, this should be a rationally explained example")

class MemoryAnalysisTool(StructuredTool):
    """ StructuredTool for the MemoryAnalysis """
    name: str = Field(default="MemoryAnalysis")
    description: str = Field(default="Analyzes the history of inputs and outputs to generate a structured memory analysis")
    json_schema: Dict[str, Any] = Field(default=MemoryAnalysis.model_json_schema())
    instruction_string: str = Field(default="Your role is to analyze the history of inputs and outputs to generate a structured memory analysis following the provided JSON schema.")

class MemoryProcessor(LLMPromptContext):
    """ Compresses the history of a SimpleAgent into a single structured output.
    User messages have the format of tuples (AuctionInput, Union[Bid, Ask]) and return a MemoryAnalysis. """
    system_string: str = Field(default="You are an expert analyst tasked with compressing the history of a SimpleAgent into a single structured output. Your role involves analyzing the sequence of inputs (AuctionInput) and outputs (Union[Bid, Ask]) to generate a comprehensive MemoryAnalysis. This analysis should detail the causal relationships and consequences derived from the agent's actions and the environment's responses. The goal is to provide a qualitative and rational explanation of the agent's behavior and suggest future actions based on the historical data.")
    structured_output: MemoryAnalysisTool = Field(default=MemoryAnalysisTool())
    use_schema_instruction: bool = Field(default=True)
    memory_processed: bool = Field(default=False)

    def update_state(self,  last_input: AuctionInput,last_output: Union[Bid, Ask]):
        """ Update the internal state of the agent with the last output and input """
        self.new_message = f"The last agent action was {last_output} and the environment responded with theobservation {last_input.observation} and the internal state {last_input.state}"

    def add_chat_turn_history(self, llm_output:'LLMOutput'):
        """ add a chat turn to the history without safely model copy just normal append """
        if llm_output.source_id != self.id:
            raise ValueError(f"LLMOutput source_id {llm_output.source_id} does not match the prompt context id {self.id}")
        if self.history is None:
            self.history = []
        self.history.append({"role": "user", "content": self.new_message})
        self.history.append({"role": "assistant", "content": llm_output.str_content or json.dumps(llm_output.json_object.object) if llm_output.json_object else "{}"})
        self.memory_processed = True

class MemoryAgentState(SimpleAgentState):
    """ State of the MemoryAgent """
    memory_analysis: MemoryAnalysis

    def __str__(self):
        simple_agent_state = super().__str__()
        memory_state = str(self.memory_analysis)
        return f"{simple_agent_state}\n\nMemory Analysis:\n{memory_state}"
    
    @classmethod
    def from_agent_and_observation(cls, agent: 'MemoryAgent', observation: AuctionLocalObservation):
        """ Create a new MemoryAgentState from a MemoryAgent and an observation """
        simple_agent_state = super().from_agent_and_observation(agent, observation)
        memory_processor_history = agent.memory_processor.history
        if memory_processor_history is not None and len(memory_processor_history) > 0:
            memory_analysis = MemoryAnalysis.model_validate(memory_processor_history[-1]["content"])
        else:
            memory_analysis = MemoryAnalysis(analysis="", action_suggestion="")
        return cls(memory_analysis=memory_analysis, **simple_agent_state.model_dump())
    
class MemoryAgent(SimpleAgent):
    """ SimpleAgent that uses a MemoryProcessor to compress its history into a single structured output, instead of implicetely analyzing the history through the message historie
     it uses a sub call to generate a memory analysis thread 
     memory maps MemoryAnalysis to Union[Bid,Ask] """
    memory_processor: MemoryProcessor = Field(description="The MemoryProcessor is used to compress the history of the agent into a single structured output")
    use_history: bool = Field(default=False, description="If false the agent will havea thread of MemoryProcessor/actions before the next action is taken")

    @computed_field
    @property
    def stage(self)-> str:
        if not self.memory_processor.memory_processed:
            return "Memory Analysis"
        else:
            return "Action"
    
    def update_state(self, local_observation: Optional[AuctionLocalObservation]=None):
        if not self.memory_processor.memory_processed and local_observation is not None:
            super().update_state(local_observation,update_message=False)
            self.memory_processor.update_state(last_input=self.input_history[-1],
                                           last_output=self.actions_history[-1])
        else:
            if local_observation is None:
                local_observation = self.input_history[-1].observation
            memory_state = MemoryAgentState.from_agent_and_observation(self, local_observation)
            self.new_message = str(memory_state) 
            self.memory_processor.memory_processed = False
