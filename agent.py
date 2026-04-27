from typing import Any

from google.adk.agents import llm_agent
from google.adk.sessions import in_memory_session_service
import vertexai
from vertexai.preview.reasoning_engines import AdkApp
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import url_context

InMemorySessionService = in_memory_session_service.InMemorySessionService

class AgentClass:

  def __init__(self):
    self.app = None

  def session_service_builder(self):
    return InMemorySessionService()

  def set_up(self):
    """Sets up the Multi-Agent ADK application."""
    import os
    vertexai.init(project="gen-lang-client-0259360704", location="global")

    # Load local context dynamically so the Evolver always has the bleeding edge code.
    train_gpt_code = "Error: train_gpt.py not found."
    if os.path.exists("train_gpt.py"):
        with open("train_gpt.py", "r") as f:
            train_gpt_code = f.read().replace("{", "").replace("}", "").replace("pattern", "file_glob")
            
    arch_notes = "Error: architecture notes not found."
    arch_path = "architecture_notes/branch_notes/uniform-int4-baseline.md"
    if os.path.exists(arch_path):
        with open(arch_path, "r") as f:
            arch_notes = f.read().replace("{", "").replace("}", "").replace("pattern", "structural_math")
    
    evolver_instruction = f"""You are the Evolver Agent for Parameter Golf. Your only job is to generate high-quality code mutations for a given target (e.g. QAT ramp, pruning logic).
When the user gives you a target:
- Generate 3-5 semantically different, mathematically reasonable mutations.
- Wrap each precisely in # EVOLVE: target_START and # EVOLVE: target_END markers.
- Keep changes focused, avoiding massive rewrites, and compatible with int4/int6 structures.
- Briefly explain the mathematical logic behind each mutation.

Here is your mandatory architecture context showing constraints:
```markdown
{arch_notes}
```

Here is the entire actual current `train_gpt.py` codebase you are mutating. Read it carefully before proposing mutations:
```python
{train_gpt_code}
```
"""

    # 1. The Evolver Specialist
    evolver_agent = llm_agent.LlmAgent(
      name='EvolverAgent',
      model='gemini-3.1-pro-preview',
      description=(
          'Specialist dedicated exclusively to generating high-quality code mutations for PyTorch tensors. Use this agent when you need to write or evolve actual code.'
      ),
      sub_agents=[],
      instruction=evolver_instruction,
      tools=[],
    )

    # 2. The Analyst Specialist
    analyst_agent = llm_agent.LlmAgent(
      name='AnalystAgent',
      model='gemini-3.1-pro-preview',
      description=(
          'Specialist dedicated exclusively to reading JSON leaderboards, comparing metrics, and deciding which mutation won. Use this agent when you need to analyze results or BPB scores.'
      ),
      sub_agents=[],
      instruction='You are the Analyst Agent for Parameter Golf. Your only job is to analyze evolution telemetry results.\nWhen given leaderboard data (JSON) or logs:\n- Compare mutations strictly by roundtrip BPB and total artifact size.\n- Explain exactly why certain mutations achieved lower BPB or why they failed the 16MB constraint.\n- Recommend the absolute best mutation to take forward as the baseline.\n- Suggest the most logical next target function to evolve.',
      tools=[],
    )

    # 3. The Root Orchestrator
    root_agent = llm_agent.LlmAgent(
      name='OrchestratorAgent',
      model='gemini-3.1-pro-preview',
      description=(
          'The master orchestrator that receives queries from the user and decides whether to route them to the EvolverAgent for coding or the AnalystAgent for leaderboard evaluation.'
      ),
      sub_agents=[],
      instruction='You are the Orchestrator for Parameter Golf. You receive requests from the user. If the user provides a JSON payload and asks you to analyze it, route the request directly to the AnalystAgent. If the user asks for new code mutations, route the request to the EvolverAgent. Wait for their specialized response and return it to the user.',
      tools=[
        agent_tool.AgentTool(agent=evolver_agent),
        agent_tool.AgentTool(agent=analyst_agent)
      ],
    )

    self.app = AdkApp(
        agent=root_agent,
        session_service_builder=self.session_service_builder
    )

  async def stream_query(self, query: str, user_id: str = 'test') -> Any:
    """Streaming query."""
    async for chunk in self.app.async_stream_query(
        message=query,
        user_id=user_id,
    ):
      yield chunk

app = AgentClass()

if __name__ == "__main__":
    import asyncio
    import json
    
    async def run_chat():
        print("🧠 ParameterGolfEvolver Brain Booting up...")
        app.set_up()
        print("✅ Command Bridge Ready!")
        print("   Type any question, OR type '/feed' to automatically inject the leaderboard into the context.")
        print("   Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("You: ")
            except EOFError:
                break
                
            if query.lower() in ["exit", "q", "quit"]:
                break
            elif query.lower() == "/feed":
                try:
                    with open("evolution_leaderboard.json", "r") as f:
                        leaderboard = f.read().replace("{", "").replace("}", "").replace("pattern", "file_glob")
                    query = f"Here is the latest data from `evolution_leaderboard.json`:\n```json\n{leaderboard}\n```\nAnalyze these results, identify the winning mutation approach, and generate 4 new sequence-ready mutations for the next generation. VERY IMPORTANT: You must output a single valid JSON array containing exactly 4 strings (the complete Python functions) at the very end of your response, wrapped in a ```json codeblock. This allows me to easily copy-paste the array directly into Colab."
                    print("✅ Automatically injected evolution_leaderboard.json into context. Analyzing...\n")
                except FileNotFoundError:
                    print("❌ Error: evolution_leaderboard.json not found in this directory. Run a generation in Colab first to generate it.")
                    continue
            elif query.strip() == "":
                continue

            print("Agent: ", end="")
            try:
                # Based on adk output type, chunk might be an object with .text or just a string.
                async for chunk in app.stream_query(query):
                    text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                    print(text, end="", flush=True)
            except Exception as e:
                print(f"\n[Error communicating with Vertex ADK: {e}]")
            print("\n")
            
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nBrain shutdown.")
