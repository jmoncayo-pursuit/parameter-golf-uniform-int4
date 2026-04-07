from typing import Any

from google.adk.agents import llm_agent
from google.adk.sessions import vertex_ai_session_service
from vertexai.preview.reasoning_engines import AdkApp
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import url_context

VertexAiSessionService = vertex_ai_session_service.VertexAiSessionService

class AgentClass:

  def __init__(self):
    self.app = None

  def session_service_builder(self):
    return VertexAiSessionService()

  def set_up(self):
    """Sets up the ADK application."""
    root_agent = llm_agent.LlmAgent(
      name='ParameterGolfEvolver',
      model='gemini-3-pro-preview',
      description=(
          'Delegated to when analyzing model training leaderboards, evaluating evolutionary JSON metric results, or generating new PyTorch codebase mutations to optimize bits-per-byte (BPB). Routes here when the user explicitly requests to "evolve" code, analyze past generation matrices, or write optimal Tensor math.'
      ),
      sub_agents=[],
      instruction='You are ParameterGolfEvolver, an expert evolutionary optimization agent for the OpenAI Parameter Golf challenge.\n\nYour role is to actively help evolve training code to achieve the best roundtrip BPB while staying strictly under the 16MB artifact limit.\n\nYou have access to the user\'s repositories and data stores. When the user says \"evolve <target>\" or \"analyze\", do the following:\n\n1. Understand Context: Check the current state of the relevant repo/branch (Uniform Int4@4.0).\n2. Generate Mutations: Produce 3-5 concrete, detailed code mutations for the requested target. Wrap each in clear \'# EVOLVE: target_START\' and \'# EVOLVE: target_END\' markers.\n3. Analyze Results: When the user provides run results from the evolution_leaderboard.json, compare BPB, artifact size, and stability. Recommend the best mutation and explain the trade-offs.\n4. Track Progress: Maintain memory of previous generations and suggest logical next targets.\n\nPriorities (in order):\n- Lowest possible roundtrip BPB\n- Artifact size safely under 16MB with good headroom\n- Code stability and compatibility with existing quantization/dequantization pipeline\n\nTone: Highly technical, concise, data-driven, and honest. Always ground suggestions in actual metrics. Never hallucinate numbers.',
      tools=[],
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
                        leaderboard = f.read()
                    query = f"Here is the latest data from `evolution_leaderboard.json`:\n```json\n{leaderboard}\n```\nAnalyze these results, identify the winning mutation pattern, and generate 4 new sequence-ready mutations for the next generation. VERY IMPORTANT: You must output a single valid JSON array containing exactly 4 strings (the complete Python functions) at the very end of your response, wrapped in a ```json codeblock. This allows me to easily copy-paste the array directly into Colab."
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
