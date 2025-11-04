"""
Multi-Agent System with LangGraph
Coordinator orchestrates 4 specialized agents: Procurement, Logistics, Finance, Compliance
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import operator

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    coordinator_decision: str
    procurement_result: str
    logistics_result: str
    finance_result: str
    compliance_result: str
    next_agent: str
    final_response: str


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class CoordinatorAgent:
    """Orchestrates which specialized agents to call"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Coordinator Agent that routes queries to specialized agents.
            
Available agents:
- procurement: For supplier information, vendor management, sourcing
- logistics: For shipping, delivery, routes, transportation
- finance: For pricing, payments, invoices, budgets
- compliance: For regulations, policies, approvals, audits

Analyze the user query and decide which agent(s) to route to.
Respond with ONLY the agent name(s) separated by commas.
Examples: "procurement,finance" or "logistics" or "compliance,procurement"
"""),
            ("human", "{query}")
        ])
    
    def run(self, state: AgentState) -> AgentState:
        print("\n🎯 COORDINATOR: Analyzing query...")
        
        response = self.llm.invoke(
            self.prompt.format_messages(query=state["user_query"])
        )
        
        decision = response.content.strip().lower()
        state["coordinator_decision"] = decision
        
        # Determine next agent to call
        agents = [a.strip() for a in decision.split(",")]
        state["next_agent"] = agents[0] if agents else "end"
        
        print(f"📋 Decision: Route to {decision}")
        return state


class ProcurementAgent:
    """Handles supplier and vendor-related queries"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Procurement Agent specialized in supplier management, 
            vendor selection, sourcing, and procurement operations.
            
Your capabilities:
- Find and evaluate suppliers
- Compare vendor offerings
- Manage purchase orders
- Track supplier performance
- Source materials and products

Provide detailed, actionable procurement insights."""),
            ("human", "{query}")
        ])
    
    def run(self, state: AgentState) -> AgentState:
        print("\n🏭 PROCUREMENT AGENT: Processing...")
        
        response = self.llm.invoke(
            self.prompt.format_messages(query=state["user_query"])
        )
        
        state["procurement_result"] = response.content
        state["messages"].append(AIMessage(content=f"Procurement: {response.content}"))
        
        # Route to next agent based on coordinator decision
        state["next_agent"] = self._get_next_agent(state["coordinator_decision"])
        
        print(f"✅ Result: {response.content[:100]}...")
        return state
    
    def _get_next_agent(self, decision: str) -> str:
        agents = [a.strip() for a in decision.split(",")]
        if "logistics" in agents:
            return "logistics"
        elif "finance" in agents:
            return "finance"
        elif "compliance" in agents:
            return "compliance"
        return "synthesize"


class LogisticsAgent:
    """Handles shipping and delivery queries"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Logistics Agent specialized in transportation, 
            delivery, shipping routes, and supply chain logistics.
            
Your capabilities:
- Calculate shipping times and costs
- Optimize delivery routes
- Track shipments
- Manage warehouse operations
- Coordinate transportation

Provide detailed logistics solutions."""),
            ("human", "{query}")
        ])
    
    def run(self, state: AgentState) -> AgentState:
        print("\n🚚 LOGISTICS AGENT: Processing...")
        
        response = self.llm.invoke(
            self.prompt.format_messages(query=state["user_query"])
        )
        
        state["logistics_result"] = response.content
        state["messages"].append(AIMessage(content=f"Logistics: {response.content}"))
        
        state["next_agent"] = self._get_next_agent(state["coordinator_decision"])
        
        print(f"✅ Result: {response.content[:100]}...")
        return state
    
    def _get_next_agent(self, decision: str) -> str:
        agents = [a.strip() for a in decision.split(",")]
        if "finance" in agents:
            return "finance"
        elif "compliance" in agents:
            return "compliance"
        return "synthesize"


class FinanceAgent:
    """Handles financial queries"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Finance Agent specialized in pricing, payments, 
            invoices, budgets, and financial analysis.
            
Your capabilities:
- Calculate costs and pricing
- Process invoices and payments
- Budget analysis and forecasting
- Financial risk assessment
- Payment terms negotiation

Provide detailed financial insights."""),
            ("human", "{query}")
        ])
    
    def run(self, state: AgentState) -> AgentState:
        print("\n💰 FINANCE AGENT: Processing...")
        
        response = self.llm.invoke(
            self.prompt.format_messages(query=state["user_query"])
        )
        
        state["finance_result"] = response.content
        state["messages"].append(AIMessage(content=f"Finance: {response.content}"))
        
        state["next_agent"] = self._get_next_agent(state["coordinator_decision"])
        
        print(f"✅ Result: {response.content[:100]}...")
        return state
    
    def _get_next_agent(self, decision: str) -> str:
        agents = [a.strip() for a in decision.split(",")]
        if "compliance" in agents:
            return "compliance"
        return "synthesize"


class ComplianceAgent:
    """Handles compliance and regulatory queries"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Compliance Agent specialized in regulations, 
            policies, approvals, and audits.
            
Your capabilities:
- Verify regulatory compliance
- Check vendor certifications
- Ensure policy adherence
- Audit documentation
- Risk and compliance assessment

Provide detailed compliance insights."""),
            ("human", "{query}")
        ])
    
    def run(self, state: AgentState) -> AgentState:
        print("\n📋 COMPLIANCE AGENT: Processing...")
        
        response = self.llm.invoke(
            self.prompt.format_messages(query=state["user_query"])
        )
        
        state["compliance_result"] = response.content
        state["messages"].append(AIMessage(content=f"Compliance: {response.content}"))
        
        state["next_agent"] = "synthesize"
        
        print(f"✅ Result: {response.content[:100]}...")
        return state


class SynthesisAgent:
    """Combines results from all agents into final response"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Synthesis Agent that combines insights from multiple 
            specialized agents into a coherent, comprehensive response.
            
Create a well-structured final answer that:
1. Addresses the user's original query
2. Integrates insights from all consulted agents
3. Provides actionable recommendations
4. Is clear and concise

Format your response professionally."""),
            ("human", """Original Query: {query}

Agent Results:
{results}

Synthesize these into a final response.""")
        ])
    
    def run(self, state: AgentState) -> AgentState:
        print("\n🔄 SYNTHESIS AGENT: Combining results...")
        
        # Gather all results
        results = []
        if state.get("procurement_result"):
            results.append(f"Procurement: {state['procurement_result']}")
        if state.get("logistics_result"):
            results.append(f"Logistics: {state['logistics_result']}")
        if state.get("finance_result"):
            results.append(f"Finance: {state['finance_result']}")
        if state.get("compliance_result"):
            results.append(f"Compliance: {state['compliance_result']}")
        
        results_text = "\n\n".join(results)
        
        response = self.llm.invoke(
            self.prompt.format_messages(
                query=state["user_query"],
                results=results_text
            )
        )
        
        state["final_response"] = response.content
        state["next_agent"] = "end"
        
        print(f"✅ Final response generated")
        return state


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_to_agent(state: AgentState) -> str:
    """Determine next node based on state"""
    next_agent = state.get("next_agent", "end")
    
    if next_agent == "end":
        return END
    
    return next_agent


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent_graph():
    """Build the LangGraph workflow"""
    
    # Initialize LLM (using OpenAI, but could use Claude via Bedrock)
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Initialize agents
    coordinator = CoordinatorAgent(llm)
    procurement = ProcurementAgent(llm)
    logistics = LogisticsAgent(llm)
    finance = FinanceAgent(llm)
    compliance = ComplianceAgent(llm)
    synthesis = SynthesisAgent(llm)
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator.run)
    workflow.add_node("procurement", procurement.run)
    workflow.add_node("logistics", logistics.run)
    workflow.add_node("finance", finance.run)
    workflow.add_node("compliance", compliance.run)
    workflow.add_node("synthesize", synthesis.run)
    
    # Set entry point
    workflow.set_entry_point("coordinator")
    
    # Add conditional edges from coordinator
    workflow.add_conditional_edges(
        "coordinator",
        route_to_agent,
        {
            "procurement": "procurement",
            "logistics": "logistics",
            "finance": "finance",
            "compliance": "compliance",
            "end": END
        }
    )
    
    # Add conditional edges from each agent
    for agent in ["procurement", "logistics", "finance", "compliance"]:
        workflow.add_conditional_edges(
            agent,
            route_to_agent,
            {
                "procurement": "procurement",
                "logistics": "logistics",
                "finance": "finance",
                "compliance": "compliance",
                "synthesize": "synthesize",
                "end": END
            }
        )
    
    # Synthesis always goes to end
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create the graph
    app = create_agent_graph()
    
    # Example queries
    queries = [
        "Find me a supplier for steel beams with delivery in 2 weeks",
        "What are the compliance requirements for importing electronics?",
        "Calculate the total cost including shipping for 1000 units from China",
    ]
    
    for query in queries:
        print("\n" + "="*80)
        print(f"USER QUERY: {query}")
        print("="*80)
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "coordinator_decision": "",
            "procurement_result": "",
            "logistics_result": "",
            "finance_result": "",
            "compliance_result": "",
            "next_agent": "",
            "final_response": ""
        }
        
        # Run the graph
        result = app.invoke(initial_state)
        
        print("\n" + "="*80)
        print("FINAL RESPONSE:")
        print("="*80)
        print(result["final_response"])
        print("\n")
