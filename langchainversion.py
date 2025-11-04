"""
Multi-Agent System with LangChain (Simpler approach)
Using LangChain's built-in agent framework
"""

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================================
# SPECIALIZED AGENT FUNCTIONS
# ============================================================================

def procurement_agent(query: str) -> str:
    """Handles supplier and vendor-related queries"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"""You are a Procurement Agent specialized in supplier management.
    
Your capabilities:
- Find and evaluate suppliers
- Compare vendor offerings
- Manage purchase orders
- Track supplier performance

Query: {query}

Provide a detailed procurement analysis."""
    
    response = llm.invoke(prompt)
    return response.content


def logistics_agent(query: str) -> str:
    """Handles shipping and delivery queries"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"""You are a Logistics Agent specialized in transportation and delivery.
    
Your capabilities:
- Calculate shipping times and costs
- Optimize delivery routes
- Track shipments
- Manage warehouse operations

Query: {query}

Provide a detailed logistics analysis."""
    
    response = llm.invoke(prompt)
    return response.content


def finance_agent(query: str) -> str:
    """Handles financial queries"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"""You are a Finance Agent specialized in pricing and payments.
    
Your capabilities:
- Calculate costs and pricing
- Process invoices and payments
- Budget analysis and forecasting
- Financial risk assessment

Query: {query}

Provide a detailed financial analysis."""
    
    response = llm.invoke(prompt)
    return response.content


def compliance_agent(query: str) -> str:
    """Handles compliance and regulatory queries"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = f"""You are a Compliance Agent specialized in regulations and policies.
    
Your capabilities:
- Verify regulatory compliance
- Check vendor certifications
- Ensure policy adherence
- Audit documentation

Query: {query}

Provide a detailed compliance analysis."""
    
    response = llm.invoke(prompt)
    return response.content


# ============================================================================
# TOOLS DEFINITION
# ============================================================================

tools = [
    Tool(
        name="Procurement",
        func=procurement_agent,
        description="Use this for supplier information, vendor management, sourcing, and procurement operations. Input should be a specific procurement question."
    ),
    Tool(
        name="Logistics",
        func=logistics_agent,
        description="Use this for shipping, delivery, routes, transportation, and logistics operations. Input should be a specific logistics question."
    ),
    Tool(
        name="Finance",
        func=finance_agent,
        description="Use this for pricing, payments, invoices, budgets, and financial analysis. Input should be a specific finance question."
    ),
    Tool(
        name="Compliance",
        func=compliance_agent,
        description="Use this for regulations, policies, approvals, audits, and compliance checks. Input should be a specific compliance question."
    ),
]


# ============================================================================
# COORDINATOR AGENT (Main Agent with Tools)
# ============================================================================

def create_coordinator_agent():
    """Create the main coordinator agent that uses specialized agents as tools"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Coordinator Agent that orchestrates specialized agents.

You have access to 4 specialized agents:
1. Procurement - for supplier and vendor queries
2. Logistics - for shipping and delivery queries
3. Finance - for pricing and payment queries
4. Compliance - for regulatory and policy queries

Your job:
1. Analyze the user's query
2. Determine which specialized agent(s) to consult
3. Call the appropriate agents using the tools available
4. Synthesize the responses into a comprehensive answer

Always provide a complete, well-structured response that addresses all aspects of the query."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    return agent_executor


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def run_query(coordinator: AgentExecutor, query: str):
    """Run a single query through the coordinator"""
    print("\n" + "="*80)
    print(f"USER QUERY: {query}")
    print("="*80 + "\n")
    
    result = coordinator.invoke({"input": query})
    
    print("\n" + "="*80)
    print("FINAL RESPONSE:")
    print("="*80)
    print(result["output"])
    print("\n")
    
    return result


if __name__ == "__main__":
    # Create coordinator agent
    coordinator = create_coordinator_agent()
    
    # Example queries
    queries = [
        "Find me a supplier for steel beams with delivery in 2 weeks and calculate the total cost",
        "What are the compliance requirements for importing electronics from Asia?",
        "I need to ship 1000 units from China to the US - what are my options and costs?",
        "Review this purchase order: $50,000 for electronics from a new supplier in Vietnam"
    ]
    
    for query in queries:
        run_query(coordinator, query)


# ============================================================================
# ALTERNATIVE: Manual Orchestration (More Control)
# ============================================================================

class ManualCoordinator:
    """Manual coordination with full control over agent flow"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    def analyze_query(self, query: str) -> list[str]:
        """Determine which agents are needed"""
        prompt = f"""Given this query, which agents should be consulted?
        
Available agents: procurement, logistics, finance, compliance

Query: {query}

Respond with only the agent names separated by commas (e.g., "procurement,finance")"""
        
        response = self.llm.invoke(prompt)
        agents = [a.strip() for a in response.content.lower().split(",")]
        return agents
    
    def run(self, query: str) -> str:
        """Orchestrate agents manually"""
        print(f"\n🎯 Analyzing query: {query}")
        
        # Determine which agents to call
        agents_needed = self.analyze_query(query)
        print(f"📋 Will consult: {', '.join(agents_needed)}")
        
        results = {}
        
        # Call each agent sequentially
        if "procurement" in agents_needed:
            print("\n🏭 Calling Procurement Agent...")
            results["procurement"] = procurement_agent(query)
        
        if "logistics" in agents_needed:
            print("\n🚚 Calling Logistics Agent...")
            results["logistics"] = logistics_agent(query)
        
        if "finance" in agents_needed:
            print("\n💰 Calling Finance Agent...")
            results["finance"] = finance_agent(query)
        
        if "compliance" in agents_needed:
            print("\n📋 Calling Compliance Agent...")
            results["compliance"] = compliance_agent(query)
        
        # Synthesize results
        print("\n🔄 Synthesizing results...")
        synthesis_prompt = f"""Original Query: {query}

Agent Results:
{self._format_results(results)}

Synthesize these into a comprehensive final response."""
        
        final_response = self.llm.invoke(synthesis_prompt)
        
        return final_response.content
    
    def _format_results(self, results: dict) -> str:
        """Format results for synthesis"""
        formatted = []
        for agent, result in results.items():
            formatted.append(f"{agent.upper()}: {result}")
        return "\n\n".join(formatted)


# Example usage of manual coordinator
def test_manual_coordinator():
    coordinator = ManualCoordinator()
    
    query = "Find a supplier for industrial parts with fast shipping and verify compliance"
    result = coordinator.run(query)
    
    print("\n" + "="*80)
    print("FINAL RESPONSE:")
    print("="*80)
    print(result)


# Uncomment to test manual coordinator
# test_manual_coordinator()
