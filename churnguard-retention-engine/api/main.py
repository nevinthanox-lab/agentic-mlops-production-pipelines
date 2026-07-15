import os
import uuid
import logging
from typing import List, Optional
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from supabase import create_client, Client

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Initialize environment configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Infrastructure credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your_supabase_service_role_key")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is missing from environment variables. Agent drafting calls will fail.")

# Database client initialization
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logger.error(f"Failed to connect to Supabase: {e}")
    raise RuntimeError("Supabase configuration invalid. Check environment variables.")

# ==========================================
# 1. STATE DEFINITIONS & SCHEMAS
# ==========================================
class AgentState(TypedDict):
    """Internal graph memory state."""
    account_id: str
    arr: float
    top_3_shap_drivers: List[str]
    churn_probability: float
    drafted_offer: str
    policy_passed: bool
    approved: bool
    human_feedback: str

class InvokeRequest(BaseModel):
    """Validates the incoming payload from upstream ML systems (Pydantic v2)."""
    model_config = ConfigDict(extra="forbid", strict=True)

    account_id: str = Field(..., description="Unique identifier for the SaaS account")
    arr: float = Field(..., gt=0.0, description="Annual Recurring Revenue in USD")
    top_3_shap_drivers: List[str] = Field(..., min_length=1, max_length=3, description="Top 3 SHAP feature drivers")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted churn probability via XGBoost")

class ResumeRequest(BaseModel):
    """Validates human interaction callbacks (Slack/n8n approval)."""
    model_config = ConfigDict(extra="forbid", strict=True)

    thread_id: str = Field(..., description="Unique execution thread ID")
    approved: bool = Field(..., description="Approval outcome")
    human_feedback: Optional[str] = Field(default="", description="Optional feedback from support agent")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def audit_transition(account_id: str, node: str, details: dict) -> None:
    """Logs transition snapshots to PostgreSQL audit layers."""
    try:
        data = {
            "account_id": account_id,
            "transition_node": node,
            "state_snapshot": details
        }
        supabase.table("churnguard_audit").insert(data).execute()
        logger.info(f"Audited transition: {account_id} @ {node}")
    except Exception as e:
        logger.error(f"Failed to commit transition audit: {e}")

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def analyze_risk_context(state: AgentState) -> dict:
    """Context analysis step to verify upstream prediction data."""
    logger.info(f"Processing context analysis for {state['account_id']}")
    audit_transition(state["account_id"], "AnalyzeRiskContext", {"churn_probability": state["churn_probability"]})
    return state

def draft_retention_offer(state: AgentState) -> dict:
    """Generates an offer tailored to key drop-off drivers using Llama-3.3 70B on Groq's inference API."""
    logger.info(f"Generating offer for {state['account_id']} via Groq ({GROQ_MODEL})")

    # Native Groq integration - high-throughput, low-latency LPU inference
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite retention AI acting as ChurnGuard. Output exactly two sentences. First sentence acknowledges the customer's specific usage drop based on the provided key drivers. Second sentence offers a targeted discount or feature access tailored to their tier."),
        ("human", "Account ARR: ${arr}. Churn Risk: {churn_probability}. Key SHAP Drivers: {drivers}.")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "arr": state["arr"],
        "churn_probability": state["churn_probability"],
        "drivers": ", ".join(state["top_3_shap_drivers"])
    })

    draft = response.content
    audit_transition(state["account_id"], "DraftRetentionOffer", {"drafted_offer": draft, "model": GROQ_MODEL})
    return {"drafted_offer": draft}

def policy_guardrail_check(state: AgentState) -> dict:
    """Ensures offers align with financial policy constraints."""
    logger.info(f"Enforcing safety policy for {state['account_id']}")
    policy_passed = state["arr"] >= 500.0
    audit_transition(state["account_id"], "PolicyGuardrailCheck", {"policy_passed": policy_passed})
    return {"policy_passed": policy_passed}

def hitl_approval_gate(state: AgentState) -> dict:
    """Consolidates human feedback into runtime state memory."""
    logger.info(f"Processing HITL decision for {state['account_id']}")
    audit_transition(state["account_id"], "HITLApprovalGate", {
        "approved": state["approved"],
        "feedback": state["human_feedback"]
    })
    return state

# ==========================================
# 4. STATE MACHINE COMPILATION
# ==========================================
workflow = StateGraph(AgentState)

# Define processing nodes
workflow.add_node("analyze_risk_context", analyze_risk_context)
workflow.add_node("draft_retention_offer", draft_retention_offer)
workflow.add_node("policy_guardrail_check", policy_guardrail_check)
workflow.add_node("hitl_approval_gate", hitl_approval_gate)

# Establish workflow execution flow
workflow.set_entry_point("analyze_risk_context")
workflow.add_edge("analyze_risk_context", "draft_retention_offer")
workflow.add_edge("draft_retention_offer", "policy_guardrail_check")
workflow.add_edge("policy_guardrail_check", "hitl_approval_gate")
workflow.add_edge("hitl_approval_gate", END)

# In-memory persistence engine
memory = MemorySaver()

# Compile graph with human-in-the-loop interrupt breakpoint
app_graph = workflow.compile(checkpointer=memory, interrupt_before=["hitl_approval_gate"])

# ==========================================
# 5. DISPATCH LAYER (FASTAPI)
# ==========================================
app = FastAPI(
    title="ChurnGuard: Predictive Churn & Agentic Retention Orchestrator",
    version="1.0.0",
    description="MLOps orchestrator implementing HITL state-machines for automated customer retention, powered by Groq (Llama-3.3-70B)."
)

@app.post("/api/v1/invoke", status_code=status.HTTP_202_ACCEPTED)
async def invoke_agent(payload: InvokeRequest):
    """Triggers the async state machine execution up to the HITL breakpoint."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "account_id": payload.account_id,
        "arr": payload.arr,
        "top_3_shap_drivers": payload.top_3_shap_drivers,
        "churn_probability": payload.churn_probability,
        "drafted_offer": "",
        "policy_passed": False,
        "approved": False,
        "human_feedback": ""
    }

    try:
        # Step through nodes until reaching the interrupt boundary
        for _ in app_graph.stream(initial_state, config=config):
            pass

        current_state = app_graph.get_state(config)

        return {
            "status": "pending_human_approval",
            "thread_id": thread_id,
            "current_node": current_state.next[0] if current_state.next else None,
            "drafted_offer": current_state.values.get("drafted_offer", ""),
            "policy_passed": current_state.values.get("policy_passed", False),
            "account_id": payload.account_id,
            "arr_context": payload.arr
        }
    except Exception as e:
        logger.error(f"Inference run aborted: {e}")
        raise HTTPException(status_code=500, detail="Internal processing failure during execution.")

@app.post("/api/v1/resume", status_code=status.HTTP_200_OK)
async def resume_agent(payload: ResumeRequest):
    """Receives callback details and advances the state machine to completion."""
    config = {"configurable": {"thread_id": payload.thread_id}}
    current_state = app_graph.get_state(config)

    if not current_state or not current_state.next:
        raise HTTPException(status_code=404, detail="Execution thread is inactive or expired.")

    try:
        # Patch human decision into memory state
        app_graph.update_state(
            config,
            {"approved": payload.approved, "human_feedback": payload.human_feedback}
        )

        # Resume execution stream to terminal state
        for _ in app_graph.stream(None, config=config):
            pass

        final_state = app_graph.get_state(config)

        return {
            "status": "completed",
            "thread_id": payload.thread_id,
            "final_decision": "approved" if final_state.values.get("approved") else "rejected",
            "account_id": final_state.values.get("account_id")
        }
    except Exception as e:
        logger.error(f"Thread resume failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing failure during execution resume.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
