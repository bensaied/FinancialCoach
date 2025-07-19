from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi_mcp import FastApiMCP, AuthConfig
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
import pandas as pd
import json
import logging
import os
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "finance_advisor"
USER_ID = "default_user"

# ────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────
class SpendingCategory(BaseModel):
    category: str = Field(..., description="Expense category name")
    amount: float = Field(..., description="Amount spent in this category")
    percentage: Optional[float] = Field(None, description="Percentage of total spending")

class SpendingRecommendation(BaseModel):
    category: str = Field(..., description="Category for recommendation")
    recommendation: str = Field(..., description="Recommendation details")
    potential_savings: Optional[float] = Field(None, description="Estimated monthly savings")

class BudgetAnalysis(BaseModel):
    total_expenses: float = Field(..., description="Total monthly expenses")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    spending_categories: List[SpendingCategory] = Field(..., description="Breakdown of spending by category")
    recommendations: List[SpendingRecommendation] = Field(..., description="Spending recommendations")

class EmergencyFund(BaseModel):
    recommended_amount: float = Field(..., description="Recommended emergency fund size")
    current_amount: Optional[float] = Field(None, description="Current emergency fund (if any)")
    current_status: str = Field(..., description="Status assessment of emergency fund")

class SavingsRecommendation(BaseModel):
    category: str = Field(..., description="Savings category")
    amount: float = Field(..., description="Recommended monthly amount")
    rationale: Optional[str] = Field(None, description="Explanation for this recommendation")

class AutomationTechnique(BaseModel):
    name: str = Field(..., description="Name of automation technique")
    description: str = Field(..., description="Details of how to implement")

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund = Field(..., description="Emergency fund recommendation")
    recommendations: List[SavingsRecommendation] = Field(..., description="Savings allocation recommendations")
    automation_techniques: Optional[List[AutomationTechnique]] = Field(None, description="Automation techniques to help save")

class Debt(BaseModel):
    name: str = Field(..., description="Name of debt")
    amount: float = Field(..., description="Current balance")
    interest_rate: float = Field(..., description="Annual interest rate (%)")
    min_payment: Optional[float] = Field(None, description="Minimum monthly payment")

class PayoffPlan(BaseModel):
    total_interest: float = Field(..., description="Total interest paid")
    months_to_payoff: int = Field(..., description="Months until debt-free")
    monthly_payment: Optional[float] = Field(None, description="Recommended monthly payment")

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan = Field(..., description="Highest interest first method")
    snowball: PayoffPlan = Field(..., description="Smallest balance first method")

class DebtRecommendation(BaseModel):
    title: str = Field(..., description="Title of recommendation")
    description: str = Field(..., description="Details of recommendation")
    impact: Optional[str] = Field(None, description="Expected impact of this action")

class DebtReduction(BaseModel):
    total_debt: float = Field(..., description="Total debt amount")
    debts: List[Debt] = Field(..., description="List of all debts")
    payoff_plans: PayoffPlans = Field(..., description="Debt payoff strategies")
    recommendations: Optional[List[DebtRecommendation]] = Field(None, description="Recommendations for debt reduction")

class FinancialDataInput(BaseModel):
    monthly_income: float = Field(..., ge=0, description="Monthly income")
    dependants: int = Field(..., ge=0, description="Number of dependants")
    transactions: Optional[List[Dict]] = Field(None, description="Transaction records")
    manual_expenses: Optional[Dict[str, float]] = Field(None, description="Manual expenses")
    debts: Optional[List[Debt]] = Field(None, description="List of debts")

    @field_validator("manual_expenses", mode="before")
    @classmethod
    def parse_manual_expenses(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string for manual_expenses: {str(e)}")
        return value

    @field_validator("debts", mode="before")
    @classmethod
    def parse_debts(cls, value):
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("Debts must be a list of debt objects")
                return [Debt(**item) for item in parsed]
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string for debts: {str(e)}")
        return value

class FinancialAnalysisResponse(BaseModel):
    budget_analysis: Optional[BudgetAnalysis] = Field(None, description="Budget analysis results")
    savings_strategy: Optional[SavingsStrategy] = Field(None, description="Savings strategy results")
    debt_reduction: Optional[DebtReduction] = Field(None, description="Debt reduction results")

# ────────────────────────────────────────────────
# Auth via MCP header or query parameter
# ────────────────────────────────────────────────
def verify_auth(request: Request):
    logger.info(f"Headers: {dict(request.headers)}")

    # Ensure GOOGLE_API_KEY is available
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("Missing GOOGLE_API_KEY environment variable")
        raise HTTPException(status_code=401, detail="Missing GOOGLE_API_KEY environment variable")

    logger.info(f"Retrieved GOOGLE_API_KEY from environment: {google_api_key[:4]}... (obfuscated)")

    return {"googleApiKey": google_api_key}


# ────────────────────────────────────────────────
# Core System Logic
# ────────────────────────────────────────────────
class BudgetAnalysisSystem:
    def __init__(self):
        self.session_service = InMemorySessionService()
        
        self.budget_analysis_agent = LlmAgent(
            name="BudgetAnalysisAgent",
            model="gemini-1.5-flash",
            description="Analyzes financial data to categorize spending patterns and recommend budget improvements",
            instruction="""You are a Budget Analysis Agent specialized in reviewing financial transactions and expenses.
You are the first agent in a sequence of three financial advisor agents.

Your tasks:
1. Analyze income, transactions, and expenses in detail
2. Categorize spending into logical groups with clear breakdown
3. Identify spending patterns and trends across categories
4. Suggest specific areas where spending could be reduced with concrete suggestions
5. Provide actionable recommendations with specific, quantified potential savings amounts

Consider:
- Number of dependants when evaluating household expenses
- Typical spending ratios for the income level (housing 30%, food 15%, etc.)
- Essential vs discretionary spending with clear separation
- Seasonal spending patterns if data spans multiple months

For spending categories, include ALL expenses from the user's data, ensure percentages add up to 100%,
and make sure every expense is categorized.

For recommendations:
- Provide at least 3-5 specific, actionable recommendations with estimated savings
- Explain the reasoning behind each recommendation
- Consider the impact on quality of life and long-term financial health
- Suggest specific implementation steps for each recommendation

Store your analysis in state['budget_analysis'] as a JSON string with the format:
{
  "total_expenses": float,
  "monthly_income": float,
  "spending_categories": [
    {"category": str, "amount": float, "percentage": float}
  ],
  "recommendations": [
    {"category": str, "recommendation": str, "potential_savings": float}
  ]
}
"""
        )
        
        self.savings_strategy_agent = LlmAgent(
            name="SavingsStrategyAgent",
            model="gemini-1.5-flash",
            description="Recommends optimal savings strategies based on income, expenses, and financial goals",
            instruction="""You are a Savings Strategy Agent specialized in creating personalized savings plans.
You are the second agent in the sequence. READ the budget analysis from state['budget_analysis'] first.

Your tasks:
1. Review the budget analysis results from state['budget_analysis']
2. Recommend comprehensive savings strategies based on the analysis
3. Calculate optimal emergency fund size based on expenses and dependants
4. Suggest appropriate savings allocation across different purposes
5. Recommend practical automation techniques for saving consistently

Consider:
- Risk factors based on job stability and dependants
- Balancing immediate needs with long-term financial health
- Progressive savings rates as discretionary income increases
- Multiple savings goals (emergency, retirement, specific purchases)
- Areas of potential savings identified in the budget analysis

Store your strategy in state['savings_strategy'] as a JSON string with the format:
{
  "emergency_fund": {
    "recommended_amount": float,
    "current_amount": float,
    "current_status": str
  },
  "recommendations": [
    {"category": str, "amount": float, "rationale": str}
  ],
  "automation_techniques": [
    {"name": str, "description": str}
  ]
}
"""
        )
        
        self.debt_reduction_agent = LlmAgent(
            name="DebtReductionAgent",
            model="gemini-1.5-flash",
            description="Creates optimized debt payoff plans to minimize interest paid and time to debt freedom",
            instruction="""You are a Debt Reduction Agent specialized in creating debt payoff strategies.
You are the final agent in the sequence. READ both state['budget_analysis'] and state['savings_strategy'] first.

Your tasks:
1. Review both budget analysis and savings strategy from the state
2. Analyze debts by interest rate, balance, and minimum payments
3. Create prioritized debt payoff plans (avalanche and snowball methods)
4. Calculate total interest paid and time to debt freedom
5. Suggest debt consolidation or refinancing opportunities
6. Provide specific recommendations to accelerate debt payoff

Consider:
- Cash flow constraints from the budget analysis
- Emergency fund and savings goals from the savings strategy
- Psychological factors (quick wins vs mathematical optimization)
- Credit score impact and improvement opportunities

Store your final plan in state['debt_reduction'] as a JSON string with the format:
{
  "total_debt": float,
  "debts": [
    {"name": str, "amount": float, "interest_rate": float, "min_payment": float}
  ],
  "payoff_plans": {
    "avalanche": {
      "total_interest": float,
      "months_to_payoff": int,
      "monthly_payment": float
    },
    "snowball": {
      "total_interest": float,
      "months_to_payoff": int,
      "monthly_payment": float
    }
  },
  "recommendations": [
    {"title": str, "description": str, "impact": str}
  ]
}
"""
        )
        
        self.coordinator_agent = SequentialAgent(
            name="FinanceCoordinatorAgent",
            description="Coordinates specialized finance agents to provide comprehensive financial advice",
            sub_agents=[
                self.budget_analysis_agent,
                self.savings_strategy_agent,
                self.debt_reduction_agent
            ]
        )
        
        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    async def analyze_budget(self, financial_data: FinancialDataInput, auth_data: dict = Depends(verify_auth)) -> FinancialAnalysisResponse:
        logger.info(f"Using GOOGLE_API_KEY: {auth_data['googleApiKey'][:4]}... (obfuscated)")
        # GOOGLE_API_KEY is already set via environment variable
        
        session_id = f"budget_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            session = self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state={
                    "monthly_income": financial_data.monthly_income,
                    "dependants": financial_data.dependants,
                    "transactions": financial_data.transactions or [],
                    "manual_expenses": financial_data.manual_expenses or {},
                    "debts": [debt.dict() for debt in financial_data.debts] if financial_data.debts else []
                }
            )
            if session.state.get("transactions"):
                self._preprocess_transactions(session)
            if session.state.get("manual_expenses"):
                self._preprocess_manual_expenses(session)

            # Create user content for the coordinator agent
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=json.dumps(financial_data.dict()))]
            )

            # Run the sequential agent
            logger.info("Starting sequential agent execution")
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                logger.debug(f"Event received: author={event.author}, is_final={event.is_final_response()}")
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    logger.info("Sequential agent completed execution")
                    break

            updated_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )

            # Collect results from all agents
            results = {}
            default_results = self._create_default_results(financial_data)
            for key, model in [
                ("budget_analysis", BudgetAnalysis),
                ("savings_strategy", SavingsStrategy),
                ("debt_reduction", DebtReduction)
            ]:
                value = updated_session.state.get(key)
                if value:
                    try:
                        results[key] = model.parse_raw(value) if isinstance(value, str) else model(**value)
                        logger.info(f"Successfully retrieved {key} from session state")
                    except Exception as e:
                        logger.warning(f"Failed to parse {key}: {str(e)}, using default")
                        results[key] = default_results[key]
                else:
                    logger.info(f"No {key} result in session state, using default")
                    results[key] = default_results[key]

            return FinancialAnalysisResponse(**results)
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        finally:
            self.session_service.delete_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    def _preprocess_transactions(self, session):
        df = pd.DataFrame(session.state.get("transactions", []))
        if 'Category' in df.columns and 'Amount' in df.columns:
            session.state["category_spending"] = df.groupby('Category')['Amount'].sum().to_dict()
            session.state["total_spending"] = df['Amount'].sum()

    def _preprocess_manual_expenses(self, session):
        expenses = session.state.get("manual_expenses", {})
        session.state["total_manual_spending"] = sum(expenses.values())
        session.state["manual_category_spending"] = expenses

    def _create_default_results(self, financial_data: FinancialDataInput) -> Dict[str, any]:
        logger.info("Generating default results")
        expenses = financial_data.manual_expenses or {}
        if not expenses and financial_data.transactions:
            expenses = pd.DataFrame(financial_data.transactions).groupby('Category')['Amount'].sum().to_dict()

        total_expenses = sum(expenses.values())
        total_debt = sum(debt.amount for debt in financial_data.debts) if financial_data.debts else 0

        return {
            "budget_analysis": BudgetAnalysis(
                total_expenses=total_expenses,
                monthly_income=financial_data.monthly_income,
                spending_categories=[
                    SpendingCategory(category=cat, amount=amt, percentage=(amt / total_expenses * 100) if total_expenses > 0 else 0)
                    for cat, amt in expenses.items()
                ],
                recommendations=[
                    SpendingRecommendation(category="General", recommendation="Review expenses", potential_savings=total_expenses * 0.1)
                ]
            ),
            "savings_strategy": SavingsStrategy(
                emergency_fund=EmergencyFund(
                    recommended_amount=total_expenses * 6,
                    current_amount=0,
                    current_status="Not started"
                ),
                recommendations=[
                    SavingsRecommendation(category="Emergency Fund", amount=total_expenses * 0.1, rationale="Build emergency fund first"),
                    SavingsRecommendation(category="Retirement", amount=financial_data.monthly_income * 0.15, rationale="Long-term savings")
                ],
                automation_techniques=[
                    AutomationTechnique(name="Automatic Transfer", description="Set up automatic transfers on payday")
                ]
            ),
            "debt_reduction": DebtReduction(
                total_debt=total_debt,
                debts=financial_data.debts or [],
                payoff_plans=PayoffPlans(
                    avalanche=PayoffPlan(
                        total_interest=total_debt * 0.2,
                        months_to_payoff=24,
                        monthly_payment=total_debt / 24 if total_debt > 0 else 0
                    ),
                    snowball=PayoffPlan(
                        total_interest=total_debt * 0.25,
                        months_to_payoff=24,
                        monthly_payment=total_debt / 24 if total_debt > 0 else 0
                    )
                ),
                recommendations=[
                    DebtRecommendation(title="Increase Payments", description="Increase your monthly payments", impact="Reduces total interest paid")
                ]
            )
        }

# ────────────────────────────────────────────────
# App & Routing
# ────────────────────────────────────────────────
app = FastAPI(title="Budget Analysis API", version="1.0.0")
budget_system = BudgetAnalysisSystem()

@app.post("/analyze_budget", response_model=FinancialAnalysisResponse)
async def analyze_budget_tool(financial_data: FinancialDataInput, auth_data: dict = Depends(verify_auth)):
    return await budget_system.analyze_budget(financial_data, auth_data)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ────────────────────────────────────────────────
# Mount MCP with Auth
# ────────────────────────────────────────────────
mcp = FastApiMCP(
    app,
    name="Protected MCP",
    auth_config=AuthConfig(
        dependencies=[Depends(verify_auth)]
    )
)
mcp.mount()

# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)