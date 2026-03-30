from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress
import json

from src.tools.api import get_financial_metrics


##### Fundamental Agent #####
def fundamentals_analyst_agent(state: AgentState, agent_id: str = "fundamentals_analyst_agent"):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        if not financial_metrics:
            progress.update_status(agent_id, ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status(agent_id, ticker, "Analyzing profitability")
        # 1. Profitability Analysis (weight: 0.35 — most predictive of long-term returns)
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        profitability_checks = [
            (return_on_equity, 0.15, 0.50),  # Strong ROE above 15% (highest weight)
            (net_margin, 0.20, 0.25),  # Healthy profit margins
            (operating_margin, 0.15, 0.25),  # Strong operating efficiency
        ]
        profitability_score = sum(weight for metric, threshold, weight in profitability_checks if metric is not None and metric > threshold)

        signals.append("bullish" if profitability_score >= 0.5 else "bearish" if profitability_score == 0 else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing growth")
        # 2. Growth Analysis (weight: 0.25)
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        growth_checks = [
            (revenue_growth, 0.10, 0.40),  # 10% revenue growth (most important)
            (earnings_growth, 0.10, 0.35),  # 10% earnings growth
            (book_value_growth, 0.10, 0.25),  # 10% book value growth
        ]
        growth_score = sum(weight for metric, threshold, weight in growth_checks if metric is not None and metric > threshold)

        signals.append("bullish" if growth_score >= 0.5 else "bearish" if growth_score == 0 else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing financial health")
        # 3. Financial Health (weight: 0.20)
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_score = 0.0
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 0.30
        if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 0.40  # Debt levels are most important for health
        if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * 0.8:  # Strong FCF conversion
            health_score += 0.30

        signals.append("bullish" if health_score >= 0.5 else "bearish" if health_score == 0 else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"),
        }

        progress.update_status(agent_id, ticker, "Analyzing valuation ratios")
        # 4. Valuation Ratios (weight: 0.20)
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        valuation_checks = [
            (pe_ratio, 25, 0.40),  # P/E overvalued (highest weight)
            (pb_ratio, 3, 0.30),  # P/B overvalued
            (ps_ratio, 5, 0.30),  # P/S overvalued
        ]
        # For valuation, exceeding threshold is bearish
        overvalued_score = sum(weight for metric, threshold, weight in valuation_checks if metric is not None and metric > threshold)

        signals.append("bearish" if overvalued_score >= 0.5 else "bullish" if overvalued_score == 0 else "neutral")
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        progress.update_status(agent_id, ticker, "Calculating final signal")
        # Determine overall signal using category weights
        category_weights = {
            0: 0.35,  # profitability
            1: 0.25,  # growth
            2: 0.20,  # health
            3: 0.20,  # valuation
        }
        signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
        weighted_total = sum(signal_values[signals[i]] * category_weights[i] for i in range(len(signals)))

        if weighted_total > 0.15:
            overall_signal = "bullish"
        elif weighted_total < -0.15:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on weighted signal strength
        confidence = round(min(abs(weighted_total) / 0.5, 1.0) * 100)

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = fundamental_analysis

    progress.update_status(agent_id, None, "Done")
    
    return {
        "messages": [message],
        "data": data,
    }
