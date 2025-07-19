# Financial Coach MCP

Financial Coach MCP is a FastAPI-based application that provides personalized financial advice using a multi-agent architecture powered by Google's Gemini API. It analyzes your budget, suggests savings strategies, and offers debt reduction plans. The app is containerized with Docker and available on [Docker Hub](https://hub.docker.com/r/bensaied/financial-coach-mcp) for easy deployment and public use.

## 🔥 Features

- **Budget Analysis**: Categorizes your expenses and provides recommendations to optimize spending.
- **Savings Strategies**: Suggests emergency fund sizes and savings plans based on your income and expenses.
- **Debt Reduction Plans**: Offers prioritized debt payoff strategies (avalanche and snowball methods).
- **Multi-Agent Architecture**: Uses three specialized AI agents to analyze your finances step-by-step.
- **FastAPI MCP**: Exposes a Machine-Readable Control Plane (MCP) for easy integration with other systems.
- **Publicly Accessible**: Deploy and use the Docker image with your Google API key.

## 🔍 How It Works

The application uses **FastAPI** to provide a RESTful API and integrates with **FastAPI MCP** to expose a control plane (`/mcp`) for discovering available tools. The core logic is powered by a **multi-agent architecture** using Google's Gemini API:

1. **Budget Analysis Agent**: Analyzes your income, expenses, and transactions to identify spending patterns and suggest savings.
2. **Savings Strategy Agent**: Builds on the budget analysis to recommend emergency funds and savings allocations.
3. **Debt Reduction Agent**: Analyzes debts and generates optimized payoff plans (avalanche prioritizes high-interest debts, snowball focuses on small balances).
4. **Coordinator Agent**: Orchestrates the three agents to ensure a cohesive financial analysis.

The app requires a **Google API key** (set as the `GOOGLE_API_KEY` environment variable) to access the Gemini API.

## 📦 Prerequisites

- **Docker**: Install Docker to run the container ([Install Docker](https://docs.docker.com/get-docker/)).
- **Google API Key**: Generate a key with Gemini API access from [Google Cloud Console](https://aistudio.google.com/apikey).

## 🚀 Quick Start

1. **Pull the Docker Image**:

   ```bash
   docker pull bensaied/financial-coach-mcp
   ```

2. **Run the Container**:

   ```bash
   docker run -p 8000:8000 -e PORT=8000 -e GOOGLE_API_KEY=your-google-api-key bensaied/financial-coach-mcp
   ```

   - Replace `your-google-api-key` with your valid Google API key.
   - The app will start on `http://localhost:8000`.

3. **Test the Health Endpoint**:

   ```bash
   curl http://localhost:8000/health
   ```

   Expected response: `{"status": "healthy"}`

4. **Explore the MCP Endpoint**:

   ```bash
   curl http://localhost:8000/mcp
   ```

   Returns a JSON description of available tools (e.g., `analyze_budget`).

5. **Analyze Your Finances**:
   ```bash
   curl -X POST http://localhost:8000/analyze_budget \
        -H "Content-Type: application/json" \
        -d '{
              "dependants": 2,
              "monthly_income": 4500,
              "manual_expenses": "{\"food\": 200, \"transport\": 100, \"education\": 800}",
              "debts": [{"name": "General Debt", "amount": 1000, "interest_rate": 15, "min_payment": 50}]
            }'
   ```
   Returns a JSON response with budget analysis, savings strategies, and debt reduction plans.

## 📁 Project Structure

- **`main.py`**: The FastAPI application with the multi-agent logic and MCP integration.
- **`Dockerfile`**: Defines the Docker image, using Python 3.11 and Uvicorn.
- **`requirements.txt`**: Lists dependencies (FastAPI, fastapi_mcp, google-cloud-aiplatform, etc.).

## ⚡ FastAPI MCP

The **FastAPI MCP** (Machine-Readable Control Plane) exposes the `/mcp` endpoint, which describes the API's tools and their schemas. This allows other systems or AI agents to discover and interact with the `analyze_budget` endpoint programmatically. No authentication key is required, making it accessible to all users.

## 🕸️ Multi-Agent Architecture

The app uses a **sequential multi-agent system** powered by Google's Gemini API:

- **Budget Analysis Agent**: Processes income, expenses, and transactions to categorize spending and suggest optimizations.
- **Savings Strategy Agent**: Uses budget analysis to recommend emergency funds and savings allocations.
- **Debt Reduction Agent**: Analyzes debts and generates payoff plans (avalanche prioritizes high-interest debts, snowball focuses on small balances).
- **Coordinator Agent**: Manages the sequence, ensuring each agent builds on the previous one’s output.

This modular design ensures comprehensive financial advice tailored to your input.

## ✨ Inspiration

This project is inspired by the [AI Financial Coach Agent](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/multi_agent_apps/ai_financial_coach_agent), a Streamlit application that uses a similar multi-agent approach for financial analysis. We adapted the concept into a FastAPI-based API, added Docker support, and integrated FastAPI MCP for broader accessibility and integration.

## 🛠️ Troubleshooting

- **401 Unauthorized**:
  - Verify `GOOGLE_API_KEY` is set correctly in the `docker run` command (e.g., `-e GOOGLE_API_KEY=your-google-api-key`).
- **500 Analysis Failed**:
  - Check if the Google API key is valid for the Gemini API.
  - Ensure the model (`gemini-1.5-flash`) is correct; try `gemini-1.5-pro` if issues persist.
- **Logs**: Check Docker logs for errors:
  ```bash
  docker logs <container-id>
  ```
  Look for `INFO:main:Retrieved GOOGLE_API_KEY` or `ERROR:main:Missing GOOGLE_API_KEY`.

## 🤝 Contributing

Feel free to fork this repository, submit pull requests, or open issues to suggest improvements or report bugs. The project is open for public use, so share your feedback to make it better!

## 🔓 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details (if added).

---

## 📝 Authors

- Github: [@bensaied](https://www.github.com/bensaied)

## 💝 Support

If you find this project helpful, please consider leaving a ⭐️!
If you are interested or have questions, contact us at **ben.saied@proton.me**

## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bensaied/)

Built with ❤️ by [@bensaied](https://www.github.com/bensaied) using FastAPI, Docker, and Google's Gemini API.
