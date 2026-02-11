# 🚀 Mongo-MCP Intelligent Agent
Agentic conversational AI system that converts natural language into real-time MongoDB queries using AutoGen and MCP servers.
An enterprise-grade AI Agent system that combines the power of **Google Gemini, MongoDB, and FastAPI.** This system uses the **Model Context Protocol (MCP)** to allow an AI agent to query databases, perform complex aggregations, calculate mathematical expressions, and draft/send automated emails based on live data.<br><br>


**#✨ Features**

**Natural Language to MongoDB**: Query your database using plain English. The agent automatically generates filters and aggregation pipelines.

**Dynamic Schema Mapping**: Automatically fetches and caches entity-specific schemas from MongoDB to help the LLM understand your data structure.

**Two-Step Email Flow**: Drafts professional HTML emails based on data findings and requires human confirmation before sending.

**MCP Architecture**: Utilizes separate Tool Servers (mail_server.py and mongo_server.py) for clean separation of concerns.

**Session-Based Memory**: Uses BoundedMemory to maintain context across conversations without exceeding token limits.

**Math & Date Handling**: Specialized logic for handling string-to-number conversions in MongoDB and complex UTC date comparisons.<br><br>




**🛠 Tech Stack**

**Framework**: FastAPI

**AI Orchestration**: Microsoft AutoGen

**LLM**: Google Gemini 2.5 Flash (via OpenAIChatCompletionClient)

**Database**: MongoDB (Async Driver)

**Caching**: Upstash Redis

**Protocol**: Model Context Protocol (MCP)
