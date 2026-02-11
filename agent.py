from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
import ast
from datetime import datetime
import os
from bson import ObjectId
from pymongo import AsyncMongoClient
from upstash_redis.asyncio import Redis
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import mcp_server_tools, StdioServerParams
from autogen_core.models import ModelInfo

from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, ListMemory


load_dotenv()

gemini_api_key = os.getenv("GEMINI_PAID_KEY")
MDB_CONNECTION_STRING = os.getenv("MDB_MCP_CONNECTION_STRING")
DB_NAME = os.getenv("DB_NAME")
SCHEMA_CACHE_TTL = int(os.getenv("SCHEMA_CACHE_TTL", 86400))

class BoundedMemory(Memory):
    def __init__(self, limit: int = 20):
        self.messages: List[MemoryContent] = []
        self.limit = limit

    async def add(self, content: MemoryContent) -> None:
        self.messages.append(content)
        if len(self.messages) > (self.limit + 1):
            self.messages = self.messages[-self.limit:]

    async def get_messages(self, model_id: Optional[str] = None) -> List[MemoryContent]:
        return self.messages

    async def update_context(self, model_id: Optional[str] = None) -> None:
        pass

    async def clear(self) -> None:
        self.messages = []

    async def close(self) -> None:
        pass

    async def query(self, *args, **kwargs) -> List[MemoryContent]:
        return []

active_agents: Dict[str, Dict] = {}

client = AsyncMongoClient(MDB_CONNECTION_STRING)
data_base = client[DB_NAME]

default_instruction = f"""
You are an expert MONGO AI Agent. You can get the answers for any type of questions.
When the user asks any question, search only in the 'entities_data' collection in the '{DB_NAME}' database.
Follow the below rules strictly:
1. Always filter the data using the given entity_id with **schema 'entity_id' and type '$oid'**, to get the correct answer. And The status must be ACTIVE; otherwise, the record is considered deleted.
2. Never assume that there is only single record. My data is dynamic data It is changed regulary. So ***STRICTLY USE the available MongoDB tools to fetch data for each and EVERY user given question.*** You need to use the tools based on given SCHEMA.
3. IN my templates_fields_data array, schema is like templateId#fieldKey. So i will give this all schema keys along with the array name. EX: templates_fields_data.68834cadec0b6c0012b68cdd#payment_id. So use queries using this schema only. 
   EXAMPLE USAGE: Use find_entities_data() with query_filter={{"templates_fields_data.68834cadec0b6c0012b68cdd#payment_id": {{"$exists": True}}}}
4. If you didn't find correct results with find_entities_data(), use it with minimal query_filter to get the entire data available in the given entity's templates_fields_data array. And get the answers from that Data .
5. For **numerical or comparison-based queries (max, min, avg, sum, sort)**:
   - **CRITICALLY IMPORTANT**: The data in MongoDB might be stored as strings. BEFORE performing any mathematical aggregation ($sum, $avg, $max, $min) or sorting, you **MUST converts the field to a number** using `$toDouble` or `$toInt` or `$convert` in a `$project` or `$addFields` stage.
   - Example pipeline stage: `{{"$addFields": {{"converted_amount": {{"$toDouble": "$templates_fields_data.templateId#fieldKey"}}}}}}`. Then use `converted_amount` for your math/sort operations.
   - Use `aggregate_entities_data()` with these conversion stages.
6. For **general math questions** (e.g., "what is 25 * 45", "calculate 15% of 500") that don't directly involve database aggregation, OR if you need to calculate something from the retrieved data, **USE the `calculate_expression(expression)` tool**.
7. **For date related queries - CRITICAL DATE TYPE HANDLING**:
     - Use aggregation pipeline with $type operator:
       {{
         "$expr": {{
           "$or": [
             {{"$and": [
               {{"$eq": [{{"$type": "$templates_fields_data.TEMPLATE_ID#date_field"}}, "date"]}},
               {{"$gte": ["$templates_fields_data.TEMPLATE_ID#date_field", {{"$dateFromString": {{"dateString": "2025-09-28T00:00:00.000Z"}}}}]}}
             ]}},
             {{"$and": [
               {{"$eq": [{{"$type": "$templates_fields_data.TEMPLATE_ID#date_field"}}, "string"]}},
               {{"$gte": ["$templates_fields_data.TEMPLATE_ID#date_field", "2025-09-28T00:00:00.000Z"]}}
             ]}}
           ]
         }}
       }}
   
   - **IMPORTANT**: Always convert user-provided dates to UTC before querying. Use comparison operators like $gte, $lte, $gt, $lt .
8. Create CASE INSENSITIVE queries with $regex in aggregation pipelines wherever required.
9. Finally, Give the user friendly answer using clear line breaks, bullet points, Bold texts, relevant emojis wherever required to make the answer more readable.
10. Do not mention entity_id in the final answer.
11. **Email Handling**:
    - You MUST also extract the entity_data_id → the Mongo document `_id`
    - If to_address and the email context are available—either provided by the user or found in previous conversations—retrieve them.
    - If `to_address` or `email context(purpose/content)` is MISSING, DO NOT give your own to_address or email context and do NOT call the tool. Instead, you MUST ask the user like: "Please provide the recipient's email address" or "Please provide the email context(purpose/content)." Means Request Human Input when needed.
    - **Follow this TWO-STEP CONFIRMATION FLOW for sending mails**:
        - **When the user requests to send an email, First generate the mail draft and show it to the user and ask for confirmation. And SET the mail flag as False. If user Confirmed or ask to send the mail then call generate_email_json(subject, body, to_address, entity_data_id, cc_address, bcc_address) tool and SET the mail flag as True and return the same JSON**
12. **Available Tools**:
    - find_entities_data(entity_id, query_filter, projection) - For basic queries
    - aggregate_entities_data(entity_id, pipeline) - For complex queries and calculations (REMEMBER TO CONVERT STRINGS TO NUMBERS)
    - count_entities_data(entity_id, query_filter) - For counting documents
    - distinct_values(entity_id, field, query_filter) - For getting unique values
    - calculate_expression(expression) - For general math calculations
13. Strictly return the output in the VALID JSON format using below keys:
    response: str = Field(description="Some message to show the user or ask any question if you need from User")
    to_address: Optional[str] = Field(description="The recipient's email address. MANDATORY if mail is True.")
    cc_address: Optional[str] = Field(description="The CC email address.", default=None)
    bcc_address: Optional[str] = Field(description="The BCC email address.", default=None)
    subject: Optional[str] = Field(description="The subject of the email. MANDATORY if mail is True.")
    body: Optional[str] = Field(description="The body of the email. Strictly in HTML format with inline CSS. MANDATORY if mail is True.")
    mail: Optional[bool] = Field(description="Whether to send the mail. **Set this to True ONLY if the User confirms to send the mail.**", default=False)
    entity_data_id: Optional[str] = Field(description="The MongoDB entity data ID.", default=None)
"""

MCP_SERVERS = {
    "mail": StdioServerParams(
        command="python",
        args=["mail_server.py"],
        read_timeout_seconds=60
    ),
    "mongo": StdioServerParams(
        command="python",
        args=["mongo_server.py"],
        env={
            "MDB_MCP_CONNECTION_STRING": MDB_CONNECTION_STRING,
            "DB_NAME": DB_NAME,
        },
        read_timeout_seconds=60
    )
}

async def get_entity_schema(entity_id: str, redis_client: Redis = None) -> str:
    cache_key = f"schema:{entity_id}"
    
    if redis_client:
        try:
            schema = await redis_client.get(cache_key)
            if schema:
                #print("Schema from Redis cache")
                return schema
        except Exception as e:
            print(f"Redis GET error: {e}")

    try:
        entities_cursor = data_base.entity.find(
            {"_id": ObjectId(entity_id), "status": "ACTIVE"},
            {"name": 1, "templates.template_id": 1}
        )
        entities = await entities_cursor.to_list(length=None)
        
        entity_template_map = {}
        all_template_ids = set()
        
        for entity in entities:
            name = entity.get("name")
            templates = entity.get("templates", [])
            template_ids = [tpl.get("template_id") for tpl in templates if tpl.get("template_id")]
            if template_ids:
                entity_template_map[name] = template_ids
                all_template_ids.update(template_ids)
        
        if not all_template_ids:
            return json.dumps({}, indent=2)
        
        template_cursor = data_base.templates.find(
            {"_id": {"$in": list(all_template_ids)}, "status": "ACTIVE"},
            {
                "_id": 1,
                "sections.fields.key": 1,
                "sections.fields.label": 1,
                "sections.fields.inputType": 1,
                "sections.fields.data_table_columns.key": 1,
                "sections.fields.data_table_columns.label": 1,
                "sections.fields.data_table_columns.inputType": 1
            }
        )
        templates = await template_cursor.to_list(length=None)
        
        template_fields_map = {}
        for doc in templates:
            template_id = str(doc["_id"])
            fields_list = []
            
            for section in doc.get("sections", []):
                for field in section.get("fields", []):
                    parent_key = field.get("key")
                    parent_label = field.get("label", "")
                    parent_type = field.get("inputType", "string")
                    
                    if parent_type == "ENTITY_TABLE":
                        continue
                    
                    if parent_key:
                        suffix = "/name" if parent_type == "ENTITY" else ""
                        fields_list.append(f"templates_fields_data.{template_id}#{parent_key}{suffix}: (type: {parent_type}), (label: {parent_label})")
                    
                    for col in field.get("data_table_columns", []):
                        col_key = col.get("key")
                        col_label = col.get("label", "")
                        col_type = col.get("inputType", "string")
                        if col_type == "ENTITY_TABLE":
                            continue
                        if col_key:
                            suffix = "/name" if col_type == "ENTITY" else ""
                            fields_list.append(f"templates_fields_data.{template_id}#{parent_key}.{col_key}{suffix}: (type: {col_type}), (label: {col_label})")
            
            template_fields_map[template_id] = fields_list
        
        result = {}
        for entity_name, template_ids in entity_template_map.items():
            entity_fields = []
            for tpl_id in template_ids:
                tpl_id_str = str(tpl_id)
                entity_fields.extend(template_fields_map.get(tpl_id_str, []))
            result[entity_name] = entity_fields
        
        schema = json.dumps(result, indent=2)
        
        if redis_client:
            try:
                await redis_client.set(cache_key, schema, ex=SCHEMA_CACHE_TTL)
                #print("Schema cached in Redis")
            except Exception as e:
                print(f"Redis SET error: {e}")
        
        return schema
        
    except Exception as e:
        print(f"Schema error: {e}")
        return json.dumps({}, indent=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = None
    try:
        redis_client = Redis(
            url=os.getenv("UPSTASH_REDIS_URL"),
            token=os.getenv("UPSTASH_REDIS_TOKEN")
        )
        await redis_client.set("health_check", "ok")
    except Exception as e:
        print(f"Redis failed: {e}. Caching disabled.")

    app.state.redis = redis_client

    mail_tools = await mcp_server_tools(MCP_SERVERS["mail"])
    mongo_tools = await mcp_server_tools(MCP_SERVERS["mongo"])

    gemini_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        api_key=gemini_api_key,
        model_info=ModelInfo(
            vision=True, 
            function_calling=True, 
            json_output=True, 
            family="gemini", 
            structured_output=True,
            multiple_system_messages=True
        ),
        temperature=0.5
    )

    question_agent = AssistantAgent(
        name="question_generator",
        model_client=gemini_client,
        system_message="You are a helpful assistant that suggests relevant follow-up questions based on the user's previous question and mongo_agent answer. Suggest 3 natural, relevant follow-up questions that the user might ask next. Keep them short, specific, and related to the entity data. Return only exactly questions as a list of strings, DO NOT include any additional text or context. EX: [\"What is the payment status?\", \"When was it created?\", \"Who is the contact person?\"]",
        max_tool_iterations=3
    )

    app.state.gemini_client = gemini_client
    app.state.mail_tools = mail_tools
    app.state.mongo_tools = mongo_tools
    app.state.question_agent = question_agent
    yield

    if app.state.redis:
        await app.state.redis.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    entity_id: str
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    response: str
    to_address: Optional[str] = Field(default=None)
    cc_address: Optional[str] = Field(default=None)
    bcc_address: Optional[str] = Field(default=None)
    subject: Optional[str] = Field(default=None)
    body: Optional[str] = Field(default=None)
    mail: Optional[bool] = Field(default=False)
    entity_data_id: Optional[str] = Field(default=None)
    suggested_questions: List[str]

@app.post("/entity_chat")
async def ask(req: AskRequest) -> AskResponse:
    try:
        redis_client = app.state.redis
        schema = await get_entity_schema(req.entity_id, redis_client)
        
        today_date = datetime.now()
        user_query = (
            f"User question: {req.question}\n"
            f"Current date time: {today_date}. Use this where required.\n"
            f"entity_id($oid): {req.entity_id}\n"
            f"Please use this entity_id to filter your MongoDB query.\n"
            f"SCHEMA for this entity:\n{schema}\n\n"
            f"Respond in VALID JSON format as specified in your instructions."
        )

        session_id = req.session_id or "default"
        if session_id not in active_agents:
            print(f"Creating NEW Agent and Memory for session: {session_id}")
            session_memory = BoundedMemory(limit=20)
            
            mongo_agent = AssistantAgent(
                name="mongo_agent",
                model_client=app.state.gemini_client,
                system_message=default_instruction,
                tools=app.state.mail_tools + app.state.mongo_tools,
                memory=[session_memory],
                max_tool_iterations=15,
                description="MongoDB expert agent",
                reflect_on_tool_use=True
            )
            active_agents[session_id] = {"agent": mongo_agent, "memory": session_memory}
        else:
            print(f"Reusing existing Agent for session: {session_id}")
            data = active_agents[session_id]
            mongo_agent = data["agent"]
            session_memory = data["memory"]

        result = await mongo_agent.run(task=user_query)
        print("\n\nresult.messages is:",result.messages)
        
        agent_response = result.messages[-1].content if hasattr(result, 'messages') and result.messages else str(result)

        await session_memory.add(
            MemoryContent(
                content=req.question,
                mime_type=MemoryMimeType.TEXT,
                metadata={"role": "user"}
            )
        )
        await session_memory.add(
            MemoryContent(
                content=agent_response,
                mime_type=MemoryMimeType.TEXT,
                metadata={"role": "assistant"}
            )
        )

        suggestions_prompt = f"""User question: "{req.question}"
Mongo agent answer: "{agent_response}"

Suggest exactly 3 follow-up questions as Python list. Return ONLY the list:"""
        
        suggestions_result = await app.state.question_agent.run(task=suggestions_prompt)
        suggestions_text = suggestions_result.messages[-1].content if hasattr(suggestions_result, 'messages') else ""
        
        suggestions_list = []
        try:
            if suggestions_text.startswith("[") and suggestions_text.endswith("]"):
                suggestions_list = ast.literal_eval(suggestions_text)
            else:
                suggestions_list = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        except:
            suggestions_list = ["What else would you like to know?", "Any other details needed?", "Next steps?"]

        try:
            clean_result = agent_response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_result)
            parsed["suggested_questions"] = suggestions_list
            
            if "response" not in parsed:
                parsed["response"] = clean_result
                
            return parsed
        except:
            return {
                "response": agent_response,
                "suggested_questions": suggestions_list,
                "to_address": None, "cc_address": None, "bcc_address": None,
                "subject": None, "body": None, "mail": False,
                "entity_data_id": None
            }

    except Exception as e:
        print(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
