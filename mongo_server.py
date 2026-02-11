from mcp.server.fastmcp import FastMCP
import json
import os
import math
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from pymongo import AsyncMongoClient
from dotenv import load_dotenv

load_dotenv()

MDB_CONNECTION_STRING=os.getenv("MDB_MCP_CONNECTION_STRING")
DB_NAME=os.getenv("DB_NAME")

if not MDB_CONNECTION_STRING or not DB_NAME:
    raise RuntimeError("\n\nMDB_MCP_CONNECTION_STRING or DB_NAME is missing")

mcp = FastMCP("mongo_server")

client = None
database = None

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

async def get_mongo_client():
    global client, database
    if client is None:
        client = AsyncMongoClient(MDB_CONNECTION_STRING)
        database = client[DB_NAME]
    return database

@mcp.tool()
async def find_entities_data(entity_id: str, query_filter: Dict[str, Any] = None, projection: Dict[str, Any] = None) -> str:
    """Find documents in entities_data collection for a given entity_id.
    
    Args:
        entity_id: The entity ID to filter by (will be converted to ObjectId)
        query_filter: Additional MongoDB filter criteria (optional)
        projection: Fields to include/exclude in results (optional)
    
    Returns:
        JSON string with the query results
    """
    try:
        db = await get_mongo_client()
        
        base_query = {
            "entity_id": ObjectId(entity_id),
            "status": "ACTIVE"
        }
        
        if query_filter:
            base_query.update(query_filter)
        
        if projection is None:
            projection = {"_id": 0}
        
        cursor = db.entities_data.find(base_query, projection)
        results = await cursor.to_list(length=100)
        
        return json.dumps({
            "success": True,
            "count": len(results),
            "data": results
        }, indent=2, cls=MongoJSONEncoder)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def aggregate_entities_data(entity_id: str, pipeline: List[Dict[str, Any]]) -> str:
    """Run aggregation pipeline on entities_data collection for complex queries.
    
    Args:
        entity_id: The entity ID to filter by
        pipeline: MongoDB aggregation pipeline as a list of stages
    
    Returns:
        JSON string with the aggregation results
    """
    try:
        db = await get_mongo_client()
        
        if not any("$match" in stage and "entity_id" in stage["$match"] for stage in pipeline):
            pipeline.insert(0, {
                "$match": {
                    "entity_id": ObjectId(entity_id),
                    "status": "ACTIVE"
                }
            })
        
        cursor = await db.entities_data.aggregate(pipeline)
        results = await cursor.to_list(length=100)
        
        return json.dumps({
            "success": True,
            "count": len(results),
            "data": results
        }, indent=2, cls=MongoJSONEncoder)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def count_entities_data(entity_id: str, query_filter: Dict[str, Any] = None) -> str:
    """Count documents in entities_data collection for a given entity_id.
    
    Args:
        entity_id: The entity ID to filter by
        query_filter: Additional MongoDB filter criteria (optional)
    
    Returns:
        JSON string with the count result
    """
    try:
        db = await get_mongo_client()
        
        base_query = {
            "entity_id": ObjectId(entity_id),
            "status": "ACTIVE"
        }
        
        if query_filter:
            base_query.update(query_filter)
        
        count = await db.entities_data.count_documents(base_query)
        
        return json.dumps({
            "success": True,
            "count": count
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def distinct_values(entity_id: str, field: str, query_filter: Dict[str, Any] = None) -> str:
    """Get distinct values for a specific field in entities_data collection.
    
    Args:
        entity_id: The entity ID to filter by
        field: The field name to get distinct values for
        query_filter: Additional MongoDB filter criteria (optional)
    
    Returns:
        JSON string with distinct values
    """
    try:
        db = await get_mongo_client()
        
        base_query = {
            "entity_id": ObjectId(entity_id),
            "status": "ACTIVE"
        }
        
        if query_filter:
            base_query.update(query_filter)
        
        values = await db.entities_data.distinct(field, base_query)
        
        return json.dumps({
            "success": True,
            "count": len(values),
            "values": values
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def calculate_expression(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: The mathematical expression to evaluate (e.g., "25 * 45", "min(10, 20)", "sqrt(16)")
    
    Returns:
        JSON string with the result
    """
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return json.dumps({
            "success": True,
            "result": result
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

if __name__ == "__main__":
    mcp.run()
