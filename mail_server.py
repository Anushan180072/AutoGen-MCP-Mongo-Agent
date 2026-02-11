from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("mail_server")

@mcp.tool()
def generate_email_json(
    subject: str, 
    body: str, 
    to_address: str,
    entity_data_id: str,
    cc_address: str = None,
    bcc_address: str = None
) -> str:
    """Generates the structured JSON for an email.
    Args:
        subject: The email subject.
        body: The email body (Strictly in HTML format with inline CSS).
        to_address: The recipient email address.
        cc_address: The CC email address (optional).
        bcc_address: The BCC email address (optional).
        entity_data_id: the Mongo document `_id` of the entity data.
    """
    if not to_address:
        return "MISSING_RECIPIENT: Please provide a recipient email address."
    
    response = {
        "to_address": to_address,
        "cc_address": cc_address,
        "bcc_address": bcc_address,
        "subject": subject,
        "body": body,
        "mail": True,
        "entity_data_id": entity_data_id,
    }
    return json.dumps(response, indent=2)

if __name__ == "__main__":
    mcp.run()
