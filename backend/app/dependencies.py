from fastapi import Header, HTTPException
from app.config import API_KEY
async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    print(f"Auth: REcieved{x_api_key} vs Expected {API_KEY}")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorised")
    return x_api_key
