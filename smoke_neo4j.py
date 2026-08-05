from dotenv import load_dotenv
import os
load_dotenv()
from langchain_neo4j import Neo4jGraph

print("URI:", os.getenv("NEO4J_URI"))
g = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)
print(g.query("RETURN 1 AS ok"))