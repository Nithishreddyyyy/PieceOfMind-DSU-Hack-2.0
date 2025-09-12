from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://admin:test1234@pieceofmind.rbtpkei.mongodb.net/?retryWrites=true&w=majority&appName=PieceOfMind"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client.todo_db
collection = db("todo_data")