from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

# Replace with your password
uri = "mongodb+srv://admin:test1234@devhack.6gqwt4w.mongodb.net/?retryWrites=true&w=majority&appName=DevHack"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("✅ Connected successfully — password is correct!")
    # Use your database
    db = client["pieceofmind"]

    # Create or get the 'products' collection
    products = db["products"]

    # Insert sample data
    sample_products = [
            {
                "name": "Focus Booster",
                "description": "A set of resources and activities to help improve concentration. Available on the Focus page.",
                "category": "Focus"
            },
            {
                "name": "Calm Audio",
                "description": "Relaxing audio for stress relief. Accessible for all users.",
                "category": "Calm"
            }
    ]

    # Insert many documents (skip if already inserted to avoid duplicates)
    result = products.insert_many(sample_products)
    print(f"Inserted IDs: {result.inserted_ids}")

    # Retrieve and print all products
    print("All products in the collection:")
    for product in products.find():
        print(product)
except OperationFailure:
    print("❌ Authentication failed — wrong username/password or insufficient role.")
except ServerSelectionTimeoutError:
    print("⚠️ Could not reach the cluster — check network or IP whitelist.")
except Exception as e:
    print(f"⚠️ Other error: {e}")
