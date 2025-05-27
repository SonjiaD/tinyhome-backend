from supabase import create_client
import json

url = "https://sjsgkndenvtzgjihermn.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNqc2drbmRlbnZ0emdqaWhlcm1uIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzMjM5MjAsImV4cCI6MjA2Mzg5OTkyMH0.AdEST3BzTwIuzcMfWmPZfRZPf4aNhC2xQG8vVWCks50"
supabase = create_client(url, key)

data = {
    "name": "Test3",
    "occupation": "Engineer",
    "location": "Fruitvale",
    "weights": {
        "transit_dist": 0.5,
        "homeless_service_dist": 0.4
    },
    "file_url": "https://example.com/testmap.csv"
}

res = supabase.table("submissions").insert(data).execute()
print(res)
