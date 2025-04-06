import sqlite3

def check_database():
    conn = sqlite3.connect('data/processed/metadata.db')
    cursor = conn.cursor()
    
    print("Checking documents table:")
    cursor.execute('SELECT * FROM documents')
    documents = cursor.fetchall()
    print(f"Found {len(documents)} documents:")
    for doc in documents:
        print(f"  - doc_id: {doc[0]}, path: {doc[1]}, hash: {doc[2]}")
    
    print("\nChecking analysis_results table:")
    cursor.execute('SELECT * FROM analysis_results')
    results = cursor.fetchall()
    print(f"Found {len(results)} analysis results:")
    for result in results:
        print(f"  - result_id: {result[0]}, doc_id: {result[1]}, type: {result[2]}")
    
    conn.close()

if __name__ == "__main__":
    check_database() 