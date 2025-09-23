"""
Run the DuckDB SQL pipeline that builds model-ready features.
"""
from pathlib import Path
import duckdb

SQL_PATH = Path(__file__).with_name("build_features.sql")

def main():
    con = duckdb.connect()
    sql_content = SQL_PATH.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    # Execute each statement separately
    for i, statement in enumerate(statements):
        try:
            print(f"Executing statement {i+1}/{len(statements)}")
            con.execute(statement)
        except Exception as e:
            print(f"Error in statement {i+1}: {e}")
            print(f"Statement: {statement[:200]}...")
            con.close()
            raise
    
    con.close()
    print("Feature Matrix created!")

if __name__ == "__main__":
    main()
