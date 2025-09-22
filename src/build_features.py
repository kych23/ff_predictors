"""
Run the DuckDB SQL pipeline that builds model-ready features.
"""
from pathlib import Path
import duckdb

SQL_PATH = Path(__file__).with_name("build_features.sql")

def main():
    con = duckdb.connect()
    con.execute(SQL_PATH.read_text(encoding="utf-8"))
    con.close()
    print("Feature Construction completed!")

if __name__ == "__main__":
    main()
