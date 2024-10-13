"""
TODO

- have the plotter x and y use json columns by re cachig the selections and ensuring type
- repair search
- add paging to replace hard sql limit
- have X and Y when table transforms to inherit colmn names
"""


from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import create_model
from typing import Optional, List, Any, Dict
import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
from datetime import datetime
import json
import re

# Load environment variables
load_dotenv()

app = FastAPI()

# Mount the static files directory
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Database connection parameters
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_PARAMS)
    except psycopg2.Error as e:
        print(f"Unable to connect to the database: {e}")
        raise

def get_column_types(cursor, table_name):
    query = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = %s
    """
    cursor.execute(query, (table_name,))
    columns = cursor.fetchall()
    
    result = {}
    for column in columns:
        column_name, data_type, is_nullable = column['column_name'], column['data_type'], column['is_nullable']
        result[column_name] = {'type': data_type, 'nullable': is_nullable}
    
    return result

def build_json_path_query(column):
    parts = column.split('.')
    if len(parts) == 1:
        return sql.Identifier(column)
    else:
        base = sql.Identifier(parts[0])
        path = [sql.Literal(part) for part in parts[1:]]
        return sql.SQL('->').join([base] + path[:-1]) + sql.SQL('->>') + path[-1]

def flatten_json(data):
    flattened = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                flattened.update(flatten_json(value))
            else:
                flattened[key] = value
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                flattened.update(flatten_json(item))
            else:
                flattened[str(i)] = item
    else:
        flattened = data
    return flattened

@app.get("/api/get-tables")
async def get_tables():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        cursor.execute(query)
        all_tables = [row[0] for row in cursor.fetchall()]
        
        non_empty_tables = []
        for table in all_tables:
            count_query = sql.SQL("SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)").format(sql.Identifier(table))
            cursor.execute(count_query)
            has_rows = cursor.fetchone()[0]
            if has_rows:
                non_empty_tables.append(table)
        
        return non_empty_tables
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching table names.")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/column-names")
async def get_column_names(table_name: str = Query(..., min_length=1)):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        columns = []
        for col, info in column_types.items():
            column_info = {
                "name": col,
                "type": info['type'],
                "is_json": info['type'] in ('json', 'jsonb')
            }
            columns.append(column_info)

        return columns
    except Exception as e:
        print(f"An error occurred: {e}")
        if cursor:
            cursor.execute("ROLLBACK")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching column names.")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/metrics-data")
async def get_metrics_data(
    table_name: str = Query(..., min_length=1),
    x_column: str = Query(None),
    y_column: str = Query(None),
    full_table: bool = Query(False)
):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        json_columns = [col for col, info in column_types.items() if info['type'] in ('json', 'jsonb')]

        if full_table:
            select_parts = [sql.Identifier(col) for col in column_types.keys()]
            query = sql.SQL("SELECT {} FROM {} LIMIT 100").format(
                sql.SQL(', ').join(select_parts),
                sql.Identifier(table_name)
            )
        else:
            if not x_column or not y_column:
                raise HTTPException(status_code=400, detail="X and Y columns must be specified when not fetching full table.")
            
            x_select = build_json_path_query(x_column)
            y_select = build_json_path_query(y_column)

            query = sql.SQL("SELECT {} as x_value, {} as y_value FROM {} LIMIT 100").format(
                x_select,
                y_select,
                sql.Identifier(table_name)
            )

        cursor.execute(query)
        rows = cursor.fetchall()

        # Process the rows
        processed_rows = []
        for row in rows:
            processed_row = {}
            for key, value in row.items():
                if key in json_columns and value is not None:
                    try:
                        json_data = json.loads(value) if isinstance(value, str) else value
                        flattened = flatten_json(json_data)
                        processed_row.update(flattened)
                    except json.JSONDecodeError:
                        processed_row[key] = value
                elif isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            processed_rows.append(processed_row)

        return processed_rows
    except Exception as e:
        print(f"An error occurred: {e}")
        if cursor:
            cursor.execute("ROLLBACK")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/search")
async def search_database(
    table_name: str = Query(..., min_length=1),
    search_term: str = Query(..., min_length=1),
    columns: Optional[List[str]] = Query(None)
):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get column types
        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        # If no specific columns are provided, search all columns
        if not columns:
            columns = list(column_types.keys())

        # Construct the WHERE clause
        where_clauses = []
        for col in columns:
            if column_types[col]['type'] in ('json', 'jsonb'):
                # For JSON columns, use the ->> operator to search as text
                where_clauses.append(
                    sql.SQL("{} ->> {} ILIKE {}").format(
                        sql.Identifier(col),
                        sql.Literal(''),  # Empty string means the entire JSON object
                        sql.Literal(f'%{search_term}%')
                    )
                )
            elif column_types[col]['type'] in ('text', 'varchar', 'char'):
                where_clauses.append(
                    sql.SQL("{} ILIKE {}").format(
                        sql.Identifier(col),
                        sql.Literal(f'%{search_term}%')
                    )
                )
            else:
                # For other types, cast to text before searching
                where_clauses.append(
                    sql.SQL("CAST({} AS TEXT) ILIKE {}").format(
                        sql.Identifier(col),
                        sql.Literal(f'%{search_term}%')
                    )
                )

        # Construct the full query
        query = sql.SQL("SELECT * FROM {} WHERE {}").format(
            sql.Identifier(table_name),
            sql.SQL(" OR ").join(where_clauses)
        )

        # Execute the query
        cursor.execute(query)
        results = cursor.fetchall()

        return results
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)