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
import argparse

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

def get_json_paths(obj, parent_path='', paths=None):
    if paths is None:
        paths = {}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{parent_path}.{key}" if parent_path else key
            if isinstance(value, (dict, list)):
                get_json_paths(value, new_path, paths)
            else:
                # Determine the type of the leaf value
                if isinstance(value, bool):
                    paths[new_path] = 'boolean'
                elif isinstance(value, int):
                    paths[new_path] = 'integer'
                elif isinstance(value, float):
                    paths[new_path] = 'numeric'
                elif isinstance(value, str):
                    # Try to detect if it's a date
                    try:
                        datetime.fromisoformat(value.replace('Z', '+00:00'))
                        paths[new_path] = 'timestamp'
                    except ValueError:
                        paths[new_path] = 'text'
                else:
                    paths[new_path] = 'text'
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:5]):
            new_path = f"{parent_path}[{i}]" if parent_path else str(i)
            if isinstance(item, (dict, list)):
                get_json_paths(item, new_path, paths)
            else:
                paths[new_path] = type(item).__name__
    
    return paths

def build_json_selector(column_path):
    parts = column_path.split('.')
    if len(parts) == 1:
        return sql.Identifier(parts[0])
    
    base = sql.Identifier(parts[0])
    path_parts = []
    
    for part in parts[1:]:
        if '[' in part and ']' in part:
            array_path = part.split('[')
            path_parts.append(sql.Literal(array_path[0]))
            index = array_path[1].rstrip(']')
            path_parts.append(sql.Literal(int(index)))
        else:
            path_parts.append(sql.Literal(part))
    
    result = base
    for part in path_parts[:-1]:
        result = sql.SQL('->').join([result, part])
    
    return sql.SQL('->>').join([result, path_parts[-1]])

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
        column_name = column['column_name']
        data_type = column['data_type']
        is_nullable = column['is_nullable']
        
        result[column_name] = {
            'type': data_type,
            'nullable': is_nullable
        }
        
        if data_type in ('json', 'jsonb'):
            sample_query = sql.SQL("""
                SELECT DISTINCT {}
                FROM {}
                WHERE {} IS NOT NULL
                LIMIT 100
            """).format(
                sql.Identifier(column_name),
                sql.Identifier(table_name),
                sql.Identifier(column_name)
            )
            
            cursor.execute(sample_query)
            samples = cursor.fetchall()
            
            for row in samples:
                if row[column_name]:
                    try:
                        data = json.loads(row[column_name]) if isinstance(row[column_name], str) else row[column_name]
                        json_paths = get_json_paths(data)
                        for path, path_type in json_paths.items():
                            full_path = f"{column_name}.{path}"
                            result[full_path] = {
                                'type': path_type,
                                'nullable': 'YES',
                                'is_json_field': True,
                                'parent_column': column_name,
                                'json_path': path
                            }
                    except (json.JSONDecodeError, AttributeError):
                        continue
    
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

# Update argparse
parser = argparse.ArgumentParser(description="SQL Dashboard with JSON handling options")
parser.add_argument("--flatten-json", action="store_true", help="Flatten JSON columns into separate columns. If false, format as line-separated JSON.")
args = parser.parse_args()

def process_json_data(value, flatten=False):
    if value is None:
        return value
    
    try:
        json_data = json.loads(value) if isinstance(value, str) else value
        if flatten:
            return flatten_json(json_data)
        else:
            return json.dumps(json_data, indent=2)  # Use indent=2 for pretty formatting
    except json.JSONDecodeError:
        return value

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
    full_table: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000)
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

        # Calculate offset
        offset = (page - 1) * page_size

        if full_table:
            select_parts = []
            for col, info in column_types.items():
                if info.get('is_json_field'):
                    select_parts.append(
                        sql.SQL("{} as {}").format(
                            build_json_selector(col),
                            sql.Identifier(col.replace('.', '_'))
                        )
                    )
                else:
                    select_parts.append(sql.Identifier(col))

            query = sql.SQL("SELECT {} FROM {} OFFSET {} LIMIT {}").format(
                sql.SQL(', ').join(select_parts),
                sql.Identifier(table_name),
                sql.Literal(offset),
                sql.Literal(page_size)
            )
        else:
            if not x_column or not y_column:
                raise HTTPException(status_code=400, detail="X and Y columns must be specified when not fetching full table.")
            
            x_select = build_json_selector(x_column)
            y_select = build_json_selector(y_column)

            query = sql.SQL("SELECT {} as x_value, {} as y_value FROM {} OFFSET {} LIMIT {}").format(
                x_select,
                y_select,
                sql.Identifier(table_name),
                sql.Literal(offset),
                sql.Literal(page_size)
            )

        # Get total count
        count_query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))
        cursor.execute(count_query)
        total_count = cursor.fetchone()['count']

        cursor.execute(query)
        rows = cursor.fetchall()

        # Process the rows (keep existing processing logic)
        processed_rows = []
        for row in rows:
            processed_row = {}
            for key, value in row.items():
                if key in json_columns and value is not None:
                    processed_row[key] = process_json_data(value, flatten=args.flatten_json)
                elif isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            
            if args.flatten_json:
                processed_row = flatten_json(processed_row)
            
            processed_rows.append(processed_row)

        return {
            "data": processed_rows,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }

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
    columns: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000)
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

        # If no columns specified, only use base columns
        if not columns:
            columns = [col for col in column_types.keys() if '.' not in col]

        # Construct the WHERE clause
        where_clauses = []
        for col in columns:
            if col in column_types:
                if column_types[col]['type'] in ('json', 'jsonb'):
                    # Simple JSON text search
                    where_clauses.append(
                        sql.SQL("CAST({} AS TEXT) ILIKE {}").format(
                            sql.Identifier(col),
                            sql.Literal(f'%{search_term}%')
                        )
                    )
                else:
                    where_clauses.append(
                        sql.SQL("CAST({} AS TEXT) ILIKE {}").format(
                            sql.Identifier(col),
                            sql.Literal(f'%{search_term}%')
                        )
                    )

        if not where_clauses:
            return {"data": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}

        # Calculate offset
        offset = (page - 1) * page_size

        # Construct the query
        query = sql.SQL("SELECT * FROM {} WHERE {} OFFSET {} LIMIT {}").format(
            sql.Identifier(table_name),
            sql.SQL(" OR ").join(where_clauses),
            sql.Literal(offset),
            sql.Literal(page_size)
        )

        # Construct the count query
        count_query = sql.SQL("SELECT COUNT(*) FROM {} WHERE {}").format(
            sql.Identifier(table_name),
            sql.SQL(" OR ").join(where_clauses)
        )

        # Execute the count query
        cursor.execute(count_query)
        total_count = cursor.fetchone()['count']

        # Execute the main query
        cursor.execute(query)
        results = cursor.fetchall()

        # Process the results
        processed_results = []
        for row in results:
            processed_row = {}
            for key, value in row.items():
                if column_types[key]['type'] in ('json', 'jsonb') and value is not None:
                    processed_row[key] = process_json_data(value, flatten=args.flatten_json)
                elif isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            
            if args.flatten_json:
                processed_row = flatten_json(processed_row)
            
            processed_results.append(processed_row)

        return {
            "data": processed_results,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }

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
