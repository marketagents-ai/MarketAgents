from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from datetime import datetime
import json
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        logger.error(f"Unable to connect to the database: {e}")
        raise

def get_valid_table_names(cursor):
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_type = 'BASE TABLE'
    ORDER BY table_name
    """
    cursor.execute(query)
    tables = [row[0] for row in cursor.fetchall()]
    return tables

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
        query = base
        for p in path[:-1]:
            query = sql.SQL("{}->{}").format(query, p)
        query = sql.SQL("{}->>{}").format(query, path[-1])
        return query

def flatten_json(data):
    flattened = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                nested = flatten_json(value)
                for nested_key, nested_value in nested.items():
                    flattened[f"{key}.{nested_key}"] = nested_value
            else:
                flattened[key] = value
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                nested = flatten_json(item)
                for nested_key, nested_value in nested.items():
                    flattened[f"{i}.{nested_key}"] = nested_value
            else:
                flattened[str(i)] = item
    else:
        return data
    return flattened

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
        
        # Fetch all table names
        all_tables = get_valid_table_names(cursor)
        
        # Filter non-empty tables
        non_empty_tables = []
        for table in all_tables:
            count_query = sql.SQL("SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)").format(sql.Identifier(table))
            cursor.execute(count_query)
            has_rows = cursor.fetchone()[0]
            if has_rows:
                non_empty_tables.append(table)
        
        return non_empty_tables
    except Exception as e:
        logger.error(f"An error occurred: {e}")
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

        # Validate table name
        valid_tables = get_valid_table_names(cursor)
        if table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name provided.")

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
    except HTTPException as he:
        logger.error(f"HTTP error occurred: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"An error occurred: {e}")
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
    page_size: int = Query(100, ge=1, le=1000),
    flatten_json: bool = Query(False)
):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Validate table name
        valid_tables = get_valid_table_names(cursor)
        if table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name provided.")

        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        valid_columns = list(column_types.keys())
        json_columns = [col for col, info in column_types.items() if info['type'] in ('json', 'jsonb')]

        # Calculate offset
        offset = (page - 1) * page_size

        if full_table:
            select_parts = [sql.Identifier(col) for col in valid_columns]
            query = sql.SQL("SELECT {} FROM {} OFFSET {} LIMIT {}").format(
                sql.SQL(', ').join(select_parts),
                sql.Identifier(table_name),
                sql.Literal(offset),
                sql.Literal(page_size)
            )
        else:
            if not x_column or not y_column:
                raise HTTPException(status_code=400, detail="X and Y columns must be specified when not fetching full table.")

            # Validate column names
            if x_column.split('.')[0] not in valid_columns:
                raise HTTPException(status_code=400, detail=f"Invalid x_column provided: {x_column}")
            if y_column.split('.')[0] not in valid_columns:
                raise HTTPException(status_code=400, detail=f"Invalid y_column provided: {y_column}")
            
            x_select = build_json_path_query(x_column)
            y_select = build_json_path_query(y_column)

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

        # Process the rows
        processed_rows = []
        for row in rows:
            processed_row = {}
            for key, value in row.items():
                if key in json_columns and value is not None:
                    processed_row[key] = process_json_data(value, flatten=flatten_json)
                elif isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            
            if flatten_json:
                processed_row = flatten_json(processed_row)
            
            processed_rows.append(processed_row)

        return {
            "data": processed_rows,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }

    except HTTPException as he:
        logger.error(f"HTTP error occurred: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"An error occurred: {e}")
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
    page_size: int = Query(100, ge=1, le=1000),
    flatten_json: bool = Query(False)
):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Validate table name
        valid_tables = get_valid_table_names(cursor)
        if table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name provided.")

        # Get column types
        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        valid_columns = list(column_types.keys())

        # If no specific columns are provided, search all columns
        if not columns:
            columns = valid_columns
        else:
            # Validate provided columns
            for col in columns:
                if col.split('.')[0] not in valid_columns:
                    raise HTTPException(status_code=400, detail=f"Invalid column provided: {col}")

        # Construct the WHERE clause
        where_clauses = []
        params = []
        for col in columns:
            base_col = col.split('.')[0]
            if column_types[base_col]['type'] in ('json', 'jsonb'):
                # For JSON columns, use CAST to text and search
                where_clauses.append(
                    sql.SQL("CAST({} AS TEXT) ILIKE %s").format(sql.Identifier(base_col))
                )
                params.append(f'%{search_term}%')
            elif column_types[base_col]['type'] in ('text', 'varchar', 'char'):
                where_clauses.append(
                    sql.SQL("{} ILIKE %s").format(sql.Identifier(base_col))
                )
                params.append(f'%{search_term}%')
            else:
                # For other types, cast to text before searching
                where_clauses.append(
                    sql.SQL("CAST({} AS TEXT) ILIKE %s").format(sql.Identifier(base_col))
                )
                params.append(f'%{search_term}%')

        # Calculate offset
        offset = (page - 1) * page_size

        # Construct the full query with pagination
        where_clause = sql.SQL(" OR ").join(where_clauses)
        query = sql.SQL("SELECT * FROM {} WHERE {} OFFSET %s LIMIT %s").format(
            sql.Identifier(table_name),
            where_clause
        )
        params.extend([offset, page_size])

        # Construct the count query
        count_query = sql.SQL("SELECT COUNT(*) FROM {} WHERE {}").format(
            sql.Identifier(table_name),
            where_clause
        )

        # Execute the count query
        cursor.execute(count_query, params[:-2])  # Exclude offset and limit
        total_count = cursor.fetchone()['count']

        # Execute the main query
        cursor.execute(query, params)
        results = cursor.fetchall()

        # Process the results
        processed_results = []
        for row in results:
            processed_row = {}
            for key, value in row.items():
                if column_types[key]['type'] in ('json', 'jsonb') and value is not None:
                    processed_row[key] = process_json_data(value, flatten=flatten_json)
                elif isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            
            if flatten_json:
                processed_row = flatten_json(processed_row)
            
            processed_results.append(processed_row)

        return {
            "data": processed_results,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }

    except HTTPException as he:
        logger.error(f"HTTP error occurred: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
