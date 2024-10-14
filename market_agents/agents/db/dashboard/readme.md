# SQL Dashboard

A VERY BASIC web-based SQL dashboard for visualizing and exploring PostgreSQL database data, specifically for use with llm agent frameworks.

## Features

- `TABLE` and `CHART` generation from SQL query results
- Simple data visualization using Chart.js
- Search functionality across tables
- Pagination
- JSON Column Extraction (`--flatten-json` as argparse)
- Pretty JSON formatting

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: HTML, CSS, JavaScript
- Database: PostgreSQL
- Visualization: Chart.js

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install fastapi uvicorn psycopg2-binary python-dotenv
   ```
3. Set up your PostgreSQL database
4. Create a `.env` file with your database credentials:
   ```
   DB_NAME=your_database_name
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_HOST=your_host
   DB_PORT=your_port
   ```
5. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```
6. Open `http://localhost:8000` in your browser

---

## Extending the Dashboard

### Adding New Chart Types

1. Modify the `initChart` function in `index.html`
2. Use Chart.js configurations to add new chart types:

   ```javascript
   chartInstance = new Chart(ctx, {
       type: 'new_chart_type',
       data: {
           // Configure data for the new chart type
       },
       options: {
           // Configure options for the new chart type
       }
   });
   ```

### Implementing Additional SQL Commands

1. Add new API endpoints in `main.py`:

   ```python
   @app.get("/api/new-endpoint")
   async def new_endpoint(param: str = Query(...)):
       conn = None
       cursor = None
       try:
           conn = get_db_connection()
           cursor = conn.cursor(cursor_factory=RealDictCursor)
           
           # Your SQL query here
           query = sql.SQL("YOUR SQL COMMAND").format(sql.Identifier(param))
           
           cursor.execute(query)
           results = cursor.fetchall()
           
           return results
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
       finally:
           if cursor:
               cursor.close()
           if conn:
               conn.close()
   ```

2. Add corresponding JavaScript functions in `index.html` to call the new endpoint and process the data

## Project Structure

- `sqlDASHBOARD.py`: FastAPI backend
- `static/index.html`: Frontend HTML and JavaScript
- `static/styles.css`: CSS styles

This SQL Dashboard provides a foundation for exploring and visualizing PostgreSQL data. Extend it by adding new chart types or implementing additional SQL commands as needed for your specific use case.
