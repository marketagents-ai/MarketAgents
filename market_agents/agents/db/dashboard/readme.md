# SQL Dashboard for LLM Agent Frameworks

A web-based SQL dashboard specifically designed for visualizing and exploring PostgreSQL database data, with enhanced features for LLM agent frameworks and JSON handling capabilities.

## Key Features

### Data Visualization
- Dual view modes: TABLE and CHART visualization
- Dynamic chart type selection based on data characteristics
- Interactive Plotly.js charts with advanced features:
  - Automatic chart type detection (scatter, bar, time series)
  - Smart data distribution analysis
  - Responsive design with hover tooltips
  - Export capabilities
- Pagination with configurable page sizes
- Column sorting capabilities
- Markdown export functionality

### JSON Handling
- Smart JSON column detection and processing
- Nested JSON path querying
- Pretty JSON formatting in table view
- Dynamic JSON schema detection

### Search & Filter
- Full-text search across all columns
- Type-aware searching (handles dates, numbers, text)
- JSON content searching
- Real-time search with debouncing

### Database Integration
- PostgreSQL integration with JSON/JSONB support
- Dynamic table and column detection
- Efficient pagination with large datasets
- Automatic data type detection
- Connection pooling
- Error handling and recovery

## Tech Stack

### Backend
- FastAPI (Python)
- Psycopg2 for PostgreSQL connectivity
- SQLAlchemy ORM (optional)
- Python-dotenv for configuration
- Pydantic for data validation

### Frontend
- Modern HTML5/CSS3
- Vanilla JavaScript (no framework dependencies)
- Plotly.js for advanced visualizations
- Support for custom styling

### Database
- PostgreSQL with JSON/JSONB support
- Efficient query optimization
- Connection pooling

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd sql-dashboard
   ```

2. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn psycopg2-binary python-dotenv plotly
   ```

3. Configure your PostgreSQL database connection in `.env`:
   ```env
   DB_NAME=your_database
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_HOST=your_host
   DB_PORT=5432
   ```

4. Launch the application:
   ```bash
   python sqlDASHBOARD.py [--flatten-json]
   ```

5. Access the dashboard at `http://localhost:8001`

## API Endpoints

### Core Endpoints
- `GET /api/get-tables`: List all available database tables
- `GET /api/column-names`: Get column information for a specific table
- `GET /api/metrics-data`: Fetch paginated data with support for JSON columns
- `GET /api/search`: Perform full-text search across tables

### Request Parameters
- `table_name`: Target table name
- `x_column`, `y_column`: Chart axis specifications
- `page`, `page_size`: Pagination controls
- `search_term`: Search query string
- `full_table`: Boolean for complete table data

## Advanced Features

### JSON Path Querying
The dashboard supports complex JSON path queries:
```python
data.nested.field[0].value
```

### Dynamic Chart Selection
The system automatically selects the most appropriate chart type based on:
- Data type analysis (categorical, numerical, temporal)
- Distribution patterns
- Data density
- Column relationships

### Data Export Options
- Markdown table format
- JSON data export
- Chart image export (PNG)

## Security Considerations

- SQL injection prevention through parameterized queries
- Environmental variable protection
- Sanitized JSON handling
- Error message sanitization
- Rate limiting capabilities

## Extending the Dashboard

### Adding Custom Chart Types
```javascript
function initChart(data, xColumn, yColumn) {
    // Configure new chart type
    const customTrace = {
        x: xData,
        y: yData,
        type: 'custom-chart',
        mode: 'custom-mode',
        // Additional configuration
    };
}
```

### Custom JSON Processing
```python
def process_json_data(value, flatten=False):
    # Add custom JSON processing logic
    if flatten:
        return flatten_json(json_data)
    return json.dumps(json_data, indent=2)
```

## Performance Optimization

- Connection pooling for database connections
- Debounced search functionality
- Efficient JSON parsing and flattening
- Pagination for large datasets
- Smart data type detection caching

# Directory Structure
- db_dash/
  - static/
    - index.html
    - styles.css
  - sqlDASHBOARD.py
  - readme.md
