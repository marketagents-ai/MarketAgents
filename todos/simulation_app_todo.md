# Simulation Dashboard TODO (using Dash)

## 1. Set up the Dash Framework
- [ ] Install Dash and its dependencies: `pip install dash dash-bootstrap-components plotly pandas`
- [ ] Create a new file `simulation_dashboard.py`
- [ ] Set up the basic structure of the Dash application

## 2. Define Dashboard Components
- [ ] Create a layout for the dashboard with the following components:
  - [ ] Market State Table
  - [ ] Order Book Table
  - [ ] Trade History Table
  - [ ] Supply and Demand Chart
  - [ ] Price vs Trade Chart
  - [ ] Cumulative Quantity vs Surplus Chart
  - [ ] Simulation Controls (Start, Stop)

## 3. Integrate Dashboard with Orchestrator
- [ ] Modify `orchestrator.py` to include a reference to the dashboard
- [ ] Create a data source function in the Orchestrator class to provide dashboard data:
  - [ ] `get_current_simulation_state()`

## 4. Implement Real-time Data Updates
- [ ] Set up a callback in Dash for periodic updates using dcc.Interval
- [ ] Implement toggle_simulation callback to control simulation start/stop

## 5. Implement Market State Table
- [ ] Create a function `generate_market_state_table()` to display current simulation state
- [ ] Include fields: Current Step, Max Steps, CE Price, CE Quantity, Efficiency, Total Utility

## 6. Implement Order Book Table
- [ ] Create a function `generate_order_book_table()` to display current bids and asks
- [ ] Separate tables for bids and asks, including Agent ID, Price, and Quantity

## 7. Implement Trade History Table
- [ ] Create a function `generate_trade_history_table()` to display recent trades
- [ ] Include fields: Trade ID, Buyer ID, Seller ID, Price, Quantity, Round

## 8. Visualize Supply and Demand Curves
- [ ] Implement `generate_supply_demand_chart()` function
- [ ] Plot supply and demand curves using Plotly
- [ ] Add CE Price and CE Quantity indicators

## 9. Create Price vs Trade Chart
- [ ] Implement `generate_price_vs_trade_chart()` function
- [ ] Plot price for each trade over time

## 10. Implement Cumulative Quantity vs Surplus Chart
- [ ] Create `generate_cumulative_quantity_surplus_chart()` function
- [ ] Plot cumulative quantity and surplus on separate y-axes

## 11. Implement Simulation Controls
- [ ] Add Start and Stop buttons
- [ ] Create callbacks to control the simulation state

## 12. Optimize Performance
- [ ] Use efficient data structures for storing and updating simulation data
- [ ] Implement partial updates where possible to reduce data transfer

## 13. Error Handling and Logging
- [ ] Implement error handling for dashboard components and data fetching
- [ ] Set up logging for dashboard events and errors

## 14. Testing
- [ ] Write unit tests for dashboard components and data processing functions
- [ ] Perform integration testing with the Orchestrator

## 15. Documentation
- [ ] Document the dashboard setup and usage
- [ ] Add inline comments explaining complex parts of the code

## 16. Polish and Refine
- [ ] Improve the visual design using Dash Bootstrap Components
- [ ] Ensure responsive design for different screen sizes
- [ ] Add loading spinners for charts and tables

## 17. Additional Features (if time permits)
- [ ] Implement a configuration panel for adjusting simulation parameters
- [ ] Add a summary statistics section
- [ ] Create a heatmap visualization of agent activities

## 18. Deployment Preparation
- [ ] Implement a function to kill any existing process on the dashboard port
- [ ] Set up error handling for dashboard startup and shutdown
