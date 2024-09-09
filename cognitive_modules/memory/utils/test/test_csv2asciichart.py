"""
pytest -v -s cognitive_modules/memory/utils/test/test_csv2asciichart.py
"""


import pytest
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(project_root)

from cognitive_modules.memory.utils.csv2asciichart import (
    read_csv, create_bar_chart, create_line_chart, create_pie_chart, create_mermaid_chart, ChartData, DynamicModel
)
import tempfile
import os

@pytest.fixture
def sample_csv():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        content = "label,value\nA,10\nB,20\nC,15\n"
        tmp.write(content)
        tmp.flush()
        print("\nGenerated CSV content:")
        print(content)
        yield tmp.name
    os.unlink(tmp.name)

def print_chart_data(chart_data):
    print("## Data")
    print("| Label | Value |")
    print("|-------|-------|")
    for dp in chart_data.data:
        print(f"| {dp.label} | {dp.value} |")

def test_read_csv(sample_csv):
    chart_data = read_csv(sample_csv)
    assert isinstance(chart_data, ChartData)
    assert len(chart_data.data) == 3
    assert chart_data.data[0].label == 'A' and chart_data.data[0].value == 10
    print("\nRead CSV data:")
    print_chart_data(chart_data)

def test_create_bar_chart(sample_csv):
    chart_data = read_csv(sample_csv)
    chart = create_bar_chart(chart_data)
    assert isinstance(chart, str)
    assert '██' in chart
    print(f"\n# {chart_data.title} - Bar Chart\n")
    print("```")
    print(chart)
    print("```")
    print_chart_data(chart_data)

def test_create_line_chart(sample_csv):
    chart_data = read_csv(sample_csv)
    chart = create_line_chart(chart_data)
    assert isinstance(chart, str)
    assert '●' in chart
    print(f"\n# {chart_data.title} - Line Chart\n")
    print("```")
    print(chart)
    print("```")
    print_chart_data(chart_data)

def test_create_pie_chart(sample_csv):
    chart_data = read_csv(sample_csv)
    chart = create_pie_chart(chart_data)
    assert isinstance(chart, str)
    assert '●' in chart
    print(f"\n# {chart_data.title} - Pie Chart\n")
    print("```")
    print(chart)
    print("```")
    print_chart_data(chart_data)

def test_create_mermaid_chart(sample_csv):
    chart_data = read_csv(sample_csv)
    chart = create_mermaid_chart(chart_data)
    assert isinstance(chart, str)
    assert 'graph TD' in chart
    print(f"\n# {chart_data.title} - Mermaid Chart\n")
    print("```mermaid")
    print(chart)
    print("```")
    print_chart_data(chart_data)

def test_dynamic_model_validation():
    with pytest.raises(ValueError):
        DynamicModel(label='Test', value=-1)
    
    model = DynamicModel(label='Test', value=10, additional_fields={'extra': 'data'})
    assert model.label == 'Test'
    assert model.value == 10
    assert model.additional_fields == {'extra': 'data'}
    print("\nDynamic Model:")
    print(f"Label: {model.label}")
    print(f"Value: {model.value}")
    print(f"Additional Fields: {model.additional_fields}")