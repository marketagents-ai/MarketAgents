import sys
import csv
from typing import List, Dict, Any
from pydantic import BaseModel, Field, validator
import argparse
from collections import defaultdict
import cmath
from pydantic import BaseModel, field_validator

class DynamicModel(BaseModel):
    label: str
    value: float
    additional_fields: dict = {}

    @field_validator('value')
    def value_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Value must be positive')
        return v

    @classmethod
    def from_dict(cls, data: dict):
        label = data.pop('label')
        value = float(data.pop('value'))
        return cls(label=label, value=value, additional_fields=data)

class ChartData(BaseModel):
    title: str
    data: List[DynamicModel]

def read_csv(file_path: str) -> ChartData:
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        data = [DynamicModel.from_dict(row) for row in reader]
    return ChartData(title=file_path.split('/')[-1].split('.')[0], data=data)

def create_bar_chart(chart_data: ChartData, width: int = 60, height: int = 20) -> str:
    max_value = max(dp.value for dp in chart_data.data)
    scale = (height - 1) / max_value if max_value > 0 else 1
    
    chart = []
    for dp in chart_data.data:
        bar_height = int(dp.value * scale)
        bar = '█' * bar_height
        label = dp.label[:10].ljust(10)
        value = f"{dp.value:>8.2f}"
        chart.append(f"{label} │{bar:<{height}}│ {value}")
    
    # Add x-axis
    x_axis = '─' * (width - 12)
    chart.append(f"{'':10} └{x_axis}┘")
    
    # Add scale
    scale_line = f"{'':10} {0:>8.2f}{max_value:>{width-20}.2f}"
    chart.append(scale_line)
    
    return '\n'.join(chart)

def create_line_chart(chart_data: ChartData, width: int = 60, height: int = 20) -> str:
    max_value = max(dp.value for dp in chart_data.data)
    min_value = min(dp.value for dp in chart_data.data)
    value_range = max_value - min_value
    scale = (height - 1) / value_range if value_range > 0 else 1
    
    chart = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Add y-axis
    for i in range(height):
        chart[i][0] = '│'
    
    # Add x-axis
    chart[-1] = ['─' for _ in range(width)]
    chart[-1][0] = '└'
    
    # Plot data points
    x_scale = (width - 1) / (len(chart_data.data) - 1)
    for i, dp in enumerate(chart_data.data):
        x = int(i * x_scale)
        y = height - 1 - int((dp.value - min_value) * scale)
        chart[y][x] = '●'
        
        # Add connecting lines
        if i > 0:
            prev_x = int((i-1) * x_scale)
            prev_y = height - 1 - int((chart_data.data[i-1].value - min_value) * scale)
            for ix in range(prev_x + 1, x):
                iy = prev_y + (y - prev_y) * (ix - prev_x) // (x - prev_x)
                chart[iy][ix] = '─' if iy == prev_y else ('/' if iy < prev_y else '\\')
    
    # Add labels and values
    for i, dp in enumerate(chart_data.data):
        x = int(i * x_scale)
        label = dp.label[:3]
        chart[-1][x] = '┴'
        chart[-2][x-1:x+2] = label.center(3)
        
    # Convert to string
    chart_str = '\n'.join(''.join(row) for row in chart)
    
    # Add scale
    scale_line = f"{min_value:<10.2f}{'':{width-20}}{max_value:>10.2f}"
    chart_str += '\n' + scale_line
    
    return chart_str

def create_pie_chart(chart_data: ChartData, diameter: int = 20) -> str:
    total = sum(dp.value for dp in chart_data.data)
    percentages = [dp.value / total for dp in chart_data.data]
    
    chart = [[' ' for _ in range(diameter)] for _ in range(diameter)]
    center = diameter // 2
    
    symbols = ['●', '○', '■', '□', '▲', '△', '◆', '◇']
    
    start_angle = 0
    for i, percentage in enumerate(percentages):
        end_angle = start_angle + percentage * 360
        symbol = symbols[i % len(symbols)]
        
        for y in range(diameter):
            for x in range(diameter):
                angle = (cmath.phase(complex(x - center, y - center)) + cmath.pi) * 180 / cmath.pi
                if start_angle <= angle < end_angle:
                    distance = abs(complex(x - center, y - center))
                    if distance <= center:
                        chart[y][x] = symbol
        
        start_angle = end_angle
    
    return '\n'.join(''.join(row) for row in chart)

def create_mermaid_chart(chart_data: ChartData) -> str:
    mermaid = "graph TD\n"
    total = sum(dp.value for dp in chart_data.data)
    
    for i, dp in enumerate(chart_data.data):
        percentage = (dp.value / total) * 100
        mermaid += f"    {i}[{dp.label}: {dp.value:.2f} ({percentage:.1f}%)]\n"
    
    # Add relationships based on value
    sorted_data = sorted(chart_data.data, key=lambda x: x.value, reverse=True)
    for i in range(len(sorted_data) - 1):
        mermaid += f"    {chart_data.data.index(sorted_data[i])} --> {chart_data.data.index(sorted_data[i+1])}\n"
    
    return mermaid

def main():
    parser = argparse.ArgumentParser(description="Generate ASCII charts from CSV data")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("--chart-type", choices=["bar", "line", "pie", "mermaid"], default="bar", help="Type of chart to generate")
    args = parser.parse_args()

    chart_data = read_csv(args.csv_file)
    
    if args.chart_type == "bar":
        chart = create_bar_chart(chart_data)
    elif args.chart_type == "line":
        chart = create_line_chart(chart_data)
    elif args.chart_type == "pie":
        chart = create_pie_chart(chart_data)
    elif args.chart_type == "mermaid":
        chart = create_mermaid_chart(chart_data)
    
    print(f"# {chart_data.title} - {args.chart_type.capitalize()} Chart\n")
    print("```")
    print(chart)
    print("```\n")
    print("## Data")
    print(f"| Label | Value |")
    print(f"|-------|-------|")
    for dp in chart_data.data:
        print(f"| {dp.label} | {dp.value} |")

if __name__ == "__main__":
    main()