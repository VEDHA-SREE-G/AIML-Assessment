# Log File Analyzer

## Objective

The objective of this project is to analyze application log files, identify ERROR and WARNING entries, group them by module, and generate a CSV report showing error frequencies.

## Features

* Reads application log files
* Detects ERROR and WARNING log entries
* Extracts module names from log records
* Counts occurrences of ERROR and WARNING messages
* Generates a CSV report summarizing the results

## Input Format

Example log entry:

2025-07-10 10:03:22 ERROR Database connection failed Module=Database

## Requirements

Install required libraries:

pip install pandas

## How to Run

Run the script:

python Q2.py

## Output

The script generates:

* log_report.csv

Example output:

Module,ERROR Count,WARNING Count
Auth,0,1
Database,2,0
Storage,1,1

## Technologies Used

* Python 3.x
* Pandas

## Author

Vedha Sree G
