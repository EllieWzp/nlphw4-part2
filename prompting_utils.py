import os
import re



def read_schema(schema_path):
    '''Read and return the schema file contents as a string.'''
    with open(schema_path, 'r') as f:
        return f.read()



def extract_sql_query(response):
    '''Extract a SQL query from a raw model response using simple heuristics.'''
    if not response:
        return ""
    text = response
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).replace("```", "")
    m = re.search(r"(SELECT|WITH|UPDATE|DELETE|INSERT)\b[\s\S]*", text, flags=re.IGNORECASE)
    if not m:
        return ""
    sql = m.group(0).strip()
    pos = sql.find(";")
    if pos != -1:
        sql = sql[:pos+1].strip()
    if not sql.endswith(";"):
        sql += ";"
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")