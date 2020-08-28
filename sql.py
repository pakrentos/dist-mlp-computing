import sqlite3 as sql
from sqlite3 import connect
from datetime import datetime

def create_users(conn):
    conn = connect('users.db')
    command = """
    CREATE TABLE IF NOT EXISTS users (
	uuid CHAR(36) PRIMARY KEY,
   	processed INT NOT NULL DEFAULT 0,
    first_time REAL,
	last_time REAL,
    mean REAL,
    version TEXT);
    """
    conn.execute(command)
    conn.commit()

def insert_user(conn, uuid, version):
    conn = connect('users.db')
    command = f"""
    INSERT INTO users VALUES ("{uuid}", 0, null, null, null, "{version}");
    """
    conn.execute(command)
    conn.commit()

def get_user(conn, uuid):
    conn = connect('users.db')
    command = f"""
    SELECT * FROM users WHERE uuid = "{uuid}";
    """
    return conn.execute(command).fetchone()

def update_user(conn, uuid, processed, first_time, last_time, mean, version):
    conn = connect('users.db')
    if mean is None:
        mean = 'null'
    if last_time is None:
        last_time = 'null'
    if first_time is None:
        first_time = 'null'
    command = f"""
    Update users set processed = {processed}, first_time = "{first_time}", last_time = "{last_time}", mean = {mean}, version="{version}" where uuid = "{uuid}";
    """
    conn.execute(command)
    conn.commit()

def update_last_time(conn, uuid):
    processed, first_time, last_time, mean, version = get_user(conn, uuid)[1:]
    processed += 1
    last_time = datetime.now().timestamp()
    proc_time = last_time - first_time if first_time is not None else 12*60
    if mean is not None:
        mean = (mean*(processed-1) + proc_time)/processed
    else:
        mean = proc_time
    update_user(conn, uuid, processed, first_time, last_time, mean, version)

def update_first_time(conn, uuid):
    processed, first_time, last_time, mean, version = get_user(conn, uuid)[1:]
    new_time = datetime.now().timestamp()
    update_user(conn, uuid, processed, new_time, last_time, mean, version)

def update_version(conn, uuid, new_v):
    processed, first_time, last_time, mean, version = get_user(conn, uuid)[1:]
    update_user(conn, uuid, processed, first_time, last_time, mean, new_v)
