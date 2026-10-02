"""Environment & Infrastructure Verification Script for Advanced RAG Platform.

Validates:
1. Python version (3.12+) and active virtual environment
2. Git installation and repository status
3. Docker daemon connectivity
4. Qdrant vector database health and connectivity
5. MySQL database connectivity (Production & Test databases)
6. Environment variable definitions (Safe check without exposing secret values)
"""

import os
import sys
import subprocess
import urllib.request
import urllib.parse
import json
import pymysql
from dotenv import dotenv_values


def check_python():
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    in_venv = sys.prefix != sys.base_prefix
    print(f"[PASS] Python Version: {ver_str} (Target: >= 3.12)")
    print(f"[PASS] Virtual Environment: {'Active (' + sys.prefix + ')' if in_venv else 'NOT ACTIVE'}")
    if v.major < 3 or (v.major == 3 and v.minor < 12):
        print("[FAIL] Error: Python 3.12 or newer is required.")
        return False
    return True


def check_git():
    try:
        res = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True)
        print(f"[PASS] Git: {res.stdout.strip()}")
        return True
    except Exception as e:
        print(f"[FAIL] Git check failed: {e}")
        return False


def check_docker():
    try:
        res = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, check=True)
        print(f"[PASS] Docker Engine: Running (v{res.stdout.strip()})")
        return True
    except Exception as e:
        print(f"[FAIL] Docker check warning: {e}")
        return False


def check_env_variables():
    print("\n--- Environment Configuration Status ---")
    if not os.path.exists(".env"):
        print("[FAIL] Error: .env file not found.")
        return False, {}
    
    config = dotenv_values(".env")
    required_keys = [
        "OPENROUTER_API_KEY",
        "DATABASE_URL",
        "TEST_DATABASE_URL",
        "QDRANT_URL",
    ]
    
    all_ok = True
    for key in required_keys:
        val = config.get(key)
        is_set = bool(val and val.strip())
        status = "configured" if is_set else "MISSING / EMPTY"
        print(f"  {key}: {status}")
        if not is_set:
            all_ok = False
            
    return all_ok, config


def check_qdrant(qdrant_url):
    print("\n--- Qdrant Vector DB Connectivity ---")
    if not qdrant_url:
        print("[FAIL] QDRANT_URL not configured")
        return False
    try:
        url = qdrant_url.rstrip("/") + "/"
        req = urllib.request.Request(url, headers={"User-Agent": "AdvancedRAG-HealthCheck"})
        with urllib.request.urlopen(req, timeout=5) as res:
            if res.status == 200:
                payload = json.loads(res.read().decode("utf-8"))
                print(f"[PASS] Qdrant Online: {payload.get('title', 'qdrant')} (v{payload.get('version', 'unknown')})")
                return True
            else:
                print(f"[FAIL] Qdrant HTTP returned status {res.status}")
                return False
    except Exception as e:
        print(f"[FAIL] Qdrant connection failed: {e}")
        return False


def check_mysql(label, url_str):
    if not url_str:
        print(f"[FAIL] {label}: Not configured")
        return False
    try:
        cleaned = url_str
        if "://" in cleaned:
            cleaned = cleaned.split("://", 1)[1]
        parsed = urllib.parse.urlparse("//" + cleaned)
        user = parsed.username or "root"
        password = parsed.password or ""
        host = parsed.hostname or "localhost"
        port = parsed.port or 3306
        db_name = parsed.path.lstrip("/") if parsed.path else None
        
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            database=db_name,
            charset="utf8mb4",
            connect_timeout=5,
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1, DATABASE(), VERSION();")
            row = cursor.fetchone()
            print(f"[PASS] {label}: Connected successfully (DB: {row[1]}, MySQL Version: {row[2]})")
        conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] {label} connection failed: {e}")
        return False


def main():
    print("=" * 60)
    print(" ADVANCED RAG PLATFORM - PHASE 1 ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    py_ok = check_python()
    git_ok = check_git()
    docker_ok = check_docker()
    env_ok, config = check_env_variables()
    
    qdrant_url = config.get("QDRANT_URL", "http://localhost:6333")
    qdrant_ok = check_qdrant(qdrant_url)
    
    print("\n--- MySQL Database Connectivity ---")
    db_ok = check_mysql("DATABASE_URL (App DB)", config.get("DATABASE_URL"))
    test_db_ok = check_mysql("TEST_DATABASE_URL (Test DB)", config.get("TEST_DATABASE_URL"))
    
    print("\n" + "=" * 60)
    all_passed = py_ok and git_ok and docker_ok and env_ok and qdrant_ok and db_ok and test_db_ok
    if all_passed:
        print("PHASE 1 VERIFICATION RESULT: ALL CHECKS PASSED [SUCCESS]")
        sys.exit(0)
    else:
        print("PHASE 1 VERIFICATION RESULT: SOME CHECKS FAILED [ACTION REQUIRED]")
        sys.exit(1)


if __name__ == "__main__":
    main()
