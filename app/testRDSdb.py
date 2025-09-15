import pymysql
# MYSQL_HOST = "myfastapi-db.cv4m82mkexvc.ap-south-1.rds.amazonaws.com"
MYSQL_HOST = "myrdsmysql.cv4m82mkexvc.ap-south-1.rds.amazonaws.com"
MYSQL_USER = "root"
MYSQL_PASSWORD = "test1234"
MYSQL_DATABASE = "mysql"

try:
    conn = pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor
    )
    cursor = conn.cursor()
    cursor.execute("SELECT VERSION() AS version")
    result = cursor.fetchone()
    print(f"Connected! MySQL version: {result['version']}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
