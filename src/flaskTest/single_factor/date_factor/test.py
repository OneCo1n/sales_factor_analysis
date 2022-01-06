import datetime

# 判断 2019年9月13号 是不是节假日（中秋节）
from chinese_calendar import is_workday, is_holiday

april_last = datetime.date(2021, 10, 1)
print(is_holiday(april_last))