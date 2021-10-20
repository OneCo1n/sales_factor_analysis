from db.dateBase import *
import time

def getMaterialsId():
    db = getDBConnect()

    # sql = "SELECT distinct  material from material_on_sale"
    #sql = "SELECT distinct  material from material_on_sale WHERE material not in (SELECT DISTINCT material FROM pearson)"
    sql = "SELECT DISTINCT a.material FROM material_on_sale a join table_material_sales_num_record b on a.material = b.material WHERE b.record >= 50"
    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()

    df_materials = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_materials = pd.DataFrame([list(i) for i in df_materials], columns=columnNames)

    return df_materials.material