import os
from lxml import etree

# 解析XML文件
def process_xml(file_path):
    tree = etree.parse(file_path)
    root = tree.getroot()

    # 遍历所有的Annotation元素
    for annotation in root.xpath('//Annotation'):
        coordinates = annotation.xpath('./Coordinates')[0]
        # 获取所有Coordinate Order的值
        coordinate_orders = [int(coord.attrib['Order']) for coord in coordinates.xpath('./Coordinate')]
        
        # 找到最大Order值
        max_order = max(coordinate_orders)
        
        # 获取Order="0"的X和Y值
        first_coordinate = coordinates.xpath('./Coordinate[@Order="0"]')[0]
        x_value = first_coordinate.attrib['X']
        y_value = first_coordinate.attrib['Y']
        
        # 创建新的Coordinate元素
        new_coordinate = etree.Element('Coordinate', Order=str(max_order + 1), X=x_value, Y=y_value)
        
        # 找到最后一个Coordinate并获取它的缩进
        last_coordinate = coordinates.xpath('./Coordinate')[-1]
        indent = ' ' * (len(etree.tostring(last_coordinate, pretty_print=True, encoding='unicode')) - len(etree.tostring(last_coordinate, pretty_print=True, encoding='unicode').lstrip()))
        
        # 将新元素添加到最后一个Coordinate之前
        coordinates.insert(len(coordinates), new_coordinate)

    # 保存更新后的XML
    tree.write(file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')

# 批量处理指定目录下的所有XML文件
def process_directory(directory_path):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # 只处理XML文件
        if os.path.isfile(file_path) and filename.endswith('.xml'):
            process_xml(file_path)
            print(f"Processed: {filename}")

# 测试批量处理目录
process_directory('function_code/xml/training')
