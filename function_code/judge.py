import os
import glob
import h5py
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon

# 解析 XML 文件，提取多边形
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    polygons = []
    
    # 遍历所有的 Annotation 标签
    for annotation in root.findall('.//Annotation'):
        points = []
        # 遍历 Annotation 下的 Coordinates 标签
        for coordinate in annotation.findall('.//Coordinate'):
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            points.append((x, y))
        if points:  # 确保点列表不为空
            polygons.append(Polygon(points))
    
    return polygons

# 判断点是否在多边形内
def is_point_in_polygon(point, polygons):
    point = Point(point)  # 创建一个点对象
    for polygon in polygons:
        if polygon.intersects(point):
            return True
    return False

# 读取 HDF5 文件中的坐标
def read_coords_from_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as file:
        coords_dataset = file['coords'][:]  # 假设数据集名字是 'coords'
    return coords_dataset

# 批量判断 HDF5 文件中的坐标是否在癌变区域内
def batch_check_coords_in_polygons(h5_file_path, xml_file_path, output_txt_path):
    # 读取 XML 文件中的多边形
    polygons = parse_xml(xml_file_path)

    # 读取 HDF5 文件中的坐标
    coords = read_coords_from_h5(h5_file_path)
    
    # 批量判断每个坐标
    results = []
    for coord in coords:
        #result = is_point_in_polygon(coord, polygons)
        coord_xy = (coord[0], coord[1])
        coord_x_plus_256 = (coord[0] + 1024, coord[1])
        coord_y_plus_256 = (coord[0], coord[1] + 1024)
        coord_xy_plus_256 = (coord[0] + 1024, coord[1] + 1024)
        result1 = is_point_in_polygon(coord_x_plus_256, polygons)
        result2 = is_point_in_polygon(coord_y_plus_256, polygons)
        result3 = is_point_in_polygon(coord_xy_plus_256, polygons)
        result4 = is_point_in_polygon(coord_xy, polygons)
        if result1 or result2 or result3 or result4:
            results.append(coord)
    
    # 将结果写入到 txt 文件
    with open(output_txt_path, 'w') as f:
        for coord in results:
            f.write(f" {coord} \n")
    
    return results
    
#base_h5_file = 'h5/tumor_{:03d}.h5'  # 格式化为如 tumor_001.h5、tumor_002.h5
#base_xml_file = 'xml/tumor_{:03d}.xml'  # 格式化为如 tumor_001.xml、tumor_002.xml
#output_txt = 'results/tumor_{:03d}.txt'

#for i in range(1, 2):  # 假设你想遍历 tumor_001 到 tumor_003
    #h5_file = base_h5_file.format(i)
    #xml_file = base_xml_file.format(i)
    #output_file = output_txt.format(i)
    
    #batch_check_coords_in_polygons(h5_file, xml_file, output_file)

    # 这里可以添加你处理 h5_file 和 xml_file 的代码，并将结果输出到 output_file
    #print(f"已处理 HDF5 文件: {h5_file}")
    #print(f"已处理 XML 文件: {xml_file}")
    #print(f"输出到文件: {output_file}")
    
#h5_file = 'h5/tumor_001.h5'

#xml_file = 'xml/tumor_001.xml'

#output_file = 'results/tumor_001.txt'

#batch_check_coords_in_polygons(h5_file, xml_file, output_file)
    
# 定义xml目录和文件名的模式
xml_dir = 'function_code/xml/test'
xml_pattern = os.path.join(xml_dir, '*.xml')

# 获取xml目录下所有xml文件的路径
xml_files = glob.glob(xml_pattern)

# 遍历所有xml文件
for xml_file in xml_files:
    # 提取文件名（去除路径和扩展名）
    file_name = os.path.splitext(os.path.basename(xml_file))[0]
    
    # 构造对应的h5_file和output_file路径
    h5_file = f'patches/Camelyon16_patch256_ostu/patches/{file_name}.h5'
    output_file = f'function_code/results/{file_name}.txt'
    batch_check_coords_in_polygons(h5_file, xml_file, output_file)
