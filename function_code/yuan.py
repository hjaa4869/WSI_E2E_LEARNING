import h5py

def inspect_h5_metadata(h5_path):
    with h5py.File(h5_path, "r") as f:
        print(f"文件: {h5_path}")
        print("内容结构:")
        
        def print_attrs(name, obj):
            print(f"🔹 数据集/组: {name}")
            # 打印属性
            for key, value in obj.attrs.items():
                print(f"   - 属性 {key}: {value}")
        
        f.visititems(print_attrs)

# 使用方法
h5_file = "patches/Camelyon16_patch256_ostu/patches/normal_001.h5"
inspect_h5_metadata(h5_file)
